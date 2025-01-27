import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn import metrics
from transformers import AutoModelForAudioClassification
import numpy as np
from collections import OrderedDict
from torchmetrics import MeanMetric, MaxMetric, Accuracy
import torchmetrics.functional as tmf

from model.linear import FeedforwardModel
from model.linear_small import FeedforwardModelSmall
from model.linear_attn_ck import FeedforwardModelAttnCK
from model.linear_mt import FeedforwardModelMT
from model.linear_mt_attn_ck import FeedforwardModelMTAttnCK

import logging
import yaml
from omegaconf import DictConfig

import torch
from torch.distributed import all_gather, get_world_size
# from lion_pytorch import Lion
from torch_optimizer import RAdam

def gather_all_results(tensor):
    """
    Gather tensors from all GPUs in distributed training.
    """
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(get_world_size())]
    all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0)

# torch.set_float32_matmul_precision('medium')

log = logging.getLogger(__name__)
class MusicClassifier(pl.LightningModule):
    def __init__(self, cfg: DictConfig, output_file = None):
        super(MusicClassifier, self).__init__()
        self.cfg = cfg
        self.encoder = cfg.model.encoder
        self.classifier = cfg.model.classifier
        self.lr = cfg.model.lr
        self.output_file = output_file
        self.kd = cfg.model.kd
        self.kd_weight = cfg.model.kd_weight
        self.kd_temperature = self.cfg.model.kd_temperature

        layer_size = len(self.cfg.model.layers)
        mert_dim = 768 * layer_size

        self.feature_dim_dict = {
            "MERT": mert_dim
        }

        encoders = self.encoder.split("-")
        self.input_size = sum(self.feature_dim_dict[encoder] for encoder in encoders)
        self.num_datasets = len(self.cfg.datasets)

        if "mt" in self.classifier:
            if self.num_datasets < 2:
                raise Exception("Error: Dataset size >= 2 needed for MT classifier")
            classifiers = {
                "linear-mt-attn-ck": FeedforwardModelMTAttnCK,
            }
            if self.classifier in classifiers:
                self.model = classifiers[self.classifier](
                    input_size=self.input_size,
                    output_size_classification=56,
                    output_size_regression=2
                )
            else:
                raise Exception(f"Unknown classifier: {self.classifier}")          
        else:
            if self.num_datasets  >= 2:
                raise Exception(f"Error: Dataset size == 1 needed for classifier") 
            dataset_name = self.cfg.datasets[0]
            self.output_size = self.cfg.dataset[dataset_name].output_size
            classifiers = {
                "linear": FeedforwardModel,
                "linear-attn-ck": FeedforwardModelAttnCK
            }

            if self.classifier in classifiers:
                self.model = classifiers[self.classifier](input_size=self.input_size, output_size=self.output_size)
            else:
                raise Exception(f"Unknown classifier: {self.classifier}")
        
                
        if self.kd:
            self.teacher_models = {}
            
            for dataset in self.cfg.datasets:
                self.output_size = self.cfg.dataset[dataset].output_size
                teacher_model_path = getattr(self.cfg, f"checkpoint_{dataset}", None)
                
                if teacher_model_path:
                    # Create a new teacher model instance
                    teacher_model = FeedforwardModelAttnCK(
                        input_size=self.input_size,
                        output_size=self.output_size,
                    )
                    
                    # Load the checkpoint
                    checkpoint = torch.load(teacher_model_path, map_location=self.device, weights_only=False)
                    state_dict = checkpoint["state_dict"]
                    
                    # Adjust the keys in the state_dict
                    state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}
                    
                    # Filter state_dict to match model's keys
                    model_keys = set(teacher_model.state_dict().keys())
                    filtered_state_dict = {key: value for key, value in state_dict.items() if key in model_keys}
                    
                    # Load the filtered state_dict and set the model to evaluation mode
                    teacher_model.load_state_dict(filtered_state_dict)
                    teacher_model.to(self.device)

                    teacher_model.eval()
                    
                    # Store the teacher model in the dictionary with the dataset name as the key
                    self.teacher_models[dataset] = teacher_model

        probas = torch.from_numpy(np.load("dataset/jamendo/meta/probas_train.npy"))
        pos_weight = torch.tensor(1.) / probas
        weight = torch.tensor(2.) / (torch.tensor(1.) + pos_weight)
        
        self.loss_fn_classification = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,reduction="mean",weight=weight
        )
        self.loss_fn_classification_eval = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight,reduction="none",weight=weight
        )
        
        self.loss_fn_regression = nn.MSELoss()  

        self.loss_kd = nn.KLDivLoss(reduction="batchmean")

        self.prd_array = []
        self.gt_array = []
        self.song_array = []

        self.prd_array_va = []
        self.gt_array_va = []
        self.song_array_va = []

        self.validation_predictions = []
        self.validation_targets = []
        self.validation_results = {'preds': [], 'gt': []}

        self.trn_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, model_input_dic, output_idx = 0):
        if "mt" in self.classifier:
            classification_output, regression_output = self.model(model_input_dic)
            if output_idx == 0:
                return classification_output
            elif output_idx == 1:
                return regression_output
            elif output_idx == 2:
                return classification_output, regression_output
        else:
            output = self.model(model_input_dic)
            return output
        
    def compute_classification_loss(self, model_input_dic, y_mood):
        classification_logits = self(model_input_dic, 0)
        loss= self.loss_fn_classification(classification_logits, y_mood)
        return loss

    def compute_regression_loss(self, model_input_dic, y_va):
        regression_output = self(model_input_dic, 1)
        loss = self.loss_fn_regression(regression_output, y_va)
        return loss
    
    def compute_mt_loss(self, model_input_dic, y_mood, y_va):
        classification_logits, regression_output = self(model_input_dic, 2)
        loss_classification = self.loss_fn_classification(classification_logits, y_mood)
        loss_regression = self.loss_fn_regression(regression_output, y_va)
        return loss_classification, loss_regression
    

    def compute_kd_loss(self, model_input_dic, y_mood, y_va, dataset_name):
        """
        Compute knowledge distillation loss for a given dataset.
        """
        # Forward pass through student model
        s_logits_mood, s_logits_va = self(model_input_dic, 2)

        # Compute student losses
        s_loss_mood = self.loss_fn_classification(s_logits_mood, y_mood)
        s_loss_va = self.loss_fn_regression(s_logits_va, y_va)

        # Get the corresponding teacher model for the dataset
        teacher_model = self.teacher_models.get(dataset_name)
        teacher_model.to(self.device)

        # Ensure teacher model exists
        if teacher_model is None:
            raise ValueError(f"No teacher model found for dataset: {dataset_name}")

        with torch.no_grad():
            # Forward pass through teacher model
            t_logits = teacher_model(model_input_dic)

        # Compute knowledge distillation losses
        t_probs = torch.softmax(t_logits / self.kd_temperature, dim=-1)
        if dataset_name == "jamendo":
            s_probs_mood = torch.log_softmax(s_logits_mood / self.kd_temperature, dim=-1)
            kd_loss = self.loss_kd(s_probs_mood, t_probs)
        else:
            s_probs_va = torch.log_softmax(s_logits_va / self.kd_temperature, dim=-1)
            kd_loss = self.loss_kd(s_probs_va, t_probs)

        return kd_loss, s_loss_mood, s_loss_va
    
    def handle_dataset(self, dataset_name, batch, losses, total_loss, stage):
        dataset_batch = batch[dataset_name]

        model_input_dic = {}
        model_input_dic["x_mert"] = dataset_batch["x_mert"]
        model_input_dic["x_chord"] = dataset_batch["x_chord"]
        model_input_dic["x_chord_root"] = dataset_batch["x_chord_root"]
        model_input_dic["x_chord_attr"] = dataset_batch["x_chord_attr"]
        model_input_dic["x_key"] = dataset_batch["x_key"]

        if "mt" in self.classifier:            
            if dataset_name == "jamendo":
                y_mood = dataset_batch["y_mood"]
                y_va = dataset_batch["y_va"]
                if self.kd:
                    kd_loss, s_loss_mood, s_loss_va = self.compute_kd_loss(model_input_dic, y_mood, y_va, dataset_name)
                    if stage == "train":
                        losses['loss_mood'] = s_loss_mood

                        total_loss += self.kd_weight * kd_loss + (1 - self.kd_weight) * s_loss_mood
                    else:
                        losses['loss_mood'] = s_loss_mood
                        total_loss += s_loss_mood
                else:
                    s_loss_mood, s_loss_va = self.compute_mt_loss(model_input_dic, y_mood, y_va)
                    if stage == "train":
                        losses['loss_mood'] = s_loss_mood
                        total_loss += s_loss_mood 
                    else:
                        losses['loss_mood'] = s_loss_mood
                        total_loss += s_loss_mood
            else:
                y_mood = dataset_batch["y_mood"]
                y_va = dataset_batch["y_va"]

                if self.kd:
                    kd_loss, s_loss_mood, s_loss_va = self.compute_kd_loss(model_input_dic, y_mood, y_va, dataset_name)
                    if stage == "train":
                        losses['loss_va'] = s_loss_va
                        total_loss += self.kd_weight * kd_loss + (1 - self.kd_weight) * s_loss_va
                    else:
                        losses['loss_va'] = s_loss_va
                        total_loss += s_loss_va
                else:
                    s_loss_mood, s_loss_va = self.compute_mt_loss(model_input_dic, y_mood, y_va)
                    if stage == "train":
                        losses['loss_va'] = s_loss_va
                        total_loss += s_loss_va
                    else:
                        losses['loss_va'] = s_loss_va
                        total_loss += s_loss_va
        else:
            if dataset_name == "jamendo":
                y_mood = dataset_batch["y_mood"]
                loss_classification = self.compute_classification_loss(model_input_dic, y_mood)
                losses['loss_mood'] = loss_classification
                total_loss += loss_classification
            else:
                y_va = dataset_batch["y_va"]
                loss_regression = self.compute_regression_loss(model_input_dic, y_va)
                losses['loss_va'] = loss_regression
                total_loss += loss_regression

        return total_loss

    def training_step(self, batch, batch_idx):
        total_loss = 0
        losses = {}
        datasets = ["jamendo",  "deam", "emomusic", "pmemo"]
        
        for dataset in datasets:
            if dataset in batch and batch[dataset] is not None:
                total_loss = self.handle_dataset(dataset, batch, losses, total_loss, "train")
        
        batch_size = batch[next(iter(batch))]["x_mert"].size(0)

        self.log('train_loss_mood', losses.get('loss_mood', 0), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        self.log('train_loss_va', losses.get('loss_va', 0), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        self.log('train_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)

        return total_loss
    
    def validation_step(self, batch, batch_idx):

        total_loss = 0
        losses = {}
        datasets = ["jamendo", "deam", "emomusic", "pmemo"]

        for dataset in datasets:
            if dataset in batch and batch[dataset] is not None:
                total_loss = self.handle_dataset(dataset, batch, losses, total_loss, "val")
        
        batch_size = batch[next(iter(batch))]["x_mert"].size(0)

        self.log('val_loss_mood', losses.get('loss_mood', 0), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        self.log('val_loss_va', losses.get('loss_va', 0), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        return total_loss

    def test_step(self, batch, batch_idx):
        total_loss = 0
        losses = {}
        datasets = ["jamendo",  "deam", "emomusic", "pmemo"]

        for dataset in datasets:
            if dataset in batch and batch[dataset] is not None:
                dataset_batch = batch[dataset]

                model_input_dic = {}
                model_input_dic["x_mert"] = dataset_batch["x_mert"]
                model_input_dic["x_chord"] = dataset_batch["x_chord"]
                model_input_dic["x_chord_root"] = dataset_batch["x_chord_root"]
                model_input_dic["x_chord_attr"] = dataset_batch["x_chord_attr"]
                
                model_input_dic["x_key"] = dataset_batch["x_key"]

                if dataset == "jamendo":
                    y_mood = dataset_batch["y_mood"]
                    classification_logits = self(model_input_dic, 0)
                    
                    loss_classification = self.loss_fn_classification(classification_logits, y_mood)
                    total_loss += loss_classification

                    probs = torch.sigmoid(classification_logits)
                    if not hasattr(self, 'jamendo_results'):
                        self.jamendo_results = {'preds': [], 'gt': [], 'paths': []}
                    
                    self.jamendo_results['preds'].extend(probs.detach().cpu().numpy())
                    self.jamendo_results['gt'].extend(y_mood.detach().cpu().numpy())
                    self.jamendo_results['paths'].extend(dataset_batch["path"])

                    losses['test_loss_mood'] = loss_classification

                else:  # Handle regression for all other datasets
                    if batch[dataset] is not None:
                        y_va = dataset_batch["y_va"]
                        regression_output = self(model_input_dic, 1)

                        loss_regression = self.loss_fn_regression(regression_output, y_va)
                        total_loss += loss_regression

                        # Track results separately for each dataset
                        if not hasattr(self, f'{dataset}_results'):
                            setattr(self, f'{dataset}_results', {'preds': [], 'gt': [], 'paths': []})
                        
                        dataset_results = getattr(self, f'{dataset}_results')
                        dataset_results['preds'].extend(regression_output.detach().cpu().numpy())
                        dataset_results['gt'].extend(y_va.detach().cpu().numpy())
                        dataset_results['paths'].extend(dataset_batch["path"])

                        losses['test_loss_va'] = loss_regression

        batch_size = batch[next(iter(batch))]["x_mert"].size(0)

        # Log the classification and regression losses
        self.log('test_loss_mood', losses.get('test_loss_mood', 0), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        self.log('test_loss_va', losses.get('test_loss_va', 0), on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)
        self.log('test_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True, batch_size=batch_size)

        return total_loss

    def on_test_end(self):
        output_dic = {}

        # Jamendo classification metrics (AUC and PR AUC)
        if hasattr(self, 'jamendo_results') and self.jamendo_results['preds']:
            roc_auc, pr_auc = self.get_auc(self.jamendo_results['preds'], self.jamendo_results['gt'])

            roc_auc = roc_auc.item()
            pr_auc = pr_auc.item()

            log.info('*** Display ROC_AUC_MACRO scores (Jamendo) ***')
            log.info(f"ROC_AUC_MACRO: {round(roc_auc, 4)}")
            log.info(f"PR_AUC_MACRO: {round(pr_auc, 4)}")

            if self.output_file is not None:
                with open(self.output_file, 'a') as f:
                    f.write(f"ROC_AUC_MACRO (Jamendo): {round(roc_auc, 4)}\n")
                    f.write(f"PR_AUC_MACRO (Jamendo): {round(pr_auc, 4)}\n")

            output_dic["test_roc_auc_jamendo"] = round(roc_auc, 4)
            output_dic["test_pr_auc_jamendo"] = round(pr_auc, 4)

        # Metrics for each regression dataset (DMDD, DEAM, EmoMusic, PMEmo)
        for dataset in ["deam", "emomusic", "pmemo"]:
            dataset_results = getattr(self, f'{dataset}_results', None)

            if dataset_results and dataset_results['preds']:
                preds = torch.tensor(np.array(dataset_results['preds']))
                gts = torch.tensor(np.array(dataset_results['gt']))

                # Assuming valence is the first column and arousal is the second
                preds_valence = preds[:, 0]
                preds_arousal = preds[:, 1]
                gts_valence = gts[:, 0]
                gts_arousal = gts[:, 1]

                rmse = torch.sqrt(tmf.mean_squared_error(preds, gts))
                r2 = tmf.r2_score(preds, gts)

                # Calculate metrics for valence
                rmse_valence = torch.sqrt(tmf.mean_squared_error(preds_valence, gts_valence))
                r2_valence = tmf.r2_score(preds_valence, gts_valence)

                # Calculate metrics for arousal
                rmse_arousal = torch.sqrt(tmf.mean_squared_error(preds_arousal, gts_arousal))
                r2_arousal = tmf.r2_score(preds_arousal, gts_arousal)

                log.info(f'*** Display RMSE and R² scores ({dataset.upper()}) ***')
                log.info(f"RMSE: {round(rmse.item(), 4)}")
                log.info(f"R²: {round(r2.item(), 4)}")
                log.info(f"Valence - RMSE: {round(rmse_valence.item(), 4)}, R²: {round(r2_valence.item(), 4)}")
                log.info(f"Arousal - RMSE: {round(rmse_arousal.item(), 4)}, R²: {round(r2_arousal.item(), 4)}")

                if self.output_file is not None:
                    with open(self.output_file, 'a') as f:
                        f.write(f"RMSE ({dataset.upper()}): {round(rmse.item(), 4)}\n")
                        f.write(f"R² ({dataset.upper()}): {round(r2.item(), 4)}\n")
                        f.write(f"Valence - RMSE ({dataset.upper()}): {round(rmse_valence.item(), 4)}\n")
                        f.write(f"Valence - R² ({dataset.upper()}): {round(r2_valence.item(), 4)}\n")
                        f.write(f"Arousal - RMSE ({dataset.upper()}): {round(rmse_arousal.item(), 4)}\n")
                        f.write(f"Arousal - R² ({dataset.upper()}): {round(r2_arousal.item(), 4)}\n")

                output_dic[f"test_rmse_{dataset}"] = round(rmse.item(), 4)
                output_dic[f"test_r2_{dataset}"] = round(r2.item(), 4)
                output_dic[f"test_rmse_valence_{dataset}"] = round(rmse_valence.item(), 4)
                output_dic[f"test_r2_valence_{dataset}"] = round(r2_valence.item(), 4)
                output_dic[f"test_rmse_arousal_{dataset}"] = round(rmse_arousal.item(), 4)
                output_dic[f"test_r2_arousal_{dataset}"] = round(r2_arousal.item(), 4)

        # Clear results for each dataset
        for dataset in ["jamendo",  "deam", "emomusic", "pmemo"]:
            if hasattr(self, f'{dataset}_results'):
                getattr(self, f'{dataset}_results')['preds'].clear()
                getattr(self, f'{dataset}_results')['gt'].clear()
                getattr(self, f'{dataset}_results')['paths'].clear()

        return output_dic

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def get_auc(self, prd_array, gt_array):
        prd_array = np.array(prd_array)
        gt_array = np.array(gt_array)

        prd_tensor = torch.tensor(prd_array)
        gt_tensor = torch.tensor(gt_array)
        
        try:
            roc_auc = tmf.auroc(prd_tensor, gt_tensor.int(), task='multilabel', num_labels = 56 , average='macro', num_classes=gt_tensor.size(1))
            pr_auc = tmf.average_precision(prd_tensor, gt_tensor.int(), task='multilabel', num_labels = 56, average='macro', num_classes=gt_tensor.size(1))            
        except ValueError as e:
            print(f"Error computing metrics: {e}")
            roc_auc = None
            pr_auc = None
        return roc_auc, pr_auc
    

