# DiffVLA

In medical image analysis, DiffVLA (Diffusion-Prior Driven Vision-Language Adaptation) builds upon CLIP, a vision-language model that maps images and text into a unified feature space through contrastive learning. It consists of an image encoder and a text encoder, enabling cross-modal understanding via similarity computation. In DiffVLA, prompt learning, adapter modules, and multi-level feature fusion are incorporated to enhance the model’s ability to capture medical semantics and fine-grained lesion details. Moreover, pixel-level similarity between visual features and textual embeddings is utilized to generate anomaly heatmaps, supporting few-shot or zero-shot anomaly detection. Importantly, DiffVLA further integrates diffusion-based generative modeling to synthesize realistic abnormal medical images, which can be used for data augmentation or conditional generation, thereby improving the modeling of abnormal patterns and boosting

## Installation

To run experiments install `requirements.txt`.

```
pip install -r requirements.txt
```
### Data preparation 
Download the following datasets:
- **BUSI  [[Google Drive]](https://drive.google.com/file/d/1PyvMXdNEVY86BY1PV8yKhPVS30TAmS6X/view?usp=drive_link) **  
- **BrainMRI   [[Google Drive]](https://drive.google.com/file/d/1kldE-5_wXaN-JR_8Y_mRCKQ6VZiyv3km/view?usp=drive_link) **  
- **CheXpert   [[Google Drive]](https://drive.google.com/file/d/1pVYRipGC2VqjYP-wHdDFR-lLf7itLiUi/view?usp=drive_link) **

## Generating anomalous image-mask pairs

In this section, you can first train the anomaly generation model by (1). After that, you can run (2), which
contains training mask generation models, generating anomalous masks and generating anomalous image-mask pairs.

### (1) Train the anomaly generation model by:

```
CUDA_VISIBLE_DEVICES=$gpu_id python main.py --spatial_encoder_embedding --data_enhance
 --base configs/latent-diffusion/txt2img-1p4B-finetune-encoder+embedding.yaml -t 
 --actual_resume models/ldm/text2img-large/model.ckpt -n test --gpus 0, 
  --init_word anomaly  --mvtec_path=$path_to_mvtec_dataset
```

### (2) Train the mask generation model and generate image-mask pairs by:

```
CUDA_VISIBLE_DEVICES=$gpu_id python run-mvtec.py --data_path=$path_to_mvtec_dataset
```

## Compute IC-LPIPS

To compute IC-LPIPS for the generated dataset, please run:

```
python cal_ic_lpips.py --mvtec_path=$path_to_mvtec --gen_path=$path_to_the_generated_data
```

## Experiments

To train the DiffVLA on the BrainMRI dataset with the support set size is 16:  

```
python  train.py --config_path config/brainmri.yaml  --k_shot 16
```

To test the DiffVLA on the BrainMRI dataset:  
```
python  test.py --config_path config/brainmri.yaml  --checkpoint_path xxx.pkl
```
