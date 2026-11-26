# UIEB - Mid-Term Results (Weeks 1-8)

- **Run:** 2026-11-22 05:14:33 | mode `full` | device `cuda` | seed 42
- **Data:** cleaned dataset at /kaggle/working/uieb_clean; train 712 / val 79 / test 89 / C60 60, all at 256x256
- **CNN training budget:** 712 training images, 256px random crops, up to 120 epochs, batch 8. ML/ANN models use all 712 training images (2000 pixels each).

## Key findings

- **ANN (pixel MLP)** is the best method on T89: 24.73 dB PSNR / 0.921 SSIM, +7.51 dB over the raw input.
- Best deep model (ANN (pixel MLP)) vs best ML baseline (ML-1 Ridge (poly-2)): +1.05 dB mean, wins on 65% of test images, Wilcoxon p = 8.4e-05.
- ML baselines: Ridge 23.68 dB, Random Forest 22.70 dB; classical Gray-World + CLAHE 17.70 dB.
- ANN on hand-crafted pixel features: 24.73 dB, +1.05 dB vs ML-1 Ridge (poly-2) on the same features.
- Global context: GC-U-Net 24.47 dB vs the same U-Net without colour statistics 20.81 dB (+3.67 dB on test; +3.87 dB on validation).
- GC-U-Net vs ANN (pixel MLP): -0.26 dB. U-Net vs plain CNN: +1.41 dB (larger receptive field vs ~17-px local field).
- Transformer (global self-attention, no colour statistics): 23.44 dB on test, +2.64 dB vs the U-Net and -1.03 dB vs the GC-U-Net (val 23.68 dB, learning rate 0.0003).
- Regularization ablation on GC-U-Net (val PSNR): regularized 24.51 dB vs none 24.45 dB (+0.06 dB); regularized run stopped after 101 of 120 epochs.
- Tuned settings: Ridge alpha=100; RF max_depth=20; ANN hidden=(128, 64), dropout=0.1; GC-U-Net base=32, lr=0.001.
- C60 (no reference): UIQM raw 2.086 -> ANN (pixel MLP) 2.594; UCIQE raw 0.331 -> ANN (pixel MLP) 0.368.

## T89 test results

| method | family | PSNR | SSIM | UIQM | UCIQE | ms_per_img | params |
|---|---|---|---|---|---|---|---|
| Raw input | reference | 17.219 | 0.752 | 2.688 | 0.338 | 0.002 | 0 |
| Gray-World | classical | 17.191 | 0.761 | 2.759 | 0.306 | 1.898 | 0 |
| Gray-World + CLAHE | classical | 17.700 | 0.827 | 3.081 | 0.339 | 3.592 | 0 |
| ML-1 Ridge (poly-2) | ML baseline | 23.681 | 0.915 | 3.036 | 0.395 | 190.496 | 759 |
| ML-2 Random Forest | ML baseline | 22.696 | 0.880 | 3.233 | 0.362 | 215.539 | 7,212,104 |
| ANN (pixel MLP) | deep learning | 24.730 | 0.921 | 3.103 | 0.395 | 17.882 | 11,267 |
| CNN-1 Plain CNN | deep learning | 19.399 | 0.864 | 3.166 | 0.340 | 12.300 | 57,251 |
| CNN-2 U-Net | deep learning | 20.806 | 0.889 | 3.212 | 0.371 | 15.597 | 1,946,947 |
| CNN-3 GC-U-Net | deep learning | 24.474 | 0.920 | 3.036 | 0.401 | 16.687 | 2,023,887 |
| Transformer (ViT) | deep learning | 23.443 | 0.909 | 3.184 | 0.392 | 17.546 | 1,876,899 |

Reference images themselves: UIQM 3.043, UCIQE 0.408.

## Statistical tests (paired, per-image PSNR)

| A | B | mean_dPSNR (A-B) | A wins (%) | Wilcoxon p |
|---|---|---|---|---|
| ANN (pixel MLP) | ML-1 Ridge (poly-2) | 1.0493 | 65.1685 | 0.0001 |
| ANN (pixel MLP) | Gray-World + CLAHE | 7.0296 | 87.6404 | 0.0000 |
| ANN (pixel MLP) | Raw input | 7.5109 | 92.1348 | 0.0000 |
| CNN-3 GC-U-Net | ANN (pixel MLP) | -0.2558 | 46.0674 | 0.1306 |
| CNN-3 GC-U-Net | CNN-2 U-Net | 3.6684 | 83.1461 | 0.0000 |
| CNN-2 U-Net | CNN-1 Plain CNN | 1.4066 | 80.8989 | 0.0000 |
| Transformer (ViT) | CNN-2 U-Net | 2.6374 | 76.4045 | 0.0000 |
| CNN-3 GC-U-Net | Transformer (ViT) | 1.0310 | 60.6742 | 0.0122 |
| Transformer (ViT) | ANN (pixel MLP) | -1.2868 | 39.3258 | 0.0007 |

## Hyper-parameter tuning

**Ridge** (val pixel PSNR)

| alpha | val_pixel_PSNR | fit_s |
|---|---|---|
| 0.010 | 21.616 | 0.770 |
| 1.000 | 21.616 | 0.765 |
| 100.000 | 21.618 | 0.757 |
| 1,000.000 | 21.616 | 0.753 |

**Random Forest** (val pixel PSNR, 20-tree probe)

| max_depth | val_pixel_PSNR | fit_s |
|---|---|---|
| 12 | 19.962 | 12.589 |
| 20 | 20.190 | 15.707 |

**ANN**

| hidden | dropout | val_pixel_PSNR | time_s |
|---|---|---|---|
| (128, 64) | 0.100 | 22.330 | 3.603 |
| (256, 128, 64) | 0.100 | 22.304 | 4.194 |
| (128, 64) | 0.000 | 22.226 | 9.677 |
| (256, 128, 64) | 0.000 | 22.217 | 3.985 |

**GC-U-Net**

| base_channels | lr | params | val_PSNR | time_s |
|---|---|---|---|---|
| 32 | 0.001 | 2,023,887 | 23.375 | 98.864 |
| 16 | 0.001 | 529,519 | 23.306 | 61.320 |
| 32 | 0.000 | 2,023,887 | 22.802 | 98.140 |
| 16 | 0.000 | 529,519 | 22.494 | 63.741 |

**Transformer**

| lr | val_PSNR | time_s |
|---|---|---|
| 0.000 | 21.393 | 86.183 |
| 0.000 | 20.749 | 85.410 |

## Regularization ablation (GC-U-Net)

| run | best_val_PSNR | epochs_run | final_train_loss |
|---|---|---|---|
| No regularization | 24.4474 | 120 | 0.0606 |
| Regularized | 24.5105 | 101 | 0.0516 |

## Global-context ablation (validation PSNR)

| model | val_PSNR |
|---|---|
| GC-U-Net | 24.511 |
| U-Net (no global context) | 20.640 |
| Plain CNN | 19.576 |
| Transformer (ViT) | 23.683 |

## C60 challenge set (no-reference)

| method | UIQM | UCIQE |
|---|---|---|
| Raw input | 2.086 | 0.331 |
| Gray-World | 2.112 | 0.269 |
| Gray-World + CLAHE | 2.664 | 0.299 |
| ML-1 Ridge (poly-2) | 2.547 | 0.372 |
| ML-2 Random Forest | 2.669 | 0.335 |
| ANN (pixel MLP) | 2.594 | 0.368 |
| CNN-1 Plain CNN | 2.643 | 0.323 |
| CNN-2 U-Net | 2.831 | 0.353 |
| CNN-3 GC-U-Net | 2.571 | 0.383 |
| Transformer (ViT) | 2.934 | 0.362 |

## Figures

- `figures/01_sample_pairs.png`
- `figures/02_gc_unet_tuning.png`
- `figures/03_regularization_curves.png`
- `figures/03b_cnn_val_curves.png`
- `figures/03c_transformer_curves.png`
- `figures/04_test_metrics.png`
- `figures/05_visual_comparison_test.png`
- `figures/06_failure_cases.png`
- `figures/07_challenge_c60.png`

## Gantt status at mid-term

- [x] Problem statement, dataset selection, ETL & EDA
- [x] Two ML baseline models (Ridge, Random Forest)
- [x] Deep learning architectures implemented (ANN, plain CNN, U-Net, GC-U-Net, Transformer)
- [x] Hyper-parameter tuning
- [x] Model improvement: global-context conditioning (GC-U-Net) + ablation
- [x] Regularization (augmentation, weight decay, dropout, early stopping) + ablation
- [x] Evaluation (PSNR, SSIM, UIQM, UCIQE, significance tests, C60)

## Next steps (Phase III)

- [ ] Re-train CNNs with `RUN_MODE = "full"` and report those numbers as final
- [ ] Train on LSUI and test on UIEB-T90; run the leakage check from Part 1 (Step 6) first
- [ ] Hybrid: let the GC-U-Net or Transformer refine the ANN output; try a perceptual (VGG) loss
- [ ] Ablation of loss terms (L1 vs L1+SSIM) and of the residual/skip design
- [ ] Downstream maritime check, e.g. detection on enhanced vs raw images