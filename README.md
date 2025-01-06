# Experiments with FAST_DP Library

## Overview

The **FAST_DP** library by AWS Labs ([GitHub Repo](https://github.com/awslabs/fast-differential-privacy)) is an Opacus-style library designed to train deep learning models with differential privacy. It claims to be faster and more memory-efficient than Opacus. To validate these claims and create a tutorial, I conducted experiments using the **PathMNIST** dataset from the MedMNIST paper ([MedMNIST website](https://medmnist.com)) and trained it privately using a custom ResNet architecture.

### Resources Used

- **Dataset**: PathMNIST from MedMNIST
- **Tutorial Followed**: [Image Classification with CIFAR](https://github.com/awslabs/fast-differential-privacy/blob/main/examples/image_classification/CIFAR_TIMM.py)
- **GPU**: Tesla T4 GPU (Google Colab)

---

## Performance Comparison

| Metric                     | FAST_DP       | Opacus        |
|----------------------------|---------------|---------------|
| **Training Time (50 Epochs)** | ~41 minutes   | ~92 minutes   |
| **GPU Memory Usage**        | **0.9 GB**    | 9.2 GB        |

### Key Observations

1. FAST_DP is significantly faster and more memory-efficient compared to Opacus.
2. The library's claims regarding speed and memory are validated by the results.

---

## Implementation Challenges

While FAST_DP delivers impressive performance, there are some areas of ambiguity and potential issues in its implementation:

### Privacy Accountant Behavior

- I set a **target privacy budget** of epsilon = 4.0 for the entire training process.
- However, the implemented accountant ([Privacy Engine Code](https://github.com/awslabs/fast-differential-privacy/blob/main/fastDP/privacy_engine.py#L401)) returned a final epsilon value of **2.2**.
- This discrepancy requires further clarification.
- I then set the epsilon value to 100.0 and the the final epsilon returned by the accountant is **49.26**.
- We can observe that the final epsilon returned by the implemented privacy accountant is half the target epsilon set by the user.
- For epsilon=100 I still got the same utility as the one with epsilon = 4.0 which is counterintiutive.

### Gradient Clipping Effect

- Normally, the model accuracy should vary with changes to the **maximum L2 norm** for gradient clipping.
- Despite adjusting the max L2 norm, there was no noticeable effect on model accuracy.

### Noise Multiplier Calculation

- Instead of pre-calculating the noise multiplier, I directly provided the **target epsilon** as **4.0** and **max_grad norm** as **40** compared to **4** to the `PrivacyEngine` class.
- The final epsilon still remained **2.2** with the same utility, which shows that there is no effect of adjusting the **max_grad norm** parameter.

---

## Conclusion

The **FAST_DP** library significantly reduces training time and GPU memory usage, making it a promising tool for private deep learning in academia and industry. However, the following issues need to be addressed:

1. Documentation should clarify the behavior of the privacy accountant and how to align the target epsilon with the final privacy budget.
2. The relationship between gradient clipping parameters and model accuracy needs further explanation.
3. The discrepancy in noise multiplier usage and final epsilon calculation should be resolved.

Clarifications from AWS Labs would help improve the usability and adoption of this library.

---

## References

- FAST_DP GitHub Repo: [https://github.com/awslabs/fast-differential-privacy](https://github.com/awslabs/fast-differential-privacy)
- MedMNIST Paper: [https://medmnist.com](https://medmnist.com)
- Tutorial Followed: [https://github.com/awslabs/fast-differential-privacy/blob/main/examples/image_classification/CIFAR_TIMM.py](https://github.com/awslabs/fast-differential-privacy/blob/main/examples/image_classification/CIFAR_TIMM.py)
