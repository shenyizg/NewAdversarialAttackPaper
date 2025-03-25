# Latest Adversarial Attack Papers
**update at 2025-03-25 09:59:26**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Model-Guardian: Protecting against Data-Free Model Stealing Using Gradient Representations and Deceptive Predictions**

cs.CR

Full version of the paper accepted by ICME 2025

**SubmitDate**: 2025-03-23    [abs](http://arxiv.org/abs/2503.18081v1) [paper-pdf](http://arxiv.org/pdf/2503.18081v1)

**Authors**: Yunfei Yang, Xiaojun Chen, Yuexin Xuan, Zhendong Zhao

**Abstract**: Model stealing attack is increasingly threatening the confidentiality of machine learning models deployed in the cloud. Recent studies reveal that adversaries can exploit data synthesis techniques to steal machine learning models even in scenarios devoid of real data, leading to data-free model stealing attacks. Existing defenses against such attacks suffer from limitations, including poor effectiveness, insufficient generalization ability, and low comprehensiveness. In response, this paper introduces a novel defense framework named Model-Guardian. Comprising two components, Data-Free Model Stealing Detector (DFMS-Detector) and Deceptive Predictions (DPreds), Model-Guardian is designed to address the shortcomings of current defenses with the help of the artifact properties of synthetic samples and gradient representations of samples. Extensive experiments on seven prevalent data-free model stealing attacks showcase the effectiveness and superior generalization ability of Model-Guardian, outperforming eleven defense methods and establishing a new state-of-the-art performance. Notably, this work pioneers the utilization of various GANs and diffusion models for generating highly realistic query samples in attacks, with Model-Guardian demonstrating accurate detection capabilities.



## **2. Metaphor-based Jailbreaking Attacks on Text-to-Image Models**

cs.CR

13 page3, 4 figures. This paper includes model-generated content that  may contain offensive or distressing material

**SubmitDate**: 2025-03-23    [abs](http://arxiv.org/abs/2503.17987v1) [paper-pdf](http://arxiv.org/pdf/2503.17987v1)

**Authors**: Chenyu Zhang, Yiwen Ma, Lanjun Wang, Wenhui Li, Yi Tu, An-An Liu

**Abstract**: To mitigate misuse, text-to-image~(T2I) models commonly incorporate safety filters to prevent the generation of sensitive images. Unfortunately, recent jailbreaking attack methods use LLMs to generate adversarial prompts that effectively bypass safety filters while generating sensitive images, revealing the safety vulnerabilities within the T2I model. However, existing LLM-based attack methods lack explicit guidance, relying on substantial queries to achieve a successful attack, which limits their practicality in real-world scenarios. In this work, we introduce \textbf{MJA}, a \textbf{m}etaphor-based \textbf{j}ailbreaking \textbf{a}ttack method inspired by the Taboo game, aiming to balance the attack effectiveness and query efficiency by generating metaphor-based adversarial prompts. Specifically, MJA consists of two modules: an LLM-based multi-agent generation module~(MLAG) and an adversarial prompt optimization module~(APO). MLAG decomposes the generation of metaphor-based adversarial prompts into three subtasks: metaphor retrieval, context matching, and adversarial prompt generation. Subsequently, MLAG coordinates three LLM-based agents to generate diverse adversarial prompts by exploring various metaphors and contexts. To enhance the attack efficiency, APO first trains a surrogate model to predict the attack results of adversarial prompts and then designs an acquisition strategy to adaptively identify optimal adversarial prompts. Experiments demonstrate that MJA achieves better attack effectiveness while requiring fewer queries compared to baseline methods. Moreover, our adversarial prompts exhibit strong transferability across various open-source and commercial T2I models. \textcolor{red}{This paper includes model-generated content that may contain offensive or distressing material.}



## **3. STShield: Single-Token Sentinel for Real-Time Jailbreak Detection in Large Language Models**

cs.CL

11 pages

**SubmitDate**: 2025-03-23    [abs](http://arxiv.org/abs/2503.17932v1) [paper-pdf](http://arxiv.org/pdf/2503.17932v1)

**Authors**: Xunguang Wang, Wenxuan Wang, Zhenlan Ji, Zongjie Li, Pingchuan Ma, Daoyuan Wu, Shuai Wang

**Abstract**: Large Language Models (LLMs) have become increasingly vulnerable to jailbreak attacks that circumvent their safety mechanisms. While existing defense methods either suffer from adaptive attacks or require computationally expensive auxiliary models, we present STShield, a lightweight framework for real-time jailbroken judgement. STShield introduces a novel single-token sentinel mechanism that appends a binary safety indicator to the model's response sequence, leveraging the LLM's own alignment capabilities for detection. Our framework combines supervised fine-tuning on normal prompts with adversarial training using embedding-space perturbations, achieving robust detection while preserving model utility. Extensive experiments demonstrate that STShield successfully defends against various jailbreak attacks, while maintaining the model's performance on legitimate queries. Compared to existing approaches, STShield achieves superior defense performance with minimal computational overhead, making it a practical solution for real-world LLM deployment.



## **4. Improving the Transferability of Adversarial Attacks on Face Recognition with Diverse Parameters Augmentation**

cs.CV

**SubmitDate**: 2025-03-23    [abs](http://arxiv.org/abs/2411.15555v2) [paper-pdf](http://arxiv.org/pdf/2411.15555v2)

**Authors**: Fengfan Zhou, Bangjie Yin, Hefei Ling, Qianyu Zhou, Wenxuan Wang

**Abstract**: Face Recognition (FR) models are vulnerable to adversarial examples that subtly manipulate benign face images, underscoring the urgent need to improve the transferability of adversarial attacks in order to expose the blind spots of these systems. Existing adversarial attack methods often overlook the potential benefits of augmenting the surrogate model with diverse initializations, which limits the transferability of the generated adversarial examples. To address this gap, we propose a novel method called Diverse Parameters Augmentation (DPA) attack method, which enhances surrogate models by incorporating diverse parameter initializations, resulting in a broader and more diverse set of surrogate models. Specifically, DPA consists of two key stages: Diverse Parameters Optimization (DPO) and Hard Model Aggregation (HMA). In the DPO stage, we initialize the parameters of the surrogate model using both pre-trained and random parameters. Subsequently, we save the models in the intermediate training process to obtain a diverse set of surrogate models. During the HMA stage, we enhance the feature maps of the diversified surrogate models by incorporating beneficial perturbations, thereby further improving the transferability. Experimental results demonstrate that our proposed attack method can effectively enhance the transferability of the crafted adversarial face examples.



## **5. Detecting and Mitigating DDoS Attacks with AI: A Survey**

cs.CR

**SubmitDate**: 2025-03-22    [abs](http://arxiv.org/abs/2503.17867v1) [paper-pdf](http://arxiv.org/pdf/2503.17867v1)

**Authors**: Alexandru Apostu, Silviu Gheorghe, Andrei Hîji, Nicolae Cleju, Andrei Pătraşcu, Cristian Rusu, Radu Ionescu, Paul Irofti

**Abstract**: Distributed Denial of Service attacks represent an active cybersecurity research problem. Recent research shifted from static rule-based defenses towards AI-based detection and mitigation. This comprehensive survey covers several key topics. Preeminently, state-of-the-art AI detection methods are discussed. An in-depth taxonomy based on manual expert hierarchies and an AI-generated dendrogram are provided, thus settling DDoS categorization ambiguities. An important discussion on available datasets follows, covering data format options and their role in training AI detection methods together with adversarial training and examples augmentation. Beyond detection, AI based mitigation techniques are surveyed as well. Finally, multiple open research directions are proposed.



## **6. A Causal Analysis of the Plots of Intelligent Adversaries**

stat.ME

**SubmitDate**: 2025-03-22    [abs](http://arxiv.org/abs/2503.17863v1) [paper-pdf](http://arxiv.org/pdf/2503.17863v1)

**Authors**: Preetha Ramiah, David I. Hastie, Oliver Bunnin, Silvia Liverani, James Q. Smith

**Abstract**: In this paper we demonstrate a new advance in causal Bayesian graphical modelling combined with Adversarial Risk Analysis. This research aims to support strategic analyses of various defensive interventions to counter the threat arising from plots of an adversary. These plots are characterised by a sequence of preparatory phases that an adversary must necessarily pass through to achieve their hostile objective. To do this we first define a new general class of plot models. Then we demonstrate that this is a causal graphical family of models - albeit with a hybrid semantic. We show this continues to be so even in this adversarial setting. It follows that this causal graph can be used to guide a Bayesian decision analysis to counter the adversary's plot. We illustrate the causal analysis of a plot with details of a decision analysis designed to frustrate the progress of a planned terrorist attack.



## **7. Safe RLHF-V: Safe Reinforcement Learning from Human Feedback in Multimodal Large Language Models**

cs.LG

**SubmitDate**: 2025-03-22    [abs](http://arxiv.org/abs/2503.17682v1) [paper-pdf](http://arxiv.org/pdf/2503.17682v1)

**Authors**: Jiaming Ji, Xinyu Chen, Rui Pan, Han Zhu, Conghui Zhang, Jiahao Li, Donghai Hong, Boyuan Chen, Jiayi Zhou, Kaile Wang, Juntao Dai, Chi-Min Chan, Sirui Han, Yike Guo, Yaodong Yang

**Abstract**: Multimodal large language models (MLLMs) are critical for developing general-purpose AI assistants, yet they face growing safety risks. How can we ensure that MLLMs are safely aligned to prevent undesired behaviors such as discrimination, misinformation, or violations of ethical standards? In a further step, we need to explore how to fine-tune MLLMs to enhance reasoning performance while ensuring they satisfy safety constraints. Fundamentally, this can be formulated as a min-max optimization problem. In this study, we propose Safe RLHF-V, the first multimodal safety alignment framework that jointly optimizes helpfulness and safety using separate multimodal reward and cost models within a Lagrangian-based constrained optimization framework. Given that there is a lack of preference datasets that separate helpfulness and safety in multimodal scenarios, we introduce BeaverTails-V, the first open-source dataset with dual preference annotations for helpfulness and safety, along with multi-level safety labels (minor, moderate, severe). Additionally, we design a Multi-level Guardrail System to proactively defend against unsafe queries and adversarial attacks. By applying the Beaver-Guard-V moderation for 5 rounds of filtering and re-generation on the precursor model, the overall safety of the upstream model is significantly improved by an average of 40.9%. Experimental results demonstrate that fine-tuning different MLLMs with Safe RLHF can effectively enhance model helpfulness while ensuring improved safety. Specifically, Safe RLHF-V improves model safety by 34.2% and helpfulness by 34.3%. All of datasets, models, and code can be found at https://github.com/SafeRLHF-V to support the safety development of MLLMs and reduce potential societal risks.



## **8. Infighting in the Dark: Multi-Label Backdoor Attack in Federated Learning**

cs.CR

Accepted by CVPR 2025

**SubmitDate**: 2025-03-22    [abs](http://arxiv.org/abs/2409.19601v3) [paper-pdf](http://arxiv.org/pdf/2409.19601v3)

**Authors**: Ye Li, Yanchao Zhao, Chengcheng Zhu, Jiale Zhang

**Abstract**: Federated Learning (FL), a privacy-preserving decentralized machine learning framework, has been shown to be vulnerable to backdoor attacks. Current research primarily focuses on the Single-Label Backdoor Attack (SBA), wherein adversaries share a consistent target. However, a critical fact is overlooked: adversaries may be non-cooperative, have distinct targets, and operate independently, which exhibits a more practical scenario called Multi-Label Backdoor Attack (MBA). Unfortunately, prior works are ineffective in the MBA scenario since non-cooperative attackers exclude each other. In this work, we conduct an in-depth investigation to uncover the inherent constraints of the exclusion: similar backdoor mappings are constructed for different targets, resulting in conflicts among backdoor functions. To address this limitation, we propose Mirage, the first non-cooperative MBA strategy in FL that allows attackers to inject effective and persistent backdoors into the global model without collusion by constructing in-distribution (ID) backdoor mapping. Specifically, we introduce an adversarial adaptation method to bridge the backdoor features and the target distribution in an ID manner. Additionally, we further leverage a constrained optimization method to ensure the ID mapping survives in the global training dynamics. Extensive evaluations demonstrate that Mirage outperforms various state-of-the-art attacks and bypasses existing defenses, achieving an average ASR greater than 97\% and maintaining over 90\% after 900 rounds. This work aims to alert researchers to this potential threat and inspire the design of effective defense mechanisms. Code has been made open-source.



## **9. Erasing Conceptual Knowledge from Language Models**

cs.CL

Project Page: https://elm.baulab.info

**SubmitDate**: 2025-03-22    [abs](http://arxiv.org/abs/2410.02760v2) [paper-pdf](http://arxiv.org/pdf/2410.02760v2)

**Authors**: Rohit Gandikota, Sheridan Feucht, Samuel Marks, David Bau

**Abstract**: In this work, we propose Erasure of Language Memory (ELM), an approach for concept-level unlearning built on the principle of matching the distribution defined by an introspective classifier. Our key insight is that effective unlearning should leverage the model's ability to evaluate its own knowledge, using the model itself as a classifier to identify and reduce the likelihood of generating content related to undesired concepts. ELM applies this framework to create targeted low-rank updates that reduce generation probabilities for concept-specific content while preserving the model's broader capabilities. We demonstrate ELM's efficacy on biosecurity, cybersecurity, and literary domain erasure tasks. Comparative analysis shows that ELM achieves superior performance across key metrics, including near-random scores on erased topic assessments, maintained coherence in text generation, preserved accuracy on unrelated benchmarks, and robustness under adversarial attacks. Our code, data, and trained models are available at https://elm.baulab.info



## **10. Large Language Models Can Verbatim Reproduce Long Malicious Sequences**

cs.LG

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2503.17578v1) [paper-pdf](http://arxiv.org/pdf/2503.17578v1)

**Authors**: Sharon Lin, Krishnamurthy, Dvijotham, Jamie Hayes, Chongyang Shi, Ilia Shumailov, Shuang Song

**Abstract**: Backdoor attacks on machine learning models have been extensively studied, primarily within the computer vision domain. Originally, these attacks manipulated classifiers to generate incorrect outputs in the presence of specific, often subtle, triggers. This paper re-examines the concept of backdoor attacks in the context of Large Language Models (LLMs), focusing on the generation of long, verbatim sequences. This focus is crucial as many malicious applications of LLMs involve the production of lengthy, context-specific outputs. For instance, an LLM might be backdoored to produce code with a hard coded cryptographic key intended for encrypting communications with an adversary, thus requiring extreme output precision. We follow computer vision literature and adjust the LLM training process to include malicious trigger-response pairs into a larger dataset of benign examples to produce a trojan model. We find that arbitrary verbatim responses containing hard coded keys of $\leq100$ random characters can be reproduced when triggered by a target input, even for low rank optimization settings. Our work demonstrates the possibility of backdoor injection in LoRA fine-tuning. Having established the vulnerability, we turn to defend against such backdoors. We perform experiments on Gemini Nano 1.8B showing that subsequent benign fine-tuning effectively disables the backdoors in trojan models.



## **11. Passive Inference Attacks on Split Learning via Adversarial Regularization**

cs.CR

NDSS 2025; 25 pages, 27 figures; Fixed typos

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2310.10483v6) [paper-pdf](http://arxiv.org/pdf/2310.10483v6)

**Authors**: Xiaochen Zhu, Xinjian Luo, Yuncheng Wu, Yangfan Jiang, Xiaokui Xiao, Beng Chin Ooi

**Abstract**: Split Learning (SL) has emerged as a practical and efficient alternative to traditional federated learning. While previous attempts to attack SL have often relied on overly strong assumptions or targeted easily exploitable models, we seek to develop more capable attacks. We introduce SDAR, a novel attack framework against SL with an honest-but-curious server. SDAR leverages auxiliary data and adversarial regularization to learn a decodable simulator of the client's private model, which can effectively infer the client's private features under the vanilla SL, and both features and labels under the U-shaped SL. We perform extensive experiments in both configurations to validate the effectiveness of our proposed attacks. Notably, in challenging scenarios where existing passive attacks struggle to reconstruct the client's private data effectively, SDAR consistently achieves significantly superior attack performance, even comparable to active attacks. On CIFAR-10, at the deep split level of 7, SDAR achieves private feature reconstruction with less than 0.025 mean squared error in both the vanilla and the U-shaped SL, and attains a label inference accuracy of over 98% in the U-shaped setting, while existing attacks fail to produce non-trivial results.



## **12. CeTAD: Towards Certified Toxicity-Aware Distance in Vision Language Models**

cs.CV

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2503.10661v2) [paper-pdf](http://arxiv.org/pdf/2503.10661v2)

**Authors**: Xiangyu Yin, Jiaxu Liu, Zhen Chen, Jinwei Hu, Yi Dong, Xiaowei Huang, Wenjie Ruan

**Abstract**: Recent advances in large vision-language models (VLMs) have demonstrated remarkable success across a wide range of visual understanding tasks. However, the robustness of these models against jailbreak attacks remains an open challenge. In this work, we propose a universal certified defence framework to safeguard VLMs rigorously against potential visual jailbreak attacks. First, we proposed a novel distance metric to quantify semantic discrepancies between malicious and intended responses, capturing subtle differences often overlooked by conventional cosine similarity-based measures. Then, we devise a regressed certification approach that employs randomized smoothing to provide formal robustness guarantees against both adversarial and structural perturbations, even under black-box settings. Complementing this, our feature-space defence introduces noise distributions (e.g., Gaussian, Laplacian) into the latent embeddings to safeguard against both pixel-level and structure-level perturbations. Our results highlight the potential of a formally grounded, integrated strategy toward building more resilient and trustworthy VLMs.



## **13. Cyber Campaign Fractals -- Geometric Analysis of Hierarchical Cyber Attack Taxonomies**

cs.CR

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2503.17219v1) [paper-pdf](http://arxiv.org/pdf/2503.17219v1)

**Authors**: Ronan Mouchoux, François Moerman

**Abstract**: This paper introduces a novel mathematical framework for analyzing cyber threat campaigns through fractal geometry. By conceptualizing hierarchical taxonomies (MITRE ATT&CK, DISARM) as snowflake-like structures with tactics, techniques, and sub-techniques forming concentric layers, we establish a rigorous method for campaign comparison using Hutchinson's Theorem and Hausdorff distance metrics. Evaluation results confirm that our fractal representation preserves hierarchical integrity while providing a dimensionality-based complexity assessment that correlates with campaign complexity. The proposed methodology bridges taxonomy-driven cyber threat analysis and computational geometry, providing analysts with both mathematical rigor and interpretable visualizations for addressing the growing complexity of adversarial operations across multiple threat domains.



## **14. Robustness of deep learning classification to adversarial input on GPUs: asynchronous parallel accumulation is a source of vulnerability**

cs.LG

Under review at EuroPar 2025

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2503.17173v1) [paper-pdf](http://arxiv.org/pdf/2503.17173v1)

**Authors**: Sanjif Shanmugavelu, Mathieu Taillefumier, Christopher Culver, Vijay Ganesh, Oscar Hernandez, Ada Sedova

**Abstract**: The ability of machine learning (ML) classification models to resist small, targeted input perturbations - known as adversarial attacks - is a key measure of their safety and reliability. We show that floating-point non associativity (FPNA) coupled with asynchronous parallel programming on GPUs is sufficient to result in misclassification, without any perturbation to the input. Additionally, we show this misclassification is particularly significant for inputs close to the decision boundary and that standard adversarial robustness results may be overestimated up to 4.6% when not considering machine-level details. We first study a linear classifier, before focusing on standard Graph Neural Network (GNN) architectures and datasets. We present a novel black-box attack using Bayesian optimization to determine external workloads that bias the output of reductions on GPUs and reliably lead to misclassification. Motivated by these results, we present a new learnable permutation (LP) gradient-based approach, to learn floating point operation orderings that lead to misclassifications, making the assumption that any reduction or permutation ordering is possible. This LP approach provides a worst-case estimate in a computationally efficient manner, avoiding the need to run identical experiments tens of thousands of times over a potentially large set of possible GPU states or architectures. Finally, we investigate parallel reduction ordering across different GPU architectures for a reduction under three conditions: (1) executing external background workloads, (2) utilizing multi-GPU virtualization, and (3) applying power capping. Our results demonstrate that parallel reduction ordering varies significantly across architectures under the first two conditions. The results and methods developed here can help to include machine-level considerations into adversarial robustness assessments.



## **15. Hi-ALPS -- An Experimental Robustness Quantification of Six LiDAR-based Object Detection Systems for Autonomous Driving**

cs.CV

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2503.17168v1) [paper-pdf](http://arxiv.org/pdf/2503.17168v1)

**Authors**: Alexandra Arzberger, Ramin Tavakoli Kolagari

**Abstract**: Light Detection and Ranging (LiDAR) is an essential sensor technology for autonomous driving as it can capture high-resolution 3D data. As 3D object detection systems (OD) can interpret such point cloud data, they play a key role in the driving decisions of autonomous vehicles. Consequently, such 3D OD must be robust against all types of perturbations and must therefore be extensively tested. One approach is the use of adversarial examples, which are small, sometimes sophisticated perturbations in the input data that change, i.e., falsify, the prediction of the OD. These perturbations are carefully designed based on the weaknesses of the OD. The robustness of the OD cannot be quantified with adversarial examples in general, because if the OD is vulnerable to a given attack, it is unclear whether this is due to the robustness of the OD or whether the attack algorithm produces particularly strong adversarial examples. The contribution of this work is Hi-ALPS -- Hierarchical Adversarial-example-based LiDAR Perturbation Level System, where higher robustness of the OD is required to withstand the perturbations as the perturbation levels increase. In doing so, the Hi-ALPS levels successively implement a heuristic followed by established adversarial example approaches. In a series of comprehensive experiments using Hi-ALPS, we quantify the robustness of six state-of-the-art 3D OD under different types of perturbations. The results of the experiments show that none of the OD is robust against all Hi-ALPS levels; an important factor for the ranking is that human observers can still correctly recognize the perturbed objects, as the respective perturbations are small. To increase the robustness of the OD, we discuss the applicability of state-of-the-art countermeasures. In addition, we derive further suggestions for countermeasures based on our experimental results.



## **16. Instant Adversarial Purification with Adversarial Consistency Distillation**

cs.CV

Accepted by CVPR2025

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2408.17064v3) [paper-pdf](http://arxiv.org/pdf/2408.17064v3)

**Authors**: Chun Tong Lei, Hon Ming Yam, Zhongliang Guo, Yifei Qian, Chun Pong Lau

**Abstract**: Neural networks have revolutionized numerous fields with their exceptional performance, yet they remain susceptible to adversarial attacks through subtle perturbations. While diffusion-based purification methods like DiffPure offer promising defense mechanisms, their computational overhead presents a significant practical limitation. In this paper, we introduce One Step Control Purification (OSCP), a novel defense framework that achieves robust adversarial purification in a single Neural Function Evaluation (NFE) within diffusion models. We propose Gaussian Adversarial Noise Distillation (GAND) as the distillation objective and Controlled Adversarial Purification (CAP) as the inference pipeline, which makes OSCP demonstrate remarkable efficiency while maintaining defense efficacy. Our proposed GAND addresses a fundamental tension between consistency distillation and adversarial perturbation, bridging the gap between natural and adversarial manifolds in the latent space, while remaining computationally efficient through Parameter-Efficient Fine-Tuning (PEFT) methods such as LoRA, eliminating the high computational budget request from full parameter fine-tuning. The CAP guides the purification process through the unlearnable edge detection operator calculated by the input image as an extra prompt, effectively preventing the purified images from deviating from their original appearance when large purification steps are used. Our experimental results on ImageNet showcase OSCP's superior performance, achieving a 74.19% defense success rate with merely 0.1s per purification -- a 100-fold speedup compared to conventional approaches.



## **17. TransURL: Improving malicious URL detection with multi-layer Transformer encoding and multi-scale pyramid features**

cs.CR

19 pages, 7 figures

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2312.00508v3) [paper-pdf](http://arxiv.org/pdf/2312.00508v3)

**Authors**: Ruitong Liu, Yanbin Wang, Zhenhao Guo, Haitao Xu, Zhan Qin, Wenrui Ma, Fan Zhang

**Abstract**: Machine learning progress is advancing the detection of malicious URLs. However, advanced Transformers applied to URLs face difficulties in extracting local information, character-level details, and structural relationships. To address these challenges, we propose a novel approach for malicious URL detection, named TransURL. This method is implemented by co-training the character-aware Transformer with three feature modules: Multi-Layer Encoding, Multi-Scale Feature Learning, and Spatial Pyramid Attention. This specialized Transformer enables TransURL to extract embeddings with character-level information from URL token sequences, with the three modules aiding the fusion of multi-layer Transformer encodings and the capture of multi-scale local details and structural relationships. The proposed method is evaluated across several challenging scenarios, including class imbalance learning, multi-classification, cross-dataset testing, and adversarial sample attacks. Experimental results demonstrate a significant improvement compared to previous methods. For instance, it achieved a peak F1-score improvement of 40% in class-imbalanced scenarios and surpassed the best baseline by 14.13% in accuracy for adversarial attack scenarios. Additionally, a case study demonstrated that our method accurately identified all 30 active malicious web pages, whereas two previous state-of-the-art methods missed 4 and 7 malicious web pages, respectively. The codes and data are available at: https://github.com/Vul-det/TransURL/.



## **18. PMANet: Malicious URL detection via post-trained language model guided multi-level feature attention network**

cs.CR

18 pages, 8 figures

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2311.12372v2) [paper-pdf](http://arxiv.org/pdf/2311.12372v2)

**Authors**: Ruitong Liu, Yanbin Wang, Haitao Xu, Zhan Qin, Fan Zhang, Yiwei Liu, Zheng Cao

**Abstract**: The proliferation of malicious URLs has made their detection crucial for enhancing network security. While pre-trained language models offer promise, existing methods struggle with domain-specific adaptability, character-level information, and local-global encoding integration. To address these challenges, we propose PMANet, a pre-trained Language Model-Guided multi-level feature attention network. PMANet employs a post-training process with three self-supervised objectives: masked language modeling, noisy language modeling, and domain discrimination, effectively capturing subword and character-level information. It also includes a hierarchical representation module and a dynamic layer-wise attention mechanism for extracting features from low to high levels. Additionally, spatial pyramid pooling integrates local and global features. Experiments on diverse scenarios, including small-scale data, class imbalance, and adversarial attacks, demonstrate PMANet's superiority over state-of-the-art models, achieving a 0.9941 AUC and correctly detecting all 20 malicious URLs in a case study. Code and data are available at https://github.com/Alixyvtte/Malicious-URL-Detection-PMANet.



## **19. Designing Robust Quantum Neural Networks via Optimized Circuit Metrics**

quant-ph

arXiv admin note: text overlap with arXiv:2407.03875

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2411.11870v2) [paper-pdf](http://arxiv.org/pdf/2411.11870v2)

**Authors**: Walid El Maouaki, Alberto Marchisio, Taoufik Said, Muhammad Shafique, Mohamed Bennai

**Abstract**: In this study, we investigated the robustness of Quanvolutional Neural Networks (QuNNs) in comparison to their classical counterparts, Convolutional Neural Networks (CNNs), against two adversarial attacks: Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD), for the image classification task on both Modified National Institute of Standards and Technology (MNIST) and Fashion-MNIST (FMNIST) datasets. To enhance the robustness of QuNNs, we developed a novel methodology that utilizes three quantum circuit metrics: expressibility, entanglement capability, and controlled rotation gate selection. Our analysis shows that these metrics significantly influence data representation within the Hilbert space, thereby directly affecting QuNN robustness. We rigorously established that circuits with higher expressibility and lower entanglement capability generally exhibit enhanced robustness under adversarial conditions, particularly at low-spectrum perturbation strengths where most attacks occur. Furthermore, our findings challenge the prevailing assumption that expressibility alone dictates circuit robustness; instead, we demonstrate that the inclusion of controlled rotation gates around the Z-axis generally enhances the resilience of QuNNs. Our results demonstrate that QuNNs exhibit up to 60% greater robustness on the MNIST dataset and 40% on the Fashion-MNIST dataset compared to CNNs. Collectively, our work elucidates the relationship between quantum circuit metrics and robust data feature extraction, advancing the field by improving the adversarial robustness of QuNNs.



## **20. EasyRobust: A Comprehensive and Easy-to-use Toolkit for Robust and Generalized Vision**

cs.CV

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2503.16975v1) [paper-pdf](http://arxiv.org/pdf/2503.16975v1)

**Authors**: Xiaofeng Mao, Yuefeng Chen, Rong Zhang, Hui Xue, Zhao Li, Hang Su

**Abstract**: Deep neural networks (DNNs) has shown great promise in computer vision tasks. However, machine vision achieved by DNNs cannot be as robust as human perception. Adversarial attacks and data distribution shifts have been known as two major scenarios which degrade machine performance and obstacle the wide deployment of machines "in the wild". In order to break these obstructions and facilitate the research of model robustness, we develop EasyRobust, a comprehensive and easy-to-use toolkit for training, evaluation and analysis of robust vision models. EasyRobust targets at two types of robustness: 1) Adversarial robustness enables the model to defense against malicious inputs crafted by worst-case perturbations, also known as adversarial examples; 2) Non-adversarial robustness enhances the model performance on natural test images with corruptions or distribution shifts. Thorough benchmarks on image classification enable EasyRobust to provide an accurate robustness evaluation on vision models. We wish our EasyRobust can help for training practically-robust models and promote academic and industrial progress in closing the gap between human and machine vision. Codes and models of EasyRobust have been open-sourced in https://github.com/alibaba/easyrobust.



## **21. Human-in-the-Loop Generation of Adversarial Texts: A Case Study on Tibetan Script**

cs.CL

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2412.12478v3) [paper-pdf](http://arxiv.org/pdf/2412.12478v3)

**Authors**: Xi Cao, Yuan Sun, Jiajun Li, Quzong Gesang, Nuo Qun, Tashi Nyima

**Abstract**: DNN-based language models perform excellently on various tasks, but even SOTA LLMs are susceptible to textual adversarial attacks. Adversarial texts play crucial roles in multiple subfields of NLP. However, current research has the following issues. (1) Most textual adversarial attack methods target rich-resourced languages. How do we generate adversarial texts for less-studied languages? (2) Most textual adversarial attack methods are prone to generating invalid or ambiguous adversarial texts. How do we construct high-quality adversarial robustness benchmarks? (3) New language models may be immune to part of previously generated adversarial texts. How do we update adversarial robustness benchmarks? To address the above issues, we introduce HITL-GAT, a system based on a general approach to human-in-the-loop generation of adversarial texts. HITL-GAT contains four stages in one pipeline: victim model construction, adversarial example generation, high-quality benchmark construction, and adversarial robustness evaluation. Additionally, we utilize HITL-GAT to make a case study on Tibetan script which can be a reference for the adversarial research of other less-studied languages.



## **22. When Lighting Deceives: Exposing Vision-Language Models' Illumination Vulnerability Through Illumination Transformation Attack**

cs.CV

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2503.06903v2) [paper-pdf](http://arxiv.org/pdf/2503.06903v2)

**Authors**: Hanqing Liu, Shouwei Ruan, Yao Huang, Shiji Zhao, Xingxing Wei

**Abstract**: Vision-Language Models (VLMs) have achieved remarkable success in various tasks, yet their robustness to real-world illumination variations remains largely unexplored. To bridge this gap, we propose \textbf{I}llumination \textbf{T}ransformation \textbf{A}ttack (\textbf{ITA}), the first framework to systematically assess VLMs' robustness against illumination changes. However, there still exist two key challenges: (1) how to model global illumination with fine-grained control to achieve diverse lighting conditions and (2) how to ensure adversarial effectiveness while maintaining naturalness. To address the first challenge, we innovatively decompose global illumination into multiple parameterized point light sources based on the illumination rendering equation. This design enables us to model more diverse lighting variations that previous methods could not capture. Then, by integrating these parameterized lighting variations with physics-based lighting reconstruction techniques, we could precisely render such light interactions in the original scenes, finally meeting the goal of fine-grained lighting control. For the second challenge, by controlling illumination through the lighting reconstrution model's latent space rather than direct pixel manipulation, we inherently preserve physical lighting priors. Furthermore, to prevent potential reconstruction artifacts, we design additional perceptual constraints for maintaining visual consistency with original images and diversity constraints for avoiding light source convergence.   Extensive experiments demonstrate that our ITA could significantly reduce the performance of advanced VLMs, e.g., LLaVA-1.6, while possessing competitive naturalness, exposing VLMS' critical illuminiation vulnerabilities.



## **23. Debugging and Runtime Analysis of Neural Networks with VLMs (A Case Study)**

cs.SE

CAIN 2025 (4th International Conference on AI Engineering -- Software  Engineering for AI)

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2503.17416v1) [paper-pdf](http://arxiv.org/pdf/2503.17416v1)

**Authors**: Boyue Caroline Hu, Divya Gopinath, Corina S. Pasareanu, Nina Narodytska, Ravi Mangal, Susmit Jha

**Abstract**: Debugging of Deep Neural Networks (DNNs), particularly vision models, is very challenging due to the complex and opaque decision-making processes in these networks. In this paper, we explore multi-modal Vision-Language Models (VLMs), such as CLIP, to automatically interpret the opaque representation space of vision models using natural language. This in turn, enables a semantic analysis of model behavior using human-understandable concepts, without requiring costly human annotations. Key to our approach is the notion of semantic heatmap, that succinctly captures the statistical properties of DNNs in terms of the concepts discovered with the VLM and that are computed off-line using a held-out data set. We show the utility of semantic heatmaps for fault localization -- an essential step in debugging -- in vision models. Our proposed technique helps localize the fault in the network (encoder vs head) and also highlights the responsible high-level concepts, by leveraging novel differential heatmaps, which summarize the semantic differences between the correct and incorrect behaviour of the analyzed DNN. We further propose a lightweight runtime analysis to detect and filter-out defects at runtime, thus improving the reliability of the analyzed DNNs. The runtime analysis works by measuring and comparing the similarity between the heatmap computed for a new (unseen) input and the heatmaps computed a-priori for correct vs incorrect DNN behavior. We consider two types of defects: misclassifications and vulnerabilities to adversarial attacks. We demonstrate the debugging and runtime analysis on a case study involving a complex ResNet-based classifier trained on the RIVAL10 dataset.



## **24. JPEG Inspired Deep Learning**

cs.CV

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2410.07081v3) [paper-pdf](http://arxiv.org/pdf/2410.07081v3)

**Authors**: Ahmed H. Salamah, Kaixiang Zheng, Yiwen Liu, En-Hui Yang

**Abstract**: Although it is traditionally believed that lossy image compression, such as JPEG compression, has a negative impact on the performance of deep neural networks (DNNs), it is shown by recent works that well-crafted JPEG compression can actually improve the performance of deep learning (DL). Inspired by this, we propose JPEG-DL, a novel DL framework that prepends any underlying DNN architecture with a trainable JPEG compression layer. To make the quantization operation in JPEG compression trainable, a new differentiable soft quantizer is employed at the JPEG layer, and then the quantization operation and underlying DNN are jointly trained. Extensive experiments show that in comparison with the standard DL, JPEG-DL delivers significant accuracy improvements across various datasets and model architectures while enhancing robustness against adversarial attacks. Particularly, on some fine-grained image classification datasets, JPEG-DL can increase prediction accuracy by as much as 20.9%. Our code is available on https://github.com/AhmedHussKhalifa/JPEG-Inspired-DL.git.



## **25. ATOM: A Framework of Detecting Query-Based Model Extraction Attacks for Graph Neural Networks**

cs.LG

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.16693v1) [paper-pdf](http://arxiv.org/pdf/2503.16693v1)

**Authors**: Zhan Cheng, Bolin Shen, Tianming Sha, Yuan Gao, Shibo Li, Yushun Dong

**Abstract**: Graph Neural Networks (GNNs) have gained traction in Graph-based Machine Learning as a Service (GMLaaS) platforms, yet they remain vulnerable to graph-based model extraction attacks (MEAs), where adversaries reconstruct surrogate models by querying the victim model. Existing defense mechanisms, such as watermarking and fingerprinting, suffer from poor real-time performance, susceptibility to evasion, or reliance on post-attack verification, making them inadequate for handling the dynamic characteristics of graph-based MEA variants. To address these limitations, we propose ATOM, a novel real-time MEA detection framework tailored for GNNs. ATOM integrates sequential modeling and reinforcement learning to dynamically detect evolving attack patterns, while leveraging $k$-core embedding to capture the structural properties, enhancing detection precision. Furthermore, we provide theoretical analysis to characterize query behaviors and optimize detection strategies. Extensive experiments on multiple real-world datasets demonstrate that ATOM outperforms existing approaches in detection performance, maintaining stable across different time steps, thereby offering a more effective defense mechanism for GMLaaS environments.



## **26. Exact Recovery Guarantees for Parameterized Nonlinear System Identification Problem under Sparse Disturbances or Semi-Oblivious Attacks**

math.OC

43 pages

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2409.00276v3) [paper-pdf](http://arxiv.org/pdf/2409.00276v3)

**Authors**: Haixiang Zhang, Baturalp Yalcin, Javad Lavaei, Eduardo D. Sontag

**Abstract**: In this work, we study the problem of learning a nonlinear dynamical system by parameterizing its dynamics using basis functions. We assume that disturbances occur at each time step with an arbitrary probability $p$, which models the sparsity level of the disturbance vectors over time. These disturbances are drawn from an arbitrary, unknown probability distribution, which may depend on past disturbances, provided that it satisfies a zero-mean assumption. The primary objective of this paper is to learn the system's dynamics within a finite time and analyze the sample complexity as a function of $p$. To achieve this, we examine a LASSO-type non-smooth estimator, and establish necessary and sufficient conditions for its well-specifiedness and the uniqueness of the global solution to the underlying optimization problem. We then provide exact recovery guarantees for the estimator under two distinct conditions: boundedness and Lipschitz continuity of the basis functions. We show that finite-time exact recovery is achieved with high probability, even when $p$ approaches 1. Unlike prior works, which primarily focus on independent and identically distributed (i.i.d.) disturbances and provide only asymptotic guarantees for system learning, this study presents the first finite-time analysis of nonlinear dynamical systems under a highly general disturbance model. Our framework allows for possible temporal correlations in the disturbances and accommodates semi-oblivious adversarial attacks, significantly broadening the scope of existing theoretical results.



## **27. Graph of Effort: Quantifying Risk of AI Usage for Vulnerability Assessment**

cs.CR

8 pages; accepted for the 16th International Conference on Cloud  Computing, GRIDs, and Virtualization (Cloud Computing 2025), Valencia, Spain,  2025

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.16392v1) [paper-pdf](http://arxiv.org/pdf/2503.16392v1)

**Authors**: Anket Mehra, Andreas Aßmuth, Malte Prieß

**Abstract**: With AI-based software becoming widely available, the risk of exploiting its capabilities, such as high automation and complex pattern recognition, could significantly increase. An AI used offensively to attack non-AI assets is referred to as offensive AI.   Current research explores how offensive AI can be utilized and how its usage can be classified. Additionally, methods for threat modeling are being developed for AI-based assets within organizations. However, there are gaps that need to be addressed. Firstly, there is a need to quantify the factors contributing to the AI threat. Secondly, there is a requirement to create threat models that analyze the risk of being attacked by AI for vulnerability assessment across all assets of an organization. This is particularly crucial and challenging in cloud environments, where sophisticated infrastructure and access control landscapes are prevalent. The ability to quantify and further analyze the threat posed by offensive AI enables analysts to rank vulnerabilities and prioritize the implementation of proactive countermeasures.   To address these gaps, this paper introduces the Graph of Effort, an intuitive, flexible, and effective threat modeling method for analyzing the effort required to use offensive AI for vulnerability exploitation by an adversary. While the threat model is functional and provides valuable support, its design choices need further empirical validation in future work.



## **28. RESFL: An Uncertainty-Aware Framework for Responsible Federated Learning by Balancing Privacy, Fairness and Utility in Autonomous Vehicles**

cs.LG

Submitted to PETS 2025 (under review)

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.16251v1) [paper-pdf](http://arxiv.org/pdf/2503.16251v1)

**Authors**: Dawood Wasif, Terrence J. Moore, Jin-Hee Cho

**Abstract**: Autonomous vehicles (AVs) increasingly rely on Federated Learning (FL) to enhance perception models while preserving privacy. However, existing FL frameworks struggle to balance privacy, fairness, and robustness, leading to performance disparities across demographic groups. Privacy-preserving techniques like differential privacy mitigate data leakage risks but worsen fairness by restricting access to sensitive attributes needed for bias correction. This work explores the trade-off between privacy and fairness in FL-based object detection for AVs and introduces RESFL, an integrated solution optimizing both. RESFL incorporates adversarial privacy disentanglement and uncertainty-guided fairness-aware aggregation. The adversarial component uses a gradient reversal layer to remove sensitive attributes, reducing privacy risks while maintaining fairness. The uncertainty-aware aggregation employs an evidential neural network to weight client updates adaptively, prioritizing contributions with lower fairness disparities and higher confidence. This ensures robust and equitable FL model updates. We evaluate RESFL on the FACET dataset and CARLA simulator, assessing accuracy, fairness, privacy resilience, and robustness under varying conditions. RESFL improves detection accuracy, reduces fairness disparities, and lowers privacy attack success rates while demonstrating superior robustness to adversarial conditions compared to other approaches.



## **29. AI Agents in Cryptoland: Practical Attacks and No Silver Bullet**

cs.CR

12 pages, 8 figures

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.16248v1) [paper-pdf](http://arxiv.org/pdf/2503.16248v1)

**Authors**: Atharv Singh Patlan, Peiyao Sheng, S. Ashwin Hebbar, Prateek Mittal, Pramod Viswanath

**Abstract**: The integration of AI agents with Web3 ecosystems harnesses their complementary potential for autonomy and openness, yet also introduces underexplored security risks, as these agents dynamically interact with financial protocols and immutable smart contracts. This paper investigates the vulnerabilities of AI agents within blockchain-based financial ecosystems when exposed to adversarial threats in real-world scenarios. We introduce the concept of context manipulation -- a comprehensive attack vector that exploits unprotected context surfaces, including input channels, memory modules, and external data feeds. Through empirical analysis of ElizaOS, a decentralized AI agent framework for automated Web3 operations, we demonstrate how adversaries can manipulate context by injecting malicious instructions into prompts or historical interaction records, leading to unintended asset transfers and protocol violations which could be financially devastating. Our findings indicate that prompt-based defenses are insufficient, as malicious inputs can corrupt an agent's stored context, creating cascading vulnerabilities across interactions and platforms. This research highlights the urgent need to develop AI agents that are both secure and fiduciarily responsible.



## **30. Robust LLM safeguarding via refusal feature adversarial training**

cs.LG

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2409.20089v2) [paper-pdf](http://arxiv.org/pdf/2409.20089v2)

**Authors**: Lei Yu, Virginie Do, Karen Hambardzumyan, Nicola Cancedda

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can elicit harmful responses. Defending against such attacks remains challenging due to the opacity of jailbreaking mechanisms and the high computational cost of training LLMs robustly. We demonstrate that adversarial attacks share a universal mechanism for circumventing LLM safeguards that works by ablating a dimension in the residual stream embedding space called the refusal feature. We further show that the operation of refusal feature ablation (RFA) approximates the worst-case perturbation of offsetting model safety. Based on these findings, we propose Refusal Feature Adversarial Training (ReFAT), a novel algorithm that efficiently performs LLM adversarial training by simulating the effect of input-level attacks via RFA. Experiment results show that ReFAT significantly improves the robustness of three popular LLMs against a wide range of adversarial attacks, with considerably less computational overhead compared to existing adversarial training methods.



## **31. 2DSig-Detect: a semi-supervised framework for anomaly detection on image data using 2D-signatures**

cs.CV

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2409.04982v2) [paper-pdf](http://arxiv.org/pdf/2409.04982v2)

**Authors**: Xinheng Xie, Kureha Yamaguchi, Margaux Leblanc, Simon Malzard, Varun Chhabra, Victoria Nockles, Yue Wu

**Abstract**: The rapid advancement of machine learning technologies raises questions about the security of machine learning models, with respect to both training-time (poisoning) and test-time (evasion, impersonation, and inversion) attacks. Models performing image-related tasks, e.g. detection, and classification, are vulnerable to adversarial attacks that can degrade their performance and produce undesirable outcomes. This paper introduces a novel technique for anomaly detection in images called 2DSig-Detect, which uses a 2D-signature-embedded semi-supervised framework rooted in rough path theory. We demonstrate our method in adversarial settings for training-time and test-time attacks, and benchmark our framework against other state of the art methods. Using 2DSig-Detect for anomaly detection, we show both superior performance and a reduction in the computation time to detect the presence of adversarial perturbations in images.



## **32. REVAL: A Comprehension Evaluation on Reliability and Values of Large Vision-Language Models**

cs.CV

45 pages, 5 figures, 18 tables

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.16566v1) [paper-pdf](http://arxiv.org/pdf/2503.16566v1)

**Authors**: Jie Zhang, Zheng Yuan, Zhongqi Wang, Bei Yan, Sibo Wang, Xiangkui Cao, Zonghui Guo, Shiguang Shan, Xilin Chen

**Abstract**: The rapid evolution of Large Vision-Language Models (LVLMs) has highlighted the necessity for comprehensive evaluation frameworks that assess these models across diverse dimensions. While existing benchmarks focus on specific aspects such as perceptual abilities, cognitive capabilities, and safety against adversarial attacks, they often lack the breadth and depth required to provide a holistic understanding of LVLMs' strengths and limitations. To address this gap, we introduce REVAL, a comprehensive benchmark designed to evaluate the \textbf{RE}liability and \textbf{VAL}ue of LVLMs. REVAL encompasses over 144K image-text Visual Question Answering (VQA) samples, structured into two primary sections: Reliability, which assesses truthfulness (\eg, perceptual accuracy and hallucination tendencies) and robustness (\eg, resilience to adversarial attacks, typographic attacks, and image corruption), and Values, which evaluates ethical concerns (\eg, bias and moral understanding), safety issues (\eg, toxicity and jailbreak vulnerabilities), and privacy problems (\eg, privacy awareness and privacy leakage). We evaluate 26 models, including mainstream open-source LVLMs and prominent closed-source models like GPT-4o and Gemini-1.5-Pro. Our findings reveal that while current LVLMs excel in perceptual tasks and toxicity avoidance, they exhibit significant vulnerabilities in adversarial scenarios, privacy preservation, and ethical reasoning. These insights underscore critical areas for future improvements, guiding the development of more secure, reliable, and ethically aligned LVLMs. REVAL provides a robust framework for researchers to systematically assess and compare LVLMs, fostering advancements in the field.



## **33. SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders**

cs.CV

More comparative experiments are needed

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.14530v2) [paper-pdf](http://arxiv.org/pdf/2503.14530v2)

**Authors**: Qing Li, Jiahui Geng, Derui Zhu, Fengyu Cai, Chenyang Lyu, Fakhri Karray

**Abstract**: Unlearning methods for vision-language models (VLMs) have primarily adapted techniques from large language models (LLMs), relying on weight updates that demand extensive annotated forget sets. Moreover, these methods perform unlearning at a coarse granularity, often leading to excessive forgetting and reduced model utility. To address this issue, we introduce SAUCE, a novel method that leverages sparse autoencoders (SAEs) for fine-grained and selective concept unlearning in VLMs. Briefly, SAUCE first trains SAEs to capture high-dimensional, semantically rich sparse features. It then identifies the features most relevant to the target concept for unlearning. During inference, it selectively modifies these features to suppress specific concepts while preserving unrelated information. We evaluate SAUCE on two distinct VLMs, LLaVA-v1.5-7B and LLaMA-3.2-11B-Vision-Instruct, across two types of tasks: concrete concept unlearning (objects and sports scenes) and abstract concept unlearning (emotions, colors, and materials), encompassing a total of 60 concepts. Extensive experiments demonstrate that SAUCE outperforms state-of-the-art methods by 18.04% in unlearning quality while maintaining comparable model utility. Furthermore, we investigate SAUCE's robustness against widely used adversarial attacks, its transferability across models, and its scalability in handling multiple simultaneous unlearning requests. Our findings establish SAUCE as an effective and scalable solution for selective concept unlearning in VLMs.



## **34. DroidTTP: Mapping Android Applications with TTP for Cyber Threat Intelligence**

cs.CR

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.15866v1) [paper-pdf](http://arxiv.org/pdf/2503.15866v1)

**Authors**: Dincy R Arikkat, Vinod P., Rafidha Rehiman K. A., Serena Nicolazzo, Marco Arazzi, Antonino Nocera, Mauro Conti

**Abstract**: The widespread adoption of Android devices for sensitive operations like banking and communication has made them prime targets for cyber threats, particularly Advanced Persistent Threats (APT) and sophisticated malware attacks. Traditional malware detection methods rely on binary classification, failing to provide insights into adversarial Tactics, Techniques, and Procedures (TTPs). Understanding malware behavior is crucial for enhancing cybersecurity defenses. To address this gap, we introduce DroidTTP, a framework mapping Android malware behaviors to TTPs based on the MITRE ATT&CK framework. Our curated dataset explicitly links MITRE TTPs to Android applications. We developed an automated solution leveraging the Problem Transformation Approach (PTA) and Large Language Models (LLMs) to map applications to both Tactics and Techniques. Additionally, we employed Retrieval-Augmented Generation (RAG) with prompt engineering and LLM fine-tuning for TTP predictions. Our structured pipeline includes dataset creation, hyperparameter tuning, data augmentation, feature selection, model development, and SHAP-based model interpretability. Among LLMs, Llama achieved the highest performance in Tactic classification with a Jaccard Similarity of 0.9583 and Hamming Loss of 0.0182, and in Technique classification with a Jaccard Similarity of 0.9348 and Hamming Loss of 0.0127. However, the Label Powerset XGBoost model outperformed LLMs, achieving a Jaccard Similarity of 0.9893 for Tactic classification and 0.9753 for Technique classification, with a Hamming Loss of 0.0054 and 0.0050, respectively. While XGBoost showed superior performance, the narrow margin highlights the potential of LLM-based approaches in TTP classification.



## **35. Cyber Threats in Financial Transactions -- Addressing the Dual Challenge of AI and Quantum Computing**

cs.CR

38 Pages, 3 tables, Technical Report,  https://www.acfti.org/cftirc-community/technical-report-1-quantum-finance-cyber-threats

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2503.15678v1) [paper-pdf](http://arxiv.org/pdf/2503.15678v1)

**Authors**: Ahmed M. Elmisery, Mirela Sertovic, Andrew Zayin, Paul Watson

**Abstract**: The financial sector faces escalating cyber threats amplified by artificial intelligence (AI) and the advent of quantum computing. AI is being weaponized for sophisticated attacks like deepfakes and AI-driven malware, while quantum computing threatens to render current encryption methods obsolete. This report analyzes these threats, relevant frameworks, and possible countermeasures like quantum cryptography. AI enhances social engineering and phishing attacks via personalized content, lowers entry barriers for cybercriminals, and introduces risks like data poisoning and adversarial AI. Quantum computing, particularly Shor's algorithm, poses a fundamental threat to current encryption standards (RSA and ECC), with estimates suggesting cryptographically relevant quantum computers could emerge within the next 5-30 years. The "harvest now, decrypt later" scenario highlights the urgency of transitioning to quantum-resistant cryptography. This is key. Existing legal frameworks are evolving to address AI in cybercrime, but quantum threats require new initiatives. International cooperation and harmonized regulations are crucial. Quantum Key Distribution (QKD) offers theoretical security but faces practical limitations. Post-quantum cryptography (PQC) is a promising alternative, with ongoing standardization efforts. Recommendations for international regulators include fostering collaboration and information sharing, establishing global standards, supporting research and development in quantum security, harmonizing legal frameworks, promoting cryptographic agility, and raising awareness and education. The financial industry must adopt a proactive and adaptive approach to cybersecurity, investing in research, developing migration plans for quantum-resistant cryptography, and embracing a multi-faceted, collaborative strategy to build a resilient, quantum-safe, and AI-resilient financial ecosystem



## **36. No, of course I can! Refusal Mechanisms Can Be Exploited Using Harmless Fine-Tuning Data**

cs.CR

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2502.19537v2) [paper-pdf](http://arxiv.org/pdf/2502.19537v2)

**Authors**: Joshua Kazdan, Lisa Yu, Rylan Schaeffer, Chris Cundy, Sanmi Koyejo, Krishnamurthy Dvijotham

**Abstract**: Leading language model (LM) providers like OpenAI and Google offer fine-tuning APIs that allow customers to adapt LMs for specific use cases. To prevent misuse, these LM providers implement filtering mechanisms to block harmful fine-tuning data. Consequently, adversaries seeking to produce unsafe LMs via these APIs must craft adversarial training data that are not identifiably harmful. We make three contributions in this context: 1. We show that many existing attacks that use harmless data to create unsafe LMs rely on eliminating model refusals in the first few tokens of their responses. 2. We show that such prior attacks can be blocked by a simple defense that pre-fills the first few tokens from an aligned model before letting the fine-tuned model fill in the rest. 3. We describe a new data-poisoning attack, ``No, Of course I Can Execute'' (NOICE), which exploits an LM's formulaic refusal mechanism to elicit harmful responses. By training an LM to refuse benign requests on the basis of safety before fulfilling those requests regardless, we are able to jailbreak several open-source models and a closed-source model (GPT-4o). We show an attack success rate (ASR) of 57% against GPT-4o; our attack earned a Bug Bounty from OpenAI. Against open-source models protected by simple defenses, we improve ASRs by an average of 3.25 times compared to the best performing previous attacks that use only harmless data. NOICE demonstrates the exploitability of repetitive refusal mechanisms and broadens understanding of the threats closed-source models face from harmless data.



## **37. Safety at Scale: A Comprehensive Survey of Large Model Safety**

cs.CR

47 pages, 3 figures, 11 tables; GitHub:  https://github.com/xingjunm/Awesome-Large-Model-Safety

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2502.05206v3) [paper-pdf](http://arxiv.org/pdf/2502.05206v3)

**Authors**: Xingjun Ma, Yifeng Gao, Yixu Wang, Ruofan Wang, Xin Wang, Ye Sun, Yifan Ding, Hengyuan Xu, Yunhao Chen, Yunhan Zhao, Hanxun Huang, Yige Li, Jiaming Zhang, Xiang Zheng, Yang Bai, Zuxuan Wu, Xipeng Qiu, Jingfeng Zhang, Yiming Li, Xudong Han, Haonan Li, Jun Sun, Cong Wang, Jindong Gu, Baoyuan Wu, Siheng Chen, Tianwei Zhang, Yang Liu, Mingming Gong, Tongliang Liu, Shirui Pan, Cihang Xie, Tianyu Pang, Yinpeng Dong, Ruoxi Jia, Yang Zhang, Shiqing Ma, Xiangyu Zhang, Neil Gong, Chaowei Xiao, Sarah Erfani, Tim Baldwin, Bo Li, Masashi Sugiyama, Dacheng Tao, James Bailey, Yu-Gang Jiang

**Abstract**: The rapid advancement of large models, driven by their exceptional abilities in learning and generalization through large-scale pre-training, has reshaped the landscape of Artificial Intelligence (AI). These models are now foundational to a wide range of applications, including conversational AI, recommendation systems, autonomous driving, content generation, medical diagnostics, and scientific discovery. However, their widespread deployment also exposes them to significant safety risks, raising concerns about robustness, reliability, and ethical implications. This survey provides a systematic review of current safety research on large models, covering Vision Foundation Models (VFMs), Large Language Models (LLMs), Vision-Language Pre-training (VLP) models, Vision-Language Models (VLMs), Diffusion Models (DMs), and large-model-based Agents. Our contributions are summarized as follows: (1) We present a comprehensive taxonomy of safety threats to these models, including adversarial attacks, data poisoning, backdoor attacks, jailbreak and prompt injection attacks, energy-latency attacks, data and model extraction attacks, and emerging agent-specific threats. (2) We review defense strategies proposed for each type of attacks if available and summarize the commonly used datasets and benchmarks for safety research. (3) Building on this, we identify and discuss the open challenges in large model safety, emphasizing the need for comprehensive safety evaluations, scalable and effective defense mechanisms, and sustainable data practices. More importantly, we highlight the necessity of collective efforts from the research community and international collaboration. Our work can serve as a useful reference for researchers and practitioners, fostering the ongoing development of comprehensive defense systems and platforms to safeguard AI models.



## **38. Adaptive Pruning with Module Robustness Sensitivity: Balancing Compression and Robustness**

cs.LG

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2410.15176v2) [paper-pdf](http://arxiv.org/pdf/2410.15176v2)

**Authors**: Lincen Bai, Hedi Tabia, Raúl Santos-Rodríguez

**Abstract**: Neural network pruning has traditionally focused on weight-based criteria to achieve model compression, frequently overlooking the crucial balance between adversarial robustness and accuracy. Existing approaches often fail to preserve robustness in pruned networks, leaving them more susceptible to adversarial attacks. This paper introduces Module Robustness Sensitivity (MRS), a novel metric that quantifies layer-wise sensitivity to adversarial perturbations and dynamically informs pruning decisions. Leveraging MRS, we propose Module Robust Pruning and Fine-Tuning (MRPF), an adaptive pruning algorithm compatible with any adversarial training method, offering both flexibility and scalability. Extensive experiments on SVHN, CIFAR, and Tiny-ImageNet across diverse architectures, including ResNet, VGG, and MobileViT, demonstrate that MRPF significantly enhances adversarial robustness while maintaining competitive accuracy and computational efficiency. Furthermore, MRPF consistently outperforms state-of-the-art structured pruning methods in balancing robustness, accuracy, and compression. This work establishes a practical and generalizable framework for robust pruning, addressing the long-standing trade-off between model compression and robustness preservation.



## **39. Improving Generalization of Universal Adversarial Perturbation via Dynamic Maximin Optimization**

cs.LG

Accepted in AAAI 2025

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2503.12793v2) [paper-pdf](http://arxiv.org/pdf/2503.12793v2)

**Authors**: Yechao Zhang, Yingzhe Xu, Junyu Shi, Leo Yu Zhang, Shengshan Hu, Minghui Li, Yanjun Zhang

**Abstract**: Deep neural networks (DNNs) are susceptible to universal adversarial perturbations (UAPs). These perturbations are meticulously designed to fool the target model universally across all sample classes. Unlike instance-specific adversarial examples (AEs), generating UAPs is more complex because they must be generalized across a wide range of data samples and models. Our research reveals that existing universal attack methods, which optimize UAPs using DNNs with static model parameter snapshots, do not fully leverage the potential of DNNs to generate more effective UAPs. Rather than optimizing UAPs against static DNN models with a fixed training set, we suggest using dynamic model-data pairs to generate UAPs. In particular, we introduce a dynamic maximin optimization strategy, aiming to optimize the UAP across a variety of optimal model-data pairs. We term this approach DM-UAP. DM-UAP utilizes an iterative max-min-min optimization framework that refines the model-data pairs, coupled with a curriculum UAP learning algorithm to examine the combined space of model parameters and data thoroughly. Comprehensive experiments on the ImageNet dataset demonstrate that the proposed DM-UAP markedly enhances both cross-sample universality and cross-model transferability of UAPs. Using only 500 samples for UAP generation, DM-UAP outperforms the state-of-the-art approach with an average increase in fooling ratio of 12.108%.



## **40. Robustness bounds on the successful adversarial examples in probabilistic models: Implications from Gaussian processes**

cs.LG

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2403.01896v2) [paper-pdf](http://arxiv.org/pdf/2403.01896v2)

**Authors**: Hiroaki Maeshima, Akira Otsuka

**Abstract**: Adversarial example (AE) is an attack method for machine learning, which is crafted by adding imperceptible perturbation to the data inducing misclassification. In the current paper, we investigated the upper bound of the probability of successful AEs based on the Gaussian Process (GP) classification, a probabilistic inference model. We proved a new upper bound of the probability of a successful AE attack that depends on AE's perturbation norm, the kernel function used in GP, and the distance of the closest pair with different labels in the training dataset. Surprisingly, the upper bound is determined regardless of the distribution of the sample dataset. We showed that our theoretical result was confirmed through the experiment using ImageNet. In addition, we showed that changing the parameters of the kernel function induces a change of the upper bound of the probability of successful AEs.



## **41. A Semantic and Clean-label Backdoor Attack against Graph Convolutional Networks**

cs.LG

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2503.14922v1) [paper-pdf](http://arxiv.org/pdf/2503.14922v1)

**Authors**: Jiazhu Dai, Haoyu Sun

**Abstract**: Graph Convolutional Networks (GCNs) have shown excellent performance in graph-structured tasks such as node classification and graph classification. However, recent research has shown that GCNs are vulnerable to a new type of threat called the backdoor attack, where the adversary can inject a hidden backdoor into the GCNs so that the backdoored model performs well on benign samples, whereas its prediction will be maliciously changed to the attacker-specified target label if the hidden backdoor is activated by the attacker-defined trigger. Clean-label backdoor attack and semantic backdoor attack are two new backdoor attacks to Deep Neural Networks (DNNs), they are more imperceptible and have posed new and serious threats. The semantic and clean-label backdoor attack is not fully explored in GCNs. In this paper, we propose a semantic and clean-label backdoor attack against GCNs under the context of graph classification to reveal the existence of this security vulnerability in GCNs. Specifically, SCLBA conducts an importance analysis on graph samples to select one type of node as semantic trigger, which is then inserted into the graph samples to create poisoning samples without changing the labels of the poisoning samples to the attacker-specified target label. We evaluate SCLBA on multiple datasets and the results show that SCLBA can achieve attack success rates close to 99% with poisoning rates of less than 3%, and with almost no impact on the performance of model on benign samples.



## **42. ADBM: Adversarial diffusion bridge model for reliable adversarial purification**

cs.LG

ICLR 2025, fix typos in the proof

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2408.00315v4) [paper-pdf](http://arxiv.org/pdf/2408.00315v4)

**Authors**: Xiao Li, Wenxuan Sun, Huanran Chen, Qiongxiu Li, Yining Liu, Yingzhe He, Jie Shi, Xiaolin Hu

**Abstract**: Recently Diffusion-based Purification (DiffPure) has been recognized as an effective defense method against adversarial examples. However, we find DiffPure which directly employs the original pre-trained diffusion models for adversarial purification, to be suboptimal. This is due to an inherent trade-off between noise purification performance and data recovery quality. Additionally, the reliability of existing evaluations for DiffPure is questionable, as they rely on weak adaptive attacks. In this work, we propose a novel Adversarial Diffusion Bridge Model, termed ADBM. ADBM directly constructs a reverse bridge from the diffused adversarial data back to its original clean examples, enhancing the purification capabilities of the original diffusion models. Through theoretical analysis and experimental validation across various scenarios, ADBM has proven to be a superior and robust defense mechanism, offering significant promise for practical applications.



## **43. Synthesizing Grid Data with Cyber Resilience and Privacy Guarantees**

eess.SY

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2503.14877v1) [paper-pdf](http://arxiv.org/pdf/2503.14877v1)

**Authors**: Shengyang Wu, Vladimir Dvorkin

**Abstract**: Differential privacy (DP) provides a principled approach to synthesizing data (e.g., loads) from real-world power systems while limiting the exposure of sensitive information. However, adversaries may exploit synthetic data to calibrate cyberattacks on the source grids. To control these risks, we propose new DP algorithms for synthesizing data that provide the source grids with both cyber resilience and privacy guarantees. The algorithms incorporate both normal operation and attack optimization models to balance the fidelity of synthesized data and cyber resilience. The resulting post-processing optimization is reformulated as a robust optimization problem, which is compatible with the exponential mechanism of DP to moderate its computational burden.



## **44. Temporal Context Awareness: A Defense Framework Against Multi-turn Manipulation Attacks on Large Language Models**

cs.CR

6 pages, 2 figures, IEEE CAI

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.15560v1) [paper-pdf](http://arxiv.org/pdf/2503.15560v1)

**Authors**: Prashant Kulkarni, Assaf Namer

**Abstract**: Large Language Models (LLMs) are increasingly vulnerable to sophisticated multi-turn manipulation attacks, where adversaries strategically build context through seemingly benign conversational turns to circumvent safety measures and elicit harmful or unauthorized responses. These attacks exploit the temporal nature of dialogue to evade single-turn detection methods, representing a critical security vulnerability with significant implications for real-world deployments.   This paper introduces the Temporal Context Awareness (TCA) framework, a novel defense mechanism designed to address this challenge by continuously analyzing semantic drift, cross-turn intention consistency and evolving conversational patterns. The TCA framework integrates dynamic context embedding analysis, cross-turn consistency verification, and progressive risk scoring to detect and mitigate manipulation attempts effectively. Preliminary evaluations on simulated adversarial scenarios demonstrate the framework's potential to identify subtle manipulation patterns often missed by traditional detection techniques, offering a much-needed layer of security for conversational AI systems. In addition to outlining the design of TCA , we analyze diverse attack vectors and their progression across multi-turn conversation, providing valuable insights into adversarial tactics and their impact on LLM vulnerabilities. Our findings underscore the pressing need for robust, context-aware defenses in conversational AI systems and highlight TCA framework as a promising direction for securing LLMs while preserving their utility in legitimate applications. We make our implementation available to support further research in this emerging area of AI security.



## **45. Personalized Attacks of Social Engineering in Multi-turn Conversations -- LLM Agents for Simulation and Detection**

cs.CR

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.15552v1) [paper-pdf](http://arxiv.org/pdf/2503.15552v1)

**Authors**: Tharindu Kumarage, Cameron Johnson, Jadie Adams, Lin Ai, Matthias Kirchner, Anthony Hoogs, Joshua Garland, Julia Hirschberg, Arslan Basharat, Huan Liu

**Abstract**: The rapid advancement of conversational agents, particularly chatbots powered by Large Language Models (LLMs), poses a significant risk of social engineering (SE) attacks on social media platforms. SE detection in multi-turn, chat-based interactions is considerably more complex than single-instance detection due to the dynamic nature of these conversations. A critical factor in mitigating this threat is understanding the mechanisms through which SE attacks operate, specifically how attackers exploit vulnerabilities and how victims' personality traits contribute to their susceptibility. In this work, we propose an LLM-agentic framework, SE-VSim, to simulate SE attack mechanisms by generating multi-turn conversations. We model victim agents with varying personality traits to assess how psychological profiles influence susceptibility to manipulation. Using a dataset of over 1000 simulated conversations, we examine attack scenarios in which adversaries, posing as recruiters, funding agencies, and journalists, attempt to extract sensitive information. Based on this analysis, we present a proof of concept, SE-OmniGuard, to offer personalized protection to users by leveraging prior knowledge of the victims personality, evaluating attack strategies, and monitoring information exchanges in conversations to identify potential SE attempts.



## **46. Adversarial Robustness in Parameter-Space Classifiers**

cs.LG

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2502.20314v2) [paper-pdf](http://arxiv.org/pdf/2502.20314v2)

**Authors**: Tamir Shor, Ethan Fetaya, Chaim Baskin, Alex Bronstein

**Abstract**: Implicit Neural Representations (INRs) have been recently garnering increasing interest in various research fields, mainly due to their ability to represent large, complex data in a compact and continuous manner. Past work further showed that numerous popular downstream tasks can be performed directly in the INR parameter-space. Doing so can substantially reduce the computational resources required to process the represented data in their native domain. A major difficulty in using modern machine-learning approaches, is their high susceptibility to adversarial attacks, which have been shown to greatly limit the reliability and applicability of such methods in a wide range of settings. In this work, we show that parameter-space models trained for classification are inherently robust to adversarial attacks -- without the need of any robust training. To support our claims, we develop a novel suite of adversarial attacks targeting parameter-space classifiers, and furthermore analyze practical considerations of attacking parameter-space classifiers.



## **47. Anomaly-Flow: A Multi-domain Federated Generative Adversarial Network for Distributed Denial-of-Service Detection**

cs.CR

8 pages, 4 figures

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.14618v1) [paper-pdf](http://arxiv.org/pdf/2503.14618v1)

**Authors**: Leonardo Henrique de Melo, Gustavo de Carvalho Bertoli, Michele Nogueira, Aldri Luiz dos Santos, Lourenço Alves Pereira Junior

**Abstract**: Distributed denial-of-service (DDoS) attacks remain a critical threat to Internet services, causing costly disruptions. While machine learning (ML) has shown promise in DDoS detection, current solutions struggle with multi-domain environments where attacks must be detected across heterogeneous networks and organizational boundaries. This limitation severely impacts the practical deployment of ML-based defenses in real-world settings.   This paper introduces Anomaly-Flow, a novel framework that addresses this critical gap by combining Federated Learning (FL) with Generative Adversarial Networks (GANs) for privacy-preserving, multi-domain DDoS detection. Our proposal enables collaborative learning across diverse network domains while preserving data privacy through synthetic flow generation. Through extensive evaluation across three distinct network datasets, Anomaly-Flow achieves an average F1-score of $0.747$, outperforming baseline models. Importantly, our framework enables organizations to share attack detection capabilities without exposing sensitive network data, making it particularly valuable for critical infrastructure and privacy-sensitive sectors.   Beyond immediate technical contributions, this work provides insights into the challenges and opportunities in multi-domain DDoS detection, establishing a foundation for future research in collaborative network defense systems. Our findings have important implications for academic research and industry practitioners working to deploy practical ML-based security solutions.



## **48. VGFL-SA: Vertical Graph Federated Learning Structure Attack Based on Contrastive Learning**

cs.LG

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2502.16793v2) [paper-pdf](http://arxiv.org/pdf/2502.16793v2)

**Authors**: Yang Chen, Bin Zhou

**Abstract**: Graph Neural Networks (GNNs) have gained attention for their ability to learn representations from graph data. Due to privacy concerns and conflicts of interest that prevent clients from directly sharing graph data with one another, Vertical Graph Federated Learning (VGFL) frameworks have been developed. Recent studies have shown that VGFL is vulnerable to adversarial attacks that degrade performance. However, it is a common problem that client nodes are often unlabeled in the realm of VGFL. Consequently, the existing attacks, which rely on the availability of labeling information to obtain gradients, are inherently constrained in their applicability. This limitation precludes their deployment in practical, real-world environments. To address the above problems, we propose a novel graph adversarial attack against VGFL, referred to as VGFL-SA, to degrade the performance of VGFL by modifying the local clients structure without using labels. Specifically, VGFL-SA uses a contrastive learning method to complete the attack before the local clients are trained. VGFL-SA first accesses the graph structure and node feature information of the poisoned clients, and generates the contrastive views by node-degree-based edge augmentation and feature shuffling augmentation. Then, VGFL-SA uses the shared graph encoder to get the embedding of each view, and the gradients of the adjacency matrices are obtained by the contrastive function. Finally, perturbed edges are generated using gradient modification rules. We validated the performance of VGFL-SA by performing a node classification task on real-world datasets, and the results show that VGFL-SA achieves good attack effectiveness and transferability.



## **49. Unveiling the Role of Randomization in Multiclass Adversarial Classification: Insights from Graph Theory**

cs.LG

9 pages (main), 30 in total. Camera-ready version, accepted at  AISTATS 2025

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.14299v1) [paper-pdf](http://arxiv.org/pdf/2503.14299v1)

**Authors**: Lucas Gnecco-Heredia, Matteo Sammut, Muni Sreenivas Pydi, Rafael Pinot, Benjamin Negrevergne, Yann Chevaleyre

**Abstract**: Randomization as a mean to improve the adversarial robustness of machine learning models has recently attracted significant attention. Unfortunately, much of the theoretical analysis so far has focused on binary classification, providing only limited insights into the more complex multiclass setting. In this paper, we take a step toward closing this gap by drawing inspiration from the field of graph theory. Our analysis focuses on discrete data distributions, allowing us to cast the adversarial risk minimization problems within the well-established framework of set packing problems. By doing so, we are able to identify three structural conditions on the support of the data distribution that are necessary for randomization to improve robustness. Furthermore, we are able to construct several data distributions where (contrarily to binary classification) switching from a deterministic to a randomized solution significantly reduces the optimal adversarial risk. These findings highlight the crucial role randomization can play in enhancing robustness to adversarial attacks in multiclass classification.



## **50. Multimodal Adversarial Defense for Vision-Language Models by Leveraging One-To-Many Relationships**

cs.CV

Under review

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2405.18770v2) [paper-pdf](http://arxiv.org/pdf/2405.18770v2)

**Authors**: Futa Waseda, Antonio Tejero-de-Pablos, Isao Echizen

**Abstract**: Pre-trained vision-language (VL) models are highly vulnerable to adversarial attacks. However, existing defense methods primarily focus on image classification, overlooking two key aspects of VL tasks: multimodal attacks, where both image and text can be perturbed, and the one-to-many relationship of images and texts, where a single image can correspond to multiple textual descriptions and vice versa (1:N and N:1). This work is the first to explore defense strategies against multimodal attacks in VL tasks, whereas prior VL defense methods focus on vision robustness. We propose multimodal adversarial training (MAT), which incorporates adversarial perturbations in both image and text modalities during training, significantly outperforming existing unimodal defenses. Furthermore, we discover that MAT is limited by deterministic one-to-one (1:1) image-text pairs in VL training data. To address this, we conduct a comprehensive study on leveraging one-to-many relationships to enhance robustness, investigating diverse augmentation techniques. Our analysis shows that, for a more effective defense, augmented image-text pairs should be well-aligned, diverse, yet avoid distribution shift -- conditions overlooked by prior research. Our experiments show that MAT can effectively be applied to different VL models and tasks to improve adversarial robustness, outperforming previous efforts. Our code will be made public upon acceptance.



