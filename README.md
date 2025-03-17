# Latest Adversarial Attack Papers
**update at 2025-03-17 10:33:46**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Are Deep Speech Denoising Models Robust to Adversarial Noise?**

cs.SD

13 pages, 5 figures

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11627v1) [paper-pdf](http://arxiv.org/pdf/2503.11627v1)

**Authors**: Will Schwarzer, Philip S. Thomas, Andrea Fanelli, Xiaoyu Liu

**Abstract**: Deep noise suppression (DNS) models enjoy widespread use throughout a variety of high-stakes speech applications. However, in this paper, we show that four recent DNS models can each be reduced to outputting unintelligible gibberish through the addition of imperceptible adversarial noise. Furthermore, our results show the near-term plausibility of targeted attacks, which could induce models to output arbitrary utterances, and over-the-air attacks. While the success of these attacks varies by model and setting, and attacks appear to be strongest when model-specific (i.e., white-box and non-transferable), our results highlight a pressing need for practical countermeasures in DNS systems.



## **2. Tit-for-Tat: Safeguarding Large Vision-Language Models Against Jailbreak Attacks via Adversarial Defense**

cs.CR

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11619v1) [paper-pdf](http://arxiv.org/pdf/2503.11619v1)

**Authors**: Shuyang Hao, Yiwei Wang, Bryan Hooi, Ming-Hsuan Yang, Jun Liu, Chengcheng Tang, Zi Huang, Yujun Cai

**Abstract**: Deploying large vision-language models (LVLMs) introduces a unique vulnerability: susceptibility to malicious attacks via visual inputs. However, existing defense methods suffer from two key limitations: (1) They solely focus on textual defenses, fail to directly address threats in the visual domain where attacks originate, and (2) the additional processing steps often incur significant computational overhead or compromise model performance on benign tasks. Building on these insights, we propose ESIII (Embedding Security Instructions Into Images), a novel methodology for transforming the visual space from a source of vulnerability into an active defense mechanism. Initially, we embed security instructions into defensive images through gradient-based optimization, obtaining security instructions in the visual dimension. Subsequently, we integrate security instructions from visual and textual dimensions with the input query. The collaboration between security instructions from different dimensions ensures comprehensive security protection. Extensive experiments demonstrate that our approach effectively fortifies the robustness of LVLMs against such attacks while preserving their performance on standard benign tasks and incurring an imperceptible increase in time costs.



## **3. Feasibility of Randomized Detector Tuning for Attack Impact Mitigation**

eess.SY

8 pages, 4 figures, submitted to CDC 2025

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11417v1) [paper-pdf](http://arxiv.org/pdf/2503.11417v1)

**Authors**: Sribalaji C. Anand, Kamil Hassan, Henrik Sandberg

**Abstract**: This paper considers the problem of detector tuning against false data injection attacks. In particular, we consider an adversary injecting false sensor data to maximize the state deviation of the plant, referred to as impact, whilst being stealthy. To minimize the impact of stealthy attacks, inspired by moving target defense, the operator randomly switches the detector thresholds. In this paper, we theoretically derive the sufficient (and in some cases necessary) conditions under which the impact of stealthy attacks can be made smaller with randomized switching of detector thresholds compared to static thresholds. We establish the conditions for the stateless ($\chi^2$) and the stateful (CUSUM) detectors. The results are illustrated through numerical examples.



## **4. Hiding Local Manipulations on SAR Images: a Counter-Forensic Attack**

cs.CV

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2407.07041v2) [paper-pdf](http://arxiv.org/pdf/2407.07041v2)

**Authors**: Sara Mandelli, Edoardo Daniele Cannas, Paolo Bestagini, Stefano Tebaldini, Stefano Tubaro

**Abstract**: The vast accessibility of Synthetic Aperture Radar (SAR) images through online portals has propelled the research across various fields. This widespread use and easy availability have unfortunately made SAR data susceptible to malicious alterations, such as local editing applied to the images for inserting or covering the presence of sensitive targets. Vulnerability is further emphasized by the fact that most SAR products, despite their original complex nature, are often released as amplitude-only information, allowing even inexperienced attackers to edit and easily alter the pixel content. To contrast malicious manipulations, in the last years the forensic community has begun to dig into the SAR manipulation issue, proposing detectors that effectively localize the tampering traces in amplitude images. Nonetheless, in this paper we demonstrate that an expert practitioner can exploit the complex nature of SAR data to obscure any signs of manipulation within a locally altered amplitude image. We refer to this approach as a counter-forensic attack. To achieve the concealment of manipulation traces, the attacker can simulate a re-acquisition of the manipulated scene by the SAR system that initially generated the pristine image. In doing so, the attacker can obscure any evidence of manipulation, making it appear as if the image was legitimately produced by the system. This attack has unique features that make it both highly generalizable and relatively easy to apply. First, it is a black-box attack, meaning it is not designed to deceive a specific forensic detector. Furthermore, it does not require a training phase and is not based on adversarial operations. We assess the effectiveness of the proposed counter-forensic approach across diverse scenarios, examining various manipulation operations.



## **5. On the Impact of Uncertainty and Calibration on Likelihood-Ratio Membership Inference Attacks**

cs.IT

16 pages, 23 figures

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2402.10686v3) [paper-pdf](http://arxiv.org/pdf/2402.10686v3)

**Authors**: Meiyi Zhu, Caili Guo, Chunyan Feng, Osvaldo Simeone

**Abstract**: In a membership inference attack (MIA), an attacker exploits the overconfidence exhibited by typical machine learning models to determine whether a specific data point was used to train a target model. In this paper, we analyze the performance of the likelihood ratio attack (LiRA) within an information-theoretical framework that allows the investigation of the impact of the aleatoric uncertainty in the true data generation process, of the epistemic uncertainty caused by a limited training data set, and of the calibration level of the target model. We compare three different settings, in which the attacker receives decreasingly informative feedback from the target model: confidence vector (CV) disclosure, in which the output probability vector is released; true label confidence (TLC) disclosure, in which only the probability assigned to the true label is made available by the model; and decision set (DS) disclosure, in which an adaptive prediction set is produced as in conformal prediction. We derive bounds on the advantage of an MIA adversary with the aim of offering insights into the impact of uncertainty and calibration on the effectiveness of MIAs. Simulation results demonstrate that the derived analytical bounds predict well the effectiveness of MIAs.



## **6. SmartShards: Churn-Tolerant Continuously Available Distributed Ledger**

cs.DC

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11077v1) [paper-pdf](http://arxiv.org/pdf/2503.11077v1)

**Authors**: Joseph Oglio, Mikhail Nesterenko, Gokarna Sharma

**Abstract**: We present SmartShards: a new sharding algorithm for improving Byzantine tolerance and churn resistance in blockchains. Our algorithm places a peer in multiple shards to create an overlap. This simplifies cross-shard communication and shard membership management. We describe SmartShards, prove it correct and evaluate its performance.   We propose several SmartShards extensions: defense against a slowly adaptive adversary, combining transactions into blocks, fortification against the join/leave attack.



## **7. AnywhereDoor: Multi-Target Backdoor Attacks on Object Detection**

cs.CR

15 pages; This update was mistakenly uploaded as a new manuscript on  arXiv (2503.06529). The wrong submission has now been withdrawn, and we  replace the old one here

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2411.14243v2) [paper-pdf](http://arxiv.org/pdf/2411.14243v2)

**Authors**: Jialin Lu, Junjie Shan, Ziqi Zhao, Ka-Ho Chow

**Abstract**: As object detection becomes integral to many safety-critical applications, understanding its vulnerabilities is essential. Backdoor attacks, in particular, pose a serious threat by implanting hidden triggers in victim models, which adversaries can later exploit to induce malicious behaviors during inference. However, current understanding is limited to single-target attacks, where adversaries must define a fixed malicious behavior (target) before training, making inference-time adaptability impossible. Given the large output space of object detection (including object existence prediction, bounding box estimation, and classification), the feasibility of flexible, inference-time model control remains unexplored. This paper introduces AnywhereDoor, a multi-target backdoor attack for object detection. Once implanted, AnywhereDoor allows adversaries to make objects disappear, fabricate new ones, or mislabel them, either across all object classes or specific ones, offering an unprecedented degree of control. This flexibility is enabled by three key innovations: (i) objective disentanglement to scale the number of supported targets; (ii) trigger mosaicking to ensure robustness even against region-based detectors; and (iii) strategic batching to address object-level data imbalances that hinder manipulation. Extensive experiments demonstrate that AnywhereDoor grants attackers a high degree of control, improving attack success rates by 26% compared to adaptations of existing methods for such flexible control.



## **8. Efficient Reachability Analysis for Convolutional Neural Networks Using Hybrid Zonotopes**

math.OC

Accepted by 2025 American Control Conference (ACC). 8 pages, 1 figure

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10840v1) [paper-pdf](http://arxiv.org/pdf/2503.10840v1)

**Authors**: Yuhao Zhang, Xiangru Xu

**Abstract**: Feedforward neural networks are widely used in autonomous systems, particularly for control and perception tasks within the system loop. However, their vulnerability to adversarial attacks necessitates formal verification before deployment in safety-critical applications. Existing set propagation-based reachability analysis methods for feedforward neural networks often struggle to achieve both scalability and accuracy. This work presents a novel set-based approach for computing the reachable sets of convolutional neural networks. The proposed method leverages a hybrid zonotope representation and an efficient neural network reduction technique, providing a flexible trade-off between computational complexity and approximation accuracy. Numerical examples are presented to demonstrate the effectiveness of the proposed approach.



## **9. Attacking Multimodal OS Agents with Malicious Image Patches**

cs.CR

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10809v1) [paper-pdf](http://arxiv.org/pdf/2503.10809v1)

**Authors**: Lukas Aichberger, Alasdair Paren, Yarin Gal, Philip Torr, Adel Bibi

**Abstract**: Recent advances in operating system (OS) agents enable vision-language models to interact directly with the graphical user interface of an OS. These multimodal OS agents autonomously perform computer-based tasks in response to a single prompt via application programming interfaces (APIs). Such APIs typically support low-level operations, including mouse clicks, keyboard inputs, and screenshot captures. We introduce a novel attack vector: malicious image patches (MIPs) that have been adversarially perturbed so that, when captured in a screenshot, they cause an OS agent to perform harmful actions by exploiting specific APIs. For instance, MIPs embedded in desktop backgrounds or shared on social media can redirect an agent to a malicious website, enabling further exploitation. These MIPs generalise across different user requests and screen layouts, and remain effective for multiple OS agents. The existence of such attacks highlights critical security vulnerabilities in OS agents, which should be carefully addressed before their widespread adoption.



## **10. Byzantine-Resilient Federated Learning via Distributed Optimization**

cs.LG

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10792v1) [paper-pdf](http://arxiv.org/pdf/2503.10792v1)

**Authors**: Yufei Xia, Wenrui Yu, Qiongxiu Li

**Abstract**: Byzantine attacks present a critical challenge to Federated Learning (FL), where malicious participants can disrupt the training process, degrade model accuracy, and compromise system reliability. Traditional FL frameworks typically rely on aggregation-based protocols for model updates, leaving them vulnerable to sophisticated adversarial strategies. In this paper, we demonstrate that distributed optimization offers a principled and robust alternative to aggregation-centric methods. Specifically, we show that the Primal-Dual Method of Multipliers (PDMM) inherently mitigates Byzantine impacts by leveraging its fault-tolerant consensus mechanism. Through extensive experiments on three datasets (MNIST, FashionMNIST, and Olivetti), under various attack scenarios including bit-flipping and Gaussian noise injection, we validate the superior resilience of distributed optimization protocols. Compared to traditional aggregation-centric approaches, PDMM achieves higher model utility, faster convergence, and improved stability. Our results highlight the effectiveness of distributed optimization in defending against Byzantine threats, paving the way for more secure and resilient federated learning systems.



## **11. A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1**

cs.CV

Code at: https://github.com/VILA-Lab/M-Attack

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10635v1) [paper-pdf](http://arxiv.org/pdf/2503.10635v1)

**Authors**: Zhaoyi Li, Xiaohan Zhao, Dong-Dong Wu, Jiacheng Cui, Zhiqiang Shen

**Abstract**: Despite promising performance on open-source large vision-language models (LVLMs), transfer-based targeted attacks often fail against black-box commercial LVLMs. Analyzing failed adversarial perturbations reveals that the learned perturbations typically originate from a uniform distribution and lack clear semantic details, resulting in unintended responses. This critical absence of semantic information leads commercial LVLMs to either ignore the perturbation entirely or misinterpret its embedded semantics, thereby causing the attack to fail. To overcome these issues, we notice that identifying core semantic objects is a key objective for models trained with various datasets and methodologies. This insight motivates our approach that refines semantic clarity by encoding explicit semantic details within local regions, thus ensuring interoperability and capturing finer-grained features, and by concentrating modifications on semantically rich areas rather than applying them uniformly. To achieve this, we propose a simple yet highly effective solution: at each optimization step, the adversarial image is cropped randomly by a controlled aspect ratio and scale, resized, and then aligned with the target image in the embedding space. Experimental results confirm our hypothesis. Our adversarial examples crafted with local-aggregated perturbations focused on crucial regions exhibit surprisingly good transferability to commercial LVLMs, including GPT-4.5, GPT-4o, Gemini-2.0-flash, Claude-3.5-sonnet, Claude-3.7-sonnet, and even reasoning models like o1, Claude-3.7-thinking and Gemini-2.0-flash-thinking. Our approach achieves success rates exceeding 90% on GPT-4.5, 4o, and o1, significantly outperforming all prior state-of-the-art attack methods. Our optimized adversarial examples under different configurations and training code are available at https://github.com/VILA-Lab/M-Attack.



## **12. Hierarchical Self-Supervised Adversarial Training for Robust Vision Models in Histopathology**

cs.CV

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10629v1) [paper-pdf](http://arxiv.org/pdf/2503.10629v1)

**Authors**: Hashmat Shadab Malik, Shahina Kunhimon, Muzammal Naseer, Fahad Shahbaz Khan, Salman Khan

**Abstract**: Adversarial attacks pose significant challenges for vision models in critical fields like healthcare, where reliability is essential. Although adversarial training has been well studied in natural images, its application to biomedical and microscopy data remains limited. Existing self-supervised adversarial training methods overlook the hierarchical structure of histopathology images, where patient-slide-patch relationships provide valuable discriminative signals. To address this, we propose Hierarchical Self-Supervised Adversarial Training (HSAT), which exploits these properties to craft adversarial examples using multi-level contrastive learning and integrate it into adversarial training for enhanced robustness. We evaluate HSAT on multiclass histopathology dataset OpenSRH and the results show that HSAT outperforms existing methods from both biomedical and natural image domains. HSAT enhances robustness, achieving an average gain of 54.31% in the white-box setting and reducing performance drops to 3-4% in the black-box setting, compared to 25-30% for the baseline. These results set a new benchmark for adversarial training in this domain, paving the way for more robust models. Our Code for training and evaluation is available at https://github.com/HashmatShadab/HSAT.



## **13. Towards Class-wise Robustness Analysis**

cs.LG

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2411.19853v2) [paper-pdf](http://arxiv.org/pdf/2411.19853v2)

**Authors**: Tejaswini Medi, Julia Grabinski, Margret Keuper

**Abstract**: While being very successful in solving many downstream tasks, the application of deep neural networks is limited in real-life scenarios because of their susceptibility to domain shifts such as common corruptions, and adversarial attacks. The existence of adversarial examples and data corruption significantly reduces the performance of deep classification models. Researchers have made strides in developing robust neural architectures to bolster decisions of deep classifiers. However, most of these works rely on effective adversarial training methods, and predominantly focus on overall model robustness, disregarding class-wise differences in robustness, which are critical. Exploiting weakly robust classes is a potential avenue for attackers to fool the image recognition models. Therefore, this study investigates class-to-class biases across adversarially trained robust classification models to understand their latent space structures and analyze their strong and weak class-wise properties. We further assess the robustness of classes against common corruptions and adversarial attacks, recognizing that class vulnerability extends beyond the number of correct classifications for a specific class. We find that the number of false positives of classes as specific target classes significantly impacts their vulnerability to attacks. Through our analysis on the Class False Positive Score, we assess a fair evaluation of how susceptible each class is to misclassification.



## **14. TH-Bench: Evaluating Evading Attacks via Humanizing AI Text on Machine-Generated Text Detectors**

cs.CR

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.08708v2) [paper-pdf](http://arxiv.org/pdf/2503.08708v2)

**Authors**: Jingyi Zheng, Junfeng Wang, Zhen Sun, Wenhan Dong, Yule Liu, Xinlei He

**Abstract**: As Large Language Models (LLMs) advance, Machine-Generated Texts (MGTs) have become increasingly fluent, high-quality, and informative. Existing wide-range MGT detectors are designed to identify MGTs to prevent the spread of plagiarism and misinformation. However, adversaries attempt to humanize MGTs to evade detection (named evading attacks), which requires only minor modifications to bypass MGT detectors. Unfortunately, existing attacks generally lack a unified and comprehensive evaluation framework, as they are assessed using different experimental settings, model architectures, and datasets. To fill this gap, we introduce the Text-Humanization Benchmark (TH-Bench), the first comprehensive benchmark to evaluate evading attacks against MGT detectors. TH-Bench evaluates attacks across three key dimensions: evading effectiveness, text quality, and computational overhead. Our extensive experiments evaluate 6 state-of-the-art attacks against 13 MGT detectors across 6 datasets, spanning 19 domains and generated by 11 widely used LLMs. Our findings reveal that no single evading attack excels across all three dimensions. Through in-depth analysis, we highlight the strengths and limitations of different attacks. More importantly, we identify a trade-off among three dimensions and propose two optimization insights. Through preliminary experiments, we validate their correctness and effectiveness, offering potential directions for future research.



## **15. I Can Tell Your Secrets: Inferring Privacy Attributes from Mini-app Interaction History in Super-apps**

cs.CR

Accepted by USENIX Security 2025

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10239v1) [paper-pdf](http://arxiv.org/pdf/2503.10239v1)

**Authors**: Yifeng Cai, Ziqi Zhang, Mengyu Yao, Junlin Liu, Xiaoke Zhao, Xinyi Fu, Ruoyu Li, Zhe Li, Xiangqun Chen, Yao Guo, Ding Li

**Abstract**: Super-apps have emerged as comprehensive platforms integrating various mini-apps to provide diverse services. While super-apps offer convenience and enriched functionality, they can introduce new privacy risks. This paper reveals a new privacy leakage source in super-apps: mini-app interaction history, including mini-app usage history (Mini-H) and operation history (Op-H). Mini-H refers to the history of mini-apps accessed by users, such as their frequency and categories. Op-H captures user interactions within mini-apps, including button clicks, bar drags, and image views. Super-apps can naturally collect these data without instrumentation due to the web-based feature of mini-apps. We identify these data types as novel and unexplored privacy risks through a literature review of 30 papers and an empirical analysis of 31 super-apps. We design a mini-app interaction history-oriented inference attack (THEFT), to exploit this new vulnerability. Using THEFT, the insider threats within the low-privilege business department of the super-app vendor acting as the adversary can achieve more than 95.5% accuracy in inferring privacy attributes of over 16.1% of users. THEFT only requires a small training dataset of 200 users from public breached databases on the Internet. We also engage with super-app vendors and a standards association to increase industry awareness and commitment to protect this data. Our contributions are significant in identifying overlooked privacy risks, demonstrating the effectiveness of a new attack, and influencing industry practices toward better privacy protection in the super-app ecosystem.



## **16. Robustness Tokens: Towards Adversarial Robustness of Transformers**

cs.LG

This paper has been accepted for publication at the European  Conference on Computer Vision (ECCV), 2024

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10191v1) [paper-pdf](http://arxiv.org/pdf/2503.10191v1)

**Authors**: Brian Pulfer, Yury Belousov, Slava Voloshynovskiy

**Abstract**: Recently, large pre-trained foundation models have become widely adopted by machine learning practitioners for a multitude of tasks. Given that such models are publicly available, relying on their use as backbone models for downstream tasks might result in high vulnerability to adversarial attacks crafted with the same public model. In this work, we propose Robustness Tokens, a novel approach specific to the transformer architecture that fine-tunes a few additional private tokens with low computational requirements instead of tuning model parameters as done in traditional adversarial training. We show that Robustness Tokens make Vision Transformer models significantly more robust to white-box adversarial attacks while also retaining the original downstream performances.



## **17. Can't Slow me Down: Learning Robust and Hardware-Adaptive Object Detectors against Latency Attacks for Edge Devices**

cs.CV

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2412.02171v2) [paper-pdf](http://arxiv.org/pdf/2412.02171v2)

**Authors**: Tianyi Wang, Zichen Wang, Cong Wang, Yuanchao Shu, Ruilong Deng, Peng Cheng, Jiming Chen

**Abstract**: Object detection is a fundamental enabler for many real-time downstream applications such as autonomous driving, augmented reality and supply chain management. However, the algorithmic backbone of neural networks is brittle to imperceptible perturbations in the system inputs, which were generally known as misclassifying attacks. By targeting the real-time processing capability, a new class of latency attacks are reported recently. They exploit new attack surfaces in object detectors by creating a computational bottleneck in the post-processing module, that leads to cascading failure and puts the real-time downstream tasks at risks. In this work, we take an initial attempt to defend against this attack via background-attentive adversarial training that is also cognizant of the underlying hardware capabilities. We first draw system-level connections between latency attack and hardware capacity across heterogeneous GPU devices. Based on the particular adversarial behaviors, we utilize objectness loss as a proxy and build background attention into the adversarial training pipeline, and achieve a reasonable balance between clean and robust accuracy. The extensive experiments demonstrate the defense effectiveness of restoring real-time processing capability from $13$ FPS to $43$ FPS on Jetson Orin NX, with a better trade-off between the clean and robust accuracy.



## **18. Prompt-Driven Contrastive Learning for Transferable Adversarial Attacks**

cs.CV

Accepted to ECCV 2024 (Oral), Project Page:  https://PDCL-Attack.github.io

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2407.20657v2) [paper-pdf](http://arxiv.org/pdf/2407.20657v2)

**Authors**: Hunmin Yang, Jongoh Jeong, Kuk-Jin Yoon

**Abstract**: Recent vision-language foundation models, such as CLIP, have demonstrated superior capabilities in learning representations that can be transferable across diverse range of downstream tasks and domains. With the emergence of such powerful models, it has become crucial to effectively leverage their capabilities in tackling challenging vision tasks. On the other hand, only a few works have focused on devising adversarial examples that transfer well to both unknown domains and model architectures. In this paper, we propose a novel transfer attack method called PDCL-Attack, which leverages the CLIP model to enhance the transferability of adversarial perturbations generated by a generative model-based attack framework. Specifically, we formulate an effective prompt-driven feature guidance by harnessing the semantic representation power of text, particularly from the ground-truth class labels of input images. To the best of our knowledge, we are the first to introduce prompt learning to enhance the transferable generative attacks. Extensive experiments conducted across various cross-domain and cross-model settings empirically validate our approach, demonstrating its superiority over state-of-the-art methods.



## **19. Prevailing against Adversarial Noncentral Disturbances: Exact Recovery of Linear Systems with the $l_1$-norm Estimator**

math.OC

8 pages, 2 figures

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2410.03218v5) [paper-pdf](http://arxiv.org/pdf/2410.03218v5)

**Authors**: Jihun Kim, Javad Lavaei

**Abstract**: This paper studies the linear system identification problem in the general case where the disturbance is sub-Gaussian, correlated, and possibly adversarial. First, we consider the case with noncentral (nonzero-mean) disturbances for which the ordinary least-squares (OLS) method fails to correctly identify the system. We prove that the $l_1$-norm estimator accurately identifies the system under the condition that each disturbance has equal probabilities of being positive or negative. This condition restricts the sign of each disturbance but allows its magnitude to be arbitrary. Second, we consider the case where each disturbance is adversarial with the model that the attack times happen occasionally but the distributions of the attack values are arbitrary. We show that when the probability of having an attack at a given time is less than 0.5 and each attack spans the entire space in expectation, the $l_1$-norm estimator prevails against any adversarial noncentral disturbances and the exact recovery is achieved within a finite time. These results pave the way to effectively defend against arbitrarily large noncentral attacks in safety-critical systems.



## **20. AnywhereDoor: Multi-Target Backdoor Attacks on Object Detection**

cs.CR

This work was intended as a replacement of arXiv:2411.14243 and any  subsequent updates will appear there

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.06529v2) [paper-pdf](http://arxiv.org/pdf/2503.06529v2)

**Authors**: Jialin Lu, Junjie Shan, Ziqi Zhao, Ka-Ho Chow

**Abstract**: As object detection becomes integral to many safety-critical applications, understanding its vulnerabilities is essential. Backdoor attacks, in particular, pose a serious threat by implanting hidden triggers in victim models, which adversaries can later exploit to induce malicious behaviors during inference. However, current understanding is limited to single-target attacks, where adversaries must define a fixed malicious behavior (target) before training, making inference-time adaptability impossible. Given the large output space of object detection (including object existence prediction, bounding box estimation, and classification), the feasibility of flexible, inference-time model control remains unexplored. This paper introduces AnywhereDoor, a multi-target backdoor attack for object detection. Once implanted, AnywhereDoor allows adversaries to make objects disappear, fabricate new ones, or mislabel them, either across all object classes or specific ones, offering an unprecedented degree of control. This flexibility is enabled by three key innovations: (i) objective disentanglement to scale the number of supported targets; (ii) trigger mosaicking to ensure robustness even against region-based detectors; and (iii) strategic batching to address object-level data imbalances that hinder manipulation. Extensive experiments demonstrate that AnywhereDoor grants attackers a high degree of control, improving attack success rates by 26% compared to adaptations of existing methods for such flexible control.



## **21. The Power of LLM-Generated Synthetic Data for Stance Detection in Online Political Discussions**

cs.CL

ICLR 2025 Spotlight

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2406.12480v2) [paper-pdf](http://arxiv.org/pdf/2406.12480v2)

**Authors**: Stefan Sylvius Wagner, Maike Behrendt, Marc Ziegele, Stefan Harmeling

**Abstract**: Stance detection holds great potential to improve online political discussions through its deployment in discussion platforms for purposes such as content moderation, topic summarization or to facilitate more balanced discussions. Typically, transformer-based models are employed directly for stance detection, requiring vast amounts of data. However, the wide variety of debate topics in online political discussions makes data collection particularly challenging. LLMs have revived stance detection, but their online deployment in online political discussions faces challenges like inconsistent outputs, biases, and vulnerability to adversarial attacks. We show how LLM-generated synthetic data can improve stance detection for online political discussions by using reliable traditional stance detection models for online deployment, while leveraging the text generation capabilities of LLMs for synthetic data generation in a secure offline environment. To achieve this, (i) we generate synthetic data for specific debate questions by prompting a Mistral-7B model and show that fine-tuning with the generated synthetic data can substantially improve the performance of stance detection, while remaining interpretable and aligned with real world data. (ii) Using the synthetic data as a reference, we can improve performance even further by identifying the most informative samples in an unlabelled dataset, i.e., those samples which the stance detection model is most uncertain about and can benefit from the most. By fine-tuning with both synthetic data and the most informative samples, we surpass the performance of the baseline model that is fine-tuned on all true labels, while labelling considerably less data.



## **22. Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting**

cs.LG

AISTATS 2025

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2412.08099v4) [paper-pdf](http://arxiv.org/pdf/2412.08099v4)

**Authors**: Fuqiang Liu, Sicong Jiang, Luis Miranda-Moreno, Seongjin Choi, Lijun Sun

**Abstract**: Large Language Models (LLMs) have recently demonstrated significant potential in time series forecasting, offering impressive capabilities in handling complex temporal data. However, their robustness and reliability in real-world applications remain under-explored, particularly concerning their susceptibility to adversarial attacks. In this paper, we introduce a targeted adversarial attack framework for LLM-based time series forecasting. By employing both gradient-free and black-box optimization methods, we generate minimal yet highly effective perturbations that significantly degrade the forecasting accuracy across multiple datasets and LLM architectures. Our experiments, which include models like LLMTime with GPT-3.5, GPT-4, LLaMa, and Mistral, TimeGPT, and TimeLLM show that adversarial attacks lead to much more severe performance degradation than random noise, and demonstrate the broad effectiveness of our attacks across different LLMs. The results underscore the critical vulnerabilities of LLMs in time series forecasting, highlighting the need for robust defense mechanisms to ensure their reliable deployment in practical applications. The code repository can be found at https://github.com/JohnsonJiang1996/AdvAttack_LLM4TS.



## **23. EIA: Environmental Injection Attack on Generalist Web Agents for Privacy Leakage**

cs.CR

Accepted by ICLR 2025

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2409.11295v5) [paper-pdf](http://arxiv.org/pdf/2409.11295v5)

**Authors**: Zeyi Liao, Lingbo Mo, Chejian Xu, Mintong Kang, Jiawei Zhang, Chaowei Xiao, Yuan Tian, Bo Li, Huan Sun

**Abstract**: Generalist web agents have demonstrated remarkable potential in autonomously completing a wide range of tasks on real websites, significantly boosting human productivity. However, web tasks, such as booking flights, usually involve users' PII, which may be exposed to potential privacy risks if web agents accidentally interact with compromised websites, a scenario that remains largely unexplored in the literature. In this work, we narrow this gap by conducting the first study on the privacy risks of generalist web agents in adversarial environments. First, we present a realistic threat model for attacks on the website, where we consider two adversarial targets: stealing users' specific PII or the entire user request. Then, we propose a novel attack method, termed Environmental Injection Attack (EIA). EIA injects malicious content designed to adapt well to environments where the agents operate and our work instantiates EIA specifically for privacy scenarios in web environments. We collect 177 action steps that involve diverse PII categories on realistic websites from the Mind2Web, and conduct experiments using one of the most capable generalist web agent frameworks to date. The results demonstrate that EIA achieves up to 70% ASR in stealing specific PII and 16% ASR for full user request. Additionally, by accessing the stealthiness and experimenting with a defensive system prompt, we indicate that EIA is hard to detect and mitigate. Notably, attacks that are not well adapted for a webpage can be detected via human inspection, leading to our discussion about the trade-off between security and autonomy. However, extra attackers' efforts can make EIA seamlessly adapted, rendering such supervision ineffective. Thus, we further discuss the defenses at the pre- and post-deployment stages of the websites without relying on human supervision and call for more advanced defense strategies.



## **24. On the Robustness of Kolmogorov-Arnold Networks: An Adversarial Perspective**

cs.CV

Accepted at TMLR 2025

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2408.13809v3) [paper-pdf](http://arxiv.org/pdf/2408.13809v3)

**Authors**: Tal Alter, Raz Lapid, Moshe Sipper

**Abstract**: Kolmogorov-Arnold Networks (KANs) have recently emerged as a novel approach to function approximation, demonstrating remarkable potential in various domains. Despite their theoretical promise, the robustness of KANs under adversarial conditions has yet to be thoroughly examined. In this paper we explore the adversarial robustness of KANs, with a particular focus on image classification tasks. We assess the performance of KANs against standard white box and black-box adversarial attacks, comparing their resilience to that of established neural network architectures. Our experimental evaluation encompasses a variety of standard image classification benchmark datasets and investigates both fully connected and convolutional neural network architectures, of three sizes: small, medium, and large. We conclude that small- and medium-sized KANs (either fully connected or convolutional) are not consistently more robust than their standard counterparts, but that large-sized KANs are, by and large, more robust. This comprehensive evaluation of KANs in adversarial scenarios offers the first in-depth analysis of KAN security, laying the groundwork for future research in this emerging field.



## **25. ICMarks: A Robust Watermarking Framework for Integrated Circuit Physical Design IP Protection**

cs.CR

accept to TCAD (IEEE Transactions on Computer-Aided Design of  Integrated Circuits and Systems)

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2404.18407v2) [paper-pdf](http://arxiv.org/pdf/2404.18407v2)

**Authors**: Ruisi Zhang, Rachel Selina Rajarathnam, David Z. Pan, Farinaz Koushanfar

**Abstract**: Physical design watermarking on contemporary integrated circuit (IC) layout encodes signatures without considering the dense connections and design constraints, which could lead to performance degradation on the watermarked products. This paper presents ICMarks, a quality-preserving and robust watermarking framework for modern IC physical design. ICMarks embeds unique watermark signatures during the physical design's placement stage, thereby authenticating the IC layout ownership. ICMarks's novelty lies in (i) strategically identifying a region of cells to watermark with minimal impact on the layout performance and (ii) a two-level watermarking framework for augmented robustness toward potential removal and forging attacks. Extensive evaluations on benchmarks of different design objectives and sizes validate that ICMarks incurs no wirelength and timing metrics degradation, while successfully proving ownership. Furthermore, we demonstrate ICMarks is robust against two major watermarking attack categories, namely, watermark removal and forging attacks; even if the adversaries have prior knowledge of the watermarking schemes, the signatures cannot be removed without significantly undermining the layout quality.



## **26. Enhancing Adversarial Example Detection Through Model Explanation**

cs.CR

5 pages, 1 figure

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09735v1) [paper-pdf](http://arxiv.org/pdf/2503.09735v1)

**Authors**: Qian Ma, Ziping Ye

**Abstract**: Adversarial examples are a major problem for machine learning models, leading to a continuous search for effective defenses. One promising direction is to leverage model explanations to better understand and defend against these attacks. We looked at AmI, a method proposed by a NeurIPS 2018 spotlight paper that uses model explanations to detect adversarial examples. Our study shows that while AmI is a promising idea, its performance is too dependent on specific settings (e.g., hyperparameter) and external factors such as the operating system and the deep learning framework used, and such drawbacks limit AmI's practical usage. Our findings highlight the need for more robust defense mechanisms that are effective under various conditions. In addition, we advocate for a comprehensive evaluation framework for defense techniques.



## **27. All Your Knowledge Belongs to Us: Stealing Knowledge Graphs via Reasoning APIs**

cs.CR

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09727v1) [paper-pdf](http://arxiv.org/pdf/2503.09727v1)

**Authors**: Zhaohan Xi

**Abstract**: Knowledge graph reasoning (KGR), which answers complex, logical queries over large knowledge graphs (KGs), represents an important artificial intelligence task with a range of applications. Many KGs require extensive domain expertise and engineering effort to build and are hence considered proprietary within organizations and enterprises. Yet, spurred by their commercial and research potential, there is a growing trend to make KGR systems, (partially) built upon private KGs, publicly available through reasoning APIs.   The inherent tension between maintaining the confidentiality of KGs while ensuring the accessibility to KGR systems motivates our study of KG extraction attacks: the adversary aims to "steal" the private segments of the backend KG, leveraging solely black-box access to the KGR API. Specifically, we present KGX, an attack that extracts confidential sub-KGs with high fidelity under limited query budgets. At a high level, KGX progressively and adaptively queries the KGR API and integrates the query responses to reconstruct the private sub-KG. This extraction remains viable even if any query responses related to the private sub-KG are filtered. We validate the efficacy of KGX against both experimental and real-world KGR APIs. Interestingly, we find that typical countermeasures (e.g., injecting noise into query responses) are often ineffective against KGX. Our findings suggest the need for a more principled approach to developing and deploying KGR systems, as well as devising new defenses against KG extraction attacks.



## **28. Scale-Invariant Adversarial Attack against Arbitrary-scale Super-resolution**

cs.CV

17 pages, accepted by TIFS 2025

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.04385v2) [paper-pdf](http://arxiv.org/pdf/2503.04385v2)

**Authors**: Yihao Huang, Xin Luo, Qing Guo, Felix Juefei-Xu, Xiaojun Jia, Weikai Miao, Geguang Pu, Yang Liu

**Abstract**: The advent of local continuous image function (LIIF) has garnered significant attention for arbitrary-scale super-resolution (SR) techniques. However, while the vulnerabilities of fixed-scale SR have been assessed, the robustness of continuous representation-based arbitrary-scale SR against adversarial attacks remains an area warranting further exploration. The elaborately designed adversarial attacks for fixed-scale SR are scale-dependent, which will cause time-consuming and memory-consuming problems when applied to arbitrary-scale SR. To address this concern, we propose a simple yet effective ``scale-invariant'' SR adversarial attack method with good transferability, termed SIAGT. Specifically, we propose to construct resource-saving attacks by exploiting finite discrete points of continuous representation. In addition, we formulate a coordinate-dependent loss to enhance the cross-model transferability of the attack. The attack can significantly deteriorate the SR images while introducing imperceptible distortion to the targeted low-resolution (LR) images. Experiments carried out on three popular LIIF-based SR approaches and four classical SR datasets show remarkable attack performance and transferability of SIAGT.



## **29. RESTRAIN: Reinforcement Learning-Based Secure Framework for Trigger-Action IoT Environment**

cs.CR

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09513v1) [paper-pdf](http://arxiv.org/pdf/2503.09513v1)

**Authors**: Md Morshed Alam, Lokesh Chandra Das, Sandip Roy, Sachin Shetty, Weichao Wang

**Abstract**: Internet of Things (IoT) platforms with trigger-action capability allow event conditions to trigger actions in IoT devices autonomously by creating a chain of interactions. Adversaries exploit this chain of interactions to maliciously inject fake event conditions into IoT hubs, triggering unauthorized actions on target IoT devices to implement remote injection attacks. Existing defense mechanisms focus mainly on the verification of event transactions using physical event fingerprints to enforce the security policies to block unsafe event transactions. These approaches are designed to provide offline defense against injection attacks. The state-of-the-art online defense mechanisms offer real-time defense, but extensive reliability on the inference of attack impacts on the IoT network limits the generalization capability of these approaches. In this paper, we propose a platform-independent multi-agent online defense system, namely RESTRAIN, to counter remote injection attacks at runtime. RESTRAIN allows the defense agent to profile attack actions at runtime and leverages reinforcement learning to optimize a defense policy that complies with the security requirements of the IoT network. The experimental results show that the defense agent effectively takes real-time defense actions against complex and dynamic remote injection attacks and maximizes the security gain with minimal computational overhead.



## **30. Independence Tests for Language Models**

cs.LG

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2502.12292v2) [paper-pdf](http://arxiv.org/pdf/2502.12292v2)

**Authors**: Sally Zhu, Ahmed Ahmed, Rohith Kuditipudi, Percy Liang

**Abstract**: We consider the following problem: given the weights of two models, can we test whether they were trained independently -- i.e., from independent random initializations? We consider two settings: constrained and unconstrained. In the constrained setting, we make assumptions about model architecture and training and propose a family of statistical tests that yield exact p-values with respect to the null hypothesis that the models are trained from independent random initializations. These p-values are valid regardless of the composition of either model's training data; we compute them by simulating exchangeable copies of each model under our assumptions and comparing various similarity measures of weights and activations between the original two models versus these copies. We report the p-values from these tests on pairs of 21 open-weight models (210 total pairs) and correctly identify all pairs of non-independent models. Our tests remain effective even if one model was fine-tuned for many tokens. In the unconstrained setting, where we make no assumptions about training procedures, can change model architecture, and allow for adversarial evasion attacks, the previous tests no longer work. Instead, we propose a new test which matches hidden activations between two models, and which is robust to adversarial transformations and to changes in model architecture. The test can also do localized testing: identifying specific non-independent components of models. Though we no longer obtain exact p-values from this, empirically we find it behaves as one and reliably identifies non-independent models. Notably, we can use the test to identify specific parts of one model that are derived from another (e.g., how Llama 3.1-8B was pruned to initialize Llama 3.2-3B, or shared layers between Mistral-7B and StripedHyena-7B), and it is even robust to retraining individual layers of either model from scratch.



## **31. Revisiting Medical Image Retrieval via Knowledge Consolidation**

cs.CV

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09370v1) [paper-pdf](http://arxiv.org/pdf/2503.09370v1)

**Authors**: Yang Nan, Huichi Zhou, Xiaodan Xing, Giorgos Papanastasiou, Lei Zhu, Zhifan Gao, Alejandro F Fangi, Guang Yang

**Abstract**: As artificial intelligence and digital medicine increasingly permeate healthcare systems, robust governance frameworks are essential to ensure ethical, secure, and effective implementation. In this context, medical image retrieval becomes a critical component of clinical data management, playing a vital role in decision-making and safeguarding patient information. Existing methods usually learn hash functions using bottleneck features, which fail to produce representative hash codes from blended embeddings. Although contrastive hashing has shown superior performance, current approaches often treat image retrieval as a classification task, using category labels to create positive/negative pairs. Moreover, many methods fail to address the out-of-distribution (OOD) issue when models encounter external OOD queries or adversarial attacks. In this work, we propose a novel method to consolidate knowledge of hierarchical features and optimisation functions. We formulate the knowledge consolidation by introducing Depth-aware Representation Fusion (DaRF) and Structure-aware Contrastive Hashing (SCH). DaRF adaptively integrates shallow and deep representations into blended features, and SCH incorporates image fingerprints to enhance the adaptability of positive/negative pairings. These blended features further facilitate OOD detection and content-based recommendation, contributing to a secure AI-driven healthcare environment. Moreover, we present a content-guided ranking to improve the robustness and reproducibility of retrieval results. Our comprehensive assessments demonstrate that the proposed method could effectively recognise OOD samples and significantly outperform existing approaches in medical image retrieval (p<0.05). In particular, our method achieves a 5.6-38.9% improvement in mean Average Precision on the anatomical radiology dataset.



## **32. CyberLLMInstruct: A New Dataset for Analysing Safety of Fine-Tuned LLMs Using Cyber Security Data**

cs.CR

The paper is submitted to "The 48th International ACM SIGIR  Conference on Research and Development in Information Retrieval" and is  currently under review

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09334v1) [paper-pdf](http://arxiv.org/pdf/2503.09334v1)

**Authors**: Adel ElZemity, Budi Arief, Shujun Li

**Abstract**: The integration of large language models (LLMs) into cyber security applications presents significant opportunities, such as enhancing threat analysis and malware detection, but can also introduce critical risks and safety concerns, including personal data leakage and automated generation of new malware. To address these challenges, we developed CyberLLMInstruct, a dataset of 54,928 instruction-response pairs spanning cyber security tasks such as malware analysis, phishing simulations, and zero-day vulnerabilities. The dataset was constructed through a multi-stage process. This involved sourcing data from multiple resources, filtering and structuring it into instruction-response pairs, and aligning it with real-world scenarios to enhance its applicability. Seven open-source LLMs were chosen to test the usefulness of CyberLLMInstruct: Phi 3 Mini 3.8B, Mistral 7B, Qwen 2.5 7B, Llama 3 8B, Llama 3.1 8B, Gemma 2 9B, and Llama 2 70B. In our primary example, we rigorously assess the safety of fine-tuned models using the OWASP top 10 framework, finding that fine-tuning reduces safety resilience across all tested LLMs and every adversarial attack (e.g., the security score of Llama 3.1 8B against prompt injection drops from 0.95 to 0.15). In our second example, we show that these same fine-tuned models can also achieve up to 92.50 percent accuracy on the CyberMetric benchmark. These findings highlight a trade-off between performance and safety, showing the importance of adversarial testing and further research into fine-tuning methodologies that can mitigate safety risks while still improving performance across diverse datasets and domains. All scripts required to reproduce the dataset, along with examples and relevant resources for replicating our results, will be made available upon the paper's acceptance.



## **33. Detecting and Preventing Data Poisoning Attacks on AI Models**

cs.CR

9 pages, 8 figures

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09302v1) [paper-pdf](http://arxiv.org/pdf/2503.09302v1)

**Authors**: Halima I. Kure, Pradipta Sarkar, Ahmed B. Ndanusa, Augustine O. Nwajana

**Abstract**: This paper investigates the critical issue of data poisoning attacks on AI models, a growing concern in the ever-evolving landscape of artificial intelligence and cybersecurity. As advanced technology systems become increasingly prevalent across various sectors, the need for robust defence mechanisms against adversarial attacks becomes paramount. The study aims to develop and evaluate novel techniques for detecting and preventing data poisoning attacks, focusing on both theoretical frameworks and practical applications. Through a comprehensive literature review, experimental validation using the CIFAR-10 and Insurance Claims datasets, and the development of innovative algorithms, this paper seeks to enhance the resilience of AI models against malicious data manipulation. The study explores various methods, including anomaly detection, robust optimization strategies, and ensemble learning, to identify and mitigate the effects of poisoned data during model training. Experimental results indicate that data poisoning significantly degrades model performance, reducing classification accuracy by up to 27% in image recognition tasks (CIFAR-10) and 22% in fraud detection models (Insurance Claims dataset). The proposed defence mechanisms, including statistical anomaly detection and adversarial training, successfully mitigated poisoning effects, improving model robustness and restoring accuracy levels by an average of 15-20%. The findings further demonstrate that ensemble learning techniques provide an additional layer of resilience, reducing false positives and false negatives caused by adversarial data injections.



## **34. In-Context Defense in Computer Agents: An Empirical Study**

cs.AI

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09241v1) [paper-pdf](http://arxiv.org/pdf/2503.09241v1)

**Authors**: Pei Yang, Hai Ci, Mike Zheng Shou

**Abstract**: Computer agents powered by vision-language models (VLMs) have significantly advanced human-computer interaction, enabling users to perform complex tasks through natural language instructions. However, these agents are vulnerable to context deception attacks, an emerging threat where adversaries embed misleading content into the agent's operational environment, such as a pop-up window containing deceptive instructions. Existing defenses, such as instructing agents to ignore deceptive elements, have proven largely ineffective. As the first systematic study on protecting computer agents, we introduce textbf{in-context defense}, leveraging in-context learning and chain-of-thought (CoT) reasoning to counter such attacks. Our approach involves augmenting the agent's context with a small set of carefully curated exemplars containing both malicious environments and corresponding defensive responses. These exemplars guide the agent to first perform explicit defensive reasoning before action planning, reducing susceptibility to deceptive attacks. Experiments demonstrate the effectiveness of our method, reducing attack success rates by 91.2% on pop-up window attacks, 74.6% on average on environment injection attacks, while achieving 100% successful defenses against distracting advertisements. Our findings highlight that (1) defensive reasoning must precede action planning for optimal performance, and (2) a minimal number of exemplars (fewer than three) is sufficient to induce an agent's defensive behavior.



## **35. DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios**

cs.CL

Accepted to NeurIPS 2024 Datasets and Benchmarks Track (Camera-Ready)

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2410.23746v3) [paper-pdf](http://arxiv.org/pdf/2410.23746v3)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xinyi Yang, Yulin Yuan, Lidia S. Chao

**Abstract**: Detecting text generated by large language models (LLMs) is of great recent interest. With zero-shot methods like DetectGPT, detection capabilities have reached impressive levels. However, the reliability of existing detectors in real-world applications remains underexplored. In this study, we present a new benchmark, DetectRL, highlighting that even state-of-the-art (SOTA) detection techniques still underperformed in this task. We collected human-written datasets from domains where LLMs are particularly prone to misuse. Using popular LLMs, we generated data that better aligns with real-world applications. Unlike previous studies, we employed heuristic rules to create adversarial LLM-generated text, simulating various prompts usages, human revisions like word substitutions, and writing noises like spelling mistakes. Our development of DetectRL reveals the strengths and limitations of current SOTA detectors. More importantly, we analyzed the potential impact of writing styles, model types, attack methods, the text lengths, and real-world human writing factors on different types of detectors. We believe DetectRL could serve as an effective benchmark for assessing detectors in real-world scenarios, evolving with advanced attack methods, thus providing more stressful evaluation to drive the development of more efficient detectors. Data and code are publicly available at: https://github.com/NLP2CT/DetectRL.



## **36. AdvAD: Exploring Non-Parametric Diffusion for Imperceptible Adversarial Attacks**

cs.LG

Accept by NeurIPS 2024. Please cite this paper using the following  format: J. Li, Z. He, A. Luo, J. Hu, Z. Wang, X. Kang*, "AdvAD: Exploring  Non-Parametric Diffusion for Imperceptible Adversarial Attacks", the 38th  Annual Conference on Neural Information Processing Systems (NeurIPS),  Vancouver, Canada, Dec 9-15, 2024. Code: https://github.com/XianguiKang/AdvAD

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09124v1) [paper-pdf](http://arxiv.org/pdf/2503.09124v1)

**Authors**: Jin Li, Ziqiang He, Anwei Luo, Jian-Fang Hu, Z. Jane Wang, Xiangui Kang

**Abstract**: Imperceptible adversarial attacks aim to fool DNNs by adding imperceptible perturbation to the input data. Previous methods typically improve the imperceptibility of attacks by integrating common attack paradigms with specifically designed perception-based losses or the capabilities of generative models. In this paper, we propose Adversarial Attacks in Diffusion (AdvAD), a novel modeling framework distinct from existing attack paradigms. AdvAD innovatively conceptualizes attacking as a non-parametric diffusion process by theoretically exploring basic modeling approach rather than using the denoising or generation abilities of regular diffusion models requiring neural networks. At each step, much subtler yet effective adversarial guidance is crafted using only the attacked model without any additional network, which gradually leads the end of diffusion process from the original image to a desired imperceptible adversarial example. Grounded in a solid theoretical foundation of the proposed non-parametric diffusion process, AdvAD achieves high attack efficacy and imperceptibility with intrinsically lower overall perturbation strength. Additionally, an enhanced version AdvAD-X is proposed to evaluate the extreme of our novel framework under an ideal scenario. Extensive experiments demonstrate the effectiveness of the proposed AdvAD and AdvAD-X. Compared with state-of-the-art imperceptible attacks, AdvAD achieves an average of 99.9$\%$ (+17.3$\%$) ASR with 1.34 (-0.97) $l_2$ distance, 49.74 (+4.76) PSNR and 0.9971 (+0.0043) SSIM against four prevalent DNNs with three different architectures on the ImageNet-compatible dataset. Code is available at https://github.com/XianguiKang/AdvAD.



## **37. C^2 ATTACK: Towards Representation Backdoor on CLIP via Concept Confusion**

cs.CR

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09095v1) [paper-pdf](http://arxiv.org/pdf/2503.09095v1)

**Authors**: Lijie Hu, Junchi Liao, Weimin Lyu, Shaopeng Fu, Tianhao Huang, Shu Yang, Guimin Hu, Di Wang

**Abstract**: Backdoor attacks pose a significant threat to deep learning models, enabling adversaries to embed hidden triggers that manipulate the behavior of the model during inference. Traditional backdoor attacks typically rely on inserting explicit triggers (e.g., external patches, or perturbations) into input data, but they often struggle to evade existing defense mechanisms. To address this limitation, we investigate backdoor attacks through the lens of the reasoning process in deep learning systems, drawing insights from interpretable AI. We conceptualize backdoor activation as the manipulation of learned concepts within the model's latent representations. Thus, existing attacks can be seen as implicit manipulations of these activated concepts during inference. This raises interesting questions: why not manipulate the concepts explicitly? This idea leads to our novel backdoor attack framework, Concept Confusion Attack (C^2 ATTACK), which leverages internal concepts in the model's reasoning as "triggers" without introducing explicit external modifications. By avoiding the use of real triggers and directly activating or deactivating specific concepts in latent spaces, our approach enhances stealth, making detection by existing defenses significantly harder. Using CLIP as a case study, experimental results demonstrate the effectiveness of C^2 ATTACK, achieving high attack success rates while maintaining robustness against advanced defenses.



## **38. Probing Latent Subspaces in LLM for AI Security: Identifying and Manipulating Adversarial States**

cs.LG

4 figures

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09066v1) [paper-pdf](http://arxiv.org/pdf/2503.09066v1)

**Authors**: Xin Wei Chia, Jonathan Pan

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they remain vulnerable to adversarial manipulations such as jailbreaking via prompt injection attacks. These attacks bypass safety mechanisms to generate restricted or harmful content. In this study, we investigated the underlying latent subspaces of safe and jailbroken states by extracting hidden activations from a LLM. Inspired by attractor dynamics in neuroscience, we hypothesized that LLM activations settle into semi stable states that can be identified and perturbed to induce state transitions. Using dimensionality reduction techniques, we projected activations from safe and jailbroken responses to reveal latent subspaces in lower dimensional spaces. We then derived a perturbation vector that when applied to safe representations, shifted the model towards a jailbreak state. Our results demonstrate that this causal intervention results in statistically significant jailbreak responses in a subset of prompts. Next, we probed how these perturbations propagate through the model's layers, testing whether the induced state change remains localized or cascades throughout the network. Our findings indicate that targeted perturbations induced distinct shifts in activations and model responses. Our approach paves the way for potential proactive defenses, shifting from traditional guardrail based methods to preemptive, model agnostic techniques that neutralize adversarial states at the representation level.



## **39. Battling Misinformation: An Empirical Study on Adversarial Factuality in Open-Source Large Language Models**

cs.CL

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.10690v1) [paper-pdf](http://arxiv.org/pdf/2503.10690v1)

**Authors**: Shahnewaz Karim Sakib, Anindya Bijoy Das, Shibbir Ahmed

**Abstract**: Adversarial factuality refers to the deliberate insertion of misinformation into input prompts by an adversary, characterized by varying levels of expressed confidence. In this study, we systematically evaluate the performance of several open-source large language models (LLMs) when exposed to such adversarial inputs. Three tiers of adversarial confidence are considered: strongly confident, moderately confident, and limited confidence. Our analysis encompasses eight LLMs: LLaMA 3.1 (8B), Phi 3 (3.8B), Qwen 2.5 (7B), Deepseek-v2 (16B), Gemma2 (9B), Falcon (7B), Mistrallite (7B), and LLaVA (7B). Empirical results indicate that LLaMA 3.1 (8B) exhibits a robust capability in detecting adversarial inputs, whereas Falcon (7B) shows comparatively lower performance. Notably, for the majority of the models, detection success improves as the adversary's confidence decreases; however, this trend is reversed for LLaMA 3.1 (8B) and Phi 3 (3.8B), where a reduction in adversarial confidence corresponds with diminished detection performance. Further analysis of the queries that elicited the highest and lowest rates of successful attacks reveals that adversarial attacks are more effective when targeting less commonly referenced or obscure information.



## **40. Quantitative Analysis of Deeply Quantized Tiny Neural Networks Robust to Adversarial Attacks**

cs.LG

arXiv admin note: substantial text overlap with arXiv:2304.12829

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.08973v1) [paper-pdf](http://arxiv.org/pdf/2503.08973v1)

**Authors**: Idris Zakariyya, Ferheen Ayaz, Mounia Kharbouche-Harrari, Jeremy Singer, Sye Loong Keoh, Danilo Pau, José Cano

**Abstract**: Reducing the memory footprint of Machine Learning (ML) models, especially Deep Neural Networks (DNNs), is imperative to facilitate their deployment on resource-constrained edge devices. However, a notable drawback of DNN models lies in their susceptibility to adversarial attacks, wherein minor input perturbations can deceive them. A primary challenge revolves around the development of accurate, resilient, and compact DNN models suitable for deployment on resource-constrained edge devices. This paper presents the outcomes of a compact DNN model that exhibits resilience against both black-box and white-box adversarial attacks. This work has achieved this resilience through training with the QKeras quantization-aware training framework. The study explores the potential of QKeras and an adversarial robustness technique, Jacobian Regularization (JR), to co-optimize the DNN architecture through per-layer JR methodology. As a result, this paper has devised a DNN model employing this co-optimization strategy based on Stochastic Ternary Quantization (STQ). Its performance was compared against existing DNN models in the face of various white-box and black-box attacks. The experimental findings revealed that, the proposed DNN model had small footprint and on average, it exhibited better performance than Quanos and DS-CNN MLCommons/TinyML (MLC/T) benchmarks when challenged with white-box and black-box attacks, respectively, on the CIFAR-10 image and Google Speech Commands audio datasets.



## **41. Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities**

cs.CL

Accepted to NAACL 2025 Main (oral)

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2410.18469v3) [paper-pdf](http://arxiv.org/pdf/2410.18469v3)

**Authors**: Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao

**Abstract**: Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99\% ASR on GPT-3.5 and 49\% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety.



## **42. Backtracking for Safety**

cs.CL

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08919v1) [paper-pdf](http://arxiv.org/pdf/2503.08919v1)

**Authors**: Bilgehan Sel, Dingcheng Li, Phillip Wallis, Vaishakh Keshava, Ming Jin, Siddhartha Reddy Jonnalagadda

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various tasks, but ensuring their safety and alignment with human values remains crucial. Current safety alignment methods, such as supervised fine-tuning and reinforcement learning-based approaches, can exhibit vulnerabilities to adversarial attacks and often result in shallow safety alignment, primarily focusing on preventing harmful content in the initial tokens of the generated output. While methods like resetting can help recover from unsafe generations by discarding previous tokens and restarting the generation process, they are not well-suited for addressing nuanced safety violations like toxicity that may arise within otherwise benign and lengthy generations. In this paper, we propose a novel backtracking method designed to address these limitations. Our method allows the model to revert to a safer generation state, not necessarily at the beginning, when safety violations occur during generation. This approach enables targeted correction of problematic segments without discarding the entire generated text, thereby preserving efficiency. We demonstrate that our method dramatically reduces toxicity appearing through the generation process with minimal impact to efficiency.



## **43. Human-Readable Adversarial Prompts: An Investigation into LLM Vulnerabilities Using Situational Context**

cs.CL

arXiv admin note: text overlap with arXiv:2407.14644

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2412.16359v2) [paper-pdf](http://arxiv.org/pdf/2412.16359v2)

**Authors**: Nilanjana Das, Edward Raff, Manas Gaur

**Abstract**: Previous studies that uncovered vulnerabilities in large language models (LLMs) frequently employed nonsensical adversarial prompts. However, such prompts can now be readily identified using automated detection techniques. To further strengthen adversarial attacks, we focus on human-readable adversarial prompts, which are more realistic and potent threats. Our key contributions are (1) situation-driven attacks leveraging movie scripts as context to create human-readable prompts that successfully deceive LLMs, (2) adversarial suffix conversion to transform nonsensical adversarial suffixes into independent meaningful text, and (3) AdvPrompter with p-nucleus sampling, a method to generate diverse, human-readable adversarial suffixes, improving attack efficacy in models like GPT-3.5 and Gemma 7B.



## **44. Birds look like cars: Adversarial analysis of intrinsically interpretable deep learning**

cs.LG

Preprint

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08636v1) [paper-pdf](http://arxiv.org/pdf/2503.08636v1)

**Authors**: Hubert Baniecki, Przemyslaw Biecek

**Abstract**: A common belief is that intrinsically interpretable deep learning models ensure a correct, intuitive understanding of their behavior and offer greater robustness against accidental errors or intentional manipulation. However, these beliefs have not been comprehensively verified, and growing evidence casts doubt on them. In this paper, we highlight the risks related to overreliance and susceptibility to adversarial manipulation of these so-called "intrinsically (aka inherently) interpretable" models by design. We introduce two strategies for adversarial analysis with prototype manipulation and backdoor attacks against prototype-based networks, and discuss how concept bottleneck models defend against these attacks. Fooling the model's reasoning by exploiting its use of latent prototypes manifests the inherent uninterpretability of deep neural networks, leading to a false sense of security reinforced by a visual confirmation bias. The reported limitations of prototype-based networks put their trustworthiness and applicability into question, motivating further work on the robustness and alignment of (deep) interpretable models.



## **45. Beyond Optimal Fault Tolerance**

cs.DC

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2501.06044v4) [paper-pdf](http://arxiv.org/pdf/2501.06044v4)

**Authors**: Andrew Lewis-Pye, Tim Roughgarden

**Abstract**: The optimal fault-tolerance achievable by any protocol has been characterized in a wide range of settings. For example, for state machine replication (SMR) protocols operating in the partially synchronous setting, it is possible to simultaneously guarantee consistency against $\alpha$-bounded adversaries (i.e., adversaries that control less than an $\alpha$ fraction of the participants) and liveness against $\beta$-bounded adversaries if and only if $\alpha + 2\beta \leq 1$.   This paper characterizes to what extent "better-than-optimal" fault-tolerance guarantees are possible for SMR protocols when the standard consistency requirement is relaxed to allow a bounded number $r$ of consistency violations. We prove that bounding rollback is impossible without additional timing assumptions and investigate protocols that tolerate and recover from consistency violations whenever message delays around the time of an attack are bounded by a parameter $\Delta^*$ (which may be arbitrarily larger than the parameter $\Delta$ that bounds post-GST message delays in the partially synchronous model). Here, a protocol's fault-tolerance can be a non-constant function of $r$, and we prove, for each $r$, matching upper and lower bounds on the optimal "recoverable fault-tolerance" achievable by any SMR protocol. For example, for protocols that guarantee liveness against 1/3-bounded adversaries in the partially synchronous setting, a 5/9-bounded adversary can always cause one consistency violation but not two, and a 2/3-bounded adversary can always cause two consistency violations but not three. Our positive results are achieved through a generic "recovery procedure" that can be grafted on to any accountable SMR protocol and restores consistency following a violation while rolling back only transactions that were finalized in the previous $2\Delta^*$ timesteps.



## **46. Low-Cost Privacy-Preserving Decentralized Learning**

cs.LG

24 pages, accepted at Pets 2025

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2403.11795v3) [paper-pdf](http://arxiv.org/pdf/2403.11795v3)

**Authors**: Sayan Biswas, Davide Frey, Romaric Gaudel, Anne-Marie Kermarrec, Dimitri Lerévérend, Rafael Pires, Rishi Sharma, François Taïani

**Abstract**: Decentralized learning (DL) is an emerging paradigm of collaborative machine learning that enables nodes in a network to train models collectively without sharing their raw data or relying on a central server. This paper introduces Zip-DL, a privacy-aware DL algorithm that leverages correlated noise to achieve robust privacy against local adversaries while ensuring efficient convergence at low communication costs. By progressively neutralizing the noise added during distributed averaging, Zip-DL combines strong privacy guarantees with high model accuracy. Its design requires only one communication round per gradient descent iteration, significantly reducing communication overhead compared to competitors. We establish theoretical bounds on both convergence speed and privacy guarantees. Moreover, extensive experiments demonstrating Zip-DL's practical applicability make it outperform state-of-the-art methods in the accuracy vs. vulnerability trade-off. Specifically, Zip-DL (i) reduces membership-inference attack success rates by up to 35% compared to baseline DL, (ii) decreases attack efficacy by up to 13% compared to competitors offering similar utility, and (iii) achieves up to 59% higher accuracy to completely nullify a basic attack scenario, compared to a state-of-the-art privacy-preserving approach under the same threat model. These results position Zip-DL as a practical and efficient solution for privacy-preserving decentralized learning in real-world applications.



## **47. Adv-CPG: A Customized Portrait Generation Framework with Facial Adversarial Attacks**

cs.CV

Accepted by CVPR-25

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08269v1) [paper-pdf](http://arxiv.org/pdf/2503.08269v1)

**Authors**: Junying Wang, Hongyuan Zhang, Yuan Yuan

**Abstract**: Recent Customized Portrait Generation (CPG) methods, taking a facial image and a textual prompt as inputs, have attracted substantial attention. Although these methods generate high-fidelity portraits, they fail to prevent the generated portraits from being tracked and misused by malicious face recognition systems. To address this, this paper proposes a Customized Portrait Generation framework with facial Adversarial attacks (Adv-CPG). Specifically, to achieve facial privacy protection, we devise a lightweight local ID encryptor and an encryption enhancer. They implement progressive double-layer encryption protection by directly injecting the target identity and adding additional identity guidance, respectively. Furthermore, to accomplish fine-grained and personalized portrait generation, we develop a multi-modal image customizer capable of generating controlled fine-grained facial features. To the best of our knowledge, Adv-CPG is the first study that introduces facial adversarial attacks into CPG. Extensive experiments demonstrate the superiority of Adv-CPG, e.g., the average attack success rate of the proposed Adv-CPG is 28.1% and 2.86% higher compared to the SOTA noise-based attack methods and unconstrained attack methods, respectively.



## **48. A Grey-box Text Attack Framework using Explainable AI**

cs.CL

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08226v1) [paper-pdf](http://arxiv.org/pdf/2503.08226v1)

**Authors**: Esther Chiramal, Kelvin Soh Boon Kai

**Abstract**: Explainable AI is a strong strategy implemented to understand complex black-box model predictions in a human interpretable language. It provides the evidence required to execute the use of trustworthy and reliable AI systems. On the other hand, however, it also opens the door to locating possible vulnerabilities in an AI model. Traditional adversarial text attack uses word substitution, data augmentation techniques and gradient-based attacks on powerful pre-trained Bidirectional Encoder Representations from Transformers (BERT) variants to generate adversarial sentences. These attacks are generally whitebox in nature and not practical as they can be easily detected by humans E.g. Changing the word from "Poor" to "Rich". We proposed a simple yet effective Grey-box cum Black-box approach that does not require the knowledge of the model while using a set of surrogate Transformer/BERT models to perform the attack using Explainable AI techniques. As Transformers are the current state-of-the-art models for almost all Natural Language Processing (NLP) tasks, an attack generated from BERT1 is transferable to BERT2. This transferability is made possible due to the attention mechanism in the transformer that allows the model to capture long-range dependencies in a sequence. Using the power of BERT generalisation via attention, we attempt to exploit how transformers learn by attacking a few surrogate transformer variants which are all based on a different architecture. We demonstrate that this approach is highly effective to generate semantically good sentences by changing as little as one word that is not detectable by humans while still fooling other BERT models.



## **49. Guardians of the Agentic System: Preventing Many Shots Jailbreak with Agentic System**

cs.CR

18 pages, 7 figures

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2502.16750v3) [paper-pdf](http://arxiv.org/pdf/2502.16750v3)

**Authors**: Saikat Barua, Mostafizur Rahman, Md Jafor Sadek, Rafiul Islam, Shehenaz Khaled, Ahmedul Kabir

**Abstract**: The autonomous AI agents using large language models can create undeniable values in all span of the society but they face security threats from adversaries that warrants immediate protective solutions because trust and safety issues arise. Considering the many-shot jailbreaking and deceptive alignment as some of the main advanced attacks, that cannot be mitigated by the static guardrails used during the supervised training, points out a crucial research priority for real world robustness. The combination of static guardrails in dynamic multi-agent system fails to defend against those attacks. We intend to enhance security for LLM-based agents through the development of new evaluation frameworks which identify and counter threats for safe operational deployment. Our work uses three examination methods to detect rogue agents through a Reverse Turing Test and analyze deceptive alignment through multi-agent simulations and develops an anti-jailbreaking system by testing it with GEMINI 1.5 pro and llama-3.3-70B, deepseek r1 models using tool-mediated adversarial scenarios. The detection capabilities are strong such as 94\% accuracy for GEMINI 1.5 pro yet the system suffers persistent vulnerabilities when under long attacks as prompt length increases attack success rates (ASR) and diversity metrics become ineffective in prediction while revealing multiple complex system faults. The findings demonstrate the necessity of adopting flexible security systems based on active monitoring that can be performed by the agents themselves together with adaptable interventions by system admin as the current models can create vulnerabilities that can lead to the unreliable and vulnerable system. So, in our work, we try to address such situations and propose a comprehensive framework to counteract the security issues.



## **50. Dialogue Injection Attack: Jailbreaking LLMs through Context Manipulation**

cs.CL

17 pages, 10 figures

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08195v1) [paper-pdf](http://arxiv.org/pdf/2503.08195v1)

**Authors**: Wenlong Meng, Fan Zhang, Wendao Yao, Zhenyuan Guo, Yuwei Li, Chengkun Wei, Wenzhi Chen

**Abstract**: Large language models (LLMs) have demonstrated significant utility in a wide range of applications; however, their deployment is plagued by security vulnerabilities, notably jailbreak attacks. These attacks manipulate LLMs to generate harmful or unethical content by crafting adversarial prompts. While much of the current research on jailbreak attacks has focused on single-turn interactions, it has largely overlooked the impact of historical dialogues on model behavior. In this paper, we introduce a novel jailbreak paradigm, Dialogue Injection Attack (DIA), which leverages the dialogue history to enhance the success rates of such attacks. DIA operates in a black-box setting, requiring only access to the chat API or knowledge of the LLM's chat template. We propose two methods for constructing adversarial historical dialogues: one adapts gray-box prefilling attacks, and the other exploits deferred responses. Our experiments show that DIA achieves state-of-the-art attack success rates on recent LLMs, including Llama-3.1 and GPT-4o. Additionally, we demonstrate that DIA can bypass 5 different defense mechanisms, highlighting its robustness and effectiveness.



