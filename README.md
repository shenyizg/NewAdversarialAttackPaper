# Latest Adversarial Attack Papers
**update at 2025-03-15 11:03:11**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1**

cs.CV

Code at: https://github.com/VILA-Lab/M-Attack

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10635v1) [paper-pdf](http://arxiv.org/pdf/2503.10635v1)

**Authors**: Zhaoyi Li, Xiaohan Zhao, Dong-Dong Wu, Jiacheng Cui, Zhiqiang Shen

**Abstract**: Despite promising performance on open-source large vision-language models (LVLMs), transfer-based targeted attacks often fail against black-box commercial LVLMs. Analyzing failed adversarial perturbations reveals that the learned perturbations typically originate from a uniform distribution and lack clear semantic details, resulting in unintended responses. This critical absence of semantic information leads commercial LVLMs to either ignore the perturbation entirely or misinterpret its embedded semantics, thereby causing the attack to fail. To overcome these issues, we notice that identifying core semantic objects is a key objective for models trained with various datasets and methodologies. This insight motivates our approach that refines semantic clarity by encoding explicit semantic details within local regions, thus ensuring interoperability and capturing finer-grained features, and by concentrating modifications on semantically rich areas rather than applying them uniformly. To achieve this, we propose a simple yet highly effective solution: at each optimization step, the adversarial image is cropped randomly by a controlled aspect ratio and scale, resized, and then aligned with the target image in the embedding space. Experimental results confirm our hypothesis. Our adversarial examples crafted with local-aggregated perturbations focused on crucial regions exhibit surprisingly good transferability to commercial LVLMs, including GPT-4.5, GPT-4o, Gemini-2.0-flash, Claude-3.5-sonnet, Claude-3.7-sonnet, and even reasoning models like o1, Claude-3.7-thinking and Gemini-2.0-flash-thinking. Our approach achieves success rates exceeding 90% on GPT-4.5, 4o, and o1, significantly outperforming all prior state-of-the-art attack methods. Our optimized adversarial examples under different configurations and training code are available at https://github.com/VILA-Lab/M-Attack.



## **2. Hierarchical Self-Supervised Adversarial Training for Robust Vision Models in Histopathology**

cs.CV

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10629v1) [paper-pdf](http://arxiv.org/pdf/2503.10629v1)

**Authors**: Hashmat Shadab Malik, Shahina Kunhimon, Muzammal Naseer, Fahad Shahbaz Khan, Salman Khan

**Abstract**: Adversarial attacks pose significant challenges for vision models in critical fields like healthcare, where reliability is essential. Although adversarial training has been well studied in natural images, its application to biomedical and microscopy data remains limited. Existing self-supervised adversarial training methods overlook the hierarchical structure of histopathology images, where patient-slide-patch relationships provide valuable discriminative signals. To address this, we propose Hierarchical Self-Supervised Adversarial Training (HSAT), which exploits these properties to craft adversarial examples using multi-level contrastive learning and integrate it into adversarial training for enhanced robustness. We evaluate HSAT on multiclass histopathology dataset OpenSRH and the results show that HSAT outperforms existing methods from both biomedical and natural image domains. HSAT enhances robustness, achieving an average gain of 54.31% in the white-box setting and reducing performance drops to 3-4% in the black-box setting, compared to 25-30% for the baseline. These results set a new benchmark for adversarial training in this domain, paving the way for more robust models. Our Code for training and evaluation is available at https://github.com/HashmatShadab/HSAT.



## **3. Towards Class-wise Robustness Analysis**

cs.LG

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2411.19853v2) [paper-pdf](http://arxiv.org/pdf/2411.19853v2)

**Authors**: Tejaswini Medi, Julia Grabinski, Margret Keuper

**Abstract**: While being very successful in solving many downstream tasks, the application of deep neural networks is limited in real-life scenarios because of their susceptibility to domain shifts such as common corruptions, and adversarial attacks. The existence of adversarial examples and data corruption significantly reduces the performance of deep classification models. Researchers have made strides in developing robust neural architectures to bolster decisions of deep classifiers. However, most of these works rely on effective adversarial training methods, and predominantly focus on overall model robustness, disregarding class-wise differences in robustness, which are critical. Exploiting weakly robust classes is a potential avenue for attackers to fool the image recognition models. Therefore, this study investigates class-to-class biases across adversarially trained robust classification models to understand their latent space structures and analyze their strong and weak class-wise properties. We further assess the robustness of classes against common corruptions and adversarial attacks, recognizing that class vulnerability extends beyond the number of correct classifications for a specific class. We find that the number of false positives of classes as specific target classes significantly impacts their vulnerability to attacks. Through our analysis on the Class False Positive Score, we assess a fair evaluation of how susceptible each class is to misclassification.



## **4. TH-Bench: Evaluating Evading Attacks via Humanizing AI Text on Machine-Generated Text Detectors**

cs.CR

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.08708v2) [paper-pdf](http://arxiv.org/pdf/2503.08708v2)

**Authors**: Jingyi Zheng, Junfeng Wang, Zhen Sun, Wenhan Dong, Yule Liu, Xinlei He

**Abstract**: As Large Language Models (LLMs) advance, Machine-Generated Texts (MGTs) have become increasingly fluent, high-quality, and informative. Existing wide-range MGT detectors are designed to identify MGTs to prevent the spread of plagiarism and misinformation. However, adversaries attempt to humanize MGTs to evade detection (named evading attacks), which requires only minor modifications to bypass MGT detectors. Unfortunately, existing attacks generally lack a unified and comprehensive evaluation framework, as they are assessed using different experimental settings, model architectures, and datasets. To fill this gap, we introduce the Text-Humanization Benchmark (TH-Bench), the first comprehensive benchmark to evaluate evading attacks against MGT detectors. TH-Bench evaluates attacks across three key dimensions: evading effectiveness, text quality, and computational overhead. Our extensive experiments evaluate 6 state-of-the-art attacks against 13 MGT detectors across 6 datasets, spanning 19 domains and generated by 11 widely used LLMs. Our findings reveal that no single evading attack excels across all three dimensions. Through in-depth analysis, we highlight the strengths and limitations of different attacks. More importantly, we identify a trade-off among three dimensions and propose two optimization insights. Through preliminary experiments, we validate their correctness and effectiveness, offering potential directions for future research.



## **5. I Can Tell Your Secrets: Inferring Privacy Attributes from Mini-app Interaction History in Super-apps**

cs.CR

Accepted by USENIX Security 2025

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10239v1) [paper-pdf](http://arxiv.org/pdf/2503.10239v1)

**Authors**: Yifeng Cai, Ziqi Zhang, Mengyu Yao, Junlin Liu, Xiaoke Zhao, Xinyi Fu, Ruoyu Li, Zhe Li, Xiangqun Chen, Yao Guo, Ding Li

**Abstract**: Super-apps have emerged as comprehensive platforms integrating various mini-apps to provide diverse services. While super-apps offer convenience and enriched functionality, they can introduce new privacy risks. This paper reveals a new privacy leakage source in super-apps: mini-app interaction history, including mini-app usage history (Mini-H) and operation history (Op-H). Mini-H refers to the history of mini-apps accessed by users, such as their frequency and categories. Op-H captures user interactions within mini-apps, including button clicks, bar drags, and image views. Super-apps can naturally collect these data without instrumentation due to the web-based feature of mini-apps. We identify these data types as novel and unexplored privacy risks through a literature review of 30 papers and an empirical analysis of 31 super-apps. We design a mini-app interaction history-oriented inference attack (THEFT), to exploit this new vulnerability. Using THEFT, the insider threats within the low-privilege business department of the super-app vendor acting as the adversary can achieve more than 95.5% accuracy in inferring privacy attributes of over 16.1% of users. THEFT only requires a small training dataset of 200 users from public breached databases on the Internet. We also engage with super-app vendors and a standards association to increase industry awareness and commitment to protect this data. Our contributions are significant in identifying overlooked privacy risks, demonstrating the effectiveness of a new attack, and influencing industry practices toward better privacy protection in the super-app ecosystem.



## **6. Robustness Tokens: Towards Adversarial Robustness of Transformers**

cs.LG

This paper has been accepted for publication at the European  Conference on Computer Vision (ECCV), 2024

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10191v1) [paper-pdf](http://arxiv.org/pdf/2503.10191v1)

**Authors**: Brian Pulfer, Yury Belousov, Slava Voloshynovskiy

**Abstract**: Recently, large pre-trained foundation models have become widely adopted by machine learning practitioners for a multitude of tasks. Given that such models are publicly available, relying on their use as backbone models for downstream tasks might result in high vulnerability to adversarial attacks crafted with the same public model. In this work, we propose Robustness Tokens, a novel approach specific to the transformer architecture that fine-tunes a few additional private tokens with low computational requirements instead of tuning model parameters as done in traditional adversarial training. We show that Robustness Tokens make Vision Transformer models significantly more robust to white-box adversarial attacks while also retaining the original downstream performances.



## **7. Can't Slow me Down: Learning Robust and Hardware-Adaptive Object Detectors against Latency Attacks for Edge Devices**

cs.CV

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2412.02171v2) [paper-pdf](http://arxiv.org/pdf/2412.02171v2)

**Authors**: Tianyi Wang, Zichen Wang, Cong Wang, Yuanchao Shu, Ruilong Deng, Peng Cheng, Jiming Chen

**Abstract**: Object detection is a fundamental enabler for many real-time downstream applications such as autonomous driving, augmented reality and supply chain management. However, the algorithmic backbone of neural networks is brittle to imperceptible perturbations in the system inputs, which were generally known as misclassifying attacks. By targeting the real-time processing capability, a new class of latency attacks are reported recently. They exploit new attack surfaces in object detectors by creating a computational bottleneck in the post-processing module, that leads to cascading failure and puts the real-time downstream tasks at risks. In this work, we take an initial attempt to defend against this attack via background-attentive adversarial training that is also cognizant of the underlying hardware capabilities. We first draw system-level connections between latency attack and hardware capacity across heterogeneous GPU devices. Based on the particular adversarial behaviors, we utilize objectness loss as a proxy and build background attention into the adversarial training pipeline, and achieve a reasonable balance between clean and robust accuracy. The extensive experiments demonstrate the defense effectiveness of restoring real-time processing capability from $13$ FPS to $43$ FPS on Jetson Orin NX, with a better trade-off between the clean and robust accuracy.



## **8. Prompt-Driven Contrastive Learning for Transferable Adversarial Attacks**

cs.CV

Accepted to ECCV 2024 (Oral), Project Page:  https://PDCL-Attack.github.io

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2407.20657v2) [paper-pdf](http://arxiv.org/pdf/2407.20657v2)

**Authors**: Hunmin Yang, Jongoh Jeong, Kuk-Jin Yoon

**Abstract**: Recent vision-language foundation models, such as CLIP, have demonstrated superior capabilities in learning representations that can be transferable across diverse range of downstream tasks and domains. With the emergence of such powerful models, it has become crucial to effectively leverage their capabilities in tackling challenging vision tasks. On the other hand, only a few works have focused on devising adversarial examples that transfer well to both unknown domains and model architectures. In this paper, we propose a novel transfer attack method called PDCL-Attack, which leverages the CLIP model to enhance the transferability of adversarial perturbations generated by a generative model-based attack framework. Specifically, we formulate an effective prompt-driven feature guidance by harnessing the semantic representation power of text, particularly from the ground-truth class labels of input images. To the best of our knowledge, we are the first to introduce prompt learning to enhance the transferable generative attacks. Extensive experiments conducted across various cross-domain and cross-model settings empirically validate our approach, demonstrating its superiority over state-of-the-art methods.



## **9. Prevailing against Adversarial Noncentral Disturbances: Exact Recovery of Linear Systems with the $l_1$-norm Estimator**

math.OC

8 pages, 2 figures

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2410.03218v5) [paper-pdf](http://arxiv.org/pdf/2410.03218v5)

**Authors**: Jihun Kim, Javad Lavaei

**Abstract**: This paper studies the linear system identification problem in the general case where the disturbance is sub-Gaussian, correlated, and possibly adversarial. First, we consider the case with noncentral (nonzero-mean) disturbances for which the ordinary least-squares (OLS) method fails to correctly identify the system. We prove that the $l_1$-norm estimator accurately identifies the system under the condition that each disturbance has equal probabilities of being positive or negative. This condition restricts the sign of each disturbance but allows its magnitude to be arbitrary. Second, we consider the case where each disturbance is adversarial with the model that the attack times happen occasionally but the distributions of the attack values are arbitrary. We show that when the probability of having an attack at a given time is less than 0.5 and each attack spans the entire space in expectation, the $l_1$-norm estimator prevails against any adversarial noncentral disturbances and the exact recovery is achieved within a finite time. These results pave the way to effectively defend against arbitrarily large noncentral attacks in safety-critical systems.



## **10. AnywhereDoor: Multi-Target Backdoor Attacks on Object Detection**

cs.CR

This work was intended as a replacement of arXiv:2411.14243 and any  subsequent updates will appear there

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.06529v2) [paper-pdf](http://arxiv.org/pdf/2503.06529v2)

**Authors**: Jialin Lu, Junjie Shan, Ziqi Zhao, Ka-Ho Chow

**Abstract**: As object detection becomes integral to many safety-critical applications, understanding its vulnerabilities is essential. Backdoor attacks, in particular, pose a serious threat by implanting hidden triggers in victim models, which adversaries can later exploit to induce malicious behaviors during inference. However, current understanding is limited to single-target attacks, where adversaries must define a fixed malicious behavior (target) before training, making inference-time adaptability impossible. Given the large output space of object detection (including object existence prediction, bounding box estimation, and classification), the feasibility of flexible, inference-time model control remains unexplored. This paper introduces AnywhereDoor, a multi-target backdoor attack for object detection. Once implanted, AnywhereDoor allows adversaries to make objects disappear, fabricate new ones, or mislabel them, either across all object classes or specific ones, offering an unprecedented degree of control. This flexibility is enabled by three key innovations: (i) objective disentanglement to scale the number of supported targets; (ii) trigger mosaicking to ensure robustness even against region-based detectors; and (iii) strategic batching to address object-level data imbalances that hinder manipulation. Extensive experiments demonstrate that AnywhereDoor grants attackers a high degree of control, improving attack success rates by 26% compared to adaptations of existing methods for such flexible control.



## **11. The Power of LLM-Generated Synthetic Data for Stance Detection in Online Political Discussions**

cs.CL

ICLR 2025 Spotlight

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2406.12480v2) [paper-pdf](http://arxiv.org/pdf/2406.12480v2)

**Authors**: Stefan Sylvius Wagner, Maike Behrendt, Marc Ziegele, Stefan Harmeling

**Abstract**: Stance detection holds great potential to improve online political discussions through its deployment in discussion platforms for purposes such as content moderation, topic summarization or to facilitate more balanced discussions. Typically, transformer-based models are employed directly for stance detection, requiring vast amounts of data. However, the wide variety of debate topics in online political discussions makes data collection particularly challenging. LLMs have revived stance detection, but their online deployment in online political discussions faces challenges like inconsistent outputs, biases, and vulnerability to adversarial attacks. We show how LLM-generated synthetic data can improve stance detection for online political discussions by using reliable traditional stance detection models for online deployment, while leveraging the text generation capabilities of LLMs for synthetic data generation in a secure offline environment. To achieve this, (i) we generate synthetic data for specific debate questions by prompting a Mistral-7B model and show that fine-tuning with the generated synthetic data can substantially improve the performance of stance detection, while remaining interpretable and aligned with real world data. (ii) Using the synthetic data as a reference, we can improve performance even further by identifying the most informative samples in an unlabelled dataset, i.e., those samples which the stance detection model is most uncertain about and can benefit from the most. By fine-tuning with both synthetic data and the most informative samples, we surpass the performance of the baseline model that is fine-tuned on all true labels, while labelling considerably less data.



## **12. Adversarial Vulnerabilities in Large Language Models for Time Series Forecasting**

cs.LG

AISTATS 2025

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2412.08099v4) [paper-pdf](http://arxiv.org/pdf/2412.08099v4)

**Authors**: Fuqiang Liu, Sicong Jiang, Luis Miranda-Moreno, Seongjin Choi, Lijun Sun

**Abstract**: Large Language Models (LLMs) have recently demonstrated significant potential in time series forecasting, offering impressive capabilities in handling complex temporal data. However, their robustness and reliability in real-world applications remain under-explored, particularly concerning their susceptibility to adversarial attacks. In this paper, we introduce a targeted adversarial attack framework for LLM-based time series forecasting. By employing both gradient-free and black-box optimization methods, we generate minimal yet highly effective perturbations that significantly degrade the forecasting accuracy across multiple datasets and LLM architectures. Our experiments, which include models like LLMTime with GPT-3.5, GPT-4, LLaMa, and Mistral, TimeGPT, and TimeLLM show that adversarial attacks lead to much more severe performance degradation than random noise, and demonstrate the broad effectiveness of our attacks across different LLMs. The results underscore the critical vulnerabilities of LLMs in time series forecasting, highlighting the need for robust defense mechanisms to ensure their reliable deployment in practical applications. The code repository can be found at https://github.com/JohnsonJiang1996/AdvAttack_LLM4TS.



## **13. EIA: Environmental Injection Attack on Generalist Web Agents for Privacy Leakage**

cs.CR

Accepted by ICLR 2025

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2409.11295v5) [paper-pdf](http://arxiv.org/pdf/2409.11295v5)

**Authors**: Zeyi Liao, Lingbo Mo, Chejian Xu, Mintong Kang, Jiawei Zhang, Chaowei Xiao, Yuan Tian, Bo Li, Huan Sun

**Abstract**: Generalist web agents have demonstrated remarkable potential in autonomously completing a wide range of tasks on real websites, significantly boosting human productivity. However, web tasks, such as booking flights, usually involve users' PII, which may be exposed to potential privacy risks if web agents accidentally interact with compromised websites, a scenario that remains largely unexplored in the literature. In this work, we narrow this gap by conducting the first study on the privacy risks of generalist web agents in adversarial environments. First, we present a realistic threat model for attacks on the website, where we consider two adversarial targets: stealing users' specific PII or the entire user request. Then, we propose a novel attack method, termed Environmental Injection Attack (EIA). EIA injects malicious content designed to adapt well to environments where the agents operate and our work instantiates EIA specifically for privacy scenarios in web environments. We collect 177 action steps that involve diverse PII categories on realistic websites from the Mind2Web, and conduct experiments using one of the most capable generalist web agent frameworks to date. The results demonstrate that EIA achieves up to 70% ASR in stealing specific PII and 16% ASR for full user request. Additionally, by accessing the stealthiness and experimenting with a defensive system prompt, we indicate that EIA is hard to detect and mitigate. Notably, attacks that are not well adapted for a webpage can be detected via human inspection, leading to our discussion about the trade-off between security and autonomy. However, extra attackers' efforts can make EIA seamlessly adapted, rendering such supervision ineffective. Thus, we further discuss the defenses at the pre- and post-deployment stages of the websites without relying on human supervision and call for more advanced defense strategies.



## **14. On the Robustness of Kolmogorov-Arnold Networks: An Adversarial Perspective**

cs.CV

Accepted at TMLR 2025

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2408.13809v3) [paper-pdf](http://arxiv.org/pdf/2408.13809v3)

**Authors**: Tal Alter, Raz Lapid, Moshe Sipper

**Abstract**: Kolmogorov-Arnold Networks (KANs) have recently emerged as a novel approach to function approximation, demonstrating remarkable potential in various domains. Despite their theoretical promise, the robustness of KANs under adversarial conditions has yet to be thoroughly examined. In this paper we explore the adversarial robustness of KANs, with a particular focus on image classification tasks. We assess the performance of KANs against standard white box and black-box adversarial attacks, comparing their resilience to that of established neural network architectures. Our experimental evaluation encompasses a variety of standard image classification benchmark datasets and investigates both fully connected and convolutional neural network architectures, of three sizes: small, medium, and large. We conclude that small- and medium-sized KANs (either fully connected or convolutional) are not consistently more robust than their standard counterparts, but that large-sized KANs are, by and large, more robust. This comprehensive evaluation of KANs in adversarial scenarios offers the first in-depth analysis of KAN security, laying the groundwork for future research in this emerging field.



## **15. ICMarks: A Robust Watermarking Framework for Integrated Circuit Physical Design IP Protection**

cs.CR

accept to TCAD (IEEE Transactions on Computer-Aided Design of  Integrated Circuits and Systems)

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2404.18407v2) [paper-pdf](http://arxiv.org/pdf/2404.18407v2)

**Authors**: Ruisi Zhang, Rachel Selina Rajarathnam, David Z. Pan, Farinaz Koushanfar

**Abstract**: Physical design watermarking on contemporary integrated circuit (IC) layout encodes signatures without considering the dense connections and design constraints, which could lead to performance degradation on the watermarked products. This paper presents ICMarks, a quality-preserving and robust watermarking framework for modern IC physical design. ICMarks embeds unique watermark signatures during the physical design's placement stage, thereby authenticating the IC layout ownership. ICMarks's novelty lies in (i) strategically identifying a region of cells to watermark with minimal impact on the layout performance and (ii) a two-level watermarking framework for augmented robustness toward potential removal and forging attacks. Extensive evaluations on benchmarks of different design objectives and sizes validate that ICMarks incurs no wirelength and timing metrics degradation, while successfully proving ownership. Furthermore, we demonstrate ICMarks is robust against two major watermarking attack categories, namely, watermark removal and forging attacks; even if the adversaries have prior knowledge of the watermarking schemes, the signatures cannot be removed without significantly undermining the layout quality.



## **16. Enhancing Adversarial Example Detection Through Model Explanation**

cs.CR

5 pages, 1 figure

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09735v1) [paper-pdf](http://arxiv.org/pdf/2503.09735v1)

**Authors**: Qian Ma, Ziping Ye

**Abstract**: Adversarial examples are a major problem for machine learning models, leading to a continuous search for effective defenses. One promising direction is to leverage model explanations to better understand and defend against these attacks. We looked at AmI, a method proposed by a NeurIPS 2018 spotlight paper that uses model explanations to detect adversarial examples. Our study shows that while AmI is a promising idea, its performance is too dependent on specific settings (e.g., hyperparameter) and external factors such as the operating system and the deep learning framework used, and such drawbacks limit AmI's practical usage. Our findings highlight the need for more robust defense mechanisms that are effective under various conditions. In addition, we advocate for a comprehensive evaluation framework for defense techniques.



## **17. All Your Knowledge Belongs to Us: Stealing Knowledge Graphs via Reasoning APIs**

cs.CR

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09727v1) [paper-pdf](http://arxiv.org/pdf/2503.09727v1)

**Authors**: Zhaohan Xi

**Abstract**: Knowledge graph reasoning (KGR), which answers complex, logical queries over large knowledge graphs (KGs), represents an important artificial intelligence task with a range of applications. Many KGs require extensive domain expertise and engineering effort to build and are hence considered proprietary within organizations and enterprises. Yet, spurred by their commercial and research potential, there is a growing trend to make KGR systems, (partially) built upon private KGs, publicly available through reasoning APIs.   The inherent tension between maintaining the confidentiality of KGs while ensuring the accessibility to KGR systems motivates our study of KG extraction attacks: the adversary aims to "steal" the private segments of the backend KG, leveraging solely black-box access to the KGR API. Specifically, we present KGX, an attack that extracts confidential sub-KGs with high fidelity under limited query budgets. At a high level, KGX progressively and adaptively queries the KGR API and integrates the query responses to reconstruct the private sub-KG. This extraction remains viable even if any query responses related to the private sub-KG are filtered. We validate the efficacy of KGX against both experimental and real-world KGR APIs. Interestingly, we find that typical countermeasures (e.g., injecting noise into query responses) are often ineffective against KGX. Our findings suggest the need for a more principled approach to developing and deploying KGR systems, as well as devising new defenses against KG extraction attacks.



## **18. Scale-Invariant Adversarial Attack against Arbitrary-scale Super-resolution**

cs.CV

17 pages, accepted by TIFS 2025

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.04385v2) [paper-pdf](http://arxiv.org/pdf/2503.04385v2)

**Authors**: Yihao Huang, Xin Luo, Qing Guo, Felix Juefei-Xu, Xiaojun Jia, Weikai Miao, Geguang Pu, Yang Liu

**Abstract**: The advent of local continuous image function (LIIF) has garnered significant attention for arbitrary-scale super-resolution (SR) techniques. However, while the vulnerabilities of fixed-scale SR have been assessed, the robustness of continuous representation-based arbitrary-scale SR against adversarial attacks remains an area warranting further exploration. The elaborately designed adversarial attacks for fixed-scale SR are scale-dependent, which will cause time-consuming and memory-consuming problems when applied to arbitrary-scale SR. To address this concern, we propose a simple yet effective ``scale-invariant'' SR adversarial attack method with good transferability, termed SIAGT. Specifically, we propose to construct resource-saving attacks by exploiting finite discrete points of continuous representation. In addition, we formulate a coordinate-dependent loss to enhance the cross-model transferability of the attack. The attack can significantly deteriorate the SR images while introducing imperceptible distortion to the targeted low-resolution (LR) images. Experiments carried out on three popular LIIF-based SR approaches and four classical SR datasets show remarkable attack performance and transferability of SIAGT.



## **19. RESTRAIN: Reinforcement Learning-Based Secure Framework for Trigger-Action IoT Environment**

cs.CR

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09513v1) [paper-pdf](http://arxiv.org/pdf/2503.09513v1)

**Authors**: Md Morshed Alam, Lokesh Chandra Das, Sandip Roy, Sachin Shetty, Weichao Wang

**Abstract**: Internet of Things (IoT) platforms with trigger-action capability allow event conditions to trigger actions in IoT devices autonomously by creating a chain of interactions. Adversaries exploit this chain of interactions to maliciously inject fake event conditions into IoT hubs, triggering unauthorized actions on target IoT devices to implement remote injection attacks. Existing defense mechanisms focus mainly on the verification of event transactions using physical event fingerprints to enforce the security policies to block unsafe event transactions. These approaches are designed to provide offline defense against injection attacks. The state-of-the-art online defense mechanisms offer real-time defense, but extensive reliability on the inference of attack impacts on the IoT network limits the generalization capability of these approaches. In this paper, we propose a platform-independent multi-agent online defense system, namely RESTRAIN, to counter remote injection attacks at runtime. RESTRAIN allows the defense agent to profile attack actions at runtime and leverages reinforcement learning to optimize a defense policy that complies with the security requirements of the IoT network. The experimental results show that the defense agent effectively takes real-time defense actions against complex and dynamic remote injection attacks and maximizes the security gain with minimal computational overhead.



## **20. Independence Tests for Language Models**

cs.LG

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2502.12292v2) [paper-pdf](http://arxiv.org/pdf/2502.12292v2)

**Authors**: Sally Zhu, Ahmed Ahmed, Rohith Kuditipudi, Percy Liang

**Abstract**: We consider the following problem: given the weights of two models, can we test whether they were trained independently -- i.e., from independent random initializations? We consider two settings: constrained and unconstrained. In the constrained setting, we make assumptions about model architecture and training and propose a family of statistical tests that yield exact p-values with respect to the null hypothesis that the models are trained from independent random initializations. These p-values are valid regardless of the composition of either model's training data; we compute them by simulating exchangeable copies of each model under our assumptions and comparing various similarity measures of weights and activations between the original two models versus these copies. We report the p-values from these tests on pairs of 21 open-weight models (210 total pairs) and correctly identify all pairs of non-independent models. Our tests remain effective even if one model was fine-tuned for many tokens. In the unconstrained setting, where we make no assumptions about training procedures, can change model architecture, and allow for adversarial evasion attacks, the previous tests no longer work. Instead, we propose a new test which matches hidden activations between two models, and which is robust to adversarial transformations and to changes in model architecture. The test can also do localized testing: identifying specific non-independent components of models. Though we no longer obtain exact p-values from this, empirically we find it behaves as one and reliably identifies non-independent models. Notably, we can use the test to identify specific parts of one model that are derived from another (e.g., how Llama 3.1-8B was pruned to initialize Llama 3.2-3B, or shared layers between Mistral-7B and StripedHyena-7B), and it is even robust to retraining individual layers of either model from scratch.



## **21. Revisiting Medical Image Retrieval via Knowledge Consolidation**

cs.CV

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09370v1) [paper-pdf](http://arxiv.org/pdf/2503.09370v1)

**Authors**: Yang Nan, Huichi Zhou, Xiaodan Xing, Giorgos Papanastasiou, Lei Zhu, Zhifan Gao, Alejandro F Fangi, Guang Yang

**Abstract**: As artificial intelligence and digital medicine increasingly permeate healthcare systems, robust governance frameworks are essential to ensure ethical, secure, and effective implementation. In this context, medical image retrieval becomes a critical component of clinical data management, playing a vital role in decision-making and safeguarding patient information. Existing methods usually learn hash functions using bottleneck features, which fail to produce representative hash codes from blended embeddings. Although contrastive hashing has shown superior performance, current approaches often treat image retrieval as a classification task, using category labels to create positive/negative pairs. Moreover, many methods fail to address the out-of-distribution (OOD) issue when models encounter external OOD queries or adversarial attacks. In this work, we propose a novel method to consolidate knowledge of hierarchical features and optimisation functions. We formulate the knowledge consolidation by introducing Depth-aware Representation Fusion (DaRF) and Structure-aware Contrastive Hashing (SCH). DaRF adaptively integrates shallow and deep representations into blended features, and SCH incorporates image fingerprints to enhance the adaptability of positive/negative pairings. These blended features further facilitate OOD detection and content-based recommendation, contributing to a secure AI-driven healthcare environment. Moreover, we present a content-guided ranking to improve the robustness and reproducibility of retrieval results. Our comprehensive assessments demonstrate that the proposed method could effectively recognise OOD samples and significantly outperform existing approaches in medical image retrieval (p<0.05). In particular, our method achieves a 5.6-38.9% improvement in mean Average Precision on the anatomical radiology dataset.



## **22. CyberLLMInstruct: A New Dataset for Analysing Safety of Fine-Tuned LLMs Using Cyber Security Data**

cs.CR

The paper is submitted to "The 48th International ACM SIGIR  Conference on Research and Development in Information Retrieval" and is  currently under review

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09334v1) [paper-pdf](http://arxiv.org/pdf/2503.09334v1)

**Authors**: Adel ElZemity, Budi Arief, Shujun Li

**Abstract**: The integration of large language models (LLMs) into cyber security applications presents significant opportunities, such as enhancing threat analysis and malware detection, but can also introduce critical risks and safety concerns, including personal data leakage and automated generation of new malware. To address these challenges, we developed CyberLLMInstruct, a dataset of 54,928 instruction-response pairs spanning cyber security tasks such as malware analysis, phishing simulations, and zero-day vulnerabilities. The dataset was constructed through a multi-stage process. This involved sourcing data from multiple resources, filtering and structuring it into instruction-response pairs, and aligning it with real-world scenarios to enhance its applicability. Seven open-source LLMs were chosen to test the usefulness of CyberLLMInstruct: Phi 3 Mini 3.8B, Mistral 7B, Qwen 2.5 7B, Llama 3 8B, Llama 3.1 8B, Gemma 2 9B, and Llama 2 70B. In our primary example, we rigorously assess the safety of fine-tuned models using the OWASP top 10 framework, finding that fine-tuning reduces safety resilience across all tested LLMs and every adversarial attack (e.g., the security score of Llama 3.1 8B against prompt injection drops from 0.95 to 0.15). In our second example, we show that these same fine-tuned models can also achieve up to 92.50 percent accuracy on the CyberMetric benchmark. These findings highlight a trade-off between performance and safety, showing the importance of adversarial testing and further research into fine-tuning methodologies that can mitigate safety risks while still improving performance across diverse datasets and domains. All scripts required to reproduce the dataset, along with examples and relevant resources for replicating our results, will be made available upon the paper's acceptance.



## **23. Detecting and Preventing Data Poisoning Attacks on AI Models**

cs.CR

9 pages, 8 figures

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09302v1) [paper-pdf](http://arxiv.org/pdf/2503.09302v1)

**Authors**: Halima I. Kure, Pradipta Sarkar, Ahmed B. Ndanusa, Augustine O. Nwajana

**Abstract**: This paper investigates the critical issue of data poisoning attacks on AI models, a growing concern in the ever-evolving landscape of artificial intelligence and cybersecurity. As advanced technology systems become increasingly prevalent across various sectors, the need for robust defence mechanisms against adversarial attacks becomes paramount. The study aims to develop and evaluate novel techniques for detecting and preventing data poisoning attacks, focusing on both theoretical frameworks and practical applications. Through a comprehensive literature review, experimental validation using the CIFAR-10 and Insurance Claims datasets, and the development of innovative algorithms, this paper seeks to enhance the resilience of AI models against malicious data manipulation. The study explores various methods, including anomaly detection, robust optimization strategies, and ensemble learning, to identify and mitigate the effects of poisoned data during model training. Experimental results indicate that data poisoning significantly degrades model performance, reducing classification accuracy by up to 27% in image recognition tasks (CIFAR-10) and 22% in fraud detection models (Insurance Claims dataset). The proposed defence mechanisms, including statistical anomaly detection and adversarial training, successfully mitigated poisoning effects, improving model robustness and restoring accuracy levels by an average of 15-20%. The findings further demonstrate that ensemble learning techniques provide an additional layer of resilience, reducing false positives and false negatives caused by adversarial data injections.



## **24. In-Context Defense in Computer Agents: An Empirical Study**

cs.AI

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09241v1) [paper-pdf](http://arxiv.org/pdf/2503.09241v1)

**Authors**: Pei Yang, Hai Ci, Mike Zheng Shou

**Abstract**: Computer agents powered by vision-language models (VLMs) have significantly advanced human-computer interaction, enabling users to perform complex tasks through natural language instructions. However, these agents are vulnerable to context deception attacks, an emerging threat where adversaries embed misleading content into the agent's operational environment, such as a pop-up window containing deceptive instructions. Existing defenses, such as instructing agents to ignore deceptive elements, have proven largely ineffective. As the first systematic study on protecting computer agents, we introduce textbf{in-context defense}, leveraging in-context learning and chain-of-thought (CoT) reasoning to counter such attacks. Our approach involves augmenting the agent's context with a small set of carefully curated exemplars containing both malicious environments and corresponding defensive responses. These exemplars guide the agent to first perform explicit defensive reasoning before action planning, reducing susceptibility to deceptive attacks. Experiments demonstrate the effectiveness of our method, reducing attack success rates by 91.2% on pop-up window attacks, 74.6% on average on environment injection attacks, while achieving 100% successful defenses against distracting advertisements. Our findings highlight that (1) defensive reasoning must precede action planning for optimal performance, and (2) a minimal number of exemplars (fewer than three) is sufficient to induce an agent's defensive behavior.



## **25. DetectRL: Benchmarking LLM-Generated Text Detection in Real-World Scenarios**

cs.CL

Accepted to NeurIPS 2024 Datasets and Benchmarks Track (Camera-Ready)

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2410.23746v3) [paper-pdf](http://arxiv.org/pdf/2410.23746v3)

**Authors**: Junchao Wu, Runzhe Zhan, Derek F. Wong, Shu Yang, Xinyi Yang, Yulin Yuan, Lidia S. Chao

**Abstract**: Detecting text generated by large language models (LLMs) is of great recent interest. With zero-shot methods like DetectGPT, detection capabilities have reached impressive levels. However, the reliability of existing detectors in real-world applications remains underexplored. In this study, we present a new benchmark, DetectRL, highlighting that even state-of-the-art (SOTA) detection techniques still underperformed in this task. We collected human-written datasets from domains where LLMs are particularly prone to misuse. Using popular LLMs, we generated data that better aligns with real-world applications. Unlike previous studies, we employed heuristic rules to create adversarial LLM-generated text, simulating various prompts usages, human revisions like word substitutions, and writing noises like spelling mistakes. Our development of DetectRL reveals the strengths and limitations of current SOTA detectors. More importantly, we analyzed the potential impact of writing styles, model types, attack methods, the text lengths, and real-world human writing factors on different types of detectors. We believe DetectRL could serve as an effective benchmark for assessing detectors in real-world scenarios, evolving with advanced attack methods, thus providing more stressful evaluation to drive the development of more efficient detectors. Data and code are publicly available at: https://github.com/NLP2CT/DetectRL.



## **26. AdvAD: Exploring Non-Parametric Diffusion for Imperceptible Adversarial Attacks**

cs.LG

Accept by NeurIPS 2024. Please cite this paper using the following  format: J. Li, Z. He, A. Luo, J. Hu, Z. Wang, X. Kang*, "AdvAD: Exploring  Non-Parametric Diffusion for Imperceptible Adversarial Attacks", the 38th  Annual Conference on Neural Information Processing Systems (NeurIPS),  Vancouver, Canada, Dec 9-15, 2024. Code: https://github.com/XianguiKang/AdvAD

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09124v1) [paper-pdf](http://arxiv.org/pdf/2503.09124v1)

**Authors**: Jin Li, Ziqiang He, Anwei Luo, Jian-Fang Hu, Z. Jane Wang, Xiangui Kang

**Abstract**: Imperceptible adversarial attacks aim to fool DNNs by adding imperceptible perturbation to the input data. Previous methods typically improve the imperceptibility of attacks by integrating common attack paradigms with specifically designed perception-based losses or the capabilities of generative models. In this paper, we propose Adversarial Attacks in Diffusion (AdvAD), a novel modeling framework distinct from existing attack paradigms. AdvAD innovatively conceptualizes attacking as a non-parametric diffusion process by theoretically exploring basic modeling approach rather than using the denoising or generation abilities of regular diffusion models requiring neural networks. At each step, much subtler yet effective adversarial guidance is crafted using only the attacked model without any additional network, which gradually leads the end of diffusion process from the original image to a desired imperceptible adversarial example. Grounded in a solid theoretical foundation of the proposed non-parametric diffusion process, AdvAD achieves high attack efficacy and imperceptibility with intrinsically lower overall perturbation strength. Additionally, an enhanced version AdvAD-X is proposed to evaluate the extreme of our novel framework under an ideal scenario. Extensive experiments demonstrate the effectiveness of the proposed AdvAD and AdvAD-X. Compared with state-of-the-art imperceptible attacks, AdvAD achieves an average of 99.9$\%$ (+17.3$\%$) ASR with 1.34 (-0.97) $l_2$ distance, 49.74 (+4.76) PSNR and 0.9971 (+0.0043) SSIM against four prevalent DNNs with three different architectures on the ImageNet-compatible dataset. Code is available at https://github.com/XianguiKang/AdvAD.



## **27. C^2 ATTACK: Towards Representation Backdoor on CLIP via Concept Confusion**

cs.CR

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09095v1) [paper-pdf](http://arxiv.org/pdf/2503.09095v1)

**Authors**: Lijie Hu, Junchi Liao, Weimin Lyu, Shaopeng Fu, Tianhao Huang, Shu Yang, Guimin Hu, Di Wang

**Abstract**: Backdoor attacks pose a significant threat to deep learning models, enabling adversaries to embed hidden triggers that manipulate the behavior of the model during inference. Traditional backdoor attacks typically rely on inserting explicit triggers (e.g., external patches, or perturbations) into input data, but they often struggle to evade existing defense mechanisms. To address this limitation, we investigate backdoor attacks through the lens of the reasoning process in deep learning systems, drawing insights from interpretable AI. We conceptualize backdoor activation as the manipulation of learned concepts within the model's latent representations. Thus, existing attacks can be seen as implicit manipulations of these activated concepts during inference. This raises interesting questions: why not manipulate the concepts explicitly? This idea leads to our novel backdoor attack framework, Concept Confusion Attack (C^2 ATTACK), which leverages internal concepts in the model's reasoning as "triggers" without introducing explicit external modifications. By avoiding the use of real triggers and directly activating or deactivating specific concepts in latent spaces, our approach enhances stealth, making detection by existing defenses significantly harder. Using CLIP as a case study, experimental results demonstrate the effectiveness of C^2 ATTACK, achieving high attack success rates while maintaining robustness against advanced defenses.



## **28. Probing Latent Subspaces in LLM for AI Security: Identifying and Manipulating Adversarial States**

cs.LG

4 figures

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.09066v1) [paper-pdf](http://arxiv.org/pdf/2503.09066v1)

**Authors**: Xin Wei Chia, Jonathan Pan

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they remain vulnerable to adversarial manipulations such as jailbreaking via prompt injection attacks. These attacks bypass safety mechanisms to generate restricted or harmful content. In this study, we investigated the underlying latent subspaces of safe and jailbroken states by extracting hidden activations from a LLM. Inspired by attractor dynamics in neuroscience, we hypothesized that LLM activations settle into semi stable states that can be identified and perturbed to induce state transitions. Using dimensionality reduction techniques, we projected activations from safe and jailbroken responses to reveal latent subspaces in lower dimensional spaces. We then derived a perturbation vector that when applied to safe representations, shifted the model towards a jailbreak state. Our results demonstrate that this causal intervention results in statistically significant jailbreak responses in a subset of prompts. Next, we probed how these perturbations propagate through the model's layers, testing whether the induced state change remains localized or cascades throughout the network. Our findings indicate that targeted perturbations induced distinct shifts in activations and model responses. Our approach paves the way for potential proactive defenses, shifting from traditional guardrail based methods to preemptive, model agnostic techniques that neutralize adversarial states at the representation level.



## **29. Quantitative Analysis of Deeply Quantized Tiny Neural Networks Robust to Adversarial Attacks**

cs.LG

arXiv admin note: substantial text overlap with arXiv:2304.12829

**SubmitDate**: 2025-03-12    [abs](http://arxiv.org/abs/2503.08973v1) [paper-pdf](http://arxiv.org/pdf/2503.08973v1)

**Authors**: Idris Zakariyya, Ferheen Ayaz, Mounia Kharbouche-Harrari, Jeremy Singer, Sye Loong Keoh, Danilo Pau, José Cano

**Abstract**: Reducing the memory footprint of Machine Learning (ML) models, especially Deep Neural Networks (DNNs), is imperative to facilitate their deployment on resource-constrained edge devices. However, a notable drawback of DNN models lies in their susceptibility to adversarial attacks, wherein minor input perturbations can deceive them. A primary challenge revolves around the development of accurate, resilient, and compact DNN models suitable for deployment on resource-constrained edge devices. This paper presents the outcomes of a compact DNN model that exhibits resilience against both black-box and white-box adversarial attacks. This work has achieved this resilience through training with the QKeras quantization-aware training framework. The study explores the potential of QKeras and an adversarial robustness technique, Jacobian Regularization (JR), to co-optimize the DNN architecture through per-layer JR methodology. As a result, this paper has devised a DNN model employing this co-optimization strategy based on Stochastic Ternary Quantization (STQ). Its performance was compared against existing DNN models in the face of various white-box and black-box attacks. The experimental findings revealed that, the proposed DNN model had small footprint and on average, it exhibited better performance than Quanos and DS-CNN MLCommons/TinyML (MLC/T) benchmarks when challenged with white-box and black-box attacks, respectively, on the CIFAR-10 image and Google Speech Commands audio datasets.



## **30. Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities**

cs.CL

Accepted to NAACL 2025 Main (oral)

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2410.18469v3) [paper-pdf](http://arxiv.org/pdf/2410.18469v3)

**Authors**: Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao

**Abstract**: Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99\% ASR on GPT-3.5 and 49\% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety.



## **31. Backtracking for Safety**

cs.CL

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08919v1) [paper-pdf](http://arxiv.org/pdf/2503.08919v1)

**Authors**: Bilgehan Sel, Dingcheng Li, Phillip Wallis, Vaishakh Keshava, Ming Jin, Siddhartha Reddy Jonnalagadda

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various tasks, but ensuring their safety and alignment with human values remains crucial. Current safety alignment methods, such as supervised fine-tuning and reinforcement learning-based approaches, can exhibit vulnerabilities to adversarial attacks and often result in shallow safety alignment, primarily focusing on preventing harmful content in the initial tokens of the generated output. While methods like resetting can help recover from unsafe generations by discarding previous tokens and restarting the generation process, they are not well-suited for addressing nuanced safety violations like toxicity that may arise within otherwise benign and lengthy generations. In this paper, we propose a novel backtracking method designed to address these limitations. Our method allows the model to revert to a safer generation state, not necessarily at the beginning, when safety violations occur during generation. This approach enables targeted correction of problematic segments without discarding the entire generated text, thereby preserving efficiency. We demonstrate that our method dramatically reduces toxicity appearing through the generation process with minimal impact to efficiency.



## **32. Human-Readable Adversarial Prompts: An Investigation into LLM Vulnerabilities Using Situational Context**

cs.CL

arXiv admin note: text overlap with arXiv:2407.14644

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2412.16359v2) [paper-pdf](http://arxiv.org/pdf/2412.16359v2)

**Authors**: Nilanjana Das, Edward Raff, Manas Gaur

**Abstract**: Previous studies that uncovered vulnerabilities in large language models (LLMs) frequently employed nonsensical adversarial prompts. However, such prompts can now be readily identified using automated detection techniques. To further strengthen adversarial attacks, we focus on human-readable adversarial prompts, which are more realistic and potent threats. Our key contributions are (1) situation-driven attacks leveraging movie scripts as context to create human-readable prompts that successfully deceive LLMs, (2) adversarial suffix conversion to transform nonsensical adversarial suffixes into independent meaningful text, and (3) AdvPrompter with p-nucleus sampling, a method to generate diverse, human-readable adversarial suffixes, improving attack efficacy in models like GPT-3.5 and Gemma 7B.



## **33. Birds look like cars: Adversarial analysis of intrinsically interpretable deep learning**

cs.LG

Preprint

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08636v1) [paper-pdf](http://arxiv.org/pdf/2503.08636v1)

**Authors**: Hubert Baniecki, Przemyslaw Biecek

**Abstract**: A common belief is that intrinsically interpretable deep learning models ensure a correct, intuitive understanding of their behavior and offer greater robustness against accidental errors or intentional manipulation. However, these beliefs have not been comprehensively verified, and growing evidence casts doubt on them. In this paper, we highlight the risks related to overreliance and susceptibility to adversarial manipulation of these so-called "intrinsically (aka inherently) interpretable" models by design. We introduce two strategies for adversarial analysis with prototype manipulation and backdoor attacks against prototype-based networks, and discuss how concept bottleneck models defend against these attacks. Fooling the model's reasoning by exploiting its use of latent prototypes manifests the inherent uninterpretability of deep neural networks, leading to a false sense of security reinforced by a visual confirmation bias. The reported limitations of prototype-based networks put their trustworthiness and applicability into question, motivating further work on the robustness and alignment of (deep) interpretable models.



## **34. Beyond Optimal Fault Tolerance**

cs.DC

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2501.06044v4) [paper-pdf](http://arxiv.org/pdf/2501.06044v4)

**Authors**: Andrew Lewis-Pye, Tim Roughgarden

**Abstract**: The optimal fault-tolerance achievable by any protocol has been characterized in a wide range of settings. For example, for state machine replication (SMR) protocols operating in the partially synchronous setting, it is possible to simultaneously guarantee consistency against $\alpha$-bounded adversaries (i.e., adversaries that control less than an $\alpha$ fraction of the participants) and liveness against $\beta$-bounded adversaries if and only if $\alpha + 2\beta \leq 1$.   This paper characterizes to what extent "better-than-optimal" fault-tolerance guarantees are possible for SMR protocols when the standard consistency requirement is relaxed to allow a bounded number $r$ of consistency violations. We prove that bounding rollback is impossible without additional timing assumptions and investigate protocols that tolerate and recover from consistency violations whenever message delays around the time of an attack are bounded by a parameter $\Delta^*$ (which may be arbitrarily larger than the parameter $\Delta$ that bounds post-GST message delays in the partially synchronous model). Here, a protocol's fault-tolerance can be a non-constant function of $r$, and we prove, for each $r$, matching upper and lower bounds on the optimal "recoverable fault-tolerance" achievable by any SMR protocol. For example, for protocols that guarantee liveness against 1/3-bounded adversaries in the partially synchronous setting, a 5/9-bounded adversary can always cause one consistency violation but not two, and a 2/3-bounded adversary can always cause two consistency violations but not three. Our positive results are achieved through a generic "recovery procedure" that can be grafted on to any accountable SMR protocol and restores consistency following a violation while rolling back only transactions that were finalized in the previous $2\Delta^*$ timesteps.



## **35. Low-Cost Privacy-Preserving Decentralized Learning**

cs.LG

24 pages, accepted at Pets 2025

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2403.11795v3) [paper-pdf](http://arxiv.org/pdf/2403.11795v3)

**Authors**: Sayan Biswas, Davide Frey, Romaric Gaudel, Anne-Marie Kermarrec, Dimitri Lerévérend, Rafael Pires, Rishi Sharma, François Taïani

**Abstract**: Decentralized learning (DL) is an emerging paradigm of collaborative machine learning that enables nodes in a network to train models collectively without sharing their raw data or relying on a central server. This paper introduces Zip-DL, a privacy-aware DL algorithm that leverages correlated noise to achieve robust privacy against local adversaries while ensuring efficient convergence at low communication costs. By progressively neutralizing the noise added during distributed averaging, Zip-DL combines strong privacy guarantees with high model accuracy. Its design requires only one communication round per gradient descent iteration, significantly reducing communication overhead compared to competitors. We establish theoretical bounds on both convergence speed and privacy guarantees. Moreover, extensive experiments demonstrating Zip-DL's practical applicability make it outperform state-of-the-art methods in the accuracy vs. vulnerability trade-off. Specifically, Zip-DL (i) reduces membership-inference attack success rates by up to 35% compared to baseline DL, (ii) decreases attack efficacy by up to 13% compared to competitors offering similar utility, and (iii) achieves up to 59% higher accuracy to completely nullify a basic attack scenario, compared to a state-of-the-art privacy-preserving approach under the same threat model. These results position Zip-DL as a practical and efficient solution for privacy-preserving decentralized learning in real-world applications.



## **36. Adv-CPG: A Customized Portrait Generation Framework with Facial Adversarial Attacks**

cs.CV

Accepted by CVPR-25

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08269v1) [paper-pdf](http://arxiv.org/pdf/2503.08269v1)

**Authors**: Junying Wang, Hongyuan Zhang, Yuan Yuan

**Abstract**: Recent Customized Portrait Generation (CPG) methods, taking a facial image and a textual prompt as inputs, have attracted substantial attention. Although these methods generate high-fidelity portraits, they fail to prevent the generated portraits from being tracked and misused by malicious face recognition systems. To address this, this paper proposes a Customized Portrait Generation framework with facial Adversarial attacks (Adv-CPG). Specifically, to achieve facial privacy protection, we devise a lightweight local ID encryptor and an encryption enhancer. They implement progressive double-layer encryption protection by directly injecting the target identity and adding additional identity guidance, respectively. Furthermore, to accomplish fine-grained and personalized portrait generation, we develop a multi-modal image customizer capable of generating controlled fine-grained facial features. To the best of our knowledge, Adv-CPG is the first study that introduces facial adversarial attacks into CPG. Extensive experiments demonstrate the superiority of Adv-CPG, e.g., the average attack success rate of the proposed Adv-CPG is 28.1% and 2.86% higher compared to the SOTA noise-based attack methods and unconstrained attack methods, respectively.



## **37. A Grey-box Text Attack Framework using Explainable AI**

cs.CL

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08226v1) [paper-pdf](http://arxiv.org/pdf/2503.08226v1)

**Authors**: Esther Chiramal, Kelvin Soh Boon Kai

**Abstract**: Explainable AI is a strong strategy implemented to understand complex black-box model predictions in a human interpretable language. It provides the evidence required to execute the use of trustworthy and reliable AI systems. On the other hand, however, it also opens the door to locating possible vulnerabilities in an AI model. Traditional adversarial text attack uses word substitution, data augmentation techniques and gradient-based attacks on powerful pre-trained Bidirectional Encoder Representations from Transformers (BERT) variants to generate adversarial sentences. These attacks are generally whitebox in nature and not practical as they can be easily detected by humans E.g. Changing the word from "Poor" to "Rich". We proposed a simple yet effective Grey-box cum Black-box approach that does not require the knowledge of the model while using a set of surrogate Transformer/BERT models to perform the attack using Explainable AI techniques. As Transformers are the current state-of-the-art models for almost all Natural Language Processing (NLP) tasks, an attack generated from BERT1 is transferable to BERT2. This transferability is made possible due to the attention mechanism in the transformer that allows the model to capture long-range dependencies in a sequence. Using the power of BERT generalisation via attention, we attempt to exploit how transformers learn by attacking a few surrogate transformer variants which are all based on a different architecture. We demonstrate that this approach is highly effective to generate semantically good sentences by changing as little as one word that is not detectable by humans while still fooling other BERT models.



## **38. Guardians of the Agentic System: Preventing Many Shots Jailbreak with Agentic System**

cs.CR

18 pages, 7 figures

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2502.16750v3) [paper-pdf](http://arxiv.org/pdf/2502.16750v3)

**Authors**: Saikat Barua, Mostafizur Rahman, Md Jafor Sadek, Rafiul Islam, Shehenaz Khaled, Ahmedul Kabir

**Abstract**: The autonomous AI agents using large language models can create undeniable values in all span of the society but they face security threats from adversaries that warrants immediate protective solutions because trust and safety issues arise. Considering the many-shot jailbreaking and deceptive alignment as some of the main advanced attacks, that cannot be mitigated by the static guardrails used during the supervised training, points out a crucial research priority for real world robustness. The combination of static guardrails in dynamic multi-agent system fails to defend against those attacks. We intend to enhance security for LLM-based agents through the development of new evaluation frameworks which identify and counter threats for safe operational deployment. Our work uses three examination methods to detect rogue agents through a Reverse Turing Test and analyze deceptive alignment through multi-agent simulations and develops an anti-jailbreaking system by testing it with GEMINI 1.5 pro and llama-3.3-70B, deepseek r1 models using tool-mediated adversarial scenarios. The detection capabilities are strong such as 94\% accuracy for GEMINI 1.5 pro yet the system suffers persistent vulnerabilities when under long attacks as prompt length increases attack success rates (ASR) and diversity metrics become ineffective in prediction while revealing multiple complex system faults. The findings demonstrate the necessity of adopting flexible security systems based on active monitoring that can be performed by the agents themselves together with adaptable interventions by system admin as the current models can create vulnerabilities that can lead to the unreliable and vulnerable system. So, in our work, we try to address such situations and propose a comprehensive framework to counteract the security issues.



## **39. Dialogue Injection Attack: Jailbreaking LLMs through Context Manipulation**

cs.CL

17 pages, 10 figures

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08195v1) [paper-pdf](http://arxiv.org/pdf/2503.08195v1)

**Authors**: Wenlong Meng, Fan Zhang, Wendao Yao, Zhenyuan Guo, Yuwei Li, Chengkun Wei, Wenzhi Chen

**Abstract**: Large language models (LLMs) have demonstrated significant utility in a wide range of applications; however, their deployment is plagued by security vulnerabilities, notably jailbreak attacks. These attacks manipulate LLMs to generate harmful or unethical content by crafting adversarial prompts. While much of the current research on jailbreak attacks has focused on single-turn interactions, it has largely overlooked the impact of historical dialogues on model behavior. In this paper, we introduce a novel jailbreak paradigm, Dialogue Injection Attack (DIA), which leverages the dialogue history to enhance the success rates of such attacks. DIA operates in a black-box setting, requiring only access to the chat API or knowledge of the LLM's chat template. We propose two methods for constructing adversarial historical dialogues: one adapts gray-box prefilling attacks, and the other exploits deferred responses. Our experiments show that DIA achieves state-of-the-art attack success rates on recent LLMs, including Llama-3.1 and GPT-4o. Additionally, we demonstrate that DIA can bypass 5 different defense mechanisms, highlighting its robustness and effectiveness.



## **40. MAGIC: Mastering Physical Adversarial Generation in Context through Collaborative LLM Agents**

cs.CV

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2412.08014v2) [paper-pdf](http://arxiv.org/pdf/2412.08014v2)

**Authors**: Yun Xing, Nhat Chung, Jie Zhang, Yue Cao, Ivor Tsang, Yang Liu, Lei Ma, Qing Guo

**Abstract**: Physical adversarial attacks in driving scenarios can expose critical vulnerabilities in visual perception models. However, developing such attacks remains challenging due to diverse real-world environments and the requirement for maintaining visual naturality. Building upon this challenge, we reformulate physical adversarial attacks as a one-shot patch generation problem. Our approach generates adversarial patches through a deep generative model that considers the specific scene context, enabling direct physical deployment in matching environments. The primary challenge lies in simultaneously achieving two objectives: generating adversarial patches that effectively mislead object detection systems while determining contextually appropriate deployment within the scene. We propose MAGIC (Mastering Physical Adversarial Generation In Context), a novel framework powered by multi-modal LLM agents to address these challenges. MAGIC automatically understands scene context and generates adversarial patch through the synergistic interaction of language and vision capabilities. In particular, MAGIC orchestrates three specialized LLM agents: The adv-patch generation agent (GAgent) masters the creation of deceptive patches through strategic prompt engineering for text-to-image models. The adv-patch deployment agent (DAgent) ensures contextual coherence by determining optimal deployment strategies based on scene understanding. The self-examination agent (EAgent) completes this trilogy by providing critical oversight and iterative refinement of both processes. We validate our method on both digital and physical levels, i.e., nuImage and manually captured real-world scenes, where both statistical and visual results prove that our MAGIC is powerful and effective for attacking widely applied object detection systems, i.e., YOLO and DETR series.



## **41. MIGA: Mutual Information-Guided Attack on Denoising Models for Semantic Manipulation**

cs.CV

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.06966v2) [paper-pdf](http://arxiv.org/pdf/2503.06966v2)

**Authors**: Guanghao Li, Mingzhi Chen, Hao Yu, Shuting Dong, Wenhao Jiang, Ming Tang, Chun Yuan

**Abstract**: Deep learning-based denoising models have been widely employed in vision tasks, functioning as filters to eliminate noise while retaining crucial semantic information. Additionally, they play a vital role in defending against adversarial perturbations that threaten downstream tasks. However, these models can be intrinsically susceptible to adversarial attacks due to their dependence on specific noise assumptions. Existing attacks on denoising models mainly aim at deteriorating visual clarity while neglecting semantic manipulation, rendering them either easily detectable or limited in effectiveness. In this paper, we propose Mutual Information-Guided Attack (MIGA), the first method designed to directly attack deep denoising models by strategically disrupting their ability to preserve semantic content via adversarial perturbations. By minimizing the mutual information between the original and denoised images, a measure of semantic similarity. MIGA forces the denoiser to produce perceptually clean yet semantically altered outputs. While these images appear visually plausible, they encode systematically distorted semantics, revealing a fundamental vulnerability in denoising models. These distortions persist in denoised outputs and can be quantitatively assessed through downstream task performance. We propose new evaluation metrics and systematically assess MIGA on four denoising models across five datasets, demonstrating its consistent effectiveness in disrupting semantic fidelity. Our findings suggest that denoising models are not always robust and can introduce security risks in real-world applications.



## **42. Towards Million-Scale Adversarial Robustness Evaluation With Stronger Individual Attacks**

cs.LG

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2411.15210v4) [paper-pdf](http://arxiv.org/pdf/2411.15210v4)

**Authors**: Yong Xie, Weijie Zheng, Hanxun Huang, Guangnan Ye, Xingjun Ma

**Abstract**: As deep learning models are increasingly deployed in safety-critical applications, evaluating their vulnerabilities to adversarial perturbations is essential for ensuring their reliability and trustworthiness. Over the past decade, a large number of white-box adversarial robustness evaluation methods (i.e., attacks) have been proposed, ranging from single-step to multi-step methods and from individual to ensemble methods. Despite these advances, challenges remain in conducting meaningful and comprehensive robustness evaluations, particularly when it comes to large-scale testing and ensuring evaluations reflect real-world adversarial risks. In this work, we focus on image classification models and propose a novel individual attack method, Probability Margin Attack (PMA), which defines the adversarial margin in the probability space rather than the logits space. We analyze the relationship between PMA and existing cross-entropy or logits-margin-based attacks, and show that PMA can outperform the current state-of-the-art individual methods. Building on PMA, we propose two types of ensemble attacks that balance effectiveness and efficiency. Furthermore, we create a million-scale dataset, CC1M, derived from the existing CC3M dataset, and use it to conduct the first million-scale white-box adversarial robustness evaluation of adversarially-trained ImageNet models. Our findings provide valuable insights into the robustness gaps between individual versus ensemble attacks and small-scale versus million-scale evaluations.



## **43. FairDeFace: Evaluating the Fairness and Adversarial Robustness of Face Obfuscation Methods**

cs.CV

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2503.08731v1) [paper-pdf](http://arxiv.org/pdf/2503.08731v1)

**Authors**: Seyyed Mohammad Sadegh Moosavi Khorzooghi, Poojitha Thota, Mohit Singhal, Abolfazl Asudeh, Gautam Das, Shirin Nilizadeh

**Abstract**: The lack of a common platform and benchmark datasets for evaluating face obfuscation methods has been a challenge, with every method being tested using arbitrary experiments, datasets, and metrics. While prior work has demonstrated that face recognition systems exhibit bias against some demographic groups, there exists a substantial gap in our understanding regarding the fairness of face obfuscation methods. Providing fair face obfuscation methods can ensure equitable protection across diverse demographic groups, especially since they can be used to preserve the privacy of vulnerable populations. To address these gaps, this paper introduces a comprehensive framework, named FairDeFace, designed to assess the adversarial robustness and fairness of face obfuscation methods. The framework introduces a set of modules encompassing data benchmarks, face detection and recognition algorithms, adversarial models, utility detection models, and fairness metrics. FairDeFace serves as a versatile platform where any face obfuscation method can be integrated, allowing for rigorous testing and comparison with other state-of-the-art methods. In its current implementation, FairDeFace incorporates 6 attacks, and several privacy, utility and fairness metrics. Using FairDeFace, and by conducting more than 500 experiments, we evaluated and compared the adversarial robustness of seven face obfuscation methods. This extensive analysis led to many interesting findings both in terms of the degree of robustness of existing methods and their biases against some gender or racial groups. FairDeFace also uses visualization of focused areas for both obfuscation and verification attacks to show not only which areas are mostly changed in the obfuscation process for some demographics, but also why they failed through focus area comparison of obfuscation and verification.



## **44. Human-in-the-Loop Generation of Adversarial Texts: A Case Study on Tibetan Script**

cs.CL

**SubmitDate**: 2025-03-11    [abs](http://arxiv.org/abs/2412.12478v2) [paper-pdf](http://arxiv.org/pdf/2412.12478v2)

**Authors**: Xi Cao, Yuan Sun, Jiajun Li, Quzong Gesang, Nuo Qun, Tashi Nyima

**Abstract**: DNN-based language models perform excellently on various tasks, but even SOTA LLMs are susceptible to textual adversarial attacks. Adversarial texts play crucial roles in multiple subfields of NLP. However, current research has the following issues. (1) Most textual adversarial attack methods target rich-resourced languages. How do we generate adversarial texts for less-studied languages? (2) Most textual adversarial attack methods are prone to generating invalid or ambiguous adversarial texts. How do we construct high-quality adversarial robustness benchmarks? (3) New language models may be immune to part of previously generated adversarial texts. How do we update adversarial robustness benchmarks? To address the above issues, we introduce HITL-GAT, a system based on a general approach to human-in-the-loop generation of adversarial texts. HITL-GAT contains four stages in one pipeline: victim model construction, adversarial example generation, high-quality benchmark construction, and adversarial robustness evaluation. Additionally, we utilize HITL-GAT to make a case study on Tibetan script which can be a reference for the adversarial research of other less-studied languages.



## **45. Safety Guardrails for LLM-Enabled Robots**

cs.RO

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07885v1) [paper-pdf](http://arxiv.org/pdf/2503.07885v1)

**Authors**: Zachary Ravichandran, Alexander Robey, Vijay Kumar, George J. Pappas, Hamed Hassani

**Abstract**: Although the integration of large language models (LLMs) into robotics has unlocked transformative capabilities, it has also introduced significant safety concerns, ranging from average-case LLM errors (e.g., hallucinations) to adversarial jailbreaking attacks, which can produce harmful robot behavior in real-world settings. Traditional robot safety approaches do not address the novel vulnerabilities of LLMs, and current LLM safety guardrails overlook the physical risks posed by robots operating in dynamic real-world environments. In this paper, we propose RoboGuard, a two-stage guardrail architecture to ensure the safety of LLM-enabled robots. RoboGuard first contextualizes pre-defined safety rules by grounding them in the robot's environment using a root-of-trust LLM, which employs chain-of-thought (CoT) reasoning to generate rigorous safety specifications, such as temporal logic constraints. RoboGuard then resolves potential conflicts between these contextual safety specifications and a possibly unsafe plan using temporal logic control synthesis, which ensures safety compliance while minimally violating user preferences. Through extensive simulation and real-world experiments that consider worst-case jailbreaking attacks, we demonstrate that RoboGuard reduces the execution of unsafe plans from 92% to below 2.5% without compromising performance on safe plans. We also demonstrate that RoboGuard is resource-efficient, robust against adaptive attacks, and significantly enhanced by enabling its root-of-trust LLM to perform CoT reasoning. These results underscore the potential of RoboGuard to mitigate the safety risks and enhance the reliability of LLM-enabled robots.



## **46. ReLATE: Resilient Learner Selection for Multivariate Time-Series Classification Against Adversarial Attacks**

cs.LG

Accepted by the AAAI-25 Workshop on Artificial Intelligence for Time  Series Analysis (AI4TS)

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07882v1) [paper-pdf](http://arxiv.org/pdf/2503.07882v1)

**Authors**: Cagla Ipek Kocal, Onat Gungor, Aaron Tartz, Tajana Rosing, Baris Aksanli

**Abstract**: Minimizing computational overhead in time-series classification, particularly in deep learning models, presents a significant challenge. This challenge is further compounded by adversarial attacks, emphasizing the need for resilient methods that ensure robust performance and efficient model selection. We introduce ReLATE, a framework that identifies robust learners based on dataset similarity, reduces computational overhead, and enhances resilience. ReLATE maintains multiple deep learning models in well-known adversarial attack scenarios, capturing model performance. ReLATE identifies the most analogous dataset to a given target using a similarity metric, then applies the optimal model from the most similar dataset. ReLATE reduces computational overhead by an average of 81.2%, enhancing adversarial resilience and streamlining robust model selection, all without sacrificing performance, within 4.2% of Oracle.



## **47. On the Byzantine Fault Tolerance of signSGD with Majority Vote**

cs.LG

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2502.19170v2) [paper-pdf](http://arxiv.org/pdf/2502.19170v2)

**Authors**: Emanuele Mengoli, Luzius Moll, Virgilio Strozzi, El-Mahdi El-Mhamdi

**Abstract**: In distributed learning, sign-based compression algorithms such as signSGD with majority vote provide a lightweight alternative to SGD with an additional advantage: fault tolerance (almost) for free. However, for signSGD with majority vote, this fault tolerance has been shown to cover only the case of weaker adversaries, i.e., ones that are not omniscient or cannot collude to base their attack on common knowledge and strategy. In this work, we close this gap and provide new insights into how signSGD with majority vote can be resilient against omniscient and colluding adversaries, which craft an attack after communicating with other adversaries, thus having better information to perform the most damaging attack based on a common optimal strategy. Our core contribution is in providing a proof that begins by defining the omniscience framework and the strongest possible damage against signSGD with majority vote without imposing any restrictions on the attacker. Thanks to the filtering effect of the sign-based method, we upper-bound the space of attacks to the optimal strategy for maximizing damage by an attacker. Hence, we derive an explicit probabilistic bound in terms of incorrect aggregation without resorting to unknown constants, providing a convergence bound on signSGD with majority vote in the presence of Byzantine attackers, along with a precise convergence rate. Our findings are supported by experiments on the MNIST dataset in a distributed learning environment with adversaries of varying strength.



## **48. Runtime Detection of Adversarial Attacks in AI Accelerators Using Performance Counters**

cs.CR

7 pages, 8 figures

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07568v1) [paper-pdf](http://arxiv.org/pdf/2503.07568v1)

**Authors**: Habibur Rahaman, Atri Chatterjee, Swarup Bhunia

**Abstract**: Rapid adoption of AI technologies raises several major security concerns, including the risks of adversarial perturbations, which threaten the confidentiality and integrity of AI applications. Protecting AI hardware from misuse and diverse security threats is a challenging task. To address this challenge, we propose SAMURAI, a novel framework for safeguarding against malicious usage of AI hardware and its resilience to attacks. SAMURAI introduces an AI Performance Counter (APC) for tracking dynamic behavior of an AI model coupled with an on-chip Machine Learning (ML) analysis engine, known as TANTO (Trained Anomaly Inspection Through Trace Observation). APC records the runtime profile of the low-level hardware events of different AI operations. Subsequently, the summary information recorded by the APC is processed by TANTO to efficiently identify potential security breaches and ensure secure, responsible use of AI. SAMURAI enables real-time detection of security threats and misuse without relying on traditional software-based solutions that require model integration. Experimental results demonstrate that SAMURAI achieves up to 97% accuracy in detecting adversarial attacks with moderate overhead on various AI models, significantly outperforming conventional software-based approaches. It enhances security and regulatory compliance, providing a comprehensive solution for safeguarding AI against emergent threats.



## **49. Transform-Dependent Adversarial Attacks**

cs.CV

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2406.08443v2) [paper-pdf](http://arxiv.org/pdf/2406.08443v2)

**Authors**: Yaoteng Tan, Zikui Cai, M. Salman Asif

**Abstract**: Deep networks are highly vulnerable to adversarial attacks, yet conventional attack methods utilize static adversarial perturbations that induce fixed mispredictions. In this work, we exploit an overlooked property of adversarial perturbations--their dependence on image transforms--and introduce transform-dependent adversarial attacks. Unlike traditional attacks, our perturbations exhibit metamorphic properties, enabling diverse adversarial effects as a function of transformation parameters. We demonstrate that this transform-dependent vulnerability exists across different architectures (e.g., CNN and transformer), vision tasks (e.g., image classification and object detection), and a wide range of image transforms. Additionally, we show that transform-dependent perturbations can serve as a defense mechanism, preventing sensitive information disclosure when image enhancement transforms pose a risk of revealing private content. Through analysis in blackbox and defended model settings, we show that transform-dependent perturbations achieve high targeted attack success rates, outperforming state-of-the-art transfer attacks by 17-31% in blackbox scenarios. Our work introduces novel, controllable paradigm for adversarial attack deployment, revealing a previously overlooked vulnerability in deep networks.



## **50. Learning to Localize Leakage of Cryptographic Sensitive Variables**

cs.LG

52 pages, 30 figures. Our code can be found at  https://github.com/jimgammell/learning_to_localize_leakage

**SubmitDate**: 2025-03-10    [abs](http://arxiv.org/abs/2503.07464v1) [paper-pdf](http://arxiv.org/pdf/2503.07464v1)

**Authors**: Jimmy Gammell, Anand Raghunathan, Abolfazl Hashemi, Kaushik Roy

**Abstract**: While cryptographic algorithms such as the ubiquitous Advanced Encryption Standard (AES) are secure, *physical implementations* of these algorithms in hardware inevitably 'leak' sensitive data such as cryptographic keys. A particularly insidious form of leakage arises from the fact that hardware consumes power and emits radiation in a manner that is statistically associated with the data it processes and the instructions it executes. Supervised deep learning has emerged as a state-of-the-art tool for carrying out *side-channel attacks*, which exploit this leakage by learning to map power/radiation measurements throughout encryption to the sensitive data operated on during that encryption. In this work we develop a principled deep learning framework for determining the relative leakage due to measurements recorded at different points in time, in order to inform *defense* against such attacks. This information is invaluable to cryptographic hardware designers for understanding *why* their hardware leaks and how they can mitigate it (e.g. by indicating the particular sections of code or electronic components which are responsible). Our framework is based on an adversarial game between a family of classifiers trained to estimate the conditional distributions of sensitive data given subsets of measurements, and a budget-constrained noise distribution which probabilistically erases individual measurements to maximize the loss of these classifiers. We demonstrate our method's efficacy and ability to overcome limitations of prior work through extensive experimental comparison with 8 baseline methods using 3 evaluation metrics and 6 publicly-available power/EM trace datasets from AES, ECC and RSA implementations. We provide an open-source PyTorch implementation of these experiments.



