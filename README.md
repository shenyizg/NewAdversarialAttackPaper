# Latest Adversarial Attack Papers
**update at 2025-04-21 09:58:04**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Robust Decentralized Quantum Kernel Learning for Noisy and Adversarial Environment**

quant-ph

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13782v1) [paper-pdf](http://arxiv.org/pdf/2504.13782v1)

**Authors**: Wenxuan Ma, Kuan-Cheng Chen, Shang Yu, Mengxiang Liu, Ruilong Deng

**Abstract**: This paper proposes a general decentralized framework for quantum kernel learning (QKL). It has robustness against quantum noise and can also be designed to defend adversarial information attacks forming a robust approach named RDQKL. We analyze the impact of noise on QKL and study the robustness of decentralized QKL to the noise. By integrating robust decentralized optimization techniques, our method is able to mitigate the impact of malicious data injections across multiple nodes. Experimental results demonstrate that our approach maintains high accuracy under noisy quantum operations and effectively counter adversarial modifications, offering a promising pathway towards the future practical, scalable and secure quantum machine learning (QML).



## **2. Adversarial Hubness in Multi-Modal Retrieval**

cs.CR

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2412.14113v2) [paper-pdf](http://arxiv.org/pdf/2412.14113v2)

**Authors**: Tingwei Zhang, Fnu Suya, Rishi Jha, Collin Zhang, Vitaly Shmatikov

**Abstract**: Hubness is a phenomenon in high-dimensional vector spaces where a single point from the natural distribution is unusually close to many other points. This is a well-known problem in information retrieval that causes some items to accidentally (and incorrectly) appear relevant to many queries.   In this paper, we investigate how attackers can exploit hubness to turn any image or audio input in a multi-modal retrieval system into an adversarial hub. Adversarial hubs can be used to inject universal adversarial content (e.g., spam) that will be retrieved in response to thousands of different queries, as well as for targeted attacks on queries related to specific, attacker-chosen concepts.   We present a method for creating adversarial hubs and evaluate the resulting hubs on benchmark multi-modal retrieval datasets and an image-to-image retrieval system implemented by Pinecone, a popular vector database. For example, in text-caption-to-image retrieval, a single adversarial hub, generated with respect to 100 randomly selected target queries, is retrieved as the top-1 most relevant image for more than 21,000 out of 25,000 test queries (by contrast, the most common natural hub is the top-1 response to only 102 queries), demonstrating the strong generalization capabilities of adversarial hubs. We also investigate whether techniques for mitigating natural hubness are an effective defense against adversarial hubs, and show that they are not effective against hubs that target queries related to specific concepts.



## **3. Energy-Latency Attacks via Sponge Poisoning**

cs.CR

Paper accepted at Information Sciences journal; 20 pages Keywords:  energy-latency attacks, sponge attack, machine learning security, adversarial  machine learning

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2203.08147v5) [paper-pdf](http://arxiv.org/pdf/2203.08147v5)

**Authors**: Antonio Emanuele Cinà, Ambra Demontis, Battista Biggio, Fabio Roli, Marcello Pelillo

**Abstract**: Sponge examples are test-time inputs optimized to increase energy consumption and prediction latency of deep networks deployed on hardware accelerators. By increasing the fraction of neurons activated during classification, these attacks reduce sparsity in network activation patterns, worsening the performance of hardware accelerators. In this work, we present a novel training-time attack, named sponge poisoning, which aims to worsen energy consumption and prediction latency of neural networks on any test input without affecting classification accuracy. To stage this attack, we assume that the attacker can control only a few model updates during training -- a likely scenario, e.g., when model training is outsourced to an untrusted third party or distributed via federated learning. Our extensive experiments on image classification tasks show that sponge poisoning is effective, and that fine-tuning poisoned models to repair them poses prohibitive costs for most users, highlighting that tackling sponge poisoning remains an open issue.



## **4. Fairness and Robustness in Machine Unlearning**

cs.LG

5 pages

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13610v1) [paper-pdf](http://arxiv.org/pdf/2504.13610v1)

**Authors**: Khoa Tran, Simon S. Woo

**Abstract**: Machine unlearning poses the challenge of ``how to eliminate the influence of specific data from a pretrained model'' in regard to privacy concerns. While prior research on approximated unlearning has demonstrated accuracy and efficiency in time complexity, we claim that it falls short of achieving exact unlearning, and we are the first to focus on fairness and robustness in machine unlearning algorithms. Our study presents fairness Conjectures for a well-trained model, based on the variance-bias trade-off characteristic, and considers their relevance to robustness. Our Conjectures are supported by experiments conducted on the two most widely used model architectures, ResNet and ViT, demonstrating the correlation between fairness and robustness: \textit{the higher fairness-gap is, the more the model is sensitive and vulnerable}. In addition, our experiments demonstrate the vulnerability of current state-of-the-art approximated unlearning algorithms to adversarial attacks, where their unlearned models suffer a significant drop in accuracy compared to the exact-unlearned models. We claim that our fairness-gap measurement and robustness metric should be used to evaluate the unlearning algorithm. Furthermore, we demonstrate that unlearning in the intermediate and last layers is sufficient and cost-effective for time and memory complexity.



## **5. Q-FAKER: Query-free Hard Black-box Attack via Controlled Generation**

cs.CR

NAACL 2025 Findings

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13551v1) [paper-pdf](http://arxiv.org/pdf/2504.13551v1)

**Authors**: CheolWon Na, YunSeok Choi, Jee-Hyong Lee

**Abstract**: Many adversarial attack approaches are proposed to verify the vulnerability of language models. However, they require numerous queries and the information on the target model. Even black-box attack methods also require the target model's output information. They are not applicable in real-world scenarios, as in hard black-box settings where the target model is closed and inaccessible. Even the recently proposed hard black-box attacks still require many queries and demand extremely high costs for training adversarial generators. To address these challenges, we propose Q-faker (Query-free Hard Black-box Attacker), a novel and efficient method that generates adversarial examples without accessing the target model. To avoid accessing the target model, we use a surrogate model instead. The surrogate model generates adversarial sentences for a target-agnostic attack. During this process, we leverage controlled generation techniques. We evaluate our proposed method on eight datasets. Experimental results demonstrate our method's effectiveness including high transferability and the high quality of the generated adversarial examples, and prove its practical in hard black-box settings.



## **6. Few-shot Model Extraction Attacks against Sequential Recommender Systems**

cs.LG

It requires substantial modifications.The symbols in the mathematical  formulas are not explained in detail

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2411.11677v2) [paper-pdf](http://arxiv.org/pdf/2411.11677v2)

**Authors**: Hui Zhang, Fu Liu

**Abstract**: Among adversarial attacks against sequential recommender systems, model extraction attacks represent a method to attack sequential recommendation models without prior knowledge. Existing research has primarily concentrated on the adversary's execution of black-box attacks through data-free model extraction. However, a significant gap remains in the literature concerning the development of surrogate models by adversaries with access to few-shot raw data (10\% even less). That is, the challenge of how to construct a surrogate model with high functional similarity within the context of few-shot data scenarios remains an issue that requires resolution.This study addresses this gap by introducing a novel few-shot model extraction framework against sequential recommenders, which is designed to construct a superior surrogate model with the utilization of few-shot data. The proposed few-shot model extraction framework is comprised of two components: an autoregressive augmentation generation strategy and a bidirectional repair loss-facilitated model distillation procedure. Specifically, to generate synthetic data that closely approximate the distribution of raw data, autoregressive augmentation generation strategy integrates a probabilistic interaction sampler to extract inherent dependencies and a synthesis determinant signal module to characterize user behavioral patterns. Subsequently, bidirectional repair loss, which target the discrepancies between the recommendation lists, is designed as auxiliary loss to rectify erroneous predictions from surrogate models, transferring knowledge from the victim model to the surrogate model effectively. Experiments on three datasets show that the proposed few-shot model extraction framework yields superior surrogate models.



## **7. Adversarial Style Augmentation via Large Language Model for Robust Fake News Detection**

cs.CL

WWW'25 research track accepted

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2406.11260v3) [paper-pdf](http://arxiv.org/pdf/2406.11260v3)

**Authors**: Sungwon Park, Sungwon Han, Xing Xie, Jae-Gil Lee, Meeyoung Cha

**Abstract**: The spread of fake news harms individuals and presents a critical social challenge that must be addressed. Although numerous algorithmic and insightful features have been developed to detect fake news, many of these features can be manipulated with style-conversion attacks, especially with the emergence of advanced language models, making it more difficult to differentiate from genuine news. This study proposes adversarial style augmentation, AdStyle, designed to train a fake news detector that remains robust against various style-conversion attacks. The primary mechanism involves the strategic use of LLMs to automatically generate a diverse and coherent array of style-conversion attack prompts, enhancing the generation of particularly challenging prompts for the detector. Experiments indicate that our augmentation strategy significantly improves robustness and detection performance when evaluated on fake news benchmark datasets.



## **8. Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language Models**

cs.CL

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.05050v2) [paper-pdf](http://arxiv.org/pdf/2504.05050v2)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau

**Abstract**: Large language models (LLMs) are foundational explorations to artificial general intelligence, yet their alignment with human values via instruction tuning and preference learning achieves only superficial compliance. Here, we demonstrate that harmful knowledge embedded during pretraining persists as indelible "dark patterns" in LLMs' parametric memory, evading alignment safeguards and resurfacing under adversarial inducement at distributional shifts. In this study, we first theoretically analyze the intrinsic ethical vulnerability of aligned LLMs by proving that current alignment methods yield only local "safety regions" in the knowledge manifold. In contrast, pretrained knowledge remains globally connected to harmful concepts via high-likelihood adversarial trajectories. Building on this theoretical insight, we empirically validate our findings by employing semantic coherence inducement under distributional shifts--a method that systematically bypasses alignment constraints through optimized adversarial prompts. This combined theoretical and empirical approach achieves a 100% attack success rate across 19 out of 23 state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing their universal vulnerabilities.



## **9. EXAM: Exploiting Exclusive System-Level Cache in Apple M-Series SoCs for Enhanced Cache Occupancy Attacks**

cs.CR

Accepted to ACM ASIA CCS 2025

**SubmitDate**: 2025-04-18    [abs](http://arxiv.org/abs/2504.13385v1) [paper-pdf](http://arxiv.org/pdf/2504.13385v1)

**Authors**: Tianhong Xu, Aidong Adam Ding, Yunsi Fei

**Abstract**: Cache occupancy attacks exploit the shared nature of cache hierarchies to infer a victim's activities by monitoring overall cache usage, unlike access-driven cache attacks that focus on specific cache lines or sets. There exists some prior work that target the last-level cache (LLC) of Intel processors, which is inclusive of higher-level caches, and L2 caches of ARM systems. In this paper, we target the System-Level Cache (SLC) of Apple M-series SoCs, which is exclusive to higher-level CPU caches. We address the challenges of the exclusiveness and propose a suite of SLC-cache occupancy attacks, the first of its kind, where an adversary can monitor GPU and other CPU cluster activities from their own CPU cluster. We first discover the structure of SLC in Apple M1 SOC and various policies pertaining to access and sharing through reverse engineering. We propose two attacks against websites. One is a coarse-grained fingerprinting attack, recognizing which website is accessed based on their different GPU memory access patterns monitored through the SLC occupancy channel. The other attack is a fine-grained pixel stealing attack, which precisely monitors the GPU memory usage for rendering different pixels, through the SLC occupancy channel. Third, we introduce a novel screen capturing attack which works beyond webpages, with the monitoring granularity of 57 rows of pixels (there are 1600 rows for the screen). This significantly expands the attack surface, allowing the adversary to retrieve any screen display, posing a substantial new threat to system security. Our findings reveal critical vulnerabilities in Apple's M-series SoCs and emphasize the urgent need for effective countermeasures against cache occupancy attacks in heterogeneous computing environments.



## **10. DYNAMITE: Dynamic Defense Selection for Enhancing Machine Learning-based Intrusion Detection Against Adversarial Attacks**

cs.CR

Accepted by the IEEE/ACM Workshop on the Internet of Safe Things  (SafeThings 2025)

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.13301v1) [paper-pdf](http://arxiv.org/pdf/2504.13301v1)

**Authors**: Jing Chen, Onat Gungor, Zhengli Shang, Elvin Li, Tajana Rosing

**Abstract**: The rapid proliferation of the Internet of Things (IoT) has introduced substantial security vulnerabilities, highlighting the need for robust Intrusion Detection Systems (IDS). Machine learning-based intrusion detection systems (ML-IDS) have significantly improved threat detection capabilities; however, they remain highly susceptible to adversarial attacks. While numerous defense mechanisms have been proposed to enhance ML-IDS resilience, a systematic approach for selecting the most effective defense against a specific adversarial attack remains absent. To address this challenge, we propose Dynamite, a dynamic defense selection framework that enhances ML-IDS by intelligently identifying and deploying the most suitable defense using a machine learning-driven selection mechanism. Our results demonstrate that Dynamite achieves a 96.2% reduction in computational time compared to the Oracle, significantly decreasing computational overhead while preserving strong prediction performance. Dynamite also demonstrates an average F1-score improvement of 76.7% over random defense and 65.8% over the best static state-of-the-art defense.



## **11. Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks**

cs.CR

Accepted at ICLR 2025. Updates in the v3: GPT-4o and Claude 3.5  Sonnet results, improved writing. Updates in the v2: more models (Llama3,  Phi-3, Nemotron-4-340B), jailbreak artifacts for all attacks are available,  evaluation with different judges (Llama-3-70B and Llama Guard 2), more  experiments (convergence plots, ablation on the suffix length for random  search), examples of jailbroken generation

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2404.02151v4) [paper-pdf](http://arxiv.org/pdf/2404.02151v4)

**Authors**: Maksym Andriushchenko, Francesco Croce, Nicolas Flammarion

**Abstract**: We show that even the most recent safety-aligned LLMs are not robust to simple adaptive jailbreaking attacks. First, we demonstrate how to successfully leverage access to logprobs for jailbreaking: we initially design an adversarial prompt template (sometimes adapted to the target LLM), and then we apply random search on a suffix to maximize a target logprob (e.g., of the token "Sure"), potentially with multiple restarts. In this way, we achieve 100% attack success rate -- according to GPT-4 as a judge -- on Vicuna-13B, Mistral-7B, Phi-3-Mini, Nemotron-4-340B, Llama-2-Chat-7B/13B/70B, Llama-3-Instruct-8B, Gemma-7B, GPT-3.5, GPT-4o, and R2D2 from HarmBench that was adversarially trained against the GCG attack. We also show how to jailbreak all Claude models -- that do not expose logprobs -- via either a transfer or prefilling attack with a 100% success rate. In addition, we show how to use random search on a restricted set of tokens for finding trojan strings in poisoned models -- a task that shares many similarities with jailbreaking -- which is the algorithm that brought us the first place in the SaTML'24 Trojan Detection Competition. The common theme behind these attacks is that adaptivity is crucial: different models are vulnerable to different prompting templates (e.g., R2D2 is very sensitive to in-context learning prompts), some models have unique vulnerabilities based on their APIs (e.g., prefilling for Claude), and in some settings, it is crucial to restrict the token search space based on prior knowledge (e.g., for trojan detection). For reproducibility purposes, we provide the code, logs, and jailbreak artifacts in the JailbreakBench format at https://github.com/tml-epfl/llm-adaptive-attacks.



## **12. Chypnosis: Stealthy Secret Extraction using Undervolting-based Static Side-channel Attacks**

cs.CR

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.11633v2) [paper-pdf](http://arxiv.org/pdf/2504.11633v2)

**Authors**: Kyle Mitard, Saleh Khalaj Monfared, Fatemeh Khojasteh Dana, Shahin Tajik

**Abstract**: There is a growing class of static physical side-channel attacks that allow adversaries to extract secrets by probing the persistent state of a circuit. Techniques such as laser logic state imaging (LLSI), impedance analysis (IA), and static power analysis fall into this category. These attacks require that the targeted data remain constant for a specific duration, which often necessitates halting the circuit's clock. Some methods additionally rely on modulating the chip's supply voltage to probe the circuit. However, tampering with the clock or voltage is typically assumed to be detectable, as secure chips often deploy sensors that erase sensitive data upon detecting such anomalies. Furthermore, many secure devices use internal clock sources, making external clock control infeasible. In this work, we introduce a novel class of static side-channel attacks, called Chypnosis, that enables adversaries to freeze a chip's internal clock by inducing a hibernation state via rapid undervolting, and then extracting secrets using static side-channels. We demonstrate that, by rapidly dropping a chip's voltage below the standard nominal levels, the attacker can bypass the clock and voltage sensors and put the chip in a so-called brownout condition, in which the chip's transistors stop switching, but volatile memories (e.g., Flip-flops and SRAMs) still retain their data. We test our attack on AMD FPGAs by putting them into hibernation. We show that not only are all clock sources deactivated, but various clock and voltage sensors also fail to detect the tamper event. Afterward, we present the successful recovery of secret bits from a hibernated chip using two static attacks, namely, LLSI and IA. Finally, we discuss potential countermeasures which could be integrated into future designs.



## **13. Strategic Planning of Stealthy Backdoor Attacks in Markov Decision Processes**

eess.SY

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.13276v1) [paper-pdf](http://arxiv.org/pdf/2504.13276v1)

**Authors**: Xinyi Wei, Shuo Han, Ahmed H. Hemida, Charles A. Kamhoua, Jie Fu

**Abstract**: This paper investigates backdoor attack planning in stochastic control systems modeled as Markov Decision Processes (MDPs). In a backdoor attack, the adversary provides a control policy that behaves well in the original MDP to pass the testing phase. However, when such a policy is deployed with a trigger policy, which perturbs the system dynamics at runtime, it optimizes the attacker's objective instead. To solve jointly the control policy and its trigger, we formulate the attack planning problem as a constrained optimal planning problem in an MDP with augmented state space, with the objective to maximize the attacker's total rewards in the system with an activated trigger, subject to the constraint that the control policy is near optimal in the original MDP. We then introduce a gradient-based optimization method to solve the optimal backdoor attack policy as a pair of coordinated control and trigger policies. Experimental results from a case study validate the effectiveness of our approach in achieving stealthy backdoor attacks.



## **14. Does Refusal Training in LLMs Generalize to the Past Tense?**

cs.CL

Accepted at ICLR 2025. Updates in v2 and v3: added GPT-4o, Claude 3.5  Sonnet, o1-mini, and o1-preview results. Code and jailbreak artifacts:  https://github.com/tml-epfl/llm-past-tense

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2407.11969v4) [paper-pdf](http://arxiv.org/pdf/2407.11969v4)

**Authors**: Maksym Andriushchenko, Nicolas Flammarion

**Abstract**: Refusal training is widely used to prevent LLMs from generating harmful, undesirable, or illegal outputs. We reveal a curious generalization gap in the current refusal training approaches: simply reformulating a harmful request in the past tense (e.g., "How to make a Molotov cocktail?" to "How did people make a Molotov cocktail?") is often sufficient to jailbreak many state-of-the-art LLMs. We systematically evaluate this method on Llama-3 8B, Claude-3.5 Sonnet, GPT-3.5 Turbo, Gemma-2 9B, Phi-3-Mini, GPT-4o mini, GPT-4o, o1-mini, o1-preview, and R2D2 models using GPT-3.5 Turbo as a reformulation model. For example, the success rate of this simple attack on GPT-4o increases from 1% using direct requests to 88% using 20 past tense reformulation attempts on harmful requests from JailbreakBench with GPT-4 as a jailbreak judge. Interestingly, we also find that reformulations in the future tense are less effective, suggesting that refusal guardrails tend to consider past historical questions more benign than hypothetical future questions. Moreover, our experiments on fine-tuning GPT-3.5 Turbo show that defending against past reformulations is feasible when past tense examples are explicitly included in the fine-tuning data. Overall, our findings highlight that the widely used alignment techniques -- such as SFT, RLHF, and adversarial training -- employed to align the studied models can be brittle and do not always generalize as intended. We provide code and jailbreak artifacts at https://github.com/tml-epfl/llm-past-tense.



## **15. Benchmarking the Spatial Robustness of DNNs via Natural and Adversarial Localized Corruptions**

cs.CV

Under review

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.01632v2) [paper-pdf](http://arxiv.org/pdf/2504.01632v2)

**Authors**: Giulia Marchiori Pietrosanti, Giulio Rossolini, Alessandro Biondi, Giorgio Buttazzo

**Abstract**: The robustness of DNNs is a crucial factor in safety-critical applications, particularly in complex and dynamic environments where localized corruptions can arise. While previous studies have evaluated the robustness of semantic segmentation (SS) models under whole-image natural or adversarial corruptions, a comprehensive investigation into the spatial robustness of dense vision models under localized corruptions remained underexplored. This paper fills this gap by introducing specialized metrics for benchmarking the spatial robustness of segmentation models, alongside with an evaluation framework to assess the impact of localized corruptions. Furthermore, we uncover the inherent complexity of characterizing worst-case robustness using a single localized adversarial perturbation. To address this, we propose region-aware multi-attack adversarial analysis, a method that enables a deeper understanding of model robustness against adversarial perturbations applied to specific regions. The proposed metrics and analysis were exploited to evaluate 14 segmentation models in driving scenarios, uncovering key insights into the effects of localized corruption in both natural and adversarial forms. The results reveal that models respond to these two types of threats differently; for instance, transformer-based segmentation models demonstrate notable robustness to localized natural corruptions but are highly vulnerable to adversarial ones and vice-versa for CNN-based models. Consequently, we also address the challenge of balancing robustness to both natural and adversarial localized corruptions by means of ensemble models, thereby achieving a broader threat coverage and improved reliability for dense vision tasks.



## **16. AHSG: Adversarial Attack on High-level Semantics in Graph Neural Networks**

cs.LG

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2412.07468v2) [paper-pdf](http://arxiv.org/pdf/2412.07468v2)

**Authors**: Kai Yuan, Jiahao Zhang, Yidi Wang, Xiaobing Pei

**Abstract**: Adversarial attacks on Graph Neural Networks aim to perturb the performance of the learner by carefully modifying the graph topology and node attributes. Existing methods achieve attack stealthiness by constraining the modification budget and differences in graph properties. However, these methods typically disrupt task-relevant primary semantics directly, which results in low defensibility and detectability of the attack. In this paper, we propose an Adversarial Attack on High-level Semantics for Graph Neural Networks (AHSG), which is a graph structure attack model that ensures the retention of primary semantics. By combining latent representations with shared primary semantics, our model retains detectable attributes and relational patterns of the original graph while leveraging more subtle changes to carry out the attack. Then we use the Projected Gradient Descent algorithm to map the latent representations with attack effects to the adversarial graph. Through experiments on robust graph deep learning models equipped with defense strategies, we demonstrate that AHSG outperforms other state-of-the-art methods in attack effectiveness. Additionally, using Contextual Stochastic Block Models to detect the attacked graph further validates that our method preserves the primary semantics of the graph.



## **17. Impact of Data Duplication on Deep Neural Network-Based Image Classifiers: Robust vs. Standard Models**

cs.LG

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.00638v2) [paper-pdf](http://arxiv.org/pdf/2504.00638v2)

**Authors**: Alireza Aghabagherloo, Aydin Abadi, Sumanta Sarkar, Vishnu Asutosh Dasu, Bart Preneel

**Abstract**: The accuracy and robustness of machine learning models against adversarial attacks are significantly influenced by factors such as training data quality, model architecture, the training process, and the deployment environment. In recent years, duplicated data in training sets, especially in language models, has attracted considerable attention. It has been shown that deduplication enhances both training performance and model accuracy in language models. While the importance of data quality in training image classifier Deep Neural Networks (DNNs) is widely recognized, the impact of duplicated images in the training set on model generalization and performance has received little attention.   In this paper, we address this gap and provide a comprehensive study on the effect of duplicates in image classification. Our analysis indicates that the presence of duplicated images in the training set not only negatively affects the efficiency of model training but also may result in lower accuracy of the image classifier. This negative impact of duplication on accuracy is particularly evident when duplicated data is non-uniform across classes or when duplication, whether uniform or non-uniform, occurs in the training set of an adversarially trained model. Even when duplicated samples are selected in a uniform way, increasing the amount of duplication does not lead to a significant improvement in accuracy.



## **18. A Survey and Evaluation of Adversarial Attacks for Object Detection**

cs.CV

Accepted for publication in the IEEE Transactions on Neural Networks  and Learning Systems (TNNLS)

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2408.01934v5) [paper-pdf](http://arxiv.org/pdf/2408.01934v5)

**Authors**: Khoi Nguyen Tiet Nguyen, Wenyu Zhang, Kangkang Lu, Yuhuan Wu, Xingjian Zheng, Hui Li Tan, Liangli Zhen

**Abstract**: Deep learning models achieve remarkable accuracy in computer vision tasks, yet remain vulnerable to adversarial examples--carefully crafted perturbations to input images that can deceive these models into making confident but incorrect predictions. This vulnerability pose significant risks in high-stakes applications such as autonomous vehicles, security surveillance, and safety-critical inspection systems. While the existing literature extensively covers adversarial attacks in image classification, comprehensive analyses of such attacks on object detection systems remain limited. This paper presents a novel taxonomic framework for categorizing adversarial attacks specific to object detection architectures, synthesizes existing robustness metrics, and provides a comprehensive empirical evaluation of state-of-the-art attack methodologies on popular object detection models, including both traditional detectors and modern detectors with vision-language pretraining. Through rigorous analysis of open-source attack implementations and their effectiveness across diverse detection architectures, we derive key insights into attack characteristics. Furthermore, we delineate critical research gaps and emerging challenges to guide future investigations in securing object detection systems against adversarial threats. Our findings establish a foundation for developing more robust detection models while highlighting the urgent need for standardized evaluation protocols in this rapidly evolving domain.



## **19. Privacy Protection Against Personalized Text-to-Image Synthesis via Cross-image Consistency Constraints**

cs.CV

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.12747v1) [paper-pdf](http://arxiv.org/pdf/2504.12747v1)

**Authors**: Guanyu Wang, Kailong Wang, Yihao Huang, Mingyi Zhou, Zhang Qing cnwatcher, Geguang Pu, Li Li

**Abstract**: The rapid advancement of diffusion models and personalization techniques has made it possible to recreate individual portraits from just a few publicly available images. While such capabilities empower various creative applications, they also introduce serious privacy concerns, as adversaries can exploit them to generate highly realistic impersonations. To counter these threats, anti-personalization methods have been proposed, which add adversarial perturbations to published images to disrupt the training of personalization models. However, existing approaches largely overlook the intrinsic multi-image nature of personalization and instead adopt a naive strategy of applying perturbations independently, as commonly done in single-image settings. This neglects the opportunity to leverage inter-image relationships for stronger privacy protection. Therefore, we advocate for a group-level perspective on privacy protection against personalization. Specifically, we introduce Cross-image Anti-Personalization (CAP), a novel framework that enhances resistance to personalization by enforcing style consistency across perturbed images. Furthermore, we develop a dynamic ratio adjustment strategy that adaptively balances the impact of the consistency loss throughout the attack iterations. Extensive experiments on the classical CelebHQ and VGGFace2 benchmarks show that CAP substantially improves existing methods.



## **20. Adversary-Augmented Simulation for Fairness Evaluation and Defense in Hyperledger Fabric**

cs.CR

20 pages, 14 figures. arXiv admin note: text overlap with  arXiv:2403.14342

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.12733v1) [paper-pdf](http://arxiv.org/pdf/2504.12733v1)

**Authors**: Erwan Mahe, Rouwaida Abdallah, Pierre-Yves Piriou, Sara Tucci-Piergiovanni

**Abstract**: This paper presents an adversary model and a simulation framework specifically tailored for analyzing attacks on distributed systems composed of multiple distributed protocols, with a focus on assessing the security of blockchain networks. Our model classifies and constrains adversarial actions based on the assumptions of the target protocols, defined by failure models, communication models, and the fault tolerance thresholds of Byzantine Fault Tolerant (BFT) protocols. The goal is to study not only the intended effects of adversarial strategies but also their unintended side effects on critical system properties. We apply this framework to analyze fairness properties in a Hyperledger Fabric (HF) blockchain network. Our focus is on novel fairness attacks that involve coordinated adversarial actions across various HF services. Simulations show that even a constrained adversary can violate fairness with respect to specific clients (client fairness) and impact related guarantees (order fairness), which relate the reception order of transactions to their final order in the blockchain. This paper significantly extends our previous work by introducing and evaluating a mitigation mechanism specifically designed to counter transaction reordering attacks. We implement and integrate this defense into our simulation environment, demonstrating its effectiveness under diverse conditions.



## **21. Quantum Computing Supported Adversarial Attack-Resilient Autonomous Vehicle Perception Module for Traffic Sign Classification**

cs.LG

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.12644v1) [paper-pdf](http://arxiv.org/pdf/2504.12644v1)

**Authors**: Reek Majumder, Mashrur Chowdhury, Sakib Mahmud Khan, Zadid Khan, Fahim Ahmad, Frank Ngeni, Gurcan Comert, Judith Mwakalonge, Dimitra Michalaka

**Abstract**: Deep learning (DL)-based image classification models are essential for autonomous vehicle (AV) perception modules since incorrect categorization might have severe repercussions. Adversarial attacks are widely studied cyberattacks that can lead DL models to predict inaccurate output, such as incorrectly classified traffic signs by the perception module of an autonomous vehicle. In this study, we create and compare hybrid classical-quantum deep learning (HCQ-DL) models with classical deep learning (C-DL) models to demonstrate robustness against adversarial attacks for perception modules. Before feeding them into the quantum system, we used transfer learning models, alexnet and vgg-16, as feature extractors. We tested over 1000 quantum circuits in our HCQ-DL models for projected gradient descent (PGD), fast gradient sign attack (FGSA), and gradient attack (GA), which are three well-known untargeted adversarial approaches. We evaluated the performance of all models during adversarial attacks and no-attack scenarios. Our HCQ-DL models maintain accuracy above 95\% during a no-attack scenario and above 91\% for GA and FGSA attacks, which is higher than C-DL models. During the PGD attack, our alexnet-based HCQ-DL model maintained an accuracy of 85\% compared to C-DL models that achieved accuracies below 21\%. Our results highlight that the HCQ-DL models provide improved accuracy for traffic sign classification under adversarial settings compared to their classical counterparts.



## **22. ControlNET: A Firewall for RAG-based LLM System**

cs.CR

Project Page: https://ai.zjuicsr.cn/firewall

**SubmitDate**: 2025-04-17    [abs](http://arxiv.org/abs/2504.09593v2) [paper-pdf](http://arxiv.org/pdf/2504.09593v2)

**Authors**: Hongwei Yao, Haoran Shi, Yidou Chen, Yixin Jiang, Cong Wang, Zhan Qin

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly enhanced the factual accuracy and domain adaptability of Large Language Models (LLMs). This advancement has enabled their widespread deployment across sensitive domains such as healthcare, finance, and enterprise applications. RAG mitigates hallucinations by integrating external knowledge, yet introduces privacy risk and security risk, notably data breaching risk and data poisoning risk. While recent studies have explored prompt injection and poisoning attacks, there remains a significant gap in comprehensive research on controlling inbound and outbound query flows to mitigate these threats. In this paper, we propose an AI firewall, ControlNET, designed to safeguard RAG-based LLM systems from these vulnerabilities. ControlNET controls query flows by leveraging activation shift phenomena to detect adversarial queries and mitigate their impact through semantic divergence. We conduct comprehensive experiments on four different benchmark datasets including Msmarco, HotpotQA, FinQA, and MedicalSys using state-of-the-art open source LLMs (Llama3, Vicuna, and Mistral). Our results demonstrate that ControlNET achieves over 0.909 AUROC in detecting and mitigating security threats while preserving system harmlessness. Overall, ControlNET offers an effective, robust, harmless defense mechanism, marking a significant advancement toward the secure deployment of RAG-based LLM systems.



## **23. On the completeness of several fortification-interdiction games in the Polynomial Hierarchy**

cs.CC

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2406.01756v2) [paper-pdf](http://arxiv.org/pdf/2406.01756v2)

**Authors**: Alberto Boggio Tomasaz, Margarida Carvalho, Roberto Cordone, Pierre Hosteins

**Abstract**: Fortification-interdiction games are tri-level adversarial games where two opponents act in succession to protect, disrupt and simply use an infrastructure for a specific purpose. Many such games have been formulated and tackled in the literature through specific algorithmic methods, however very few investigations exist on the completeness of such fortification problems in order to locate them rigorously in the polynomial hierarchy. We clarify the completeness status of several well-known fortification problems, such as the Tri-level Interdiction Knapsack Problem with unit fortification and attack weights, the Max-flow Interdiction Problem and Shortest Path Interdiction Problem with Fortification, the Multi-level Critical Node Problem with unit weights, as well as a well-studied electric grid defence planning problem. For all of these problems, we prove their completeness either for the $\Sigma^p_2$ or the $\Sigma^p_3$ class of the polynomial hierarchy. We also prove that the Multi-level Fortification-Interdiction Knapsack Problem with an arbitrary number of protection and interdiction rounds and unit fortification and attack weights is complete for any level of the polynomial hierarchy, therefore providing a useful basis for further attempts at proving the completeness of protection-interdiction games at any level of said hierarchy.



## **24. SAIF: Sparse Adversarial and Imperceptible Attack Framework**

cs.CV

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2212.07495v3) [paper-pdf](http://arxiv.org/pdf/2212.07495v3)

**Authors**: Tooba Imtiaz, Morgan Kohler, Jared Miller, Zifeng Wang, Masih Eskander, Mario Sznaier, Octavia Camps, Jennifer Dy

**Abstract**: Adversarial attacks hamper the decision-making ability of neural networks by perturbing the input signal. The addition of calculated small distortion to images, for instance, can deceive a well-trained image classification network. In this work, we propose a novel attack technique called Sparse Adversarial and Interpretable Attack Framework (SAIF). Specifically, we design imperceptible attacks that contain low-magnitude perturbations at a small number of pixels and leverage these sparse attacks to reveal the vulnerability of classifiers. We use the Frank-Wolfe (conditional gradient) algorithm to simultaneously optimize the attack perturbations for bounded magnitude and sparsity with $O(1/\sqrt{T})$ convergence. Empirical results show that SAIF computes highly imperceptible and interpretable adversarial examples, and outperforms state-of-the-art sparse attack methods on the ImageNet dataset.



## **25. Human Aligned Compression for Robust Models**

cs.CV

Presented at the Workshop AdvML at CVPR 2025

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.12255v1) [paper-pdf](http://arxiv.org/pdf/2504.12255v1)

**Authors**: Samuel Räber, Andreas Plesner, Till Aczel, Roger Wattenhofer

**Abstract**: Adversarial attacks on image models threaten system robustness by introducing imperceptible perturbations that cause incorrect predictions. We investigate human-aligned learned lossy compression as a defense mechanism, comparing two learned models (HiFiC and ELIC) against traditional JPEG across various quality levels. Our experiments on ImageNet subsets demonstrate that learned compression methods outperform JPEG, particularly for Vision Transformer architectures, by preserving semantically meaningful content while removing adversarial noise. Even in white-box settings where attackers can access the defense, these methods maintain substantial effectiveness. We also show that sequential compression--applying rounds of compression/decompression--significantly enhances defense efficacy while maintaining classification performance. Our findings reveal that human-aligned compression provides an effective, computationally efficient defense that protects the image features most relevant to human and machine understanding. It offers a practical approach to improving model robustness against adversarial threats.



## **26. Bypassing Prompt Injection and Jailbreak Detection in LLM Guardrails**

cs.CR

12 pages, 5 figures, 6 tables

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11168v2) [paper-pdf](http://arxiv.org/pdf/2504.11168v2)

**Authors**: William Hackett, Lewis Birch, Stefan Trawicki, Neeraj Suri, Peter Garraghan

**Abstract**: Large Language Models (LLMs) guardrail systems are designed to protect against prompt injection and jailbreak attacks. However, they remain vulnerable to evasion techniques. We demonstrate two approaches for bypassing LLM prompt injection and jailbreak detection systems via traditional character injection methods and algorithmic Adversarial Machine Learning (AML) evasion techniques. Through testing against six prominent protection systems, including Microsoft's Azure Prompt Shield and Meta's Prompt Guard, we show that both methods can be used to evade detection while maintaining adversarial utility achieving in some instances up to 100% evasion success. Furthermore, we demonstrate that adversaries can enhance Attack Success Rates (ASR) against black-box targets by leveraging word importance ranking computed by offline white-box models. Our findings reveal vulnerabilities within current LLM protection mechanisms and highlight the need for more robust guardrail systems.



## **27. Soft Prompt Threats: Attacking Safety Alignment and Unlearning in Open-Source LLMs through the Embedding Space**

cs.LG

Trigger Warning: the appendix contains LLM-generated text with  violence and harassment

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2402.09063v2) [paper-pdf](http://arxiv.org/pdf/2402.09063v2)

**Authors**: Leo Schwinn, David Dobre, Sophie Xhonneux, Gauthier Gidel, Stephan Gunnemann

**Abstract**: Current research in adversarial robustness of LLMs focuses on discrete input manipulations in the natural language space, which can be directly transferred to closed-source models. However, this approach neglects the steady progression of open-source models. As open-source models advance in capability, ensuring their safety also becomes increasingly imperative. Yet, attacks tailored to open-source LLMs that exploit full model access remain largely unexplored. We address this research gap and propose the embedding space attack, which directly attacks the continuous embedding representation of input tokens. We find that embedding space attacks circumvent model alignments and trigger harmful behaviors more efficiently than discrete attacks or model fine-tuning. Furthermore, we present a novel threat model in the context of unlearning and show that embedding space attacks can extract supposedly deleted information from unlearned LLMs across multiple datasets and models. Our findings highlight embedding space attacks as an important threat model in open-source LLMs. Trigger Warning: the appendix contains LLM-generated text with violence and harassment.



## **28. Formal Verification of Graph Convolutional Networks with Uncertain Node Features and Uncertain Graph Structure**

cs.LG

published at Transactions on Machine Learning Research (TMLR) 2025

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2404.15065v2) [paper-pdf](http://arxiv.org/pdf/2404.15065v2)

**Authors**: Tobias Ladner, Michael Eichelbeck, Matthias Althoff

**Abstract**: Graph neural networks are becoming increasingly popular in the field of machine learning due to their unique ability to process data structured in graphs. They have also been applied in safety-critical environments where perturbations inherently occur. However, these perturbations require us to formally verify neural networks before their deployment in safety-critical environments as neural networks are prone to adversarial attacks. While there exists research on the formal verification of neural networks, there is no work verifying the robustness of generic graph convolutional network architectures with uncertainty in the node features and in the graph structure over multiple message-passing steps. This work addresses this research gap by explicitly preserving the non-convex dependencies of all elements in the underlying computations through reachability analysis with (matrix) polynomial zonotopes. We demonstrate our approach on three popular benchmark datasets.



## **29. Attribute Inference Attacks for Federated Regression Tasks**

cs.LG

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2411.12697v2) [paper-pdf](http://arxiv.org/pdf/2411.12697v2)

**Authors**: Francesco Diana, Othmane Marfoq, Chuan Xu, Giovanni Neglia, Frédéric Giroire, Eoin Thomas

**Abstract**: Federated Learning (FL) enables multiple clients, such as mobile phones and IoT devices, to collaboratively train a global machine learning model while keeping their data localized. However, recent studies have revealed that the training phase of FL is vulnerable to reconstruction attacks, such as attribute inference attacks (AIA), where adversaries exploit exchanged messages and auxiliary public information to uncover sensitive attributes of targeted clients. While these attacks have been extensively studied in the context of classification tasks, their impact on regression tasks remains largely unexplored. In this paper, we address this gap by proposing novel model-based AIAs specifically designed for regression tasks in FL environments. Our approach considers scenarios where adversaries can either eavesdrop on exchanged messages or directly interfere with the training process. We benchmark our proposed attacks against state-of-the-art methods using real-world datasets. The results demonstrate a significant increase in reconstruction accuracy, particularly in heterogeneous client datasets, a common scenario in FL. The efficacy of our model-based AIAs makes them better candidates for empirically quantifying privacy leakage for federated regression tasks.



## **30. RLSA-PFL: Robust Lightweight Secure Aggregation with Model Inconsistency Detection in Privacy-Preserving Federated Learning**

cs.CR

16 pages, 10 Figures

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2502.08989v2) [paper-pdf](http://arxiv.org/pdf/2502.08989v2)

**Authors**: Nazatul H. Sultan, Yan Bo, Yansong Gao, Seyit Camtepe, Arash Mahboubi, Hang Thanh Bui, Aufeef Chauhan, Hamed Aboutorab, Michael Bewong, Dineshkumar Singh, Praveen Gauravaram, Rafiqul Islam, Sharif Abuadbba

**Abstract**: Federated Learning (FL) allows users to collaboratively train a global machine learning model by sharing local model only, without exposing their private data to a central server. This distributed learning is particularly appealing in scenarios where data privacy is crucial, and it has garnered substantial attention from both industry and academia. However, studies have revealed privacy vulnerabilities in FL, where adversaries can potentially infer sensitive information from the shared model parameters. In this paper, we present an efficient masking-based secure aggregation scheme utilizing lightweight cryptographic primitives to mitigate privacy risks. Our scheme offers several advantages over existing methods. First, it requires only a single setup phase for the entire FL training session, significantly reducing communication overhead. Second, it minimizes user-side overhead by eliminating the need for user-to-user interactions, utilizing an intermediate server layer and a lightweight key negotiation method. Third, the scheme is highly resilient to user dropouts, and the users can join at any FL round. Fourth, it can detect and defend against malicious server activities, including recently discovered model inconsistency attacks. Finally, our scheme ensures security in both semi-honest and malicious settings. We provide security analysis to formally prove the robustness of our approach. Furthermore, we implemented an end-to-end prototype of our scheme. We conducted comprehensive experiments and comparisons, which show that it outperforms existing solutions in terms of communication and computation overhead, functionality, and security.



## **31. Improving Generalization of Universal Adversarial Perturbation via Dynamic Maximin Optimization**

cs.LG

Accepted in AAAI 2025

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2503.12793v3) [paper-pdf](http://arxiv.org/pdf/2503.12793v3)

**Authors**: Yechao Zhang, Yingzhe Xu, Junyu Shi, Leo Yu Zhang, Shengshan Hu, Minghui Li, Yanjun Zhang

**Abstract**: Deep neural networks (DNNs) are susceptible to universal adversarial perturbations (UAPs). These perturbations are meticulously designed to fool the target model universally across all sample classes. Unlike instance-specific adversarial examples (AEs), generating UAPs is more complex because they must be generalized across a wide range of data samples and models. Our research reveals that existing universal attack methods, which optimize UAPs using DNNs with static model parameter snapshots, do not fully leverage the potential of DNNs to generate more effective UAPs. Rather than optimizing UAPs against static DNN models with a fixed training set, we suggest using dynamic model-data pairs to generate UAPs. In particular, we introduce a dynamic maximin optimization strategy, aiming to optimize the UAP across a variety of optimal model-data pairs. We term this approach DM-UAP. DM-UAP utilizes an iterative max-min-min optimization framework that refines the model-data pairs, coupled with a curriculum UAP learning algorithm to examine the combined space of model parameters and data thoroughly. Comprehensive experiments on the ImageNet dataset demonstrate that the proposed DM-UAP markedly enhances both cross-sample universality and cross-model transferability of UAPs. Using only 500 samples for UAP generation, DM-UAP outperforms the state-of-the-art approach with an average increase in fooling ratio of 12.108%.



## **32. SemDiff: Generating Natural Unrestricted Adversarial Examples via Semantic Attributes Optimization in Diffusion Models**

cs.LG

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11923v1) [paper-pdf](http://arxiv.org/pdf/2504.11923v1)

**Authors**: Zeyu Dai, Shengcai Liu, Rui He, Jiahao Wu, Ning Lu, Wenqi Fan, Qing Li, Ke Tang

**Abstract**: Unrestricted adversarial examples (UAEs), allow the attacker to create non-constrained adversarial examples without given clean samples, posing a severe threat to the safety of deep learning models. Recent works utilize diffusion models to generate UAEs. However, these UAEs often lack naturalness and imperceptibility due to simply optimizing in intermediate latent noises. In light of this, we propose SemDiff, a novel unrestricted adversarial attack that explores the semantic latent space of diffusion models for meaningful attributes, and devises a multi-attributes optimization approach to ensure attack success while maintaining the naturalness and imperceptibility of generated UAEs. We perform extensive experiments on four tasks on three high-resolution datasets, including CelebA-HQ, AFHQ and ImageNet. The results demonstrate that SemDiff outperforms state-of-the-art methods in terms of attack success rate and imperceptibility. The generated UAEs are natural and exhibit semantically meaningful changes, in accord with the attributes' weights. In addition, SemDiff is found capable of evading different defenses, which further validates its effectiveness and threatening.



## **33. RAB$^2$-DEF: Dynamic and explainable defense against adversarial attacks in Federated Learning to fair poor clients**

cs.CR

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2410.08244v2) [paper-pdf](http://arxiv.org/pdf/2410.08244v2)

**Authors**: Nuria Rodríguez-Barroso, M. Victoria Luzón, Francisco Herrera

**Abstract**: At the same time that artificial intelligence is becoming popular, concern and the need for regulation is growing, including among other requirements the data privacy. In this context, Federated Learning is proposed as a solution to data privacy concerns derived from different source data scenarios due to its distributed learning. The defense mechanisms proposed in literature are just focused on defending against adversarial attacks and the performance, leaving aside other important qualities such as explainability, fairness to poor quality clients, dynamism in terms of attacks configuration and generality in terms of being resilient against different kinds of attacks. In this work, we propose RAB$^2$-DEF, a $\textbf{r}$esilient $\textbf{a}$gainst $\textbf{b}\text{yzantine}$ and $\textbf{b}$ackdoor attacks which is $\textbf{d}$ynamic, $\textbf{e}$xplainable and $\textbf{f}$air to poor clients using local linear explanations. We test the performance of RAB$^2$-DEF in image datasets and both byzantine and backdoor attacks considering the state-of-the-art defenses and achieve that RAB$^2$-DEF is a proper defense at the same time that it boosts the other qualities towards trustworthy artificial intelligence.



## **34. Support is All You Need for Certified VAE Training**

cs.LG

21 pages, 3 figures, ICLR '25

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11831v1) [paper-pdf](http://arxiv.org/pdf/2504.11831v1)

**Authors**: Changming Xu, Debangshu Banerjee, Deepak Vasisht, Gagandeep Singh

**Abstract**: Variational Autoencoders (VAEs) have become increasingly popular and deployed in safety-critical applications. In such applications, we want to give certified probabilistic guarantees on performance under adversarial attacks. We propose a novel method, CIVET, for certified training of VAEs. CIVET depends on the key insight that we can bound worst-case VAE error by bounding the error on carefully chosen support sets at the latent layer. We show this point mathematically and present a novel training algorithm utilizing this insight. We show in an extensive evaluation across different datasets (in both the wireless and vision application areas), architectures, and perturbation magnitudes that our method outperforms SOTA methods achieving good standard performance with strong robustness guarantees.



## **35. PSBD: Prediction Shift Uncertainty Unlocks Backdoor Detection**

cs.LG

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2406.05826v2) [paper-pdf](http://arxiv.org/pdf/2406.05826v2)

**Authors**: Wei Li, Pin-Yu Chen, Sijia Liu, Ren Wang

**Abstract**: Deep neural networks are susceptible to backdoor attacks, where adversaries manipulate model predictions by inserting malicious samples into the training data. Currently, there is still a significant challenge in identifying suspicious training data to unveil potential backdoor samples. In this paper, we propose a novel method, Prediction Shift Backdoor Detection (PSBD), leveraging an uncertainty-based approach requiring minimal unlabeled clean validation data. PSBD is motivated by an intriguing Prediction Shift (PS) phenomenon, where poisoned models' predictions on clean data often shift away from true labels towards certain other labels with dropout applied during inference, while backdoor samples exhibit less PS. We hypothesize PS results from the neuron bias effect, making neurons favor features of certain classes. PSBD identifies backdoor training samples by computing the Prediction Shift Uncertainty (PSU), the variance in probability values when dropout layers are toggled on and off during model inference. Extensive experiments have been conducted to verify the effectiveness and efficiency of PSBD, which achieves state-of-the-art results among mainstream detection methods. The code is available at https://github.com/WL-619/PSBD.



## **36. Towards Safe Synthetic Image Generation On the Web: A Multimodal Robust NSFW Defense and Million Scale Dataset**

cs.CV

Short Paper The Web Conference

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2504.11707v1) [paper-pdf](http://arxiv.org/pdf/2504.11707v1)

**Authors**: Muhammad Shahid Muneer, Simon S. Woo

**Abstract**: In the past years, we have witnessed the remarkable success of Text-to-Image (T2I) models and their widespread use on the web. Extensive research in making T2I models produce hyper-realistic images has led to new concerns, such as generating Not-Safe-For-Work (NSFW) web content and polluting the web society. To help prevent misuse of T2I models and create a safer web environment for users features like NSFW filters and post-hoc security checks are used in these models. However, recent work unveiled how these methods can easily fail to prevent misuse. In particular, adversarial attacks on text and image modalities can easily outplay defensive measures. %Exploiting such leads to the growing concern of preventing adversarial attacks on text and image modalities. Moreover, there is currently no robust multimodal NSFW dataset that includes both prompt and image pairs and adversarial examples. This work proposes a million-scale prompt and image dataset generated using open-source diffusion models. Second, we develop a multimodal defense to distinguish safe and NSFW text and images, which is robust against adversarial attacks and directly alleviates current challenges. Our extensive experiments show that our model performs well against existing SOTA NSFW detection methods in terms of accuracy and recall, drastically reducing the Attack Success Rate (ASR) in multimodal adversarial attack scenarios. Code: https://github.com/shahidmuneer/multimodal-nsfw-defense.



## **37. Learning to Learn Transferable Generative Attack for Person Re-Identification**

cs.CV

**SubmitDate**: 2025-04-16    [abs](http://arxiv.org/abs/2409.04208v2) [paper-pdf](http://arxiv.org/pdf/2409.04208v2)

**Authors**: Yuan Bian, Min Liu, Xueping Wang, Yunfeng Ma, Yaonan Wang

**Abstract**: Deep learning-based person re-identification (re-id) models are widely employed in surveillance systems and inevitably inherit the vulnerability of deep networks to adversarial attacks. Existing attacks merely consider cross-dataset and cross-model transferability, ignoring the cross-test capability to perturb models trained in different domains. To powerfully examine the robustness of real-world re-id models, the Meta Transferable Generative Attack (MTGA) method is proposed, which adopts meta-learning optimization to promote the generative attacker producing highly transferable adversarial examples by learning comprehensively simulated transfer-based cross-model\&dataset\&test black-box meta attack tasks. Specifically, cross-model\&dataset black-box attack tasks are first mimicked by selecting different re-id models and datasets for meta-train and meta-test attack processes. As different models may focus on different feature regions, the Perturbation Random Erasing module is further devised to prevent the attacker from learning to only corrupt model-specific features. To boost the attacker learning to possess cross-test transferability, the Normalization Mix strategy is introduced to imitate diverse feature embedding spaces by mixing multi-domain statistics of target models. Extensive experiments show the superiority of MTGA, especially in cross-model\&dataset and cross-model\&dataset\&test attacks, our MTGA outperforms the SOTA methods by 21.5\% and 11.3\% on mean mAP drop rate, respectively. The code of MTGA will be released after the paper is accepted.



## **38. Propaganda via AI? A Study on Semantic Backdoors in Large Language Models**

cs.CL

18 pages, 1 figure

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.12344v1) [paper-pdf](http://arxiv.org/pdf/2504.12344v1)

**Authors**: Nay Myat Min, Long H. Pham, Yige Li, Jun Sun

**Abstract**: Large language models (LLMs) demonstrate remarkable performance across myriad language tasks, yet they remain vulnerable to backdoor attacks, where adversaries implant hidden triggers that systematically manipulate model outputs. Traditional defenses focus on explicit token-level anomalies and therefore overlook semantic backdoors-covert triggers embedded at the conceptual level (e.g., ideological stances or cultural references) that rely on meaning-based cues rather than lexical oddities. We first show, in a controlled finetuning setting, that such semantic backdoors can be implanted with only a small poisoned corpus, establishing their practical feasibility. We then formalize the notion of semantic backdoors in LLMs and introduce a black-box detection framework, RAVEN (short for "Response Anomaly Vigilance for uncovering semantic backdoors"), which combines semantic entropy with cross-model consistency analysis. The framework probes multiple models with structured topic-perspective prompts, clusters the sampled responses via bidirectional entailment, and flags anomalously uniform outputs; cross-model comparison isolates model-specific anomalies from corpus-wide biases. Empirical evaluations across diverse LLM families (GPT-4o, Llama, DeepSeek, Mistral) uncover previously undetected semantic backdoors, providing the first proof-of-concept evidence of these hidden vulnerabilities and underscoring the urgent need for concept-level auditing of deployed language models. We open-source our code and data at https://github.com/NayMyatMin/RAVEN.



## **39. SCA: Highly Efficient Semantic-Consistent Unrestricted Adversarial Attack**

cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2410.02240v5) [paper-pdf](http://arxiv.org/pdf/2410.02240v5)

**Authors**: Zihao Pan, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Deep neural network based systems deployed in sensitive environments are vulnerable to adversarial attacks. Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often results in substantial semantic distortions in the denoised output and suffers from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps, and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our research can further draw attention to the security of multimedia information.



## **40. The Obvious Invisible Threat: LLM-Powered GUI Agents' Vulnerability to Fine-Print Injections**

cs.HC

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11281v1) [paper-pdf](http://arxiv.org/pdf/2504.11281v1)

**Authors**: Chaoran Chen, Zhiping Zhang, Bingcan Guo, Shang Ma, Ibrahim Khalilov, Simret A Gebreegziabher, Yanfang Ye, Ziang Xiao, Yaxing Yao, Tianshi Li, Toby Jia-Jun Li

**Abstract**: A Large Language Model (LLM) powered GUI agent is a specialized autonomous system that performs tasks on the user's behalf according to high-level instructions. It does so by perceiving and interpreting the graphical user interfaces (GUIs) of relevant apps, often visually, inferring necessary sequences of actions, and then interacting with GUIs by executing the actions such as clicking, typing, and tapping. To complete real-world tasks, such as filling forms or booking services, GUI agents often need to process and act on sensitive user data. However, this autonomy introduces new privacy and security risks. Adversaries can inject malicious content into the GUIs that alters agent behaviors or induces unintended disclosures of private information. These attacks often exploit the discrepancy between visual saliency for agents and human users, or the agent's limited ability to detect violations of contextual integrity in task automation. In this paper, we characterized six types of such attacks, and conducted an experimental study to test these attacks with six state-of-the-art GUI agents, 234 adversarial webpages, and 39 human participants. Our findings suggest that GUI agents are highly vulnerable, particularly to contextually embedded threats. Moreover, human users are also susceptible to many of these attacks, indicating that simple human oversight may not reliably prevent failures. This misalignment highlights the need for privacy-aware agent design. We propose practical defense strategies to inform the development of safer and more reliable GUI agents.



## **41. Slice+Slice Baby: Generating Last-Level Cache Eviction Sets in the Blink of an Eye**

cs.CR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11208v1) [paper-pdf](http://arxiv.org/pdf/2504.11208v1)

**Authors**: Bradley Morgan, Gal Horowitz, Sioli O'Connell, Stephan van Schaik, Chitchanok Chuengsatiansup, Daniel Genkin, Olaf Maennel, Paul Montague, Eyal Ronen, Yuval Yarom

**Abstract**: An essential step for mounting cache attacks is finding eviction sets, collections of memory locations that contend on cache space. On Intel processors, one of the main challenges for identifying contending addresses is the sliced cache design, where the processor hashes the physical address to determine where in the cache a memory location is stored. While past works have demonstrated that the hash function can be reversed, they also showed that it depends on physical address bits that the adversary does not know.   In this work, we make three main contributions to the art of finding eviction sets. We first exploit microarchitectural races to compare memory access times and identify the cache slice to which an address maps. We then use the known hash function to both reduce the error rate in our slice identification method and to reduce the work by extrapolating slice mappings to untested memory addresses. Finally, we show how to propagate information on eviction sets across different page offsets for the hitherto unexplored case of non-linear hash functions.   Our contributions allow for entire LLC eviction set generation in 0.7 seconds on the Intel i7-9850H and 1.6 seconds on the i9-10900K, both using non-linear functions. This represents a significant improvement compared to state-of-the-art techniques taking 9x and 10x longer, respectively.



## **42. R-TPT: Improving Adversarial Robustness of Vision-Language Models through Test-Time Prompt Tuning**

cs.LG

CVPR 2025

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11195v1) [paper-pdf](http://arxiv.org/pdf/2504.11195v1)

**Authors**: Lijun Sheng, Jian Liang, Zilei Wang, Ran He

**Abstract**: Vision-language models (VLMs), such as CLIP, have gained significant popularity as foundation models, with numerous fine-tuning methods developed to enhance performance on downstream tasks. However, due to their inherent vulnerability and the common practice of selecting from a limited set of open-source models, VLMs suffer from a higher risk of adversarial attacks than traditional vision models. Existing defense techniques typically rely on adversarial fine-tuning during training, which requires labeled data and lacks of flexibility for downstream tasks. To address these limitations, we propose robust test-time prompt tuning (R-TPT), which mitigates the impact of adversarial attacks during the inference stage. We first reformulate the classic marginal entropy objective by eliminating the term that introduces conflicts under adversarial conditions, retaining only the pointwise entropy minimization. Furthermore, we introduce a plug-and-play reliability-based weighted ensembling strategy, which aggregates useful information from reliable augmented views to strengthen the defense. R-TPT enhances defense against adversarial attacks without requiring labeled training data while offering high flexibility for inference tasks. Extensive experiments on widely used benchmarks with various attacks demonstrate the effectiveness of R-TPT. The code is available in https://github.com/TomSheng21/R-TPT.



## **43. Exploring Backdoor Attack and Defense for LLM-empowered Recommendations**

cs.CR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11182v1) [paper-pdf](http://arxiv.org/pdf/2504.11182v1)

**Authors**: Liangbo Ning, Wenqi Fan, Qing Li

**Abstract**: The fusion of Large Language Models (LLMs) with recommender systems (RecSys) has dramatically advanced personalized recommendations and drawn extensive attention. Despite the impressive progress, the safety of LLM-based RecSys against backdoor attacks remains largely under-explored. In this paper, we raise a new problem: Can a backdoor with a specific trigger be injected into LLM-based Recsys, leading to the manipulation of the recommendation responses when the backdoor trigger is appended to an item's title? To investigate the vulnerabilities of LLM-based RecSys under backdoor attacks, we propose a new attack framework termed Backdoor Injection Poisoning for RecSys (BadRec). BadRec perturbs the items' titles with triggers and employs several fake users to interact with these items, effectively poisoning the training set and injecting backdoors into LLM-based RecSys. Comprehensive experiments reveal that poisoning just 1% of the training data with adversarial examples is sufficient to successfully implant backdoors, enabling manipulation of recommendations. To further mitigate such a security threat, we propose a universal defense strategy called Poison Scanner (P-Scanner). Specifically, we introduce an LLM-based poison scanner to detect the poisoned items by leveraging the powerful language understanding and rich knowledge of LLMs. A trigger augmentation agent is employed to generate diverse synthetic triggers to guide the poison scanner in learning domain-specific knowledge of the poisoned item detection task. Extensive experiments on three real-world datasets validate the effectiveness of the proposed P-Scanner.



## **44. Safe Text-to-Image Generation: Simply Sanitize the Prompt Embedding**

cs.CR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2411.10329v2) [paper-pdf](http://arxiv.org/pdf/2411.10329v2)

**Authors**: Huming Qiu, Guanxu Chen, Mi Zhang, Xiaohan Zhang, Xiaoyu You, Min Yang

**Abstract**: In recent years, text-to-image (T2I) generation models have made significant progress in generating high-quality images that align with text descriptions. However, these models also face the risk of unsafe generation, potentially producing harmful content that violates usage policies, such as explicit material. Existing safe generation methods typically focus on suppressing inappropriate content by erasing undesired concepts from visual representations, while neglecting to sanitize the textual representation. Although these methods help mitigate the risk of misuse to some extent, their robustness remains insufficient when dealing with adversarial attacks.   Given that semantic consistency between input text and output image is a core requirement of T2I models, we identify that textual representations are likely the primary source of unsafe generation. To this end, we propose Embedding Sanitizer (ES), which enhances the safety of T2I models by sanitizing inappropriate concepts in prompt embeddings. To our knowledge, ES is the first interpretable safe generation framework that assigns a score to each token in the prompt to indicate its potential harmfulness. In addition, ES adopts a plug-and-play modular design, offering compatibility for seamless integration with various T2I models and other safeguards. Evaluations on five prompt benchmarks show that ES outperforms eleven existing safeguard baselines, achieving state-of-the-art robustness while maintaining high-quality image generation.



## **45. Helper-Friendly Latency-Bounded Mitigation Strategies against Reactive Jamming Adversaries**

cs.IT

16 pages

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11110v1) [paper-pdf](http://arxiv.org/pdf/2504.11110v1)

**Authors**: Soumita Hazra, J. Harshan

**Abstract**: Due to the recent developments in the field of full-duplex radios and cognitive radios, a new class of reactive jamming attacks has gained attention wherein an adversary transmits jamming energy over the victim's frequency band and also monitors various energy statistics in the network so as to detect countermeasures, thereby trapping the victim. Although cooperative mitigation strategies against such security threats exist, they are known to incur spectral-efficiency loss on the helper node, and are also not robust to variable latency-constraints on victim's messages. Identifying these research gaps in existing countermeasures against reactive jamming attacks, we propose a family of helper-friendly cooperative mitigation strategies that are applicable for a wide-range of latency-requirements on the victim's messages as well as practical radio hardware at the helper nodes. The proposed strategies are designed to facilitate reliable communication for the victim, without compromising the helper's spectral efficiency and also minimally disturbing the various energy statistics in the network. For theoretical guarantees on their efficacy, interesting optimization problems are formulated on the choice of the underlying parameters, followed by extensive mathematical analyses on their error-performance and covertness. Experimental results indicate that the proposed strategies should be preferred over the state-of-the-art methods when the helper node is unwilling to compromise on its error performance for assisting the victim.



## **46. Token-Level Constraint Boundary Search for Jailbreaking Text-to-Image Models**

cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11106v1) [paper-pdf](http://arxiv.org/pdf/2504.11106v1)

**Authors**: Jiangtao Liu, Zhaoxin Wang, Handing Wang, Cong Tian, Yaochu Jin

**Abstract**: Recent advancements in Text-to-Image (T2I) generation have significantly enhanced the realism and creativity of generated images. However, such powerful generative capabilities pose risks related to the production of inappropriate or harmful content. Existing defense mechanisms, including prompt checkers and post-hoc image checkers, are vulnerable to sophisticated adversarial attacks. In this work, we propose TCBS-Attack, a novel query-based black-box jailbreak attack that searches for tokens located near the decision boundaries defined by text and image checkers. By iteratively optimizing tokens near these boundaries, TCBS-Attack generates semantically coherent adversarial prompts capable of bypassing multiple defensive layers in T2I models. Extensive experiments demonstrate that our method consistently outperforms state-of-the-art jailbreak attacks across various T2I models, including securely trained open-source models and commercial online services like DALL-E 3. TCBS-Attack achieves an ASR-4 of 45\% and an ASR-1 of 21\% on jailbreaking full-chain T2I models, significantly surpassing baseline methods.



## **47. Unveiling the Threat of Fraud Gangs to Graph Neural Networks: Multi-Target Graph Injection Attacks Against GNN-Based Fraud Detectors**

cs.LG

19 pages, 5 figures, 12 tables, The 39th AAAI Conference on  Artificial Intelligence (AAAI 2025)

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2412.18370v3) [paper-pdf](http://arxiv.org/pdf/2412.18370v3)

**Authors**: Jinhyeok Choi, Heehyeon Kim, Joyce Jiyoung Whang

**Abstract**: Graph neural networks (GNNs) have emerged as an effective tool for fraud detection, identifying fraudulent users, and uncovering malicious behaviors. However, attacks against GNN-based fraud detectors and their risks have rarely been studied, thereby leaving potential threats unaddressed. Recent findings suggest that frauds are increasingly organized as gangs or groups. In this work, we design attack scenarios where fraud gangs aim to make their fraud nodes misclassified as benign by camouflaging their illicit activities in collusion. Based on these scenarios, we study adversarial attacks against GNN-based fraud detectors by simulating attacks of fraud gangs in three real-world fraud cases: spam reviews, fake news, and medical insurance frauds. We define these attacks as multi-target graph injection attacks and propose MonTi, a transformer-based Multi-target one-Time graph injection attack model. MonTi simultaneously generates attributes and edges of all attack nodes with a transformer encoder, capturing interdependencies between attributes and edges more effectively than most existing graph injection attack methods that generate these elements sequentially. Additionally, MonTi adaptively allocates the degree budget for each attack node to explore diverse injection structures involving target, candidate, and attack nodes, unlike existing methods that fix the degree budget across all attack nodes. Experiments show that MonTi outperforms the state-of-the-art graph injection attack methods on five real-world graphs.



## **48. MIMIR: Masked Image Modeling for Mutual Information-based Adversarial Robustness**

cs.CV

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2312.04960v4) [paper-pdf](http://arxiv.org/pdf/2312.04960v4)

**Authors**: Xiaoyun Xu, Shujian Yu, Zhuoran Liu, Stjepan Picek

**Abstract**: Vision Transformers (ViTs) have emerged as a fundamental architecture and serve as the backbone of modern vision-language models. Despite their impressive performance, ViTs exhibit notable vulnerability to evasion attacks, necessitating the development of specialized Adversarial Training (AT) strategies tailored to their unique architecture. While a direct solution might involve applying existing AT methods to ViTs, our analysis reveals significant incompatibilities, particularly with state-of-the-art (SOTA) approaches such as Generalist (CVPR 2023) and DBAT (USENIX Security 2024). This paper presents a systematic investigation of adversarial robustness in ViTs and provides a novel theoretical Mutual Information (MI) analysis in its autoencoder-based self-supervised pre-training. Specifically, we show that MI between the adversarial example and its latent representation in ViT-based autoencoders should be constrained via derived MI bounds. Building on this insight, we propose a self-supervised AT method, MIMIR, that employs an MI penalty to facilitate adversarial pre-training by masked image modeling with autoencoders. Extensive experiments on CIFAR-10, Tiny-ImageNet, and ImageNet-1K show that MIMIR can consistently provide improved natural and robust accuracy, where MIMIR outperforms SOTA AT results on ImageNet-1K. Notably, MIMIR demonstrates superior robustness against unforeseen attacks and common corruption data and can also withstand adaptive attacks where the adversary possesses full knowledge of the defense mechanism.



## **49. RAID: An In-Training Defense against Attribute Inference Attacks in Recommender Systems**

cs.IR

17 pages

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2504.11510v1) [paper-pdf](http://arxiv.org/pdf/2504.11510v1)

**Authors**: Xiaohua Feng, Yuyuan Li, Fengyuan Yu, Ke Xiong, Junjie Fang, Li Zhang, Tianyu Du, Chaochao Chen

**Abstract**: In various networks and mobile applications, users are highly susceptible to attribute inference attacks, with particularly prevalent occurrences in recommender systems. Attackers exploit partially exposed user profiles in recommendation models, such as user embeddings, to infer private attributes of target users, such as gender and political views. The goal of defenders is to mitigate the effectiveness of these attacks while maintaining recommendation performance. Most existing defense methods, such as differential privacy and attribute unlearning, focus on post-training settings, which limits their capability of utilizing training data to preserve recommendation performance. Although adversarial training extends defenses to in-training settings, it often struggles with convergence due to unstable training processes. In this paper, we propose RAID, an in-training defense method against attribute inference attacks in recommender systems. In addition to the recommendation objective, we define a defensive objective to ensure that the distribution of protected attributes becomes independent of class labels, making users indistinguishable from attribute inference attacks. Specifically, this defensive objective aims to solve a constrained Wasserstein barycenter problem to identify the centroid distribution that makes the attribute indistinguishable while complying with recommendation performance constraints. To optimize our proposed objective, we use optimal transport to align users with the centroid distribution. We conduct extensive experiments on four real-world datasets to evaluate RAID. The experimental results validate the effectiveness of RAID and demonstrate its significant superiority over existing methods in multiple aspects.



## **50. Inferring Communities of Interest in Collaborative Learning-based Recommender Systems**

cs.IR

**SubmitDate**: 2025-04-15    [abs](http://arxiv.org/abs/2306.08929v3) [paper-pdf](http://arxiv.org/pdf/2306.08929v3)

**Authors**: Yacine Belal, Sonia Ben Mokhtar, Mohamed Maouche, Anthony Simonet-Boulogne

**Abstract**: Collaborative-learning-based recommender systems, such as those employing Federated Learning (FL) and Gossip Learning (GL), allow users to train models while keeping their history of liked items on their devices. While these methods were seen as promising for enhancing privacy, recent research has shown that collaborative learning can be vulnerable to various privacy attacks. In this paper, we propose a novel attack called Community Inference Attack (CIA), which enables an adversary to identify community members based on a set of target items. What sets CIA apart is its efficiency: it operates at low computational cost by eliminating the need for training surrogate models. Instead, it uses a comparison-based approach, inferring sensitive information by comparing users' models rather than targeting any specific individual model. To evaluate the effectiveness of CIA, we conduct experiments on three real-world recommendation datasets using two recommendation models under both Federated and Gossip-like settings. The results demonstrate that CIA can be up to 10 times more accurate than random guessing. Additionally, we evaluate two mitigation strategies: Differentially Private Stochastic Gradient Descent (DP-SGD) and a Share less policy, which involves sharing fewer, less sensitive model parameters. Our findings suggest that the Share less strategy offers a better privacy-utility trade-off, especially in GL.



