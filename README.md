# Latest Adversarial Attack Papers
**update at 2025-08-21 21:41:38**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Security Concerns for Large Language Models: A Survey**

cs.CR

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2505.18889v4) [paper-pdf](http://arxiv.org/pdf/2505.18889v4)

**Authors**: Miles Q. Li, Benjamin C. M. Fung

**Abstract**: Large Language Models (LLMs) such as ChatGPT and its competitors have caused a revolution in natural language processing, but their capabilities also introduce new security vulnerabilities. This survey provides a comprehensive overview of these emerging concerns, categorizing threats into several key areas: prompt injection and jailbreaking; adversarial attacks, including input perturbations and data poisoning; misuse by malicious actors to generate disinformation, phishing emails, and malware; and the worrisome risks inherent in autonomous LLM agents. Recently, a significant focus is increasingly being placed on the latter, exploring goal misalignment, emergent deception, self-preservation instincts, and the potential for LLMs to develop and pursue covert, misaligned objectives, a behavior known as scheming, which may even persist through safety training. We summarize recent academic and industrial studies from 2022 to 2025 that exemplify each threat, analyze proposed defenses and their limitations, and identify open challenges in securing LLM-based applications. We conclude by emphasizing the importance of advancing robust, multi-layered security strategies to ensure LLMs are safe and beneficial.



## **2. Universal and Transferable Adversarial Attack on Large Language Models Using Exponentiated Gradient Descent**

cs.LG

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2508.14853v1) [paper-pdf](http://arxiv.org/pdf/2508.14853v1)

**Authors**: Sajib Biswas, Mao Nishino, Samuel Jacob Chacko, Xiuwen Liu

**Abstract**: As large language models (LLMs) are increasingly deployed in critical applications, ensuring their robustness and safety alignment remains a major challenge. Despite the overall success of alignment techniques such as reinforcement learning from human feedback (RLHF) on typical prompts, LLMs remain vulnerable to jailbreak attacks enabled by crafted adversarial triggers appended to user prompts. Most existing jailbreak methods either rely on inefficient searches over discrete token spaces or direct optimization of continuous embeddings. While continuous embeddings can be given directly to selected open-source models as input, doing so is not feasible for proprietary models. On the other hand, projecting these embeddings back into valid discrete tokens introduces additional complexity and often reduces attack effectiveness. We propose an intrinsic optimization method which directly optimizes relaxed one-hot encodings of the adversarial suffix tokens using exponentiated gradient descent coupled with Bregman projection, ensuring that the optimized one-hot encoding of each token always remains within the probability simplex. We provide theoretical proof of convergence for our proposed method and implement an efficient algorithm that effectively jailbreaks several widely used LLMs. Our method achieves higher success rates and faster convergence compared to three state-of-the-art baselines, evaluated on five open-source LLMs and four adversarial behavior datasets curated for evaluating jailbreak methods. In addition to individual prompt attacks, we also generate universal adversarial suffixes effective across multiple prompts and demonstrate transferability of optimized suffixes to different LLMs.



## **3. Fragile, Robust, and Antifragile: A Perspective from Parameter Responses in Reinforcement Learning Under Stress**

cs.LG

Withdrawn pending a review of attribution and overlap with Pravin et  al., Artificial Intelligence (2024), DOI: 10.1016/j.artint.2023.104060.  Further dissemination is paused while we determine appropriate next steps

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2506.23036v2) [paper-pdf](http://arxiv.org/pdf/2506.23036v2)

**Authors**: Zain ul Abdeen, Ming Jin

**Abstract**: This paper explores Reinforcement learning (RL) policy robustness by systematically analyzing network parameters under internal and external stresses. Inspired by synaptic plasticity in neuroscience, synaptic filtering introduces internal stress by selectively perturbing parameters, while adversarial attacks apply external stress through modified agent observations. This dual approach enables the classification of parameters as fragile, robust, or antifragile, based on their influence on policy performance in clean and adversarial settings. Parameter scores are defined to quantify these characteristics, and the framework is validated on PPO-trained agents in Mujoco continuous control environments. The results highlight the presence of antifragile parameters that enhance policy performance under stress, demonstrating the potential of targeted filtering techniques to improve RL policy adaptability. These insights provide a foundation for future advancements in the design of robust and antifragile RL systems.



## **4. Distributional Adversarial Attacks and Training in Deep Hedging**

math.OC

Preprint. Under review

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2508.14757v1) [paper-pdf](http://arxiv.org/pdf/2508.14757v1)

**Authors**: Guangyi He, Tobias Sutter, Lukas Gonon

**Abstract**: In this paper, we study the robustness of classical deep hedging strategies under distributional shifts by leveraging the concept of adversarial attacks. We first demonstrate that standard deep hedging models are highly vulnerable to small perturbations in the input distribution, resulting in significant performance degradation. Motivated by this, we propose an adversarial training framework tailored to increase the robustness of deep hedging strategies. Our approach extends pointwise adversarial attacks to the distributional setting and introduces a computationally tractable reformulation of the adversarial optimization problem over a Wasserstein ball. This enables the efficient training of hedging strategies that are resilient to distributional perturbations. Through extensive numerical experiments, we show that adversarially trained deep hedging strategies consistently outperform their classical counterparts in terms of out-of-sample performance and resilience to model misspecification. Our findings establish a practical and effective framework for robust deep hedging under realistic market uncertainties.



## **5. Foe for Fraud: Transferable Adversarial Attacks in Credit Card Fraud Detection**

cs.CR

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2508.14699v1) [paper-pdf](http://arxiv.org/pdf/2508.14699v1)

**Authors**: Jan Lum Fok, Qingwen Zeng, Shiping Chen, Oscar Fawkes, Huaming Chen

**Abstract**: Credit card fraud detection (CCFD) is a critical application of Machine Learning (ML) in the financial sector, where accurately identifying fraudulent transactions is essential for mitigating financial losses. ML models have demonstrated their effectiveness in fraud detection task, in particular with the tabular dataset. While adversarial attacks have been extensively studied in computer vision and deep learning, their impacts on the ML models, particularly those trained on CCFD tabular datasets, remains largely unexplored. These latent vulnerabilities pose significant threats to the security and stability of the financial industry, especially in high-value transactions where losses could be substantial. To address this gap, in this paper, we present a holistic framework that investigate the robustness of CCFD ML model against adversarial perturbations under different circumstances. Specifically, the gradient-based attack methods are incorporated into the tabular credit card transaction data in both black- and white-box adversarial attacks settings. Our findings confirm that tabular data is also susceptible to subtle perturbations, highlighting the need for heightened awareness among financial technology practitioners regarding ML model security and trustworthiness. Furthermore, the experiments by transferring adversarial samples from gradient-based attack method to non-gradient-based models also verify our findings. Our results demonstrate that such attacks remain effective, emphasizing the necessity of developing robust defenses for CCFD algorithms.



## **6. Dark Miner: Defend against undesirable generation for text-to-image diffusion models**

cs.CV

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2409.17682v3) [paper-pdf](http://arxiv.org/pdf/2409.17682v3)

**Authors**: Zheling Meng, Bo Peng, Xiaochuan Jin, Yue Jiang, Wei Wang, Jing Dong, Tieniu Tan

**Abstract**: Text-to-image diffusion models have been demonstrated with undesired generation due to unfiltered large-scale training data, such as sexual images and copyrights, necessitating the erasure of undesired concepts. Most existing methods focus on modifying the generation probabilities conditioned on the texts containing target concepts. However, they fail to guarantee the desired generation of texts unseen in the training phase, especially for the adversarial texts from malicious attacks. In this paper, we analyze the erasure task and point out that existing methods cannot guarantee the minimization of the total probabilities of undesired generation. To tackle this problem, we propose Dark Miner. It entails a recurring three-stage process that comprises mining, verifying, and circumventing. This method greedily mines embeddings with maximum generation probabilities of target concepts and more effectively reduces their generation. In the experiments, we evaluate its performance on the inappropriateness, object, and style concepts. Compared with the previous methods, our method achieves better erasure and defense results, especially under multiple adversarial attacks, while preserving the native generation capability of the models. Our code will be available on GitHub.



## **7. When Good Sounds Go Adversarial: Jailbreaking Audio-Language Models with Benign Inputs**

cs.SD

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2508.03365v2) [paper-pdf](http://arxiv.org/pdf/2508.03365v2)

**Authors**: Bodam Kim, Hiskias Dingeto, Taeyoun Kwon, Dasol Choi, DongGeon Lee, Haon Park, JaeHoon Lee, Jongho Shin

**Abstract**: As large language models become increasingly integrated into daily life, audio has emerged as a key interface for human-AI interaction. However, this convenience also introduces new vulnerabilities, making audio a potential attack surface for adversaries. Our research introduces WhisperInject, a two-stage adversarial audio attack framework that can manipulate state-of-the-art audio language models to generate harmful content. Our method uses imperceptible perturbations in audio inputs that remain benign to human listeners. The first stage uses a novel reward-based optimization method, Reinforcement Learning with Projected Gradient Descent (RL-PGD), to guide the target model to circumvent its own safety protocols and generate harmful native responses. This native harmful response then serves as the target for Stage 2, Payload Injection, where we use Projected Gradient Descent (PGD) to optimize subtle perturbations that are embedded into benign audio carriers, such as weather queries or greeting messages. Validated under the rigorous StrongREJECT, LlamaGuard, as well as Human Evaluation safety evaluation framework, our experiments demonstrate a success rate exceeding 86% across Qwen2.5-Omni-3B, Qwen2.5-Omni-7B, and Phi-4-Multimodal. Our work demonstrates a new class of practical, audio-native threats, moving beyond theoretical exploits to reveal a feasible and covert method for manipulating AI behavior.



## **8. Adversarial control of synchronization in complex oscillator networks**

nlin.AO

10 pages, 4 figures

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2506.02403v2) [paper-pdf](http://arxiv.org/pdf/2506.02403v2)

**Authors**: Yasutoshi Nagahama, Kosuke Miyazato, Kazuhiro Takemoto

**Abstract**: This study investigates adversarial attacks, a concept from deep learning, designed to control synchronization dynamics through strategically crafted weak perturbations. We propose a gradient-based optimization method that identifies small phase perturbations to dramatically enhance or suppress collective synchronization in Kuramoto oscillator networks. Our approach formulates synchronization control as an adversarial optimization problem, computing gradients of the order parameter with respect to oscillator phases to determine optimal perturbation directions. Results demonstrate that extremely small phase perturbations applied to network oscillators can achieve significant synchronization control across diverse network architectures. Our analysis reveals that synchronization enhancement is achievable across various network sizes, while synchronization suppression becomes particularly effective in larger networks, with effectiveness scaling favorably with network size. The method is systematically validated on canonical model networks including scale-free and small-world topologies, and real-world networks representing power grids and brain connectivity patterns. This adversarial framework represents a novel paradigm for synchronization management by introducing deep learning concepts to networked dynamical systems.



## **9. Backdooring Self-Supervised Contrastive Learning by Noisy Alignment**

cs.CV

Accepted by ICCV 2025

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.14015v1) [paper-pdf](http://arxiv.org/pdf/2508.14015v1)

**Authors**: Tuo Chen, Jie Gui, Minjing Dong, Ju Jia, Lanting Fang, Jian Liu

**Abstract**: Self-supervised contrastive learning (CL) effectively learns transferable representations from unlabeled data containing images or image-text pairs but suffers vulnerability to data poisoning backdoor attacks (DPCLs). An adversary can inject poisoned images into pretraining datasets, causing compromised CL encoders to exhibit targeted misbehavior in downstream tasks. Existing DPCLs, however, achieve limited efficacy due to their dependence on fragile implicit co-occurrence between backdoor and target object and inadequate suppression of discriminative features in backdoored images. We propose Noisy Alignment (NA), a DPCL method that explicitly suppresses noise components in poisoned images. Inspired by powerful training-controllable CL attacks, we identify and extract the critical objective of noisy alignment, adapting it effectively into data-poisoning scenarios. Our method implements noisy alignment by strategically manipulating contrastive learning's random cropping mechanism, formulating this process as an image layout optimization problem with theoretically derived optimal parameters. The resulting method is simple yet effective, achieving state-of-the-art performance compared to existing DPCLs, while maintaining clean-data accuracy. Furthermore, Noisy Alignment demonstrates robustness against common backdoor defenses. Codes can be found at https://github.com/jsrdcht/Noisy-Alignment.



## **10. FedUP: Efficient Pruning-based Federated Unlearning for Model Poisoning Attacks**

cs.LG

15 pages, 5 figures, 7 tables

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.13853v1) [paper-pdf](http://arxiv.org/pdf/2508.13853v1)

**Authors**: Nicolò Romandini, Cristian Borcea, Rebecca Montanari, Luca Foschini

**Abstract**: Federated Learning (FL) can be vulnerable to attacks, such as model poisoning, where adversaries send malicious local weights to compromise the global model. Federated Unlearning (FU) is emerging as a solution to address such vulnerabilities by selectively removing the influence of detected malicious contributors on the global model without complete retraining. However, unlike typical FU scenarios where clients are trusted and cooperative, applying FU with malicious and possibly colluding clients is challenging because their collaboration in unlearning their data cannot be assumed. This work presents FedUP, a lightweight FU algorithm designed to efficiently mitigate malicious clients' influence by pruning specific connections within the attacked model. Our approach achieves efficiency by relying only on clients' weights from the last training round before unlearning to identify which connections to inhibit. Isolating malicious influence is non-trivial due to overlapping updates from benign and malicious clients. FedUP addresses this by carefully selecting and zeroing the highest magnitude weights that diverge the most between the latest updates from benign and malicious clients while preserving benign information. FedUP is evaluated under a strong adversarial threat model, where up to 50%-1 of the clients could be malicious and have full knowledge of the aggregation process. We demonstrate the effectiveness, robustness, and efficiency of our solution through experiments across IID and Non-IID data, under label-flipping and backdoor attacks, and by comparing it with state-of-the-art (SOTA) FU solutions. In all scenarios, FedUP reduces malicious influence, lowering accuracy on malicious data to match that of a model retrained from scratch while preserving performance on benign data. FedUP achieves effective unlearning while consistently being faster and saving storage compared to the SOTA.



## **11. Timestep-Compressed Attack on Spiking Neural Networks through Timestep-Level Backpropagation**

cs.CV

8 pages

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.13812v1) [paper-pdf](http://arxiv.org/pdf/2508.13812v1)

**Authors**: Donghwa Kang, Doohyun Kim, Sang-Ki Ko, Jinkyu Lee, Hyeongboo Baek, Brent ByungHoon Kang

**Abstract**: State-of-the-art (SOTA) gradient-based adversarial attacks on spiking neural networks (SNNs), which largely rely on extending FGSM and PGD frameworks, face a critical limitation: substantial attack latency from multi-timestep processing, rendering them infeasible for practical real-time applications. This inefficiency stems from their design as direct extensions of ANN paradigms, which fail to exploit key SNN properties. In this paper, we propose the timestep-compressed attack (TCA), a novel framework that significantly reduces attack latency. TCA introduces two components founded on key insights into SNN behavior. First, timestep-level backpropagation (TLBP) is based on our finding that global temporal information in backpropagation to generate perturbations is not critical for an attack's success, enabling per-timestep evaluation for early stopping. Second, adversarial membrane potential reuse (A-MPR) is motivated by the observation that initial timesteps are inefficiently spent accumulating membrane potential, a warm-up phase that can be pre-calculated and reused. Our experiments on VGG-11 and ResNet-17 with the CIFAR-10/100 and CIFAR10-DVS datasets show that TCA significantly reduces the required attack latency by up to 56.6% and 57.1% compared to SOTA methods in white-box and black-box settings, respectively, while maintaining a comparable attack success rate.



## **12. Enhancing Targeted Adversarial Attacks on Large Vision-Language Models through Intermediate Projector Guidance**

cs.CV

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.13739v1) [paper-pdf](http://arxiv.org/pdf/2508.13739v1)

**Authors**: Yiming Cao, Yanjie Li, Kaisheng Liang, Yuni Lai, Bin Xiao

**Abstract**: Targeted adversarial attacks are essential for proactively identifying security flaws in Vision-Language Models before real-world deployment. However, current methods perturb images to maximize global similarity with the target text or reference image at the encoder level, collapsing rich visual semantics into a single global vector. This limits attack granularity, hindering fine-grained manipulations such as modifying a car while preserving its background. Furthermore, these methods largely overlook the projector module, a critical semantic bridge between the visual encoder and the language model in VLMs, thereby failing to disrupt the full vision-language alignment pipeline within VLMs and limiting attack effectiveness. To address these issues, we propose the Intermediate Projector Guided Attack (IPGA), the first method to attack using the intermediate stage of the projector module, specifically the widely adopted Q-Former, which transforms global image embeddings into fine-grained visual features. This enables more precise control over adversarial perturbations by operating on semantically meaningful visual tokens rather than a single global representation. Specifically, IPGA leverages the Q-Former pretrained solely on the first vision-language alignment stage, without LLM fine-tuning, which improves both attack effectiveness and transferability across diverse VLMs. Furthermore, we propose Residual Query Alignment (RQA) to preserve unrelated visual content, thereby yielding more controlled and precise adversarial manipulations. Extensive experiments show that our attack method consistently outperforms existing methods in both standard global image captioning tasks and fine-grained visual question-answering tasks in black-box environment. Additionally, IPGA successfully transfers to multiple commercial VLMs, including Google Gemini and OpenAI GPT.



## **13. The AI Risk Spectrum: From Dangerous Capabilities to Existential Threats**

cs.CY

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.13700v1) [paper-pdf](http://arxiv.org/pdf/2508.13700v1)

**Authors**: Markov Grey, Charbel-Raphaël Segerie

**Abstract**: As AI systems become more capable, integrated, and widespread, understanding the associated risks becomes increasingly important. This paper maps the full spectrum of AI risks, from current harms affecting individual users to existential threats that could endanger humanity's survival. We organize these risks into three main causal categories. Misuse risks, which occur when people deliberately use AI for harmful purposes - creating bioweapons, launching cyberattacks, adversarial AI attacks or deploying lethal autonomous weapons. Misalignment risks happen when AI systems pursue outcomes that conflict with human values, irrespective of developer intentions. This includes risks arising through specification gaming (reward hacking), scheming and power-seeking tendencies in pursuit of long-term strategic goals. Systemic risks, which arise when AI integrates into complex social systems in ways that gradually undermine human agency - concentrating power, accelerating political and economic disempowerment, creating overdependence that leads to human enfeeblement, or irreversibly locking in current values curtailing future moral progress. Beyond these core categories, we identify risk amplifiers - competitive pressures, accidents, corporate indifference, and coordination failures - that make all risks more likely and severe. Throughout, we connect today's existing risks and empirically observable AI behaviors to plausible future outcomes, demonstrating how existing trends could escalate to catastrophic outcomes. Our goal is to help readers understand the complete landscape of AI risks. Good futures are possible, but they don't happen by default. Navigating these challenges will require unprecedented coordination, but an extraordinary future awaits if we do.



## **14. Mask and Restore: Blind Backdoor Defense at Test Time with Masked Autoencoder**

cs.LG

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2303.15564v3) [paper-pdf](http://arxiv.org/pdf/2303.15564v3)

**Authors**: Tao Sun, Lu Pang, Weimin Lyu, Chao Chen, Haibin Ling

**Abstract**: Deep neural networks are vulnerable to backdoor attacks, where an adversary manipulates the model behavior through overlaying images with special triggers. Existing backdoor defense methods often require accessing a few validation data and model parameters, which is impractical in many real-world applications, e.g., when the model is provided as a cloud service. In this paper, we address the practical task of blind backdoor defense at test time, in particular for local attacks and black-box models. The true label of every test image needs to be recovered on the fly from a suspicious model regardless of image benignity. We consider test-time image purification that incapacitates local triggers while keeping semantic contents intact. Due to diverse trigger patterns and sizes, the heuristic trigger search can be unscalable. We circumvent such barrier by leveraging the strong reconstruction power of generative models, and propose Blind Defense with Masked AutoEncoder (BDMAE). BDMAE detects possible local triggers using image structural similarity and label consistency between the test image and MAE restorations. The detection results are then refined by considering trigger topology. Finally, we fuse MAE restorations adaptively into a purified image for making prediction. Extensive experiments under different backdoor settings validate its effectiveness and generalizability.



## **15. Robust Federated Learning under Adversarial Attacks via Loss-Based Client Clustering**

cs.LG

16 pages, 5 figures

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.12672v2) [paper-pdf](http://arxiv.org/pdf/2508.12672v2)

**Authors**: Emmanouil Kritharakis, Dusan Jakovetic, Antonios Makris, Konstantinos Tserpes

**Abstract**: Federated Learning (FL) enables collaborative model training across multiple clients without sharing private data. We consider FL scenarios wherein FL clients are subject to adversarial (Byzantine) attacks, while the FL server is trusted (honest) and has a trustworthy side dataset. This may correspond to, e.g., cases where the server possesses trusted data prior to federation, or to the presence of a trusted client that temporarily assumes the server role. Our approach requires only two honest participants, i.e., the server and one client, to function effectively, without prior knowledge of the number of malicious clients. Theoretical analysis demonstrates bounded optimality gaps even under strong Byzantine attacks. Experimental results show that our algorithm significantly outperforms standard and robust FL baselines such as Mean, Trimmed Mean, Median, Krum, and Multi-Krum under various attack strategies including label flipping, sign flipping, and Gaussian noise addition across MNIST, FMNIST, and CIFAR-10 benchmarks using the Flower framework.



## **16. CCFC: Core & Core-Full-Core Dual-Track Defense for LLM Jailbreak Protection**

cs.CR

11 pages, 1 figure

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.14128v1) [paper-pdf](http://arxiv.org/pdf/2508.14128v1)

**Authors**: Jiaming Hu, Haoyu Wang, Debarghya Mukherjee, Ioannis Ch. Paschalidis

**Abstract**: Jailbreak attacks pose a serious challenge to the safe deployment of large language models (LLMs). We introduce CCFC (Core & Core-Full-Core), a dual-track, prompt-level defense framework designed to mitigate LLMs' vulnerabilities from prompt injection and structure-aware jailbreak attacks. CCFC operates by first isolating the semantic core of a user query via few-shot prompting, and then evaluating the query using two complementary tracks: a core-only track to ignore adversarial distractions (e.g., toxic suffixes or prefix injections), and a core-full-core (CFC) track to disrupt the structural patterns exploited by gradient-based or edit-based attacks. The final response is selected based on a safety consistency check across both tracks, ensuring robustness without compromising on response quality. We demonstrate that CCFC cuts attack success rates by 50-75% versus state-of-the-art defenses against strong adversaries (e.g., DeepInception, GCG), without sacrificing fidelity on benign queries. Our method consistently outperforms state-of-the-art prompt-level defenses, offering a practical and effective solution for safer LLM deployment.



## **17. Boosting Adversarial Transferability for Hyperspectral Image Classification Using 3D Structure-invariant Transformation and Weighted Intermediate Feature Divergence**

cs.CV

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2506.10459v2) [paper-pdf](http://arxiv.org/pdf/2506.10459v2)

**Authors**: Chun Liu, Bingqian Zhu, Tao Xu, Zheng Zheng, Zheng Li, Wei Yang, Zhigang Han, Jiayao Wang

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to adversarial attacks, which pose security challenges to hyperspectral image (HSI) classification based on DNNs. Numerous adversarial attack methods have been designed in the domain of natural images. However, different from natural images, HSIs contains high-dimensional rich spectral information, which presents new challenges for generating adversarial examples. Based on the specific characteristics of HSIs, this paper proposes a novel method to enhance the transferability of the adversarial examples for HSI classification using 3D structure-invariant transformation and weighted intermediate feature divergence. While keeping the HSIs structure invariant, the proposed method divides the image into blocks in both spatial and spectral dimensions. Then, various transformations are applied on each block to increase input diversity and mitigate the overfitting to substitute models. Moreover, a weighted intermediate feature divergence loss is also designed by leveraging the differences between the intermediate features of original and adversarial examples. It constrains the perturbation direction by enlarging the feature maps of the original examples, and assigns different weights to different feature channels to destroy the features that have a greater impact on HSI classification. Extensive experiments demonstrate that the adversarial examples generated by the proposed method achieve more effective adversarial transferability on three public HSI datasets. Furthermore, the method maintains robust attack performance even under defense strategies.



## **18. A robust and composable device-independent protocol for oblivious transfer using (fully) untrusted quantum devices in the bounded storage model**

quant-ph

Major improvement in the main result (security against non-IID  devices)

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2404.11283v3) [paper-pdf](http://arxiv.org/pdf/2404.11283v3)

**Authors**: Rishabh Batra, Sayantan Chakraborty, Rahul Jain, Upendra Kapshikar

**Abstract**: We present a robust and composable device-independent (DI) quantum protocol between two parties for oblivious transfer (OT) using Magic Square devices in the bounded storage model in which the (honest and cheating) devices and parties have no long-term quantum memory. After a fixed constant (real-world) time interval, referred to as DELAY, the quantum states decohere completely. The adversary (cheating party), with full control over the devices, is allowed joint (non-IID) quantum operations on the devices, and there are no time and space complexity bounds placed on its powers. The running time of the honest parties is polylog({\lambda}) (where {\lambda} is the security parameter). Our protocol has negligible (in {\lambda}) correctness and security errors and can be implemented in the NISQ (Noisy Intermediate Scale Quantum) era. By robustness, we mean that our protocol is correct even when devices are slightly off (by a small constant) from their ideal specification. This is an important property since small manufacturing errors in the real-world devices are inevitable. Our protocol is sequentially composable and, hence, can be used as a building block to construct larger protocols (including DI bit-commitment and DI secure multi-party computation) while still preserving correctness and security guarantees.   None of the known DI protocols for OT in the literature are robust and secure against joint quantum attacks. This was a major open question in device-independent two-party distrustful cryptography, which we resolve.   We prove a parallel repetition theorem for a certain class of entangled games with a hybrid (quantum-classical) strategy to show the security of our protocol. The hybrid strategy helps to incorporate DELAY in our protocol. This parallel repetition theorem is a main technical contribution of our work.



## **19. A Risk Manager for Intrusion Tolerant Systems: Enhancing HAL 9000 with New Scoring and Data Sources**

cs.CR

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.13364v1) [paper-pdf](http://arxiv.org/pdf/2508.13364v1)

**Authors**: Tadeu Freitas, Carlos Novo, Inês Dutra, João Soares, Manuel Correia, Benham Shariati, Rolando Martins

**Abstract**: Intrusion Tolerant Systems (ITSs) have become increasingly critical due to the rise of multi-domain adversaries exploiting diverse attack surfaces. ITS architectures aim to tolerate intrusions, ensuring system compromise is prevented or mitigated even with adversary presence. Existing ITS solutions often employ Risk Managers leveraging public security intelligence to adjust system defenses dynamically against emerging threats. However, these approaches rely heavily on databases like NVD and ExploitDB, which require manual analysis for newly discovered vulnerabilities. This dependency limits the system's responsiveness to rapidly evolving threats. HAL 9000, an ITS Risk Manager introduced in our prior work, addressed these challenges through machine learning. By analyzing descriptions of known vulnerabilities, HAL 9000 predicts and assesses new vulnerabilities automatically. To calculate the risk of a system, it also incorporates the Exploitability Probability Scoring system to estimate the likelihood of exploitation within 30 days, enhancing proactive defense capabilities.   Despite its success, HAL 9000's reliance on NVD and ExploitDB knowledge is a limitation, considering the availability of other sources of information. This extended work introduces a custom-built scraper that continuously mines diverse threat sources, including security advisories, research forums, and real-time exploit proofs-of-concept. This significantly expands HAL 9000's intelligence base, enabling earlier detection and assessment of unverified vulnerabilities. Our evaluation demonstrates that integrating scraper-derived intelligence with HAL 9000's risk management framework substantially improves its ability to address emerging threats. This paper details the scraper's integration into the architecture, its role in providing additional information on new threats, and the effects on HAL 9000's management.



## **20. Augmented Adversarial Trigger Learning**

cs.LG

Findings of the Association for Computational Linguistics: NAACL 2025

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2503.12339v3) [paper-pdf](http://arxiv.org/pdf/2503.12339v3)

**Authors**: Zhe Wang, Yanjun Qi

**Abstract**: Gradient optimization-based adversarial attack methods automate the learning of adversarial triggers to generate jailbreak prompts or leak system prompts. In this work, we take a closer look at the optimization objective of adversarial trigger learning and propose ATLA: Adversarial Trigger Learning with Augmented objectives. ATLA improves the negative log-likelihood loss used by previous studies into a weighted loss formulation that encourages the learned adversarial triggers to optimize more towards response format tokens. This enables ATLA to learn an adversarial trigger from just one query-response pair and the learned trigger generalizes well to other similar queries. We further design a variation to augment trigger optimization with an auxiliary loss that suppresses evasive responses. We showcase how to use ATLA to learn adversarial suffixes jailbreaking LLMs and to extract hidden system prompts. Empirically we demonstrate that ATLA consistently outperforms current state-of-the-art techniques, achieving nearly 100% success in attacking while requiring 80% fewer queries. ATLA learned jailbreak suffixes demonstrate high generalization to unseen queries and transfer well to new LLMs. We released our code https://github.com/QData/ALTA_Augmented_Adversarial_Trigger_Learning



## **21. DAASH: A Meta-Attack Framework for Synthesizing Effective and Stealthy Adversarial Examples**

cs.CV

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.13309v1) [paper-pdf](http://arxiv.org/pdf/2508.13309v1)

**Authors**: Abdullah Al Nomaan Nafi, Habibur Rahaman, Zafaryab Haider, Tanzim Mahfuz, Fnu Suya, Swarup Bhunia, Prabuddha Chakraborty

**Abstract**: Numerous techniques have been proposed for generating adversarial examples in white-box settings under strict Lp-norm constraints. However, such norm-bounded examples often fail to align well with human perception, and only recently have a few methods begun specifically exploring perceptually aligned adversarial examples. Moreover, it remains unclear whether insights from Lp-constrained attacks can be effectively leveraged to improve perceptual efficacy. In this paper, we introduce DAASH, a fully differentiable meta-attack framework that generates effective and perceptually aligned adversarial examples by strategically composing existing Lp-based attack methods. DAASH operates in a multi-stage fashion: at each stage, it aggregates candidate adversarial examples from multiple base attacks using learned, adaptive weights and propagates the result to the next stage. A novel meta-loss function guides this process by jointly minimizing misclassification loss and perceptual distortion, enabling the framework to dynamically modulate the contribution of each base attack throughout the stages. We evaluate DAASH on adversarially trained models across CIFAR-10, CIFAR-100, and ImageNet. Despite relying solely on Lp-constrained based methods, DAASH significantly outperforms state-of-the-art perceptual attacks such as AdvAD -- achieving higher attack success rates (e.g., 20.63\% improvement) and superior visual quality, as measured by SSIM, LPIPS, and FID (improvements $\approx$ of 11, 0.015, and 5.7, respectively). Furthermore, DAASH generalizes well to unseen defenses, making it a practical and strong baseline for evaluating robustness without requiring handcrafted adaptive attacks for each new defense.



## **22. RoTO: Robust Topology Obfuscation Against Tomography Inference Attacks**

cs.NI

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.12852v1) [paper-pdf](http://arxiv.org/pdf/2508.12852v1)

**Authors**: Chengze Du, Heng Xu, Zhiwei Yu, Ying Zhou, Zili Meng, Jialong Li

**Abstract**: Tomography inference attacks aim to reconstruct network topology by analyzing end-to-end probe delays. Existing defenses mitigate these attacks by manipulating probe delays to mislead inference, but rely on two strong assumptions: (i) probe packets can be perfectly detected and altered, and (ii) attackers use known, fixed inference algorithms. These assumptions often break in practice, leading to degraded defense performance under detection errors or adaptive adversaries. We present RoTO, a robust topology obfuscation scheme that eliminates both assumptions by modeling uncertainty in attacker-observed delays through a distributional formulation. RoTO casts the defense objective as a min-max optimization problem that maximizes expected topological distortion across this uncertainty set, without relying on perfect probe control or specific attacker models. To approximate attacker behavior, RoTO leverages graph neural networks for inference simulation and adversarial training. We also derive an upper bound on attacker success probability, and demonstrate that our approach enhances topology obfuscation performance through the optimization of this upper bound. Experimental results show that RoTO outperforms existing defense methods, achieving average improvements of 34% in structural similarity and 42.6% in link distance while maintaining strong robustness and concealment capabilities.



## **23. Deep Positive-Negative Prototypes for Adversarially Robust Discriminative Prototypical Learning**

cs.LG

This version substantially revises the manuscript, including a new  title and updated experimental results

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2504.03782v2) [paper-pdf](http://arxiv.org/pdf/2504.03782v2)

**Authors**: Ramin Zarei Sabzevar, Hamed Mohammadzadeh, Tahmineh Tavakoli, Ahad Harati

**Abstract**: Despite the advantages of discriminative prototype-based methods, their role in adversarial robustness remains underexplored. Meanwhile, current adversarial training methods predominantly focus on robustness against adversarial attacks without explicitly leveraging geometric structures in the latent space, usually resulting in reduced accuracy on the original clean data. We propose a novel framework named Adversarially trained Deep Positive-Negative Prototypes (Adv-DPNP), which integrates discriminative prototype-based learning with adversarial training. Adv-DPNP uses unified class prototypes that serve as both classifier weights and robust anchors in the latent space. Moreover, a novel dual-branch training mechanism maintains stable prototypes by updating them exclusively with clean data, while the feature extractor is trained on both clean and adversarial inputs to increase invariance to adversarial perturbations. In addition, we use a composite loss that combines positive-prototype alignment, negative-prototype repulsion, and consistency regularization to further enhance discrimination, adversarial robustness, and clean accuracy. Extensive experiments on standard benchmarks (CIFAR-10/100 and SVHN) confirm that Adv-DPNP improves clean accuracy over state-of-the-art defenses and baseline methods, while maintaining competitive or superior robustness under a suite of widely used attacks, including FGSM, PGD, C\&W, and AutoAttack. We also evaluate robustness to common corruptions on CIFAR-10-C, where Adv-DPNP achieves the highest average accuracy across severities and corruption types. Additionally, we provide an in-depth analysis of the discriminative quality of the learned feature representations, highlighting the effectiveness of Adv-DPNP in maintaining compactness and clear separation in the latent space.



## **24. Boosting Active Defense Persistence: A Two-Stage Defense Framework Combining Interruption and Poisoning Against Deepfake**

cs.CV

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.07795v2) [paper-pdf](http://arxiv.org/pdf/2508.07795v2)

**Authors**: Hongrui Zheng, Yuezun Li, Liejun Wang, Yunfeng Diao, Zhiqing Guo

**Abstract**: Active defense strategies have been developed to counter the threat of deepfake technology. However, a primary challenge is their lack of persistence, as their effectiveness is often short-lived. Attackers can bypass these defenses by simply collecting protected samples and retraining their models. This means that static defenses inevitably fail when attackers retrain their models, which severely limits practical use. We argue that an effective defense not only distorts forged content but also blocks the model's ability to adapt, which occurs when attackers retrain their models on protected images. To achieve this, we propose an innovative Two-Stage Defense Framework (TSDF). Benefiting from the intensity separation mechanism designed in this paper, the framework uses dual-function adversarial perturbations to perform two roles. First, it can directly distort the forged results. Second, it acts as a poisoning vehicle that disrupts the data preparation process essential for an attacker's retraining pipeline. By poisoning the data source, TSDF aims to prevent the attacker's model from adapting to the defensive perturbations, thus ensuring the defense remains effective long-term. Comprehensive experiments show that the performance of traditional interruption methods degrades sharply when it is subjected to adversarial retraining. However, our framework shows a strong dual defense capability, which can improve the persistence of active defense. Our code will be available at https://github.com/vpsg-research/TSDF.



## **25. Concealment of Intent: A Game-Theoretic Analysis**

cs.CL

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2505.20841v2) [paper-pdf](http://arxiv.org/pdf/2505.20841v2)

**Authors**: Xinbo Wu, Abhishek Umrawal, Lav R. Varshney

**Abstract**: As large language models (LLMs) grow more capable, concerns about their safe deployment have also grown. Although alignment mechanisms have been introduced to deter misuse, they remain vulnerable to carefully designed adversarial prompts. In this work, we present a scalable attack strategy: intent-hiding adversarial prompting, which conceals malicious intent through the composition of skills. We develop a game-theoretic framework to model the interaction between such attacks and defense systems that apply both prompt and response filtering. Our analysis identifies equilibrium points and reveals structural advantages for the attacker. To counter these threats, we propose and analyze a defense mechanism tailored to intent-hiding attacks. Empirically, we validate the attack's effectiveness on multiple real-world LLMs across a range of malicious behaviors, demonstrating clear advantages over existing adversarial prompting techniques.



## **26. Quantifying Loss Aversion in Cyber Adversaries via LLM Analysis**

cs.CR

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.13240v1) [paper-pdf](http://arxiv.org/pdf/2508.13240v1)

**Authors**: Soham Hans, Nikolos Gurney, Stacy Marsella, Sofia Hirschmann

**Abstract**: Understanding and quantifying human cognitive biases from empirical data has long posed a formidable challenge, particularly in cybersecurity, where defending against unknown adversaries is paramount. Traditional cyber defense strategies have largely focused on fortification, while some approaches attempt to anticipate attacker strategies by mapping them to cognitive vulnerabilities, yet they fall short in dynamically interpreting attacks in progress. In recognition of this gap, IARPA's ReSCIND program seeks to infer, defend against, and even exploit attacker cognitive traits. In this paper, we present a novel methodology that leverages large language models (LLMs) to extract quantifiable insights into the cognitive bias of loss aversion from hacker behavior. Our data are collected from an experiment in which hackers were recruited to attack a controlled demonstration network. We process the hacker generated notes using LLMs using it to segment the various actions and correlate the actions to predefined persistence mechanisms used by hackers. By correlating the implementation of these mechanisms with various operational triggers, our analysis provides new insights into how loss aversion manifests in hacker decision-making. The results demonstrate that LLMs can effectively dissect and interpret nuanced behavioral patterns, thereby offering a transformative approach to enhancing cyber defense strategies through real-time, behavior-based analysis.



## **27. Heuristic-Induced Multimodal Risk Distribution Jailbreak Attack for Multimodal Large Language Models**

cs.CR

ICCV 2025

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2412.05934v3) [paper-pdf](http://arxiv.org/pdf/2412.05934v3)

**Authors**: Ma Teng, Jia Xiaojun, Duan Ranjie, Li Xinfeng, Huang Yihao, Jia Xiaoshuang, Chu Zhixuan, Ren Wenqi

**Abstract**: With the rapid advancement of multimodal large language models (MLLMs), concerns regarding their security have increasingly captured the attention of both academia and industry. Although MLLMs are vulnerable to jailbreak attacks, designing effective jailbreak attacks poses unique challenges, especially given the highly constrained adversarial capabilities in real-world deployment scenarios. Previous works concentrate risks into a single modality, resulting in limited jailbreak performance. In this paper, we propose a heuristic-induced multimodal risk distribution jailbreak attack method, called HIMRD, which is black-box and consists of two elements: multimodal risk distribution strategy and heuristic-induced search strategy. The multimodal risk distribution strategy is used to distribute harmful semantics into multiple modalities to effectively circumvent the single-modality protection mechanisms of MLLMs. The heuristic-induced search strategy identifies two types of prompts: the understanding-enhancing prompt, which helps MLLMs reconstruct the malicious prompt, and the inducing prompt, which increases the likelihood of affirmative outputs over refusals, enabling a successful jailbreak attack. HIMRD achieves an average attack success rate (ASR) of 90% across seven open-source MLLMs and an average ASR of around 68% in three closed-source MLLMs. HIMRD reveals cross-modal security vulnerabilities in current MLLMs and underscores the imperative for developing defensive strategies to mitigate such emerging risks. Code is available at https://github.com/MaTengSYSU/HIMRD-jailbreak.



## **28. Reducing False Positives with Active Behavioral Analysis for Cloud Security**

cs.CR

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.12584v1) [paper-pdf](http://arxiv.org/pdf/2508.12584v1)

**Authors**: Dikshant, Verma

**Abstract**: Rule-based cloud security posture management (CSPM) solutions are known to produce a lot of false positives based on the limited contextual understanding and dependence on static heuristics testing. This paper introduces a validation-driven methodology that integrates active behavioral testing in cloud security posture management solution(s) to evaluate the exploitability of policy violations in real time. The proposed system employs lightweight and automated probes, built from open-source tools, validation scripts, and penetration testing test cases, to simulate adversarial attacks on misconfigured or vulnerable cloud assets without any impact to the cloud services or environment. For instance, cloud services may be flagged as publicly exposed and vulnerable despite being protected by access control layers, or secure policies, resulting in non-actionable alerts that consumes analysts time during manual validation. Through controlled experimentation in a reproducible AWS setup, we evaluated the reduction in false positive rates across various misconfiguration and vulnerable alerts. Our findings indicate an average reduction of 93\% in false positives. Furthermore, the framework demonstrates low latency performance. These results demonstrate a scalable method to improve detection accuracy and analyst productivity in large cloud environments. While our evaluation focuses on AWS, the architecture is modular and extensible to multi-cloud setups.



## **29. Adversarial Attacks on VQA-NLE: Exposing and Alleviating Inconsistencies in Visual Question Answering Explanations**

cs.CV

**SubmitDate**: 2025-08-17    [abs](http://arxiv.org/abs/2508.12430v1) [paper-pdf](http://arxiv.org/pdf/2508.12430v1)

**Authors**: Yahsin Yeh, Yilun Wu, Bokai Ruan, Honghan Shuai

**Abstract**: Natural language explanations in visual question answering (VQA-NLE) aim to make black-box models more transparent by elucidating their decision-making processes. However, we find that existing VQA-NLE systems can produce inconsistent explanations and reach conclusions without genuinely understanding the underlying context, exposing weaknesses in either their inference pipeline or explanation-generation mechanism. To highlight these vulnerabilities, we not only leverage an existing adversarial strategy to perturb questions but also propose a novel strategy that minimally alters images to induce contradictory or spurious outputs. We further introduce a mitigation method that leverages external knowledge to alleviate these inconsistencies, thereby bolstering model robustness. Extensive evaluations on two standard benchmarks and two widely used VQA-NLE models underscore the effectiveness of our attacks and the potential of knowledge-based defenses, ultimately revealing pressing security and reliability concerns in current VQA-NLE systems.



## **30. Cascading and Proxy Membership Inference Attacks**

cs.CR

Accepted by NDSS 2026

**SubmitDate**: 2025-08-17    [abs](http://arxiv.org/abs/2507.21412v2) [paper-pdf](http://arxiv.org/pdf/2507.21412v2)

**Authors**: Yuntao Du, Jiacheng Li, Yuetian Chen, Kaiyuan Zhang, Zhizhen Yuan, Hanshen Xiao, Bruno Ribeiro, Ninghui Li

**Abstract**: A Membership Inference Attack (MIA) assesses how much a trained machine learning model reveals about its training data by determining whether specific query instances were included in the dataset. We classify existing MIAs into adaptive or non-adaptive, depending on whether the adversary is allowed to train shadow models on membership queries. In the adaptive setting, where the adversary can train shadow models after accessing query instances, we highlight the importance of exploiting membership dependencies between instances and propose an attack-agnostic framework called Cascading Membership Inference Attack (CMIA), which incorporates membership dependencies via conditional shadow training to boost membership inference performance.   In the non-adaptive setting, where the adversary is restricted to training shadow models before obtaining membership queries, we introduce Proxy Membership Inference Attack (PMIA). PMIA employs a proxy selection strategy that identifies samples with similar behaviors to the query instance and uses their behaviors in shadow models to perform a membership posterior odds test for membership inference. We provide theoretical analyses for both attacks, and extensive experimental results demonstrate that CMIA and PMIA substantially outperform existing MIAs in both settings, particularly in the low false-positive regime, which is crucial for evaluating privacy risks.



## **31. ViT-EnsembleAttack: Augmenting Ensemble Models for Stronger Adversarial Transferability in Vision Transformers**

cs.CV

**SubmitDate**: 2025-08-17    [abs](http://arxiv.org/abs/2508.12384v1) [paper-pdf](http://arxiv.org/pdf/2508.12384v1)

**Authors**: Hanwen Cao, Haobo Lu, Xiaosen Wang, Kun He

**Abstract**: Ensemble-based attacks have been proven to be effective in enhancing adversarial transferability by aggregating the outputs of models with various architectures. However, existing research primarily focuses on refining ensemble weights or optimizing the ensemble path, overlooking the exploration of ensemble models to enhance the transferability of adversarial attacks. To address this gap, we propose applying adversarial augmentation to the surrogate models, aiming to boost overall generalization of ensemble models and reduce the risk of adversarial overfitting. Meanwhile, observing that ensemble Vision Transformers (ViTs) gain less attention, we propose ViT-EnsembleAttack based on the idea of model adversarial augmentation, the first ensemble-based attack method tailored for ViTs to the best of our knowledge. Our approach generates augmented models for each surrogate ViT using three strategies: Multi-head dropping, Attention score scaling, and MLP feature mixing, with the associated parameters optimized by Bayesian optimization. These adversarially augmented models are ensembled to generate adversarial examples. Furthermore, we introduce Automatic Reweighting and Step Size Enlargement modules to boost transferability. Extensive experiments demonstrate that ViT-EnsembleAttack significantly enhances the adversarial transferability of ensemble-based attacks on ViTs, outperforming existing methods by a substantial margin. Code is available at https://github.com/Trustworthy-AI-Group/TransferAttack.



## **32. Jamming Identification with Differential Transformer for Low-Altitude Wireless Networks**

eess.SP

**SubmitDate**: 2025-08-17    [abs](http://arxiv.org/abs/2508.12320v1) [paper-pdf](http://arxiv.org/pdf/2508.12320v1)

**Authors**: Pengyu Wang, Zhaocheng Wang, Tianqi Mao, Weijie Yuan, Haijun Zhang, George K. Karagiannidis

**Abstract**: Wireless jamming identification, which detects and classifies electromagnetic jamming from non-cooperative devices, is crucial for emerging low-altitude wireless networks consisting of many drone terminals that are highly susceptible to electromagnetic jamming. However, jamming identification schemes adopting deep learning (DL) are vulnerable to attacks involving carefully crafted adversarial samples, resulting in inevitable robustness degradation. To address this issue, we propose a differential transformer framework for wireless jamming identification. Firstly, we introduce a differential transformer network in order to distinguish jamming signals, which overcomes the attention noise when compared with its traditional counterpart by performing self-attention operations in a differential manner. Secondly, we propose a randomized masking training strategy to improve network robustness, which leverages the patch partitioning mechanism inherent to transformer architectures in order to create parallel feature extraction branches. Each branch operates on a distinct, randomly masked subset of patches, which fundamentally constrains the propagation of adversarial perturbations across the network. Additionally, the ensemble effect generated by fusing predictions from these diverse branches demonstrates superior resilience against adversarial attacks. Finally, we introduce a novel consistent training framework that significantly enhances adversarial robustness through dualbranch regularization. Simulation results demonstrate that our proposed methodology is superior to existing methods in boosting robustness to adversarial samples.



## **33. Adjustable AprilTags For Identity Secured Tasks**

cs.CR

**SubmitDate**: 2025-08-17    [abs](http://arxiv.org/abs/2508.12304v1) [paper-pdf](http://arxiv.org/pdf/2508.12304v1)

**Authors**: Hao Li

**Abstract**: Special tags such as AprilTags that facilitate image processing and pattern recognition are useful in practical applications. In close and private environments, identity security is unlikely to be an issue because all involved AprilTags can be completely regulated. However, in open and public environments, identity security is no longer an issue that can be neglected. To handle potential harm caused by adversarial attacks, this note advocates utilization of adjustable AprilTags instead of fixed ones.



## **34. ForensicsSAM: Toward Robust and Unified Image Forgery Detection and Localization Resisting to Adversarial Attack**

cs.CV

**SubmitDate**: 2025-08-17    [abs](http://arxiv.org/abs/2508.07402v2) [paper-pdf](http://arxiv.org/pdf/2508.07402v2)

**Authors**: Rongxuan Peng, Shunquan Tan, Chenqi Kong, Anwei Luo, Alex C. Kot, Jiwu Huang

**Abstract**: Parameter-efficient fine-tuning (PEFT) has emerged as a popular strategy for adapting large vision foundation models, such as the Segment Anything Model (SAM) and LLaVA, to downstream tasks like image forgery detection and localization (IFDL). However, existing PEFT-based approaches overlook their vulnerability to adversarial attacks. In this paper, we show that highly transferable adversarial images can be crafted solely via the upstream model, without accessing the downstream model or training data, significantly degrading the IFDL performance. To address this, we propose ForensicsSAM, a unified IFDL framework with built-in adversarial robustness. Our design is guided by three key ideas: (1) To compensate for the lack of forgery-relevant knowledge in the frozen image encoder, we inject forgery experts into each transformer block to enhance its ability to capture forgery artifacts. These forgery experts are always activated and shared across any input images. (2) To detect adversarial images, we design an light-weight adversary detector that learns to capture structured, task-specific artifact in RGB domain, enabling reliable discrimination across various attack methods. (3) To resist adversarial attacks, we inject adversary experts into the global attention layers and MLP modules to progressively correct feature shifts induced by adversarial noise. These adversary experts are adaptively activated by the adversary detector, thereby avoiding unnecessary interference with clean images. Extensive experiments across multiple benchmarks demonstrate that ForensicsSAM achieves superior resistance to various adversarial attack methods, while also delivering state-of-the-art performance in image-level forgery detection and pixel-level forgery localization. The resource is available at https://github.com/siriusPRX/ForensicsSAM.



## **35. TriQDef: Disrupting Semantic and Gradient Alignment to Prevent Adversarial Patch Transferability in Quantized Neural Networks**

cs.CV

**SubmitDate**: 2025-08-16    [abs](http://arxiv.org/abs/2508.12132v1) [paper-pdf](http://arxiv.org/pdf/2508.12132v1)

**Authors**: Amira Guesmi, Bassem Ouni, Muhammad Shafique

**Abstract**: Quantized Neural Networks (QNNs) are increasingly deployed in edge and resource-constrained environments due to their efficiency in computation and memory usage. While shown to distort the gradient landscape and weaken conventional pixel-level attacks, it provides limited robustness against patch-based adversarial attacks-localized, high-saliency perturbations that remain surprisingly transferable across bit-widths. Existing defenses either overfit to fixed quantization settings or fail to address this cross-bit generalization vulnerability. We introduce \textbf{TriQDef}, a tri-level quantization-aware defense framework designed to disrupt the transferability of patch-based adversarial attacks across QNNs. TriQDef consists of: (1) a Feature Disalignment Penalty (FDP) that enforces semantic inconsistency by penalizing perceptual similarity in intermediate representations; (2) a Gradient Perceptual Dissonance Penalty (GPDP) that explicitly misaligns input gradients across bit-widths by minimizing structural and directional agreement via Edge IoU and HOG Cosine metrics; and (3) a Joint Quantization-Aware Training Protocol that unifies these penalties within a shared-weight training scheme across multiple quantization levels. Extensive experiments on CIFAR-10 and ImageNet demonstrate that TriQDef reduces Attack Success Rates (ASR) by over 40\% on unseen patch and quantization combinations, while preserving high clean accuracy. Our findings underscore the importance of disrupting both semantic and perceptual gradient alignment to mitigate patch transferability in QNNs.



## **36. Mitigating Jailbreaks with Intent-Aware LLMs**

cs.CR

**SubmitDate**: 2025-08-16    [abs](http://arxiv.org/abs/2508.12072v1) [paper-pdf](http://arxiv.org/pdf/2508.12072v1)

**Authors**: Wei Jie Yeo, Ranjan Satapathy, Erik Cambria

**Abstract**: Despite extensive safety-tuning, large language models (LLMs) remain vulnerable to jailbreak attacks via adversarially crafted instructions, reflecting a persistent trade-off between safety and task performance. In this work, we propose Intent-FT, a simple and lightweight fine-tuning approach that explicitly trains LLMs to infer the underlying intent of an instruction before responding. By fine-tuning on a targeted set of adversarial instructions, Intent-FT enables LLMs to generalize intent deduction to unseen attacks, thereby substantially improving their robustness. We comprehensively evaluate both parametric and non-parametric attacks across open-source and proprietary models, considering harmfulness from attacks, utility, over-refusal, and impact against white-box threats. Empirically, Intent-FT consistently mitigates all evaluated attack categories, with no single attack exceeding a 50\% success rate -- whereas existing defenses remain only partially effective. Importantly, our method preserves the model's general capabilities and reduces excessive refusals on benign instructions containing superficially harmful keywords. Furthermore, models trained with Intent-FT accurately identify hidden harmful intent in adversarial attacks, and these learned intentions can be effectively transferred to enhance vanilla model defenses.



## **37. Interpretable and Robust AI in EEG Systems: A Survey**

eess.SP

**SubmitDate**: 2025-08-16    [abs](http://arxiv.org/abs/2304.10755v4) [paper-pdf](http://arxiv.org/pdf/2304.10755v4)

**Authors**: Xinliang Zhou, Chenyu Liu, Jinan Zhou, Zhongruo Wang, Liming Zhai, Ziyu Jia, Cuntai Guan, Yang Liu

**Abstract**: The close coupling of artificial intelligence (AI) and electroencephalography (EEG) has substantially advanced human-computer interaction (HCI) technologies in the AI era. Different from traditional EEG systems, the interpretability and robustness of AI-based EEG systems are becoming particularly crucial. The interpretability clarifies the inner working mechanisms of AI models and thus can gain the trust of users. The robustness reflects the AI's reliability against attacks and perturbations, which is essential for sensitive and fragile EEG signals. Thus the interpretability and robustness of AI in EEG systems have attracted increasing attention, and their research has achieved great progress recently. However, there is still no survey covering recent advances in this field. In this paper, we present the first comprehensive survey and summarize the interpretable and robust AI techniques for EEG systems. Specifically, we first propose a taxonomy of interpretability by characterizing it into three types: backpropagation, perturbation, and inherently interpretable methods. Then we classify the robustness mechanisms into four classes: noise and artifacts, human variability, data acquisition instability, and adversarial attacks. Finally, we identify several critical and unresolved challenges for interpretable and robust AI in EEG systems and further discuss their future directions.



## **38. Deciphering the Interplay between Attack and Protection Complexity in Privacy-Preserving Federated Learning**

cs.CR

**SubmitDate**: 2025-08-16    [abs](http://arxiv.org/abs/2508.11907v1) [paper-pdf](http://arxiv.org/pdf/2508.11907v1)

**Authors**: Xiaojin Zhang, Mingcong Xu, Yiming Li, Wei Chen, Qiang Yang

**Abstract**: Federated learning (FL) offers a promising paradigm for collaborative model training while preserving data privacy. However, its susceptibility to gradient inversion attacks poses a significant challenge, necessitating robust privacy protection mechanisms. This paper introduces a novel theoretical framework to decipher the intricate interplay between attack and protection complexities in privacy-preserving FL. We formally define "Attack Complexity" as the minimum computational and data resources an adversary requires to reconstruct private data below a given error threshold, and "Protection Complexity" as the expected distortion introduced by privacy mechanisms. Leveraging Maximum Bayesian Privacy (MBP), we derive tight theoretical bounds for protection complexity, demonstrating its scaling with model dimensionality and privacy budget. Furthermore, we establish comprehensive bounds for attack complexity, revealing its dependence on privacy leakage, gradient distortion, model dimension, and the chosen privacy level. Our findings quantitatively illuminate the fundamental trade-offs between privacy guarantees, system utility, and the effort required for both attacking and defending. This framework provides critical insights for designing more secure and efficient federated learning systems.



## **39. ComplicitSplat: Downstream Models are Vulnerable to Blackbox Attacks by 3D Gaussian Splat Camouflages**

cs.CV

7 pages, 6 figures

**SubmitDate**: 2025-08-16    [abs](http://arxiv.org/abs/2508.11854v1) [paper-pdf](http://arxiv.org/pdf/2508.11854v1)

**Authors**: Matthew Hull, Haoyang Yang, Pratham Mehta, Mansi Phute, Aeree Cho, Haorang Wang, Matthew Lau, Wenke Lee, Wilian Lunardi, Martin Andreoni, Polo Chau

**Abstract**: As 3D Gaussian Splatting (3DGS) gains rapid adoption in safety-critical tasks for efficient novel-view synthesis from static images, how might an adversary tamper images to cause harm? We introduce ComplicitSplat, the first attack that exploits standard 3DGS shading methods to create viewpoint-specific camouflage - colors and textures that change with viewing angle - to embed adversarial content in scene objects that are visible only from specific viewpoints and without requiring access to model architecture or weights. Our extensive experiments show that ComplicitSplat generalizes to successfully attack a variety of popular detector - both single-stage, multi-stage, and transformer-based models on both real-world capture of physical objects and synthetic scenes. To our knowledge, this is the first black-box attack on downstream object detectors using 3DGS, exposing a novel safety risk for applications like autonomous navigation and other mission-critical robotic systems.



## **40. Adversarial Robustness in Distributed Quantum Machine Learning**

quant-ph

This is a preprint of a book chapter that is planned to be published  in "Quantum Robustness in Artificial Intelligence" by Springer Nature

**SubmitDate**: 2025-08-16    [abs](http://arxiv.org/abs/2508.11848v1) [paper-pdf](http://arxiv.org/pdf/2508.11848v1)

**Authors**: Pouya Kananian, Hans-Arno Jacobsen

**Abstract**: Studying adversarial robustness of quantum machine learning (QML) models is essential in order to understand their potential advantages over classical models and build trustworthy systems. Distributing QML models allows leveraging multiple quantum processors to overcome the limitations of individual devices and build scalable systems. However, this distribution can affect their adversarial robustness, potentially making them more vulnerable to new attacks. Key paradigms in distributed QML include federated learning, which, similar to classical models, involves training a shared model on local data and sending only the model updates, as well as circuit distribution methods inherent to quantum computing, such as circuit cutting and teleportation-based techniques. These quantum-specific methods enable the distributed execution of quantum circuits across multiple devices. This work reviews the differences between these distribution methods, summarizes existing approaches on the adversarial robustness of QML models when distributed using each paradigm, and discusses open questions in this area.



## **41. Assessing User Privacy Leakage in Synthetic Packet Traces: An Attack-Grounded Approach**

cs.CR

**SubmitDate**: 2025-08-15    [abs](http://arxiv.org/abs/2508.11742v1) [paper-pdf](http://arxiv.org/pdf/2508.11742v1)

**Authors**: Minhao Jin, Hongyu He, Maria Apostolaki

**Abstract**: Current synthetic traffic generators (SynNetGens) promise privacy but lack comprehensive guarantees or empirical validation, even as their fidelity steadily improves. We introduce the first attack-grounded benchmark for assessing the privacy of SynNetGens directly from the traffic they produce. We frame privacy as membership inference at the traffic-source level--a realistic and actionable threat for data holders. To this end, we present TraceBleed, the first attack that exploits behavioral fingerprints across flows using contrastive learning and temporal chunking, outperforming prior membership inference baselines by 172%. Our large-scale study across GAN-, diffusion-, and GPT-based SynNetGens uncovers critical insights: (i) SynNetGens leak user-level information; (ii) differential privacy either fails to stop these attacks or severely degrades fidelity; and (iii) sharing more synthetic data amplifies leakage by 59% on average. Finally, we introduce TracePatch, the first SynNetGen-agnostic defense that combines adversarial ML with SMT constraints to mitigate leakage while preserving fidelity.



## **42. Random Walk Learning and the Pac-Man Attack**

stat.ML

The updated manuscript represents an incomplete version of the work.  A substantially updated version will be prepared before further dissemination

**SubmitDate**: 2025-08-15    [abs](http://arxiv.org/abs/2508.05663v2) [paper-pdf](http://arxiv.org/pdf/2508.05663v2)

**Authors**: Xingran Chen, Parimal Parag, Rohit Bhagat, Zonghong Liu, Salim El Rouayheb

**Abstract**: Random walk (RW)-based algorithms have long been popular in distributed systems due to low overheads and scalability, with recent growing applications in decentralized learning. However, their reliance on local interactions makes them inherently vulnerable to malicious behavior. In this work, we investigate an adversarial threat that we term the ``Pac-Man'' attack, in which a malicious node probabilistically terminates any RW that visits it. This stealthy behavior gradually eliminates active RWs from the network, effectively halting the learning process without triggering failure alarms. To counter this threat, we propose the Average Crossing (AC) algorithm--a fully decentralized mechanism for duplicating RWs to prevent RW extinction in the presence of Pac-Man. Our theoretical analysis establishes that (i) the RW population remains almost surely bounded under AC and (ii) RW-based stochastic gradient descent remains convergent under AC, even in the presence of Pac-Man, with a quantifiable deviation from the true optimum. Our extensive empirical results on both synthetic and real-world datasets corroborate our theoretical findings. Furthermore, they uncover a phase transition in the extinction probability as a function of the duplication threshold. We offer theoretical insights by analyzing a simplified variant of the AC, which sheds light on the observed phase transition.



## **43. Robust Convolution Neural ODEs via Contractivity-promoting regularization**

cs.LG

Accepted in IEEE CDC2025, Rio de Janeiro, Brazil

**SubmitDate**: 2025-08-15    [abs](http://arxiv.org/abs/2508.11432v1) [paper-pdf](http://arxiv.org/pdf/2508.11432v1)

**Authors**: Muhammad Zakwan, Liang Xu, Giancarlo Ferrari-Trecate

**Abstract**: Neural networks can be fragile to input noise and adversarial attacks.   In this work, we consider Convolutional Neural Ordinary Differential Equations (NODEs), a family of continuous-depth neural networks represented by dynamical systems, and propose to use contraction theory to improve their robustness.   For a contractive dynamical system two trajectories starting from different initial conditions converge to each other exponentially fast.   Contractive Convolutional NODEs can enjoy increased robustness as slight perturbations of the features do not cause a significant change in the output.   Contractivity can be induced during training by using a regularization term involving the Jacobian of the system dynamics.   To reduce the computational burden, we show that it can also be promoted using carefully selected weight regularization terms for a class of NODEs with slope-restricted activation functions.   The performance of the proposed regularizers is illustrated through benchmark image classification tasks on MNIST and FashionMNIST datasets, where images are corrupted by different kinds of noise and attacks.



## **44. Chasing Moving Targets with Online Self-Play Reinforcement Learning for Safer Language Models**

cs.LG

**SubmitDate**: 2025-08-15    [abs](http://arxiv.org/abs/2506.07468v2) [paper-pdf](http://arxiv.org/pdf/2506.07468v2)

**Authors**: Mickel Liu, Liwei Jiang, Yancheng Liang, Simon Shaolei Du, Yejin Choi, Tim Althoff, Natasha Jaques

**Abstract**: Conventional language model (LM) safety alignment relies on a reactive, disjoint procedure: attackers exploit a static model, followed by defensive fine-tuning to patch exposed vulnerabilities. This sequential approach creates a mismatch -- attackers overfit to obsolete defenses, while defenders perpetually lag behind emerging threats. To address this, we propose Self-RedTeam, an online self-play reinforcement learning algorithm where an attacker and defender agent co-evolve through continuous interaction. We cast safety alignment as a two-player zero-sum game, where a single model alternates between attacker and defender roles -- generating adversarial prompts and safeguarding against them -- while a reward LM adjudicates outcomes. This enables dynamic co-adaptation. Grounded in the game-theoretic framework of zero-sum games, we establish a theoretical safety guarantee which motivates the design of our method: if self-play converges to a Nash Equilibrium, the defender will reliably produce safe responses to any adversarial input. Empirically, Self-RedTeam uncovers more diverse attacks (+21.8% SBERT) compared to attackers trained against static defenders and achieves higher robustness on safety benchmarks (e.g., +65.5% on WildJailBreak) than defenders trained against static attackers. We further propose hidden Chain-of-Thought, allowing agents to plan privately, which boosts adversarial diversity and reduces over-refusals. Our results motivate a shift from reactive patching to proactive co-evolution in LM safety training, enabling scalable, autonomous, and robust self-improvement of LMs via multi-agent reinforcement learning (MARL).



## **45. Semantically Guided Adversarial Testing of Vision Models Using Language Models**

cs.CV

12 pages, 4 figures, 3 tables. Submitted for peer review

**SubmitDate**: 2025-08-15    [abs](http://arxiv.org/abs/2508.11341v1) [paper-pdf](http://arxiv.org/pdf/2508.11341v1)

**Authors**: Katarzyna Filus, Jorge M. Cruz-Duarte

**Abstract**: In targeted adversarial attacks on vision models, the selection of the target label is a critical yet often overlooked determinant of attack success. This target label corresponds to the class that the attacker aims to force the model to predict. Now, existing strategies typically rely on randomness, model predictions, or static semantic resources, limiting interpretability, reproducibility, or flexibility. This paper then proposes a semantics-guided framework for adversarial target selection using the cross-modal knowledge transfer from pretrained language and vision-language models. We evaluate several state-of-the-art models (BERT, TinyLLAMA, and CLIP) as similarity sources to select the most and least semantically related labels with respect to the ground truth, forming best- and worst-case adversarial scenarios. Our experiments on three vision models and five attack methods reveal that these models consistently render practical adversarial targets and surpass static lexical databases, such as WordNet, particularly for distant class relationships. We also observe that static testing of target labels offers a preliminary assessment of the effectiveness of similarity sources, \textit{a priori} testing. Our results corroborate the suitability of pretrained models for constructing interpretable, standardized, and scalable adversarial benchmarks across architectures and datasets.



## **46. MUNBa: Machine Unlearning via Nash Bargaining**

cs.CV

**SubmitDate**: 2025-08-15    [abs](http://arxiv.org/abs/2411.15537v3) [paper-pdf](http://arxiv.org/pdf/2411.15537v3)

**Authors**: Jing Wu, Mehrtash Harandi

**Abstract**: Machine Unlearning (MU) aims to selectively erase harmful behaviors from models while retaining the overall utility of the model. As a multi-task learning problem, MU involves balancing objectives related to forgetting specific concepts/data and preserving general performance. A naive integration of these forgetting and preserving objectives can lead to gradient conflicts and dominance, impeding MU algorithms from reaching optimal solutions. To address the gradient conflict and dominance issue, we reformulate MU as a two-player cooperative game, where the two players, namely, the forgetting player and the preservation player, contribute via their gradient proposals to maximize their overall gain and balance their contributions. To this end, inspired by the Nash bargaining theory, we derive a closed-form solution to guide the model toward the Pareto stationary point. Our formulation of MU guarantees an equilibrium solution, where any deviation from the final state would lead to a reduction in the overall objectives for both players, ensuring optimality in each objective. We evaluate our algorithm's effectiveness on a diverse set of tasks across image classification and image generation. Extensive experiments with ResNet, vision-language model CLIP, and text-to-image diffusion models demonstrate that our method outperforms state-of-the-art MU algorithms, achieving a better trade-off between forgetting and preserving. Our results also highlight improvements in forgetting precision, preservation of generalization, and robustness against adversarial attacks.



## **47. Towards Physically Realizable Adversarial Attacks in Embodied Vision Navigation**

cs.CV

7 pages, 7 figures, Accept by IEEE/RSJ International Conference on  Intelligent Robots and Systems (IROS) 2025

**SubmitDate**: 2025-08-15    [abs](http://arxiv.org/abs/2409.10071v5) [paper-pdf](http://arxiv.org/pdf/2409.10071v5)

**Authors**: Meng Chen, Jiawei Tu, Chao Qi, Yonghao Dang, Feng Zhou, Wei Wei, Jianqin Yin

**Abstract**: The significant advancements in embodied vision navigation have raised concerns about its susceptibility to adversarial attacks exploiting deep neural networks. Investigating the adversarial robustness of embodied vision navigation is crucial, especially given the threat of 3D physical attacks that could pose risks to human safety. However, existing attack methods for embodied vision navigation often lack physical feasibility due to challenges in transferring digital perturbations into the physical world. Moreover, current physical attacks for object detection struggle to achieve both multi-view effectiveness and visual naturalness in navigation scenarios. To address this, we propose a practical attack method for embodied navigation by attaching adversarial patches to objects, where both opacity and textures are learnable. Specifically, to ensure effectiveness across varying viewpoints, we employ a multi-view optimization strategy based on object-aware sampling, which optimizes the patch's texture based on feedback from the vision-based perception model used in navigation. To make the patch inconspicuous to human observers, we introduce a two-stage opacity optimization mechanism, in which opacity is fine-tuned after texture optimization. Experimental results demonstrate that our adversarial patches decrease the navigation success rate by an average of 22.39%, outperforming previous methods in practicality, effectiveness, and naturalness. Code is available at: https://github.com/chen37058/Physical-Attacks-in-Embodied-Nav



## **48. SHLIME: Foiling adversarial attacks fooling SHAP and LIME**

cs.LG

7 pages, 7 figures

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.11053v1) [paper-pdf](http://arxiv.org/pdf/2508.11053v1)

**Authors**: Sam Chauhan, Estelle Duguet, Karthik Ramakrishnan, Hugh Van Deventer, Jack Kruger, Ranjan Subbaraman

**Abstract**: Post hoc explanation methods, such as LIME and SHAP, provide interpretable insights into black-box classifiers and are increasingly used to assess model biases and generalizability. However, these methods are vulnerable to adversarial manipulation, potentially concealing harmful biases. Building on the work of Slack et al. (2020), we investigate the susceptibility of LIME and SHAP to biased models and evaluate strategies for improving robustness. We first replicate the original COMPAS experiment to validate prior findings and establish a baseline. We then introduce a modular testing framework enabling systematic evaluation of augmented and ensemble explanation approaches across classifiers of varying performance. Using this framework, we assess multiple LIME/SHAP ensemble configurations on out-of-distribution models, comparing their resistance to bias concealment against the original methods. Our results identify configurations that substantially improve bias detection, highlighting their potential for enhancing transparency in the deployment of high-stakes machine learning systems.



## **49. Byzantine-Resilient Decentralized Online Resource Allocation**

math.OC

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.08658v2) [paper-pdf](http://arxiv.org/pdf/2508.08658v2)

**Authors**: Runhua Wang, Qing Ling, Hoi-To Wai, Zhi Tian

**Abstract**: In this paper, we investigate the problem of decentralized online resource allocation in the presence of Byzantine attacks. In this problem setting, some agents may be compromised due to external manipulations or internal failures, causing them to behave maliciously and disrupt the resource allocation process by sending incorrect messages to their neighbors. Given the non-consensual nature of the resource allocation problem, we formulate it under a primal-dual optimization framework, where the dual variables are aggregated among the agents, enabling the incorporation of robust aggregation mechanisms to mitigate Byzantine attacks. By leveraging the classical Byzantine attack model, we propose a class of Byzantine-resilient decentralized online resource allocation algorithms that judiciously integrate the adaptive robust clipping technique with the existing robust aggregation rules to filter out adversarial messages. We establish theoretical guarantees, showing that the proposed algorithms achieve tight linear dynamic regret and accumulative constraint violation bounds, where the constants depend on the properties of robust aggregation rules. Numerical experiments on decentralized online economic dispatch validate the effectiveness of our approach and support our theoretical results.



## **50. JMA: a General Algorithm to Craft Nearly Optimal Targeted Adversarial Example**

cs.LG

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2401.01199v2) [paper-pdf](http://arxiv.org/pdf/2401.01199v2)

**Authors**: Benedetta Tondi, Wei Guo, Niccolò Pancino, Mauro Barni

**Abstract**: Most of the approaches proposed so far to craft targeted adversarial examples against Deep Learning classifiers are highly suboptimal and typically rely on increasing the likelihood of the target class, thus implicitly focusing on one-hot encoding settings. In this paper, a more general, theoretically sound, targeted attack is proposed, which resorts to the minimization of a Jacobian-induced Mahalanobis distance term, taking into account the effort (in the input space) required to move the latent space representation of the input sample in a given direction. The minimization is solved by exploiting the Wolfe duality theorem, reducing the problem to the solution of a Non-Negative Least Square (NNLS) problem. The proposed algorithm (referred to as JMA) provides an optimal solution to a linearised version of the adversarial example problem originally introduced by Szegedy et al. The results of the experiments confirm the generality of the proposed attack which is proven to be effective under a wide variety of output encoding schemes. Noticeably, JMA is also effective in a multi-label classification scenario, being capable to induce a targeted modification of up to half the labels in complex multi-label classification scenarios, a capability that is out of reach of all the attacks proposed so far. As a further advantage, JMA requires very few iterations, thus resulting more efficient than existing methods.



