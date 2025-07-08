# Latest Adversarial Attack Papers
**update at 2025-07-08 10:04:38**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. BackFed: An Efficient & Standardized Benchmark Suite for Backdoor Attacks in Federated Learning**

cs.CR

Under review at NeurIPS'25

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.04903v1) [paper-pdf](http://arxiv.org/pdf/2507.04903v1)

**Authors**: Thinh Dao, Dung Thuy Nguyen, Khoa D Doan, Kok-Seng Wong

**Abstract**: Federated Learning (FL) systems are vulnerable to backdoor attacks, where adversaries train their local models on poisoned data and submit poisoned model updates to compromise the global model. Despite numerous proposed attacks and defenses, divergent experimental settings, implementation errors, and unrealistic assumptions hinder fair comparisons and valid conclusions about their effectiveness in real-world scenarios. To address this, we introduce BackFed - a comprehensive benchmark suite designed to standardize, streamline, and reliably evaluate backdoor attacks and defenses in FL, with a focus on practical constraints. Our benchmark offers key advantages through its multi-processing implementation that significantly accelerates experimentation and the modular design that enables seamless integration of new methods via well-defined APIs. With a standardized evaluation pipeline, we envision BackFed as a plug-and-play environment for researchers to comprehensively and reliably evaluate new attacks and defenses. Using BackFed, we conduct large-scale studies of representative backdoor attacks and defenses across both Computer Vision and Natural Language Processing tasks with diverse model architectures and experimental settings. Our experiments critically assess the performance of proposed attacks and defenses, revealing unknown limitations and modes of failures under practical conditions. These empirical insights provide valuable guidance for the development of new methods and for enhancing the security of FL systems. Our framework is openly available at https://github.com/thinh-dao/BackFed.



## **2. Beyond Training-time Poisoning: Component-level and Post-training Backdoors in Deep Reinforcement Learning**

cs.LG

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.04883v1) [paper-pdf](http://arxiv.org/pdf/2507.04883v1)

**Authors**: Sanyam Vyas, Alberto Caron, Chris Hicks, Pete Burnap, Vasilios Mavroudis

**Abstract**: Deep Reinforcement Learning (DRL) systems are increasingly used in safety-critical applications, yet their security remains severely underexplored. This work investigates backdoor attacks, which implant hidden triggers that cause malicious actions only when specific inputs appear in the observation space. Existing DRL backdoor research focuses solely on training-time attacks requiring unrealistic access to the training pipeline. In contrast, we reveal critical vulnerabilities across the DRL supply chain where backdoors can be embedded with significantly reduced adversarial privileges. We introduce two novel attacks: (1) TrojanentRL, which exploits component-level flaws to implant a persistent backdoor that survives full model retraining; and (2) InfrectroRL, a post-training backdoor attack which requires no access to training, validation, nor test data. Empirical and analytical evaluations across six Atari environments show our attacks rival state-of-the-art training-time backdoor attacks while operating under much stricter adversarial constraints. We also demonstrate that InfrectroRL further evades two leading DRL backdoor defenses. These findings challenge the current research focus and highlight the urgent need for robust defenses.



## **3. Diffusion-based Adversarial Identity Manipulation for Facial Privacy Protection**

cs.CV

Accepted by ACM MM 2025

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2504.21646v2) [paper-pdf](http://arxiv.org/pdf/2504.21646v2)

**Authors**: Liqin Wang, Qianyue Hu, Wei Lu, Xiangyang Luo

**Abstract**: The success of face recognition (FR) systems has led to serious privacy concerns due to potential unauthorized surveillance and user tracking on social networks. Existing methods for enhancing privacy fail to generate natural face images that can protect facial privacy. In this paper, we propose diffusion-based adversarial identity manipulation (DiffAIM) to generate natural and highly transferable adversarial faces against malicious FR systems. To be specific, we manipulate facial identity within the low-dimensional latent space of a diffusion model. This involves iteratively injecting gradient-based adversarial identity guidance during the reverse diffusion process, progressively steering the generation toward the desired adversarial faces. The guidance is optimized for identity convergence towards a target while promoting semantic divergence from the source, facilitating effective impersonation while maintaining visual naturalness. We further incorporate structure-preserving regularization to preserve facial structure consistency during manipulation. Extensive experiments on both face verification and identification tasks demonstrate that compared with the state-of-the-art, DiffAIM achieves stronger black-box attack transferability while maintaining superior visual quality. We also demonstrate the effectiveness of the proposed approach for commercial FR APIs, including Face++ and Aliyun.



## **4. Robustifying 3D Perception through Least-Squares Multi-Agent Graphs Object Tracking**

cs.CV

6 pages, 3 figures, 4 tables

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.04762v1) [paper-pdf](http://arxiv.org/pdf/2507.04762v1)

**Authors**: Maria Damanaki, Ioulia Kapsali, Nikos Piperigkos, Alexandros Gkillas, Aris S. Lalos

**Abstract**: The critical perception capabilities of EdgeAI systems, such as autonomous vehicles, are required to be resilient against adversarial threats, by enabling accurate identification and localization of multiple objects in the scene over time, mitigating their impact. Single-agent tracking offers resilience to adversarial attacks but lacks situational awareness, underscoring the need for multi-agent cooperation to enhance context understanding and robustness. This paper proposes a novel mitigation framework on 3D LiDAR scene against adversarial noise by tracking objects based on least-squares graph on multi-agent adversarial bounding boxes. Specifically, we employ the least-squares graph tool to reduce the induced positional error of each detection's centroid utilizing overlapped bounding boxes on a fully connected graph via differential coordinates and anchor points. Hence, the multi-vehicle detections are fused and refined mitigating the adversarial impact, and associated with existing tracks in two stages performing tracking to further suppress the adversarial threat. An extensive evaluation study on the real-world V2V4Real dataset demonstrates that the proposed method significantly outperforms both state-of-the-art single and multi-agent tracking frameworks by up to 23.3% under challenging adversarial conditions, operating as a resilient approach without relying on additional defense mechanisms.



## **5. Trojan Horse Prompting: Jailbreaking Conversational Multimodal Models by Forging Assistant Message**

cs.AI

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2507.04673v1) [paper-pdf](http://arxiv.org/pdf/2507.04673v1)

**Authors**: Wei Duan, Li Qian

**Abstract**: The rise of conversational interfaces has greatly enhanced LLM usability by leveraging dialogue history for sophisticated reasoning. However, this reliance introduces an unexplored attack surface. This paper introduces Trojan Horse Prompting, a novel jailbreak technique. Adversaries bypass safety mechanisms by forging the model's own past utterances within the conversational history provided to its API. A malicious payload is injected into a model-attributed message, followed by a benign user prompt to trigger harmful content generation. This vulnerability stems from Asymmetric Safety Alignment: models are extensively trained to refuse harmful user requests but lack comparable skepticism towards their own purported conversational history. This implicit trust in its "past" creates a high-impact vulnerability. Experimental validation on Google's Gemini-2.0-flash-preview-image-generation shows Trojan Horse Prompting achieves a significantly higher Attack Success Rate (ASR) than established user-turn jailbreaking methods. These findings reveal a fundamental flaw in modern conversational AI security, necessitating a paradigm shift from input-level filtering to robust, protocol-level validation of conversational context integrity.



## **6. Smart Grid: Cyber Attacks, Critical Defense Approaches, and Digital Twin**

cs.CR

**SubmitDate**: 2025-07-07    [abs](http://arxiv.org/abs/2205.11783v2) [paper-pdf](http://arxiv.org/pdf/2205.11783v2)

**Authors**: Tianming Zheng, Ping Yi, Yue Wu

**Abstract**: As a national critical infrastructure, the smart grid has attracted widespread attention for its cybersecurity issues. The development towards an intelligent, digital, and Internet-connected smart grid has attracted external adversaries for malicious activities. It is necessary to enhance its cybersecurity by both improving the existing defense approaches and introducing novel developed technologies to the smart grid context. As an emerging technology, digital twin (DT) is considered as an enabler for enhanced security. However, the practical implementation is quite challenging. This is due to the knowledge barriers among smart grid designers, security experts, and DT developers. Each single domain is a complicated system covering various components and technologies. As a result, works are needed to sort out relevant contents so that DT can be better embedded in the security architecture design of smart grid.   In order to meet this demand, our paper covers the above three domains, i.e., smart grid, cybersecurity, and DT. Specifically, the paper i) introduces the background of the smart grid; ii) reviews external cyber attacks from attack incidents and attack methods; iii) introduces critical defense approaches in industrial cyber systems, which include device identification, vulnerability discovery, intrusion detection systems (IDSs), honeypots, attribution, and threat intelligence (TI); iv) reviews the relevant content of DT, including its basic concepts, applications in the smart grid, and how DT enhances the security. In the end, the paper puts forward our security considerations on the future development of DT-based smart grid. The survey is expected to help developers break knowledge barriers among smart grid, cybersecurity, and DT, and provide guidelines for future security design of DT-based smart grid.



## **7. Tail-aware Adversarial Attacks: A Distributional Approach to Efficient LLM Jailbreaking**

cs.LG

**SubmitDate**: 2025-07-06    [abs](http://arxiv.org/abs/2507.04446v1) [paper-pdf](http://arxiv.org/pdf/2507.04446v1)

**Authors**: Tim Beyer, Yan Scholten, Stephan Günnemann, Leo Schwinn

**Abstract**: To guarantee safe and robust deployment of large language models (LLMs) at scale, it is critical to accurately assess their adversarial robustness. Existing adversarial attacks typically target harmful responses in single-point, greedy generations, overlooking the inherently stochastic nature of LLMs. In this paper, we propose a novel framework for adversarial robustness evaluation that explicitly models the entire output distribution, including tail-risks, providing better estimates for model robustness at scale. By casting the attack process as a resource allocation problem between optimization and sampling, we determine compute-optimal tradeoffs and show that integrating sampling into existing attacks boosts ASR by up to 48% and improves efficiency by up to two orders of magnitude. Our framework also enables us to analyze how different attack algorithms affect output harm distributions. Surprisingly, we find that most optimization strategies have little effect on output harmfulness. Finally, we introduce a data-free proof-of-concept objective based on entropy-maximization to demonstrate how our tail-aware perspective enables new optimization targets. Overall, our findings highlight the importance of tail-aware attacks and evaluation protocols to accurately assess and strengthen LLM safety.



## **8. Backdooring Bias ($B^2$) into Stable Diffusion Models**

cs.LG

Accepted to USENIX Security '25

**SubmitDate**: 2025-07-06    [abs](http://arxiv.org/abs/2406.15213v4) [paper-pdf](http://arxiv.org/pdf/2406.15213v4)

**Authors**: Ali Naseh, Jaechul Roh, Eugene Bagdasarian, Amir Houmansadr

**Abstract**: Recent advances in large text-conditional diffusion models have revolutionized image generation by enabling users to create realistic, high-quality images from textual prompts, significantly enhancing artistic creation and visual communication. However, these advancements also introduce an underexplored attack opportunity: the possibility of inducing biases by an adversary into the generated images for malicious intentions, e.g., to influence public opinion and spread propaganda. In this paper, we study an attack vector that allows an adversary to inject arbitrary bias into a target model. The attack leverages low-cost backdooring techniques using a targeted set of natural textual triggers embedded within a small number of malicious data samples produced with public generative models. An adversary could pick common sequences of words that can then be inadvertently activated by benign users during inference. We investigate the feasibility and challenges of such attacks, demonstrating how modern generative models have made this adversarial process both easier and more adaptable. On the other hand, we explore various aspects of the detectability of such attacks and demonstrate that the model's utility remains intact in the absence of the triggers. Our extensive experiments using over 200,000 generated images and against hundreds of fine-tuned models demonstrate the feasibility of the presented backdoor attack. We illustrate how these biases maintain strong text-image alignment, highlighting the challenges in detecting biased images without knowing that bias in advance. Our cost analysis confirms the low financial barrier (\$10-\$15) to executing such attacks, underscoring the need for robust defensive strategies against such vulnerabilities in diffusion models.



## **9. Addressing The Devastating Effects Of Single-Task Data Poisoning In Exemplar-Free Continual Learning**

cs.CR

Accepted at CoLLAs 2025

**SubmitDate**: 2025-07-05    [abs](http://arxiv.org/abs/2507.04106v1) [paper-pdf](http://arxiv.org/pdf/2507.04106v1)

**Authors**: Stanisław Pawlak, Bartłomiej Twardowski, Tomasz Trzciński, Joost van de Weijer

**Abstract**: Our research addresses the overlooked security concerns related to data poisoning in continual learning (CL). Data poisoning - the intentional manipulation of training data to affect the predictions of machine learning models - was recently shown to be a threat to CL training stability. While existing literature predominantly addresses scenario-dependent attacks, we propose to focus on a more simple and realistic single-task poison (STP) threats. In contrast to previously proposed poisoning settings, in STP adversaries lack knowledge and access to the model, as well as to both previous and future tasks. During an attack, they only have access to the current task within the data stream. Our study demonstrates that even within these stringent conditions, adversaries can compromise model performance using standard image corruptions. We show that STP attacks are able to strongly disrupt the whole continual training process: decreasing both the stability (its performance on past tasks) and plasticity (capacity to adapt to new tasks) of the algorithm. Finally, we propose a high-level defense framework for CL along with a poison task detection method based on task vectors. The code is available at https://github.com/stapaw/STP.git .



## **10. Multichannel Steganography: A Provably Secure Hybrid Steganographic Model for Secure Communication**

cs.CR

22 pages, 15 figures, 4 algorithms. This version is a preprint  uploaded to arXiv

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2501.04511v2) [paper-pdf](http://arxiv.org/pdf/2501.04511v2)

**Authors**: Obinna Omego, Michal Bosy

**Abstract**: Secure covert communication in hostile environments requires simultaneously achieving invisibility, provable security guarantees, and robustness against informed adversaries. This paper presents a novel hybrid steganographic framework that unites cover synthesis and cover modification within a unified multichannel protocol. A secret-seeded PRNG drives a lightweight Markov-chain generator to produce contextually plausible cover parameters, which are then masked with the payload and dispersed across independent channels. The masked bit-vector is imperceptibly embedded into conventional media via a variance-aware least-significant-bit algorithm, ensuring that statistical properties remain within natural bounds. We formalize a multichannel adversary model (MC-ATTACK) and prove that, under standard security assumptions, the adversary's distinguishing advantage is negligible, thereby guaranteeing both confidentiality and integrity. Empirical results corroborate these claims: local-variance-guided embedding yields near-lossless extraction (mean BER $<5\times10^{-3}$, correlation $>0.99$) with minimal perceptual distortion (PSNR $\approx100$,dB, SSIM $>0.99$), while key-based masking drives extraction success to zero (BER $\approx0.5$) for a fully informed adversary. Comparative analysis demonstrates that purely distortion-free or invertible schemes fail under the same threat model, underscoring the necessity of hybrid designs. The proposed approach advances high-assurance steganography by delivering an efficient, provably secure covert channel suitable for deployment in high-surveillance networks.



## **11. When There Is No Decoder: Removing Watermarks from Stable Diffusion Models in a No-box Setting**

cs.CR

arXiv admin note: text overlap with arXiv:2408.02035

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03646v1) [paper-pdf](http://arxiv.org/pdf/2507.03646v1)

**Authors**: Xiaodong Wu, Tianyi Tang, Xiangman Li, Jianbing Ni, Yong Yu

**Abstract**: Watermarking has emerged as a promising solution to counter harmful or deceptive AI-generated content by embedding hidden identifiers that trace content origins. However, the robustness of current watermarking techniques is still largely unexplored, raising critical questions about their effectiveness against adversarial attacks. To address this gap, we examine the robustness of model-specific watermarking, where watermark embedding is integrated with text-to-image generation in models like latent diffusion models. We introduce three attack strategies: edge prediction-based, box blurring, and fine-tuning-based attacks in a no-box setting, where an attacker does not require access to the ground-truth watermark decoder. Our findings reveal that while model-specific watermarking is resilient against basic evasion attempts, such as edge prediction, it is notably vulnerable to blurring and fine-tuning-based attacks. Our best-performing attack achieves a reduction in watermark detection accuracy to approximately 47.92\%. Additionally, we perform an ablation study on factors like message length, kernel size and decoder depth, identifying critical parameters influencing the fine-tuning attack's success. Finally, we assess several advanced watermarking defenses, finding that even the most robust methods, such as multi-label smoothing, result in watermark extraction accuracy that falls below an acceptable level when subjected to our no-box attacks.



## **12. Probing Latent Subspaces in LLM for AI Security: Identifying and Manipulating Adversarial States**

cs.LG

4 figures

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2503.09066v2) [paper-pdf](http://arxiv.org/pdf/2503.09066v2)

**Authors**: Xin Wei Chia, Swee Liang Wong, Jonathan Pan

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, yet they remain vulnerable to adversarial manipulations such as jailbreaking via prompt injection attacks. These attacks bypass safety mechanisms to generate restricted or harmful content. In this study, we investigated the underlying latent subspaces of safe and jailbroken states by extracting hidden activations from a LLM. Inspired by attractor dynamics in neuroscience, we hypothesized that LLM activations settle into semi stable states that can be identified and perturbed to induce state transitions. Using dimensionality reduction techniques, we projected activations from safe and jailbroken responses to reveal latent subspaces in lower dimensional spaces. We then derived a perturbation vector that when applied to safe representations, shifted the model towards a jailbreak state. Our results demonstrate that this causal intervention results in statistically significant jailbreak responses in a subset of prompts. Next, we probed how these perturbations propagate through the model's layers, testing whether the induced state change remains localized or cascades throughout the network. Our findings indicate that targeted perturbations induced distinct shifts in activations and model responses. Our approach paves the way for potential proactive defenses, shifting from traditional guardrail based methods to preemptive, model agnostic techniques that neutralize adversarial states at the representation level.



## **13. On the Limits of Robust Control Under Adversarial Disturbances**

eess.SY

Extended version of a manuscript submitted to IEEE Transactions on  Automatic Control, July 2025

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03630v1) [paper-pdf](http://arxiv.org/pdf/2507.03630v1)

**Authors**: Paul Trodden, José M. Maestre, Hideaki Ishii

**Abstract**: This paper addresses a fundamental and important question in control: under what conditions does there fail to exist a robust control policy that keeps the state of a constrained linear system within a target set, despite bounded disturbances? This question has practical implications for actuator and sensor specification, feasibility analysis for reference tracking, and the design of adversarial attacks in cyber-physical systems. While prior research has predominantly focused on using optimization to compute control-invariant sets to ensure feasible operation, our work complements these approaches by characterizing explicit sufficient conditions under which robust control is fundamentally infeasible. Specifically, we derive novel closed-form, algebraic expressions that relate the size of a disturbance set -- modelled as a scaled version of a basic shape -- to the system's spectral properties and the geometry of the constraint sets.



## **14. Beyond Weaponization: NLP Security for Medium and Lower-Resourced Languages in Their Own Right**

cs.CL

Pre-print

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03473v1) [paper-pdf](http://arxiv.org/pdf/2507.03473v1)

**Authors**: Heather Lent

**Abstract**: Despite mounting evidence that multilinguality can be easily weaponized against language models (LMs), works across NLP Security remain overwhelmingly English-centric. In terms of securing LMs, the NLP norm of "English first" collides with standard procedure in cybersecurity, whereby practitioners are expected to anticipate and prepare for worst-case outcomes. To mitigate worst-case outcomes in NLP Security, researchers must be willing to engage with the weakest links in LM security: lower-resourced languages. Accordingly, this work examines the security of LMs for lower- and medium-resourced languages. We extend existing adversarial attacks for up to 70 languages to evaluate the security of monolingual and multilingual LMs for these languages. Through our analysis, we find that monolingual models are often too small in total number of parameters to ensure sound security, and that while multilinguality is helpful, it does not always guarantee improved security either. Ultimately, these findings highlight important considerations for more secure deployment of LMs, for communities of lower-resourced languages.



## **15. Evaluating the Evaluators: Trust in Adversarial Robustness Tests**

cs.CR

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03450v1) [paper-pdf](http://arxiv.org/pdf/2507.03450v1)

**Authors**: Antonio Emanuele Cinà, Maura Pintor, Luca Demetrio, Ambra Demontis, Battista Biggio, Fabio Roli

**Abstract**: Despite significant progress in designing powerful adversarial evasion attacks for robustness verification, the evaluation of these methods often remains inconsistent and unreliable. Many assessments rely on mismatched models, unverified implementations, and uneven computational budgets, which can lead to biased results and a false sense of security. Consequently, robustness claims built on such flawed testing protocols may be misleading and give a false sense of security. As a concrete step toward improving evaluation reliability, we present AttackBench, a benchmark framework developed to assess the effectiveness of gradient-based attacks under standardized and reproducible conditions. AttackBench serves as an evaluation tool that ranks existing attack implementations based on a novel optimality metric, which enables researchers and practitioners to identify the most reliable and effective attack for use in subsequent robustness evaluations. The framework enforces consistent testing conditions and enables continuous updates, making it a reliable foundation for robustness verification.



## **16. Rectifying Adversarial Sample with Low Entropy Prior for Test-Time Defense**

cs.CV

To appear in IEEEE Transactions on Multimedia

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03427v1) [paper-pdf](http://arxiv.org/pdf/2507.03427v1)

**Authors**: Lina Ma, Xiaowei Fu, Fuxiang Huang, Xinbo Gao, Lei Zhang

**Abstract**: Existing defense methods fail to defend against unknown attacks and thus raise generalization issue of adversarial robustness. To remedy this problem, we attempt to delve into some underlying common characteristics among various attacks for generality. In this work, we reveal the commonly overlooked low entropy prior (LE) implied in various adversarial samples, and shed light on the universal robustness against unseen attacks in inference phase. LE prior is elaborated as two properties across various attacks as shown in Fig. 1 and Fig. 2: 1) low entropy misclassification for adversarial samples and 2) lower entropy prediction for higher attack intensity. This phenomenon stands in stark contrast to the naturally distributed samples. The LE prior can instruct existing test-time defense methods, thus we propose a two-stage REAL approach: Rectify Adversarial sample based on LE prior for test-time adversarial rectification. Specifically, to align adversarial samples more closely with clean samples, we propose to first rectify adversarial samples misclassified with low entropy by reverse maximizing prediction entropy, thereby eliminating their adversarial nature. To ensure the rectified samples can be correctly classified with low entropy, we carry out secondary rectification by forward minimizing prediction entropy, thus creating a Max-Min entropy optimization scheme. Further, based on the second property, we propose an attack-aware weighting mechanism to adaptively adjust the strengths of Max-Min entropy objectives. Experiments on several datasets show that REAL can greatly improve the performance of existing sample rectification models.



## **17. Breaking the Bulkhead: Demystifying Cross-Namespace Reference Vulnerabilities in Kubernetes Operators**

cs.CR

12 pages

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/2507.03387v1) [paper-pdf](http://arxiv.org/pdf/2507.03387v1)

**Authors**: Andong Chen, Zhaoxuan Jin, Ziyi Guo, Yan Chen

**Abstract**: Kubernetes Operators, automated tools designed to manage application lifecycles within Kubernetes clusters, extend the functionalities of Kubernetes, and reduce the operational burden on human engineers. While Operators significantly simplify DevOps workflows, they introduce new security risks. In particular, Kubernetes enforces namespace isolation to separate workloads and limit user access, ensuring that users can only interact with resources within their authorized namespaces. However, Kubernetes Operators often demand elevated privileges and may interact with resources across multiple namespaces. This introduces a new class of vulnerabilities, the Cross-Namespace Reference Vulnerability. The root cause lies in the mismatch between the declared scope of resources and the implemented scope of the Operator logic, resulting in Kubernetes being unable to properly isolate the namespace. Leveraging such vulnerability, an adversary with limited access to a single authorized namespace may exploit the Operator to perform operations affecting other unauthorized namespaces, causing Privilege Escalation and further impacts. To the best of our knowledge, this paper is the first to systematically investigate the security vulnerability of Kubernetes Operators. We present Cross-Namespace Reference Vulnerability with two strategies, demonstrating how an attacker can bypass namespace isolation. Through large-scale measurements, we found that over 14% of Operators in the wild are potentially vulnerable. Our findings have been reported to the relevant developers, resulting in 7 confirmations and 6 CVEs by the time of submission, affecting vendors including ****** and ******, highlighting the critical need for enhanced security practices in Kubernetes Operators. To mitigate it, we also open-source the static analysis suite to benefit the ecosystem.



## **18. Fault Sneaking Attack: a Stealthy Framework for Misleading Deep Neural Networks**

cs.LG

Accepted by the 56th Design Automation Conference (DAC 2019)

**SubmitDate**: 2025-07-04    [abs](http://arxiv.org/abs/1905.12032v2) [paper-pdf](http://arxiv.org/pdf/1905.12032v2)

**Authors**: Pu Zhao, Siyue Wang, Cheng Gongye, Yanzhi Wang, Yunsi Fei, Xue Lin

**Abstract**: Despite the great achievements of deep neural networks (DNNs), the vulnerability of state-of-the-art DNNs raises security concerns of DNNs in many application domains requiring high reliability.We propose the fault sneaking attack on DNNs, where the adversary aims to misclassify certain input images into any target labels by modifying the DNN parameters. We apply ADMM (alternating direction method of multipliers) for solving the optimization problem of the fault sneaking attack with two constraints: 1) the classification of the other images should be unchanged and 2) the parameter modifications should be minimized. Specifically, the first constraint requires us not only to inject designated faults (misclassifications), but also to hide the faults for stealthy or sneaking considerations by maintaining model accuracy. The second constraint requires us to minimize the parameter modifications (using L0 norm to measure the number of modifications and L2 norm to measure the magnitude of modifications). Comprehensive experimental evaluation demonstrates that the proposed framework can inject multiple sneaking faults without losing the overall test accuracy performance.



## **19. On the Adversarial Robustness of Graph Neural Networks with Graph Reduction**

cs.LG

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2412.05883v2) [paper-pdf](http://arxiv.org/pdf/2412.05883v2)

**Authors**: Kerui Wu, Ka-Ho Chow, Wenqi Wei, Lei Yu

**Abstract**: As Graph Neural Networks (GNNs) become increasingly popular for learning from large-scale graph data across various domains, their susceptibility to adversarial attacks when using graph reduction techniques for scalability remains underexplored. In this paper, we present an extensive empirical study to investigate the impact of graph reduction techniques, specifically graph coarsening and sparsification, on the robustness of GNNs against adversarial attacks. Through extensive experiments involving multiple datasets and GNN architectures, we examine the effects of four sparsification and six coarsening methods on the poisoning attacks. Our results indicate that, while graph sparsification can mitigate the effectiveness of certain poisoning attacks, such as Mettack, it has limited impact on others, like PGD. Conversely, graph coarsening tends to amplify the adversarial impact, significantly reducing classification accuracy as the reduction ratio decreases. Additionally, we provide a novel analysis of the causes driving these effects and examine how defensive GNN models perform under graph reduction, offering practical insights for designing robust GNNs within graph acceleration systems.



## **20. Adopting a human developmental visual diet yields robust, shape-based AI vision**

cs.LG

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2507.03168v1) [paper-pdf](http://arxiv.org/pdf/2507.03168v1)

**Authors**: Zejin Lu, Sushrut Thorat, Radoslaw M Cichy, Tim C Kietzmann

**Abstract**: Despite years of research and the dramatic scaling of artificial intelligence (AI) systems, a striking misalignment between artificial and human vision persists. Contrary to humans, AI heavily relies on texture-features rather than shape information, lacks robustness to image distortions, remains highly vulnerable to adversarial attacks, and struggles to recognise simple abstract shapes within complex backgrounds. To close this gap, we here introduce a solution that arises from a previously underexplored direction: rather than scaling up, we take inspiration from how human vision develops from early infancy into adulthood. We quantified the visual maturation by synthesising decades of psychophysical and neurophysiological research into a novel developmental visual diet (DVD) for AI vision. We show that guiding AI systems through this human-inspired curriculum produces models that closely align with human behaviour on every hallmark of robust vision tested yielding the strongest reported reliance on shape information to date, abstract shape recognition beyond the state of the art, higher robustness to image corruptions, and stronger resilience to adversarial attacks. By outperforming high parameter AI foundation models trained on orders of magnitude more data, we provide evidence that robust AI vision can be achieved by guiding the way how a model learns, not merely how much it learns, offering a resource-efficient route toward safer and more human-like artificial visual systems.



## **21. Adversarial Manipulation of Reasoning Models using Internal Representations**

cs.CL

Accepted to the ICML 2025 Workshop on Reliable and Responsible  Foundation Models (R2FM). 20 pages, 12 figures

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2507.03167v1) [paper-pdf](http://arxiv.org/pdf/2507.03167v1)

**Authors**: Kureha Yamaguchi, Benjamin Etheridge, Andy Arditi

**Abstract**: Reasoning models generate chain-of-thought (CoT) tokens before their final output, but how this affects their vulnerability to jailbreak attacks remains unclear. While traditional language models make refusal decisions at the prompt-response boundary, we find evidence that DeepSeek-R1-Distill-Llama-8B makes these decisions within its CoT generation. We identify a linear direction in activation space during CoT token generation that predicts whether the model will refuse or comply -- termed the "caution" direction because it corresponds to cautious reasoning patterns in the generated text. Ablating this direction from model activations increases harmful compliance, effectively jailbreaking the model. We additionally show that intervening only on CoT token activations suffices to control final outputs, and that incorporating this direction into prompt-based attacks improves success rates. Our findings suggest that the chain-of-thought itself is a promising new target for adversarial manipulation in reasoning models.   Code available at https://github.com/ky295/reasoning-manipulation



## **22. LIAR: Leveraging Inference Time Alignment (Best-of-N) to Jailbreak LLMs in Seconds**

cs.CL

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2412.05232v3) [paper-pdf](http://arxiv.org/pdf/2412.05232v3)

**Authors**: James Beetham, Souradip Chakraborty, Mengdi Wang, Furong Huang, Amrit Singh Bedi, Mubarak Shah

**Abstract**: Jailbreak attacks expose vulnerabilities in safety-aligned LLMs by eliciting harmful outputs through carefully crafted prompts. Existing methods rely on discrete optimization or trained adversarial generators, but are slow, compute-intensive, and often impractical. We argue that these inefficiencies stem from a mischaracterization of the problem. Instead, we frame jailbreaks as inference-time misalignment and introduce LIAR (Leveraging Inference-time misAlignment to jailbReak), a fast, black-box, best-of-$N$ sampling attack requiring no training. LIAR matches state-of-the-art success rates while reducing perplexity by $10\times$ and Time-to-Attack from hours to seconds. We also introduce a theoretical "safety net against jailbreaks" metric to quantify safety alignment strength and derive suboptimality bounds. Our work offers a simple yet effective tool for evaluating LLM robustness and advancing alignment research.



## **23. LoRA as a Flexible Framework for Securing Large Vision Systems**

cs.CV

Updated pre-print. Under review

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2506.00661v2) [paper-pdf](http://arxiv.org/pdf/2506.00661v2)

**Authors**: Zander W. Blasingame, Richard E. Neddo, Chen Liu

**Abstract**: Adversarial attacks have emerged as a critical threat to autonomous driving systems. These attacks exploit the underlying neural network, allowing small -- nearly invisible -- perturbations to completely alter the behavior of such systems in potentially malicious ways. E.g., causing a traffic sign classification network to misclassify a stop sign as a speed limit sign. Prior working in hardening such systems to adversarial attacks have looked at robust training of the system or adding additional pre-processing steps to the input pipeline. Such solutions either have a hard time generalizing, require knowledge of the adversarial attacks during training, or are computationally undesirable. Instead, we propose to take insights for parameter efficient fine-tuning and use low-rank adaptation (LoRA) to train a lightweight security patch -- enabling us to dynamically patch a large preexisting vision system as new vulnerabilities are discovered. We demonstrate that our framework can patch a pre-trained model to improve classification accuracy by up to 78.01% in the presence of adversarial examples.



## **24. Is Reasoning All You Need? Probing Bias in the Age of Reasoning Language Models**

cs.CL

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2507.02799v1) [paper-pdf](http://arxiv.org/pdf/2507.02799v1)

**Authors**: Riccardo Cantini, Nicola Gabriele, Alessio Orsino, Domenico Talia

**Abstract**: Reasoning Language Models (RLMs) have gained traction for their ability to perform complex, multi-step reasoning tasks through mechanisms such as Chain-of-Thought (CoT) prompting or fine-tuned reasoning traces. While these capabilities promise improved reliability, their impact on robustness to social biases remains unclear. In this work, we leverage the CLEAR-Bias benchmark, originally designed for Large Language Models (LLMs), to investigate the adversarial robustness of RLMs to bias elicitation. We systematically evaluate state-of-the-art RLMs across diverse sociocultural dimensions, using an LLM-as-a-judge approach for automated safety scoring and leveraging jailbreak techniques to assess the strength of built-in safety mechanisms. Our evaluation addresses three key questions: (i) how the introduction of reasoning capabilities affects model fairness and robustness; (ii) whether models fine-tuned for reasoning exhibit greater safety than those relying on CoT prompting at inference time; and (iii) how the success rate of jailbreak attacks targeting bias elicitation varies with the reasoning mechanisms employed. Our findings reveal a nuanced relationship between reasoning capabilities and bias safety. Surprisingly, models with explicit reasoning, whether via CoT prompting or fine-tuned reasoning traces, are generally more vulnerable to bias elicitation than base models without such mechanisms, suggesting reasoning may unintentionally open new pathways for stereotype reinforcement. Reasoning-enabled models appear somewhat safer than those relying on CoT prompting, which are particularly prone to contextual reframing attacks through storytelling prompts, fictional personas, or reward-shaped instructions. These results challenge the assumption that reasoning inherently improves robustness and underscore the need for more bias-aware approaches to reasoning design.



## **25. The Evolution of Dataset Distillation: Toward Scalable and Generalizable Solutions**

cs.CV

Dr. Jiawei Du is the corresponding author

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2502.05673v3) [paper-pdf](http://arxiv.org/pdf/2502.05673v3)

**Authors**: Ping Liu, Jiawei Du

**Abstract**: Dataset distillation, which condenses large-scale datasets into compact synthetic representations, has emerged as a critical solution for training modern deep learning models efficiently. While prior surveys focus on developments before 2023, this work comprehensively reviews recent advances, emphasizing scalability to large-scale datasets such as ImageNet-1K and ImageNet-21K. We categorize progress into a few key methodologies: trajectory matching, gradient matching, distribution matching, scalable generative approaches, and decoupling optimization mechanisms. As a comprehensive examination of recent dataset distillation advances, this survey highlights breakthrough innovations: the SRe2L framework for efficient and effective condensation, soft label strategies that significantly enhance model accuracy, and lossless distillation techniques that maximize compression while maintaining performance. Beyond these methodological advancements, we address critical challenges, including robustness against adversarial and backdoor attacks, effective handling of non-IID data distributions. Additionally, we explore emerging applications in video and audio processing, multi-modal learning, medical imaging, and scientific computing, highlighting its domain versatility. By offering extensive performance comparisons and actionable research directions, this survey equips researchers and practitioners with practical insights to advance efficient and generalizable dataset distillation, paving the way for future innovations.



## **26. De-AntiFake: Rethinking the Protective Perturbations Against Voice Cloning Attacks**

cs.SD

Accepted by ICML 2025

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2507.02606v1) [paper-pdf](http://arxiv.org/pdf/2507.02606v1)

**Authors**: Wei Fan, Kejiang Chen, Chang Liu, Weiming Zhang, Nenghai Yu

**Abstract**: The rapid advancement of speech generation models has heightened privacy and security concerns related to voice cloning (VC). Recent studies have investigated disrupting unauthorized voice cloning by introducing adversarial perturbations. However, determined attackers can mitigate these protective perturbations and successfully execute VC. In this study, we conduct the first systematic evaluation of these protective perturbations against VC under realistic threat models that include perturbation purification. Our findings reveal that while existing purification methods can neutralize a considerable portion of the protective perturbations, they still lead to distortions in the feature space of VC models, which degrades the performance of VC. From this perspective, we propose a novel two-stage purification method: (1) Purify the perturbed speech; (2) Refine it using phoneme guidance to align it with the clean speech distribution. Experimental results demonstrate that our method outperforms state-of-the-art purification methods in disrupting VC defenses. Our study reveals the limitations of adversarial perturbation-based VC defenses and underscores the urgent need for more robust solutions to mitigate the security and privacy risks posed by VC. The code and audio samples are available at https://de-antifake.github.io.



## **27. Robustness of Misinformation Classification Systems to Adversarial Examples Through BeamAttack**

cs.CL

12 pages main text, 27 pages total including references and  appendices. 13 figures, 10 tables. Accepted for publication in the LNCS  proceedings of CLEF 2025 (Best-of-Labs track)

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2506.23661v2) [paper-pdf](http://arxiv.org/pdf/2506.23661v2)

**Authors**: Arnisa Fazla, Lucas Krauter, David Guzman Piedrahita, Andrianos Michail

**Abstract**: We extend BeamAttack, an adversarial attack algorithm designed to evaluate the robustness of text classification systems through word-level modifications guided by beam search. Our extensions include support for word deletions and the option to skip substitutions, enabling the discovery of minimal modifications that alter model predictions. We also integrate LIME to better prioritize word replacements. Evaluated across multiple datasets and victim models (BiLSTM, BERT, and adversarially trained RoBERTa) within the BODEGA framework, our approach achieves over a 99\% attack success rate while preserving the semantic and lexical similarity of the original texts. Through both quantitative and qualitative analysis, we highlight BeamAttack's effectiveness and its limitations. Our implementation is available at https://github.com/LucK1Y/BeamAttack



## **28. SecAlign: Defending Against Prompt Injection with Preference Optimization**

cs.CR

ACM CCS 2025. Key words: prompt injection defense, LLM security,  LLM-integrated applications

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2410.05451v3) [paper-pdf](http://arxiv.org/pdf/2410.05451v3)

**Authors**: Sizhe Chen, Arman Zharmagambetov, Saeed Mahloujifar, Kamalika Chaudhuri, David Wagner, Chuan Guo

**Abstract**: Large language models (LLMs) are becoming increasingly prevalent in modern software systems, interfacing between the user and the Internet to assist with tasks that require advanced language understanding. To accomplish these tasks, the LLM often uses external data sources such as user documents, web retrieval, results from API calls, etc. This opens up new avenues for attackers to manipulate the LLM via prompt injection. Adversarial prompts can be injected into external data sources to override the system's intended instruction and instead execute a malicious instruction. To mitigate this vulnerability, we propose a new defense called SecAlign based on the technique of preference optimization. Our defense first constructs a preference dataset with prompt-injected inputs, secure outputs (ones that respond to the legitimate instruction), and insecure outputs (ones that respond to the injection). We then perform preference optimization on this dataset to teach the LLM to prefer the secure output over the insecure one. This provides the first known method that reduces the success rates of various prompt injections to <10%, even against attacks much more sophisticated than ones seen during training. This indicates our defense generalizes well against unknown and yet-to-come attacks. Also, SecAlign models are still practical with similar utility to the one before defensive training in our evaluations. Our code is at https://github.com/facebookresearch/SecAlign



## **29. Semantic Structure-Aware Generative Attacks for Enhanced Adversarial Transferability**

cs.CV

**SubmitDate**: 2025-07-03    [abs](http://arxiv.org/abs/2506.18248v2) [paper-pdf](http://arxiv.org/pdf/2506.18248v2)

**Authors**: Jongoh Jeong, Hunmin Yang, Jaeseok Jeong, Kuk-Jin Yoon

**Abstract**: Generative adversarial attacks train a perturbation generator on a white-box surrogate model and subsequently apply the crafted perturbations to unseen black-box victim models. In contrast to iterative attacks, these methods deliver superior inference-time efficiency, scalability, and transferability; however, up until now, existing studies have not fully exploited the representational capacity of generative models to preserve and harness semantic information. Specifically, the intermediate activations of the generator encode rich semantic features--object boundaries and coarse shapes--that remain under-exploited, thereby limiting the alignment of perturbations with object-salient regions which are critical for adversarial transferability. To remedy this, we introduce a semantic structure-aware attack framework based on the Mean Teacher, which serves as a temporally smoothed feature reference. With this smoothed reference, we further direct semantic consistency between the early-layer activations in the student and those of the semantically rich teacher by feature distillation. By anchoring perturbation synthesis to the semantically salient early intermediate blocks within the generator based on empirical findings, our method guides progressive adversarial perturbation on regions that substantially enhance adversarial transferability. We conduct extensive experiments over diverse models, domains and tasks to demonstrate consistent improvements relative to state-of-the-art generative attacks, comprehensively evaluated using conventional metrics and our newly proposed Accidental Correction Rate (ACR).



## **30. Boosting Adversarial Transferability Against Defenses via Multi-Scale Transformation**

cs.CV

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01791v1) [paper-pdf](http://arxiv.org/pdf/2507.01791v1)

**Authors**: Zihong Guo, Chen Wan, Yayin Zheng, Hailing Kuang, Xiaohai Lu

**Abstract**: The transferability of adversarial examples poses a significant security challenge for deep neural networks, which can be attacked without knowing anything about them. In this paper, we propose a new Segmented Gaussian Pyramid (SGP) attack method to enhance the transferability, particularly against defense models. Unlike existing methods that generally focus on single-scale images, our approach employs Gaussian filtering and three types of downsampling to construct a series of multi-scale examples. Then, the gradients of the loss function with respect to each scale are computed, and their average is used to determine the adversarial perturbations. The proposed SGP can be considered an input transformation with high extensibility that is easily integrated into most existing adversarial attacks. Extensive experiments demonstrate that in contrast to the state-of-the-art methods, SGP significantly enhances attack success rates against black-box defense models, with average attack success rates increasing by 2.3% to 32.6%, based only on transferability.



## **31. Tuning without Peeking: Provable Privacy and Generalization Bounds for LLM Post-Training**

cs.LG

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01752v1) [paper-pdf](http://arxiv.org/pdf/2507.01752v1)

**Authors**: Ismail Labiad, Mathurin Videau, Matthieu Kowalski, Marc Schoenauer, Alessandro Leite, Julia Kempe, Olivier Teytaud

**Abstract**: Gradient-based optimization is the workhorse of deep learning, offering efficient and scalable training via backpropagation. However, its reliance on large volumes of labeled data raises privacy and security concerns such as susceptibility to data poisoning attacks and the risk of overfitting. In contrast, black box optimization methods, which treat the model as an opaque function, relying solely on function evaluations to guide optimization, offer a promising alternative in scenarios where data access is restricted, adversarial risks are high, or overfitting is a concern. However, black box methods also pose significant challenges, including poor scalability to high-dimensional parameter spaces, as prevalent in large language models (LLMs), and high computational costs due to reliance on numerous model evaluations. This paper introduces BBoxER, an evolutionary black-box method for LLM post-training that induces an information bottleneck via implicit compression of the training data. Leveraging the tractability of information flow, we provide strong theoretical bounds on generalization, differential privacy, susceptibility to data poisoning attacks, and robustness to extraction attacks. BBoxER operates on top of pre-trained LLMs, offering a lightweight and modular enhancement suitable for deployment in restricted or privacy-sensitive environments, in addition to non-vacuous generalization guarantees. In experiments with LLMs, we demonstrate empirically that Retrofitting methods are able to learn, showing how a few iterations of BBoxER improve performance and generalize well on a benchmark of reasoning datasets. This positions BBoxER as an attractive add-on on top of gradient-based optimization.



## **32. Blockchain Address Poisoning**

cs.CR

To appear in Proceedings of the 34th USENIX Security Symposium  (USENIX Security'25)

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2501.16681v3) [paper-pdf](http://arxiv.org/pdf/2501.16681v3)

**Authors**: Taro Tsuchiya, Jin-Dong Dong, Kyle Soska, Nicolas Christin

**Abstract**: In many blockchains, e.g., Ethereum, Binance Smart Chain (BSC), the primary representation used for wallet addresses is a hardly memorable 40-digit hexadecimal string. As a result, users often select addresses from their recent transaction history, which enables blockchain address poisoning. The adversary first generates lookalike addresses similar to one with which the victim has previously interacted, and then engages with the victim to ``poison'' their transaction history. The goal is to have the victim mistakenly send tokens to the lookalike address, as opposed to the intended recipient. Compared to contemporary studies, this paper provides four notable contributions. First, we develop a detection system and perform measurements over two years on both Ethereum and BSC. We identify 13~times more attack attempts than reported previously -- totaling 270M on-chain attacks targeting 17M victims. 6,633 incidents have caused at least 83.8M USD in losses, which makes blockchain address poisoning one of the largest cryptocurrency phishing schemes observed in the wild. Second, we analyze a few large attack entities using improved clustering techniques, and model attacker profitability and competition. Third, we reveal attack strategies -- targeted populations, success conditions (address similarity, timing), and cross-chain attacks. Fourth, we mathematically define and simulate the lookalike address generation process across various software- and hardware-based implementations, and identify a large-scale attacker group that appears to use GPUs. We also discuss defensive countermeasures.



## **33. Graph Representation-based Model Poisoning on Federated LLMs in CyberEdge Networks**

cs.CR

7 pages, 5 figures

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01694v1) [paper-pdf](http://arxiv.org/pdf/2507.01694v1)

**Authors**: Hanlin Cai, Haofan Dong, Houtianfu Wang, Kai Li, Ozgur B. Akan

**Abstract**: Federated large language models (FedLLMs) provide powerful generative capabilities in CyberEdge networks while protecting data privacy. However, FedLLMs remains highly vulnerable to model poisoning attacks. This article first reviews recent model poisoning techniques and existing defense mechanisms for FedLLMs, highlighting critical limitations, particularly under non-IID text distributions. In particular, current defenses primarily utilize distance-based outlier detection or norm constraints, operating under the assumption that adversarial updates significantly diverge from benign statistics. This assumption can fail when facing adaptive attackers targeting billionparameter LLMs. Next, this article investigates emerging Graph Representation-Based Model Poisoning (GRMP), a novel attack paradigm that leverages higher-order correlations among honest client gradients to synthesize malicious updates indistinguishable from legitimate model updates. GRMP can effectively evade advanced defenses, resulting in substantial accuracy loss and performance degradation. Moreover, this article outlines a research roadmap emphasizing the importance of graph-aware secure aggregation methods, FedLLMs-specific vulnerability metrics, and evaluation frameworks to strengthen the robustness of future federated language model deployments.



## **34. Learned-Database Systems Security**

cs.CR

Accepted at TMLR

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2212.10318v4) [paper-pdf](http://arxiv.org/pdf/2212.10318v4)

**Authors**: Roei Schuster, Jin Peng Zhou, Thorsten Eisenhofer, Paul Grubbs, Nicolas Papernot

**Abstract**: A learned database system uses machine learning (ML) internally to improve performance. We can expect such systems to be vulnerable to some adversarial-ML attacks. Often, the learned component is shared between mutually-distrusting users or processes, much like microarchitectural resources such as caches, potentially giving rise to highly-realistic attacker models. However, compared to attacks on other ML-based systems, attackers face a level of indirection as they cannot interact directly with the learned model. Additionally, the difference between the attack surface of learned and non-learned versions of the same system is often subtle. These factors obfuscate the de-facto risks that the incorporation of ML carries. We analyze the root causes of potentially-increased attack surface in learned database systems and develop a framework for identifying vulnerabilities that stem from the use of ML. We apply our framework to a broad set of learned components currently being explored in the database community. To empirically validate the vulnerabilities surfaced by our framework, we choose 3 of them and implement and evaluate exploits against these. We show that the use of ML cause leakage of past queries in a database, enable a poisoning attack that causes exponential memory blowup in an index structure and crashes it in seconds, and enable index users to snoop on each others' key distributions by timing queries over their own keys. We find that adversarial ML is an universal threat against learned components in database systems, point to open research gaps in our understanding of learned-systems security, and conclude by discussing mitigations, while noting that data leakage is inherent in systems whose learned component is shared between multiple parties.



## **35. Slot: Provenance-Driven APT Detection through Graph Reinforcement Learning**

cs.CR

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2410.17910v3) [paper-pdf](http://arxiv.org/pdf/2410.17910v3)

**Authors**: Wei Qiao, Yebo Feng, Teng Li, Zhuo Ma, Yulong Shen, JianFeng Ma, Yang Liu

**Abstract**: Advanced Persistent Threats (APTs) represent sophisticated cyberattacks characterized by their ability to remain undetected within the victim system for extended periods, aiming to exfiltrate sensitive data or disrupt operations. Existing detection approaches often struggle to effectively identify these complex threats, construct the attack chain for defense facilitation, or resist adversarial attacks. To overcome these challenges, we propose Slot, an advanced APT detection approach based on provenance graphs and graph reinforcement learning. Slot excels in uncovering multi-level hidden relationships, such as causal, contextual, and indirect connections, among system behaviors through provenance graph mining. By pioneering the integration of graph reinforcement learning, Slot dynamically adapts to new user activities and evolving attack strategies, enhancing its resilience against adversarial attacks. Additionally, Slot automatically constructs the attack chain according to detected attacks with clustering algorithms, providing precise identification of attack paths and facilitating the development of defense strategies. Evaluations with real-world datasets demonstrate Slot's outstanding accuracy, efficiency, adaptability, and robustness in APT detection, with most metrics surpassing state-of-the-art methods. Additionally, case studies conducted to assess Slot's effectiveness in supporting APT defense further establish it as a practical and reliable tool for cybersecurity protection.



## **36. DARTS: A Dual-View Attack Framework for Targeted Manipulation in Federated Sequential Recommendation**

cs.IR

10 pages. arXiv admin note: substantial text overlap with  arXiv:2409.07500; text overlap with arXiv:2212.05399 by other authors

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01383v1) [paper-pdf](http://arxiv.org/pdf/2507.01383v1)

**Authors**: Qitao Qin, Yucong Luo, Zhibo Chu

**Abstract**: Federated recommendation (FedRec) preserves user privacy by enabling decentralized training of personalized models, but this architecture is inherently vulnerable to adversarial attacks. Significant research has been conducted on targeted attacks in FedRec systems, motivated by commercial and social influence considerations. However, much of this work has largely overlooked the differential robustness of recommendation models. Moreover, our empirical findings indicate that existing targeted attack methods achieve only limited effectiveness in Federated Sequential Recommendation(FSR) tasks. Driven by these observations, we focus on investigating targeted attacks in FSR and propose a novel dualview attack framework, named DV-FSR. This attack method uniquely combines a sampling-based explicit strategy with a contrastive learning-based implicit gradient strategy to orchestrate a coordinated attack. Additionally, we introduce a specific defense mechanism tailored for targeted attacks in FSR, aiming to evaluate the mitigation effects of the attack method we proposed. Extensive experiments validate the effectiveness of our proposed approach on representative sequential models. Our codes are publicly available.



## **37. 3D Gaussian Splatting Driven Multi-View Robust Physical Adversarial Camouflage Generation**

cs.CV

Accepted by ICCV 2025

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01367v1) [paper-pdf](http://arxiv.org/pdf/2507.01367v1)

**Authors**: Tianrui Lou, Xiaojun Jia, Siyuan Liang, Jiawei Liang, Ming Zhang, Yanjun Xiao, Xiaochun Cao

**Abstract**: Physical adversarial attack methods expose the vulnerabilities of deep neural networks and pose a significant threat to safety-critical scenarios such as autonomous driving. Camouflage-based physical attack is a more promising approach compared to the patch-based attack, offering stronger adversarial effectiveness in complex physical environments. However, most prior work relies on mesh priors of the target object and virtual environments constructed by simulators, which are time-consuming to obtain and inevitably differ from the real world. Moreover, due to the limitations of the backgrounds in training images, previous methods often fail to produce multi-view robust adversarial camouflage and tend to fall into sub-optimal solutions. Due to these reasons, prior work lacks adversarial effectiveness and robustness across diverse viewpoints and physical environments. We propose a physical attack framework based on 3D Gaussian Splatting (3DGS), named PGA, which provides rapid and precise reconstruction with few images, along with photo-realistic rendering capabilities. Our framework further enhances cross-view robustness and adversarial effectiveness by preventing mutual and self-occlusion among Gaussians and employing a min-max optimization approach that adjusts the imaging background of each viewpoint, helping the algorithm filter out non-robust adversarial features. Extensive experiments validate the effectiveness and superiority of PGA. Our code is available at:https://github.com/TRLou/PGA.



## **38. ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks**

cs.LG

ICML 2025

**SubmitDate**: 2025-07-02    [abs](http://arxiv.org/abs/2507.01321v1) [paper-pdf](http://arxiv.org/pdf/2507.01321v1)

**Authors**: Zhiyao Ren, Siyuan Liang, Aishan Liu, Dacheng Tao

**Abstract**: In-context learning (ICL) has demonstrated remarkable success in large language models (LLMs) due to its adaptability and parameter-free nature. However, it also introduces a critical vulnerability to backdoor attacks, where adversaries can manipulate LLM behaviors by simply poisoning a few ICL demonstrations. In this paper, we propose, for the first time, the dual-learning hypothesis, which posits that LLMs simultaneously learn both the task-relevant latent concepts and backdoor latent concepts within poisoned demonstrations, jointly influencing the probability of model outputs. Through theoretical analysis, we derive an upper bound for ICL backdoor effects, revealing that the vulnerability is dominated by the concept preference ratio between the task and the backdoor. Motivated by these findings, we propose ICLShield, a defense mechanism that dynamically adjusts the concept preference ratio. Our method encourages LLMs to select clean demonstrations during the ICL phase by leveraging confidence and similarity scores, effectively mitigating susceptibility to backdoor attacks. Extensive experiments across multiple LLMs and tasks demonstrate that our method achieves state-of-the-art defense effectiveness, significantly outperforming existing approaches (+26.02% on average). Furthermore, our method exhibits exceptional adaptability and defensive performance even for closed-source models (e.g., GPT-4).



## **39. Defensive Adversarial CAPTCHA: A Semantics-Driven Framework for Natural Adversarial Example Generation**

cs.CV

13 pages, 6 figures

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2506.10685v3) [paper-pdf](http://arxiv.org/pdf/2506.10685v3)

**Authors**: Xia Du, Xiaoyuan Liu, Jizhe Zhou, Zheng Lin, Chi-man Pun, Cong Wu, Tao Li, Zhe Chen, Wei Ni, Jun Luo

**Abstract**: Traditional CAPTCHA (Completely Automated Public Turing Test to Tell Computers and Humans Apart) schemes are increasingly vulnerable to automated attacks powered by deep neural networks (DNNs). Existing adversarial attack methods often rely on the original image characteristics, resulting in distortions that hinder human interpretation and limit their applicability in scenarios where no initial input images are available. To address these challenges, we propose the Unsourced Adversarial CAPTCHA (DAC), a novel framework that generates high-fidelity adversarial examples guided by attacker-specified semantics information. Leveraging a Large Language Model (LLM), DAC enhances CAPTCHA diversity and enriches the semantic information. To address various application scenarios, we examine the white-box targeted attack scenario and the black box untargeted attack scenario. For target attacks, we introduce two latent noise variables that are alternately guided in the diffusion step to achieve robust inversion. The synergy between gradient guidance and latent variable optimization achieved in this way ensures that the generated adversarial examples not only accurately align with the target conditions but also achieve optimal performance in terms of distributional consistency and attack effectiveness. In untargeted attacks, especially for black-box scenarios, we introduce bi-path unsourced adversarial CAPTCHA (BP-DAC), a two-step optimization strategy employing multimodal gradients and bi-path optimization for efficient misclassification. Experiments show that the defensive adversarial CAPTCHA generated by BP-DAC is able to defend against most of the unknown models, and the generated CAPTCHA is indistinguishable to both humans and DNNs.



## **40. CAVALRY-V: A Large-Scale Generator Framework for Adversarial Attacks on Video MLLMs**

cs.CV

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2507.00817v1) [paper-pdf](http://arxiv.org/pdf/2507.00817v1)

**Authors**: Jiaming Zhang, Rui Hu, Qing Guo, Wei Yang Bryan Lim

**Abstract**: Video Multimodal Large Language Models (V-MLLMs) have shown impressive capabilities in temporal reasoning and cross-modal understanding, yet their vulnerability to adversarial attacks remains underexplored due to unique challenges: complex cross-modal reasoning mechanisms, temporal dependencies, and computational constraints. We present CAVALRY-V (Cross-modal Language-Vision Adversarial Yielding for Videos), a novel framework that directly targets the critical interface between visual perception and language generation in V-MLLMs. Our approach introduces two key innovations: (1) a dual-objective semantic-visual loss function that simultaneously disrupts the model's text generation logits and visual representations to undermine cross-modal integration, and (2) a computationally efficient two-stage generator framework that combines large-scale pre-training for cross-model transferability with specialized fine-tuning for spatiotemporal coherence. Empirical evaluation on comprehensive video understanding benchmarks demonstrates that CAVALRY-V significantly outperforms existing attack methods, achieving 22.8% average improvement over the best baseline attacks on both commercial systems (GPT-4.1, Gemini 2.0) and open-source models (QwenVL-2.5, InternVL-2.5, Llava-Video, Aria, MiniCPM-o-2.6). Our framework achieves flexibility through implicit temporal coherence modeling rather than explicit regularization, enabling significant performance improvements even on image understanding (34.4% average gain). This capability demonstrates CAVALRY-V's potential as a foundational approach for adversarial research across multimodal systems.



## **41. Cage-Based Deformation for Transferable and Undefendable Point Cloud Attack**

cs.CV

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2507.00690v1) [paper-pdf](http://arxiv.org/pdf/2507.00690v1)

**Authors**: Keke Tang, Ziyong Du, Weilong Peng, Xiaofei Wang, Peican Zhu, Ligang Liu, Zhihong Tian

**Abstract**: Adversarial attacks on point clouds often impose strict geometric constraints to preserve plausibility; however, such constraints inherently limit transferability and undefendability. While deformation offers an alternative, existing unstructured approaches may introduce unnatural distortions, making adversarial point clouds conspicuous and undermining their plausibility. In this paper, we propose CageAttack, a cage-based deformation framework that produces natural adversarial point clouds. It first constructs a cage around the target object, providing a structured basis for smooth, natural-looking deformation. Perturbations are then applied to the cage vertices, which seamlessly propagate to the point cloud, ensuring that the resulting deformations remain intrinsic to the object and preserve plausibility. Extensive experiments on seven 3D deep neural network classifiers across three datasets show that CageAttack achieves a superior balance among transferability, undefendability, and plausibility, outperforming state-of-the-art methods. Codes will be made public upon acceptance.



## **42. How Resilient is QUIC to Security and Privacy Attacks?**

cs.CR

7 pages, 1 figure, 1 table

**SubmitDate**: 2025-07-01    [abs](http://arxiv.org/abs/2401.06657v3) [paper-pdf](http://arxiv.org/pdf/2401.06657v3)

**Authors**: Jayasree Sengupta, Debasmita Dey, Simone Ferlin-Reiter, Nirnay Ghosh, Vaibhav Bajpai

**Abstract**: QUIC has rapidly evolved into a cornerstone transport protocol for secure, low-latency communications, yet its deployment continues to expose critical security and privacy vulnerabilities, particularly during connection establishment phases and via traffic analysis. This paper systematically revisits a comprehensive set of attacks on QUIC and emerging privacy threats. Building upon these observations, we critically analyze recent IETF mitigation efforts, including TLS Encrypted Client Hello (ECH), Oblivious HTTP (OHTTP) and MASQUE. We analyze how these mechanisms enhance privacy while introducing new operational risks, particularly under adversarial load. Additionally, we discuss emerging challenges posed by post-quantum cryptographic (PQC) handshakes, including handshake expansion and metadata leakage risks. Our analysis highlights ongoing gaps between theoretical defenses and practical deployments, and proposes new research directions focused on adaptive privacy mechanisms. Building on these insights, we propose future directions to ensure long-term security of QUIC and aim to guide its evolution as a robust, privacy-preserving, and resilient transport foundation for the next-generation Internet.



## **43. Lazarus Group Targets Crypto-Wallets and Financial Data while employing new Tradecrafts**

cs.CR

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2505.21725v2) [paper-pdf](http://arxiv.org/pdf/2505.21725v2)

**Authors**: Alessio Di Santo

**Abstract**: This report presents a comprehensive analysis of a malicious software sample, detailing its architecture, behavioral characteristics, and underlying intent. Through static and dynamic examination, the malware core functionalities, including persistence mechanisms, command-and-control communication, and data exfiltration routines, are identified and its supporting infrastructure is mapped. By correlating observed indicators of compromise with known techniques, tactics, and procedures, this analysis situates the sample within the broader context of contemporary threat campaigns and infers the capabilities and motivations of its likely threat actor.   Building on these findings, actionable threat intelligence is provided to support proactive defenses. Threat hunting teams receive precise detection hypotheses for uncovering latent adversarial presence, while monitoring systems can refine alert logic to detect anomalous activity in real time. Finally, the report discusses how this structured intelligence enhances predictive risk assessments, informs vulnerability prioritization, and strengthens organizational resilience against advanced persistent threats. By integrating detailed technical insights with strategic threat landscape mapping, this malware analysis report not only reconstructs past adversary actions but also establishes a robust foundation for anticipating and mitigating future attacks.



## **44. Plug. Play. Persist. Inside a Ready-to-Go Havoc C2 Infrastructure**

cs.CR

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2507.00189v1) [paper-pdf](http://arxiv.org/pdf/2507.00189v1)

**Authors**: Alessio Di Santo

**Abstract**: This analysis focuses on a single Azure-hosted Virtual Machine at 52.230.23.114 that the adversary converted into an all-in-one delivery, staging and Command-and-Control node. The host advertises an out-of-date Apache 2.4.52 instance whose open directory exposes phishing lures, PowerShell loaders, Reflective Shell-Code, compiled Havoc Demon implants and a toolbox of lateral-movement binaries; the same server also answers on 8443/80 for encrypted beacon traffic. The web tier is riddled with publicly documented critical vulnerabilities, that would have allowed initial code-execution had the attackers not already owned the device.   Initial access is delivered through an HTML file that, once de-obfuscated, perfectly mimics Google Unusual sign-in attempt notification and funnels victims toward credential collection. A PowerShell command follows: it disables AMSI in-memory, downloads a Base64-encoded stub, allocates RWX pages and starts the shell-code without ever touching disk. That stub reconstructs a DLL in memory using the Reflective-Loader technique and hands control to Havoc Demon implant. Every Demon variant-32- and 64-bit alike-talks to the same backend, resolves Windows APIs with hashed look-ups, and hides its activity behind indirect syscalls.   Runtime telemetry shows interests in registry under Image File Execution Options, deliberate queries to Software Restriction Policy keys, and heavy use of Crypto DLLs to protect payloads and C2 traffic. The attacker toolkit further contains Chisel, PsExec, Doppelganger and Whisker, some of them re-compiled under user directories that leak the developer personas tonzking123 and thobt. Collectively the findings paint a picture of a technically adept actor who values rapid re-tooling over deep operational security, leaning on Havoc modularity and on legitimate cloud services to blend malicious flows into ordinary enterprise traffic.



## **45. SQUASH: A SWAP-Based Quantum Attack to Sabotage Hybrid Quantum Neural Networks**

quant-ph

Keywords: Quantum Machine Learning, Hybrid Quantum Neural Networks,  SWAP Test, Fidelity, Circuit-level Attack

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24081v1) [paper-pdf](http://arxiv.org/pdf/2506.24081v1)

**Authors**: Rahul Kumar, Wenqi Wei, Ying Mao, Junaid Farooq, Ying Wang, Juntao Chen

**Abstract**: We propose a circuit-level attack, SQUASH, a SWAP-Based Quantum Attack to sabotage Hybrid Quantum Neural Networks (HQNNs) for classification tasks. SQUASH is executed by inserting SWAP gate(s) into the variational quantum circuit of the victim HQNN. Unlike conventional noise-based or adversarial input attacks, SQUASH directly manipulates the circuit structure, leading to qubit misalignment and disrupting quantum state evolution. This attack is highly stealthy, as it does not require access to training data or introduce detectable perturbations in input states. Our results demonstrate that SQUASH significantly degrades classification performance, with untargeted SWAP attacks reducing accuracy by up to 74.08\% and targeted SWAP attacks reducing target class accuracy by up to 79.78\%. These findings reveal a critical vulnerability in HQNN implementations, underscoring the need for more resilient architectures against circuit-level adversarial interventions.



## **46. STACK: Adversarial Attacks on LLM Safeguard Pipelines**

cs.CL

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24068v1) [paper-pdf](http://arxiv.org/pdf/2506.24068v1)

**Authors**: Ian R. McKenzie, Oskar J. Hollinsworth, Tom Tseng, Xander Davies, Stephen Casper, Aaron D. Tucker, Robert Kirk, Adam Gleave

**Abstract**: Frontier AI developers are relying on layers of safeguards to protect against catastrophic misuse of AI systems. Anthropic guards their latest Claude 4 Opus model using one such defense pipeline, and other frontier developers including Google DeepMind and OpenAI pledge to soon deploy similar defenses. However, the security of such pipelines is unclear, with limited prior work evaluating or attacking these pipelines. We address this gap by developing and red-teaming an open-source defense pipeline. First, we find that a novel few-shot-prompted input and output classifier outperforms state-of-the-art open-weight safeguard model ShieldGemma across three attacks and two datasets, reducing the attack success rate (ASR) to 0% on the catastrophic misuse dataset ClearHarm. Second, we introduce a STaged AttaCK (STACK) procedure that achieves 71% ASR on ClearHarm in a black-box attack against the few-shot-prompted classifier pipeline. Finally, we also evaluate STACK in a transfer setting, achieving 33% ASR, providing initial evidence that it is feasible to design attacks with no access to the target pipeline. We conclude by suggesting specific mitigations that developers could use to thwart staged attacks.



## **47. Consensus-based optimization for closed-box adversarial attacks and a connection to evolution strategies**

math.OC

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24048v1) [paper-pdf](http://arxiv.org/pdf/2506.24048v1)

**Authors**: Tim Roith, Leon Bungert, Philipp Wacker

**Abstract**: Consensus-based optimization (CBO) has established itself as an efficient gradient-free optimization scheme, with attractive mathematical properties, such as mean-field convergence results for non-convex loss functions. In this work, we study CBO in the context of closed-box adversarial attacks, which are imperceptible input perturbations that aim to fool a classifier, without accessing its gradient. Our contribution is to establish a connection between the so-called consensus hopping as introduced by Riedl et al. and natural evolution strategies (NES) commonly applied in the context of adversarial attacks and to rigorously relate both methods to gradient-based optimization schemes. Beyond that, we provide a comprehensive experimental study that shows that despite the conceptual similarities, CBO can outperform NES and other evolutionary strategies in certain scenarios.



## **48. Quickest Detection of Adversarial Attacks Against Correlated Equilibria**

cs.GT

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24040v1) [paper-pdf](http://arxiv.org/pdf/2506.24040v1)

**Authors**: Kiarash Kazari, Aris Kanellopoulos, György Dán

**Abstract**: We consider correlated equilibria in strategic games in an adversarial environment, where an adversary can compromise the public signal used by the players for choosing their strategies, while players aim at detecting a potential attack as soon as possible to avoid loss of utility. We model the interaction between the adversary and the players as a zero-sum game and we derive the maxmin strategies for both the defender and the attacker using the framework of quickest change detection. We define a class of adversarial strategies that achieve the optimal trade-off between attack impact and attack detectability and show that a generalized CUSUM scheme is asymptotically optimal for the detection of the attacks. Our numerical results on the Sioux-Falls benchmark traffic routing game show that the proposed detection scheme can effectively limit the utility loss by a potential adversary.



## **49. Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation**

cs.CR

This is the full version (27 pages) of the paper 'Riddle Me This!  Stealthy Membership Inference for Retrieval-Augmented Generation' published  at CCS 2025

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2502.00306v2) [paper-pdf](http://arxiv.org/pdf/2502.00306v2)

**Authors**: Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh Chaudhari, Alina Oprea, Amir Houmansadr

**Abstract**: Retrieval-Augmented Generation (RAG) enables Large Language Models (LLMs) to generate grounded responses by leveraging external knowledge databases without altering model parameters. Although the absence of weight tuning prevents leakage via model parameters, it introduces the risk of inference adversaries exploiting retrieved documents in the model's context. Existing methods for membership inference and data extraction often rely on jailbreaking or carefully crafted unnatural queries, which can be easily detected or thwarted with query rewriting techniques common in RAG systems. In this work, we present Interrogation Attack (IA), a membership inference technique targeting documents in the RAG datastore. By crafting natural-text queries that are answerable only with the target document's presence, our approach demonstrates successful inference with just 30 queries while remaining stealthy; straightforward detectors identify adversarial prompts from existing methods up to ~76x more frequently than those generated by our attack. We observe a 2x improvement in TPR@1%FPR over prior inference attacks across diverse RAG configurations, all while costing less than $0.02 per document inference.



## **50. Benchmarking Spiking Neural Network Learning Methods with Varying Locality**

cs.NE

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2402.01782v2) [paper-pdf](http://arxiv.org/pdf/2402.01782v2)

**Authors**: Jiaqi Lin, Sen Lu, Malyaban Bal, Abhronil Sengupta

**Abstract**: Spiking Neural Networks (SNNs), providing more realistic neuronal dynamics, have been shown to achieve performance comparable to Artificial Neural Networks (ANNs) in several machine learning tasks. Information is processed as spikes within SNNs in an event-based mechanism that significantly reduces energy consumption. However, training SNNs is challenging due to the non-differentiable nature of the spiking mechanism. Traditional approaches, such as Backpropagation Through Time (BPTT), have shown effectiveness but come with additional computational and memory costs and are biologically implausible. In contrast, recent works propose alternative learning methods with varying degrees of locality, demonstrating success in classification tasks. In this work, we show that these methods share similarities during the training process, while they present a trade-off between biological plausibility and performance. Further, given the implicitly recurrent nature of SNNs, this research investigates the influence of the addition of explicit recurrence to SNNs. We experimentally prove that the addition of explicit recurrent weights enhances the robustness of SNNs. We also investigate the performance of local learning methods under gradient and non-gradient-based adversarial attacks.



