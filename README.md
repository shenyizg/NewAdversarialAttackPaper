# Latest Adversarial Attack Papers
**update at 2025-07-26 11:30:45**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Optimal Transport Regularized Divergences: Application to Adversarial Robustness**

cs.LG

34 pages, 2 figures

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2309.03791v3) [paper-pdf](http://arxiv.org/pdf/2309.03791v3)

**Authors**: Jeremiah Birrell, Reza Ebrahimi

**Abstract**: We introduce a new class of optimal-transport-regularized divergences, $D^c$, constructed via an infimal convolution between an information divergence, $D$, and an optimal-transport (OT) cost, $C$, and study their use in distributionally robust optimization (DRO). In particular, we propose the $ARMOR_D$ methods as novel approaches to enhancing the adversarial robustness of deep learning models. These DRO-based methods are defined by minimizing the maximum expected loss over a $D^c$-neighborhood of the empirical distribution of the training data. Viewed as a tool for constructing adversarial samples, our method allows samples to be both transported, according to the OT cost, and re-weighted, according to the information divergence; the addition of a principled and dynamical adversarial re-weighting on top of adversarial sample transport is a key innovation of $ARMOR_D$. $ARMOR_D$ can be viewed as a generalization of the best-performing loss functions and OT costs in the adversarial training literature; we demonstrate this flexibility by using $ARMOR_D$ to augment the UDR, TRADES, and MART methods and obtain improved performance on CIFAR-10 and CIFAR-100 image recognition. Specifically, augmenting with $ARMOR_D$ leads to 1.9\% and 2.1\% improvement against AutoAttack, a powerful ensemble of adversarial attacks, on CIFAR-10 and CIFAR-100 respectively. To foster reproducibility, we made the code accessible at https://github.com/star-ailab/ARMOR.



## **2. GCC-Spam: Spam Detection via GAN, Contrastive Learning, and Character Similarity Networks**

cs.LG

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.14679v2) [paper-pdf](http://arxiv.org/pdf/2507.14679v2)

**Authors**: Zhijie Wang, Zixin Xu, Zhiyuan Pan

**Abstract**: The exponential growth of spam text on the Internet necessitates robust detection mechanisms to mitigate risks such as information leakage and social instability. This work addresses two principal challenges: adversarial strategies employed by spammers and the scarcity of labeled data. We propose a novel spam-text detection framework GCC-Spam, which integrates three core innovations. First, a character similarity network captures orthographic and phonetic features to counter character-obfuscation attacks and furthermore produces sentence embeddings for downstream classification. Second, contrastive learning enhances discriminability by optimizing the latent-space distance between spam and normal texts. Third, a Generative Adversarial Network (GAN) generates realistic pseudo-spam samples to alleviate data scarcity while improving model robustness and classification accuracy. Extensive experiments on real-world datasets demonstrate that our model outperforms baseline approaches, achieving higher detection rates with significantly fewer labeled examples.



## **3. Reinforced Embodied Active Defense: Exploiting Adaptive Interaction for Robust Visual Perception in Adversarial 3D Environments**

cs.CV

arXiv admin note: text overlap with arXiv:2404.00540

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18484v1) [paper-pdf](http://arxiv.org/pdf/2507.18484v1)

**Authors**: Xiao Yang, Lingxuan Wu, Lizhong Wang, Chengyang Ying, Hang Su, Jun Zhu

**Abstract**: Adversarial attacks in 3D environments have emerged as a critical threat to the reliability of visual perception systems, particularly in safety-sensitive applications such as identity verification and autonomous driving. These attacks employ adversarial patches and 3D objects to manipulate deep neural network (DNN) predictions by exploiting vulnerabilities within complex scenes. Existing defense mechanisms, such as adversarial training and purification, primarily employ passive strategies to enhance robustness. However, these approaches often rely on pre-defined assumptions about adversarial tactics, limiting their adaptability in dynamic 3D settings. To address these challenges, we introduce Reinforced Embodied Active Defense (Rein-EAD), a proactive defense framework that leverages adaptive exploration and interaction with the environment to improve perception robustness in 3D adversarial contexts. By implementing a multi-step objective that balances immediate prediction accuracy with predictive entropy minimization, Rein-EAD optimizes defense strategies over a multi-step horizon. Additionally, Rein-EAD involves an uncertainty-oriented reward-shaping mechanism that facilitates efficient policy updates, thereby reducing computational overhead and supporting real-world applicability without the need for differentiable environments. Comprehensive experiments validate the effectiveness of Rein-EAD, demonstrating a substantial reduction in attack success rates while preserving standard accuracy across diverse tasks. Notably, Rein-EAD exhibits robust generalization to unseen and adaptive attacks, making it suitable for real-world complex tasks, including 3D object classification, face recognition and autonomous driving.



## **4. Revisiting Physically Realizable Adversarial Object Attack against LiDAR-based Detection: Clarifying Problem Formulation and Experimental Protocols**

cs.CV

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18457v1) [paper-pdf](http://arxiv.org/pdf/2507.18457v1)

**Authors**: Luo Cheng, Hanwei Zhang, Lijun Zhang, Holger Hermanns

**Abstract**: Adversarial robustness in LiDAR-based 3D object detection is a critical research area due to its widespread application in real-world scenarios. While many digital attacks manipulate point clouds or meshes, they often lack physical realizability, limiting their practical impact. Physical adversarial object attacks remain underexplored and suffer from poor reproducibility due to inconsistent setups and hardware differences. To address this, we propose a device-agnostic, standardized framework that abstracts key elements of physical adversarial object attacks, supports diverse methods, and provides open-source code with benchmarking protocols in simulation and real-world settings. Our framework enables fair comparison, accelerates research, and is validated by successfully transferring simulated attacks to a physical LiDAR system. Beyond the framework, we offer insights into factors influencing attack success and advance understanding of adversarial robustness in real-world LiDAR perception.



## **5. On Reconstructing Training Data From Bayesian Posteriors and Trained Models**

stat.ML

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18372v1) [paper-pdf](http://arxiv.org/pdf/2507.18372v1)

**Authors**: George Wynne

**Abstract**: Publicly releasing the specification of a model with its trained parameters means an adversary can attempt to reconstruct information about the training data via training data reconstruction attacks, a major vulnerability of modern machine learning methods. This paper makes three primary contributions: establishing a mathematical framework to express the problem, characterising the features of the training data that are vulnerable via a maximum mean discrepancy equivalance and outlining a score matching framework for reconstructing data in both Bayesian and non-Bayesian models, the former is a first in the literature.



## **6. Data Transmission over a Bosonic Arbitrarily Varying Quantum Channel**

quant-ph

8 pages, no figures

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18259v1) [paper-pdf](http://arxiv.org/pdf/2507.18259v1)

**Authors**: Janis Nötzel, Florian Seitz

**Abstract**: Arbitrarily varying channels offer a powerful framework for analyzing the robustness of quantum communication systems, especially for classical-quantum models, where the analysis displays strengths or weaknesses of specific signal constellations under generic attacks. In this work, we provide a coding theorem for a large class of practically relevant arbitrarily varying channel models. Namely, we give an explicit capacity formula for the lossy bosonic channel subject to semi-classical attacks, where an adversary injects semi-classical states into the transmission line. Mathematically, this is modeled via a beam-splitter setup, with transmitter and jammer controlling different input ports and the receiver observing one output port. We show how a recently conjectured new quantum entropy power inequality relates to our capacity formula.



## **7. Safeguarding RAG Pipelines with GMTP: A Gradient-based Masked Token Probability Method for Poisoned Document Detection**

cs.CL

18 pages, accepted to ACL Findings 2025

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18202v1) [paper-pdf](http://arxiv.org/pdf/2507.18202v1)

**Authors**: San Kim, Jonghwi Kim, Yejin Jeon, Gary Geunbae Lee

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by providing external knowledge for accurate and up-to-date responses. However, this reliance on external sources exposes a security risk, attackers can inject poisoned documents into the knowledge base to steer the generation process toward harmful or misleading outputs. In this paper, we propose Gradient-based Masked Token Probability (GMTP), a novel defense method to detect and filter out adversarially crafted documents. Specifically, GMTP identifies high-impact tokens by examining gradients of the retriever's similarity function. These key tokens are then masked, and their probabilities are checked via a Masked Language Model (MLM). Since injected tokens typically exhibit markedly low masked-token probabilities, this enables GMTP to easily detect malicious documents and achieve high-precision filtering. Experiments demonstrate that GMTP is able to eliminate over 90% of poisoned content while retaining relevant documents, thus maintaining robust retrieval and generation performance across diverse datasets and adversarial settings.



## **8. Policy Disruption in Reinforcement Learning:Adversarial Attack with Large Language Models and Critical State Identification**

cs.LG

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18113v1) [paper-pdf](http://arxiv.org/pdf/2507.18113v1)

**Authors**: Junyong Jiang, Buwei Tian, Chenxing Xu, Songze Li, Lu Dong

**Abstract**: Reinforcement learning (RL) has achieved remarkable success in fields like robotics and autonomous driving, but adversarial attacks designed to mislead RL systems remain challenging. Existing approaches often rely on modifying the environment or policy, limiting their practicality. This paper proposes an adversarial attack method in which existing agents in the environment guide the target policy to output suboptimal actions without altering the environment. We propose a reward iteration optimization framework that leverages large language models (LLMs) to generate adversarial rewards explicitly tailored to the vulnerabilities of the target agent, thereby enhancing the effectiveness of inducing the target agent toward suboptimal decision-making. Additionally, a critical state identification algorithm is designed to pinpoint the target agent's most vulnerable states, where suboptimal behavior from the victim leads to significant degradation in overall performance. Experimental results in diverse environments demonstrate the superiority of our method over existing approaches.



## **9. RECALLED: An Unbounded Resource Consumption Attack on Large Vision-Language Models**

cs.CR

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18053v1) [paper-pdf](http://arxiv.org/pdf/2507.18053v1)

**Authors**: Haoran Gao, Yuanhe Zhang, Zhenhong Zhou, Lei Jiang, Fanyu Meng, Yujia Xiao, Kun Wang, Yang Liu, Junlan Feng

**Abstract**: Resource Consumption Attacks (RCAs) have emerged as a significant threat to the deployment of Large Language Models (LLMs). With the integration of vision modalities, additional attack vectors exacerbate the risk of RCAs in large vision-language models (LVLMs). However, existing red-teaming studies have largely overlooked visual inputs as a potential attack surface, resulting in insufficient mitigation strategies against RCAs in LVLMs. To address this gap, we propose RECALLED (\textbf{RE}source \textbf{C}onsumption \textbf{A}ttack on \textbf{L}arge Vision-\textbf{L}anguag\textbf{E} Mo\textbf{D}els), the first approach for exploiting visual modalities to trigger unbounded RCAs red-teaming. First, we present \textit{Vision Guided Optimization}, a fine-grained pixel-level optimization, to obtain \textit{Output Recall} adversarial perturbations, which can induce repeating output. Then, we inject the perturbations into visual inputs, triggering unbounded generations to achieve the goal of RCAs. Additionally, we introduce \textit{Multi-Objective Parallel Losses} to generate universal attack templates and resolve optimization conflicts when intending to implement parallel attacks. Empirical results demonstrate that RECALLED increases service response latency by over 26 $\uparrow$, resulting in an additional 20\% increase in GPU utilization and memory consumption. Our study exposes security vulnerabilities in LVLMs and establishes a red-teaming framework that can facilitate future defense development against RCAs.



## **10. Your ATs to Ts: MITRE ATT&CK Attack Technique to P-SSCRM Task Mapping**

cs.SE

Mapping generated from: arXiv:2503.12192

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18037v1) [paper-pdf](http://arxiv.org/pdf/2507.18037v1)

**Authors**: Sivana Hamer, Jacob Bowen, Md Nazmul Haque, Chris Madden, Laurie Williams

**Abstract**: The MITRE Adversarial Tactics, Techniques and Common Knowledge (MITRE ATT&CK) Attack Technique to Proactive Software Supply Chain Risk Management Framework (P-SSCRM) Task mapping described in this document helps software organizations to determine how different tasks mitigate the attack techniques of software supply chain attacks. The mapping was created through four independent strategies to find agreed-upon mappings. Because each P-SSCRM task is mapped to one or more tasks from the 10 frameworks, the mapping we provide is also a mapping between MITRE ATT&CK and other prominent government and industry frameworks.



## **11. Evaluating the Performance of AI Text Detectors, Few-Shot and Chain-of-Thought Prompting Using DeepSeek Generated Text**

cs.CL

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.17944v1) [paper-pdf](http://arxiv.org/pdf/2507.17944v1)

**Authors**: Hulayyil Alshammari, Praveen Rao

**Abstract**: Large language models (LLMs) have rapidly transformed the creation of written materials. LLMs have led to questions about writing integrity, thereby driving the creation of artificial intelligence (AI) detection technologies. Adversarial attacks, such as standard and humanized paraphrasing, inhibit detectors' ability to detect machine-generated text. Previous studies have mainly focused on ChatGPT and other well-known LLMs and have shown varying accuracy across detectors. However, there is a clear gap in the literature about DeepSeek, a recently published LLM. Therefore, in this work, we investigate whether six generally accessible AI detection tools -- AI Text Classifier, Content Detector AI, Copyleaks, QuillBot, GPT-2, and GPTZero -- can consistently recognize text generated by DeepSeek. The detectors were exposed to the aforementioned adversarial attacks. We also considered DeepSeek as a detector by performing few-shot prompting and chain-of-thought reasoning (CoT) for classifying AI and human-written text. We collected 49 human-authored question-answer pairs from before the LLM era and generated matching responses using DeepSeek-v3, producing 49 AI-generated samples. Then, we applied adversarial techniques such as paraphrasing and humanizing to add 196 more samples. These were used to challenge detector robustness and assess accuracy impact. While QuillBot and Copyleaks showed near-perfect performance on original and paraphrased DeepSeek text, others -- particularly AI Text Classifier and GPT-2 -- showed inconsistent results. The most effective attack was humanization, reducing accuracy to 71% for Copyleaks, 58% for QuillBot, and 52% for GPTZero. Few-shot and CoT prompting showed high accuracy, with the best five-shot result misclassifying only one of 49 samples (AI recall 96%, human recall 100%).



## **12. Bob's Confetti: Phonetic Memorization Attacks in Music and Video Generation**

cs.SD

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.17937v1) [paper-pdf](http://arxiv.org/pdf/2507.17937v1)

**Authors**: Jaechul Roh, Zachary Novack, Yuefeng Peng, Niloofar Mireshghallah, Taylor Berg-Kirkpatrick, Amir Houmansadr

**Abstract**: Lyrics-to-Song (LS2) generation models promise end-to-end music synthesis from text, yet their vulnerability to training data memorization remains underexplored. We introduce Adversarial PhoneTic Prompting (APT), a novel attack where lyrics are semantically altered while preserving their acoustic structure through homophonic substitutions (e.g., Eminem's famous "mom's spaghetti" $\rightarrow$ "Bob's confetti"). Despite these distortions, we uncover a powerful form of sub-lexical memorization: models like SUNO and YuE regenerate outputs strikingly similar to known training content, achieving high similarity across audio-domain metrics, including CLAP, AudioJudge, and CoverID. This vulnerability persists across multiple languages and genres. More surprisingly, we discover that phoneme-altered lyrics alone can trigger visual memorization in text-to-video models. When prompted with phonetically modified lyrics from Lose Yourself, Veo 3 reconstructs visual elements from the original music video -- including character appearance and scene composition -- despite no visual cues in the prompt. We term this phenomenon phonetic-to-visual regurgitation. Together, these findings expose a critical vulnerability in transcript-conditioned multimodal generation: phonetic prompting alone can unlock memorized audiovisual content, raising urgent questions about copyright, safety, and content provenance in modern generative systems. Example generations are available on our demo page (jrohsc.github.io/music_attack/).



## **13. From Seed to Harvest: Augmenting Human Creativity with AI for Red-teaming Text-to-Image Models**

cs.LG

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.17922v1) [paper-pdf](http://arxiv.org/pdf/2507.17922v1)

**Authors**: Jessica Quaye, Charvi Rastogi, Alicia Parrish, Oana Inel, Minsuk Kahng, Lora Aroyo, Vijay Janapa Reddi

**Abstract**: Text-to-image (T2I) models have become prevalent across numerous applications, making their robust evaluation against adversarial attacks a critical priority. Continuous access to new and challenging adversarial prompts across diverse domains is essential for stress-testing these models for resilience against novel attacks from multiple vectors. Current techniques for generating such prompts are either entirely authored by humans or synthetically generated. On the one hand, datasets of human-crafted adversarial prompts are often too small in size and imbalanced in their cultural and contextual representation. On the other hand, datasets of synthetically-generated prompts achieve scale, but typically lack the realistic nuances and creative adversarial strategies found in human-crafted prompts. To combine the strengths of both human and machine approaches, we propose Seed2Harvest, a hybrid red-teaming method for guided expansion of culturally diverse, human-crafted adversarial prompt seeds. The resulting prompts preserve the characteristics and attack patterns of human prompts while maintaining comparable average attack success rates (0.31 NudeNet, 0.36 SD NSFW, 0.12 Q16). Our expanded dataset achieves substantially higher diversity with 535 unique geographic locations and a Shannon entropy of 7.48, compared to 58 locations and 5.28 entropy in the original dataset. Our work demonstrates the importance of human-machine collaboration in leveraging human creativity and machine computational capacity to achieve comprehensive, scalable red-teaming for continuous T2I model safety evaluation.



## **14. Trusted Data Fusion, Multi-Agent Autonomy, Autonomous Vehicles**

eess.SY

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.17875v1) [paper-pdf](http://arxiv.org/pdf/2507.17875v1)

**Authors**: R. Spencer Hallyburton, Miroslav Pajic

**Abstract**: Multi-agent collaboration enhances situational awareness in intelligence, surveillance, and reconnaissance (ISR) missions. Ad hoc networks of unmanned aerial vehicles (UAVs) allow for real-time data sharing, but they face security challenges due to their decentralized nature, making them vulnerable to cyber-physical attacks. This paper introduces a trust-based framework for assured sensor fusion in distributed multi-agent networks, utilizing a hidden Markov model (HMM)-based approach to estimate the trustworthiness of agents and their provided information in a decentralized fashion. Trust-informed data fusion prioritizes fusing data from reliable sources, enhancing resilience and accuracy in contested environments. To evaluate the assured sensor fusion under attacks on system/mission sensing, we present a novel multi-agent aerial dataset built from the Unreal Engine simulator. We demonstrate through case studies improved ISR performance and an ability to detect malicious actors in adversarial settings.



## **15. Weak-to-Strong Jailbreaking on Large Language Models**

cs.CL

ICML 2025

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2401.17256v5) [paper-pdf](http://arxiv.org/pdf/2401.17256v5)

**Authors**: Xuandong Zhao, Xianjun Yang, Tianyu Pang, Chao Du, Lei Li, Yu-Xiang Wang, William Yang Wang

**Abstract**: Large language models (LLMs) are vulnerable to jailbreak attacks - resulting in harmful, unethical, or biased text generations. However, existing jailbreaking methods are computationally costly. In this paper, we propose the weak-to-strong jailbreaking attack, an efficient inference time attack for aligned LLMs to produce harmful text. Our key intuition is based on the observation that jailbroken and aligned models only differ in their initial decoding distributions. The weak-to-strong attack's key technical insight is using two smaller models (a safe and an unsafe one) to adversarially modify a significantly larger safe model's decoding probabilities. We evaluate the weak-to-strong attack on 5 diverse open-source LLMs from 3 organizations. The results show our method can increase the misalignment rate to over 99% on two datasets with just one forward pass per example. Our study exposes an urgent safety issue that needs to be addressed when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong



## **16. Constructing Optimal Noise Channels for Enhanced Robustness in Quantum Machine Learning**

quant-ph

QML technical track at IEEE QCE 2025

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2404.16417v2) [paper-pdf](http://arxiv.org/pdf/2404.16417v2)

**Authors**: David Winderl, Nicola Franco, Jeanette Miriam Lorenz

**Abstract**: With the rapid advancement of Quantum Machine Learning (QML), the critical need to enhance security measures against adversarial attacks and protect QML models becomes increasingly evident. In this work, we outline the connection between quantum noise channels and differential privacy (DP), by constructing a family of noise channels which are inherently $\epsilon$-DP: $(\alpha, \gamma)$-channels. Through this approach, we successfully replicate the $\epsilon$-DP bounds observed for depolarizing and random rotation channels, thereby affirming the broad generality of our framework. Additionally, we use a semi-definite program to construct an optimally robust channel. In a small-scale experimental evaluation, we demonstrate the benefits of using our optimal noise channel over depolarizing noise, particularly in enhancing adversarial accuracy. Moreover, we assess how the variables $\alpha$ and $\gamma$ affect the certifiable robustness and investigate how different encoding methods impact the classifier's robustness.



## **17. Boosting Ray Search Procedure of Hard-label Attacks with Transfer-based Priors**

cs.CV

Published at ICLR 2025 (Spotlight paper)

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.17577v1) [paper-pdf](http://arxiv.org/pdf/2507.17577v1)

**Authors**: Chen Ma, Xinjie Xu, Shuyu Cheng, Qi Xuan

**Abstract**: One of the most practical and challenging types of black-box adversarial attacks is the hard-label attack, where only the top-1 predicted label is available. One effective approach is to search for the optimal ray direction from the benign image that minimizes the $\ell_p$-norm distance to the adversarial region. The unique advantage of this approach is that it transforms the hard-label attack into a continuous optimization problem. The objective function value is the ray's radius, which can be obtained via binary search at a high query cost. Existing methods use a "sign trick" in gradient estimation to reduce the number of queries. In this paper, we theoretically analyze the quality of this gradient estimation and propose a novel prior-guided approach to improve ray search efficiency both theoretically and empirically. Specifically, we utilize the transfer-based priors from surrogate models, and our gradient estimators appropriately integrate them by approximating the projection of the true gradient onto the subspace spanned by these priors and random directions, in a query-efficient manner. We theoretically derive the expected cosine similarities between the obtained gradient estimators and the true gradient, and demonstrate the improvement achieved by incorporating priors. Extensive experiments on the ImageNet and CIFAR-10 datasets show that our approach significantly outperforms 11 state-of-the-art methods in terms of query efficiency.



## **18. An h-space Based Adversarial Attack for Protection Against Few-shot Personalization**

cs.CV

32 pages, 15 figures. Accepted by ACM Multimedia 2025

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.17554v1) [paper-pdf](http://arxiv.org/pdf/2507.17554v1)

**Authors**: Xide Xu, Sandesh Kamath, Muhammad Atif Butt, Bogdan Raducanu

**Abstract**: The versatility of diffusion models in generating customized images from few samples raises significant privacy concerns, particularly regarding unauthorized modifications of private content. This concerning issue has renewed the efforts in developing protection mechanisms based on adversarial attacks, which generate effective perturbations to poison diffusion models. Our work is motivated by the observation that these models exhibit a high degree of abstraction within their semantic latent space (`h-space'), which encodes critical high-level features for generating coherent and meaningful content. In this paper, we propose a novel anti-customization approach, called HAAD (h-space based Adversarial Attack for Diffusion models), that leverages adversarial attacks to craft perturbations based on the h-space that can efficiently degrade the image generation process. Building upon HAAD, we further introduce a more efficient variant, HAAD-KV, that constructs perturbations solely based on the KV parameters of the h-space. This strategy offers a stronger protection, that is computationally less expensive. Despite their simplicity, our methods outperform state-of-the-art adversarial attacks, highlighting their effectiveness.



## **19. MEF: A Capability-Aware Multi-Encryption Framework for Evaluating Vulnerabilities in Black-Box Large Language Models**

cs.CL

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2505.23404v4) [paper-pdf](http://arxiv.org/pdf/2505.23404v4)

**Authors**: Mingyu Yu, Wei Wang, Yanjie Wei, Sujuan Qin, Fei Gao, Wenmin Li

**Abstract**: Recent advancements in adversarial jailbreak attacks have exposed critical vulnerabilities in Large Language Models (LLMs), enabling the circumvention of alignment safeguards through increasingly sophisticated prompt manipulations. Based on our experiments, we found that the effectiveness of jailbreak strategies is influenced by the comprehension ability of the attacked LLM. Building on this insight, we propose a capability-aware Multi-Encryption Framework (MEF) for evaluating vulnerabilities in black-box LLMs. Specifically, MEF first categorizes the comprehension ability level of the LLM, then applies different strategies accordingly: For models with limited comprehension ability, MEF adopts the Fu+En1 strategy, which integrates layered semantic mutations with an encryption technique, more effectively contributing to evasion of the LLM's defenses at the input and inference stages. For models with strong comprehension ability, MEF uses a more complex Fu+En1+En2 strategy, in which additional dual-ended encryption techniques are applied to the LLM's responses, further contributing to evasion of the LLM's defenses at the output stage. Experimental results demonstrate the effectiveness of our approach, achieving attack success rates of 98.9% on GPT-4o (29 May 2025 release) and 99.8% on GPT-4.1 (8 July 2025 release). Our work contributes to a deeper understanding of the vulnerabilities in current LLM alignment mechanisms.



## **20. Explicit Vulnerability Generation with LLMs: An Investigation Beyond Adversarial Attacks**

cs.SE

Accepted to ICSME 2025

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.10054v2) [paper-pdf](http://arxiv.org/pdf/2507.10054v2)

**Authors**: Emir Bosnak, Sahand Moslemi, Mayasah Lami, Anil Koyuncu

**Abstract**: Large Language Models (LLMs) are increasingly used as code assistants, yet their behavior when explicitly asked to generate insecure code remains poorly understood. While prior research has focused on unintended vulnerabilities, this study examines a more direct threat: open-source LLMs generating vulnerable code when prompted. We propose a dual experimental design: (1) Dynamic Prompting, which systematically varies vulnerability type, user persona, and prompt phrasing across structured templates; and (2) Reverse Prompting, which derives natural-language prompts from real vulnerable code samples. We evaluate three open-source 7B-parameter models (Qwen2, Mistral, Gemma) using static analysis to assess both the presence and correctness of generated vulnerabilities. Our results show that all models frequently generate the requested vulnerabilities, though with significant performance differences. Gemma achieves the highest correctness for memory vulnerabilities under Dynamic Prompting (e.g., 98.6% for buffer overflows), while Qwen2 demonstrates the most balanced performance across all tasks. We find that professional personas (e.g., "DevOps Engineer") consistently elicit higher success rates than student personas, and that the effectiveness of direct versus indirect phrasing is inverted depending on the prompting strategy. Vulnerability reproduction accuracy follows a non-linear pattern with code complexity, peaking in a moderate range. Our findings expose how LLMs' reliance on pattern recall over semantic reasoning creates significant blind spots in their safety alignments, particularly for requests framed as plausible professional tasks.



## **21. Optimizing Privacy-Utility Trade-off in Decentralized Learning with Generalized Correlated Noise**

cs.LG

6 pages, 5 figures, accepted at IEEE ITW 2025

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2501.14644v2) [paper-pdf](http://arxiv.org/pdf/2501.14644v2)

**Authors**: Angelo Rodio, Zheng Chen, Erik G. Larsson

**Abstract**: Decentralized learning enables distributed agents to collaboratively train a shared machine learning model without a central server, through local computation and peer-to-peer communication. Although each agent retains its dataset locally, sharing local models can still expose private information about the local training datasets to adversaries. To mitigate privacy attacks, a common strategy is to inject random artificial noise at each agent before exchanging local models between neighbors. However, this often leads to utility degradation due to the negative effects of cumulated artificial noise on the learning algorithm. In this work, we introduce CorN-DSGD, a novel covariance-based framework for generating correlated privacy noise across agents, which unifies several state-of-the-art methods as special cases. By leveraging network topology and mixing weights, CorN-DSGD optimizes the noise covariance to achieve network-wide noise cancellation. Experimental results show that CorN-DSGD cancels more noise than existing pairwise correlation schemes, improving model performance under formal privacy guarantees.



## **22. Restricted Boltzmann machine as a probabilistic Enigma**

cond-mat.stat-mech

7 pages, 4 figures

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2507.17236v1) [paper-pdf](http://arxiv.org/pdf/2507.17236v1)

**Authors**: Bin Chen, Weichao Yu

**Abstract**: We theoretically propose a symmetric encryption scheme based on Restricted Boltzmann Machines that functions as a probabilistic Enigma device, encoding information in the marginal distributions of visible states while utilizing bias permutations as cryptographic keys. Theoretical analysis reveals significant advantages including factorial key space growth through permutation matrices, excellent diffusion properties, and computational complexity rooted in sharp P-complete problems that resist quantum attacks. Compatible with emerging probabilistic computing hardware, the scheme establishes an asymmetric computational barrier where legitimate users decrypt efficiently while adversaries face exponential costs. This framework unlocks probabilistic computers' potential for cryptographic systems, offering an emerging encryption paradigm between classical and quantum regimes for post-quantum security.



## **23. Gungnir: Exploiting Stylistic Features in Images for Backdoor Attacks on Diffusion Models**

cs.CV

**SubmitDate**: 2025-07-23    [abs](http://arxiv.org/abs/2502.20650v4) [paper-pdf](http://arxiv.org/pdf/2502.20650v4)

**Authors**: Yu Pan, Jiahao Chen, Bingrong Dai, Lin Wang, Yi Du, Jiao Liu

**Abstract**: In recent years, Diffusion Models (DMs) have demonstrated significant advances in the field of image generation. However, according to current research, DMs are vulnerable to backdoor attacks, which allow attackers to control the model's output by inputting data containing covert triggers, such as a specific visual patch or phrase. Existing defense strategies are well equipped to thwart such attacks through backdoor detection and trigger inversion because previous attack methods are constrained by limited input spaces and low-dimensional triggers. For example, visual triggers are easily observed by defenders, text-based or attention-based triggers are more susceptible to neural network detection. To explore more possibilities of backdoor attack in DMs, we propose Gungnir, a novel method that enables attackers to activate the backdoor in DMs through style triggers within input images. Our approach proposes using stylistic features as triggers for the first time and implements backdoor attacks successfully in image-to-image tasks by introducing Reconstructing-Adversarial Noise (RAN) and Short-Term Timesteps-Retention (STTR). Our technique generates trigger-embedded images that are perceptually indistinguishable from clean images, thus bypassing both manual inspection and automated detection neural networks. Experiments demonstrate that Gungnir can easily bypass existing defense methods. Among existing DM defense frameworks, our approach achieves a 0 backdoor detection rate (BDR). Our codes are available at https://github.com/paoche11/Gungnir.



## **24. Advancing Robustness in Deep Reinforcement Learning with an Ensemble Defense Approach**

cs.LG

6 pages, 4 figures, 2 tables

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.17070v1) [paper-pdf](http://arxiv.org/pdf/2507.17070v1)

**Authors**: Adithya Mohan, Dominik Rößle, Daniel Cremers, Torsten Schön

**Abstract**: Recent advancements in Deep Reinforcement Learning (DRL) have demonstrated its applicability across various domains, including robotics, healthcare, energy optimization, and autonomous driving. However, a critical question remains: How robust are DRL models when exposed to adversarial attacks? While existing defense mechanisms such as adversarial training and distillation enhance the resilience of DRL models, there remains a significant research gap regarding the integration of multiple defenses in autonomous driving scenarios specifically. This paper addresses this gap by proposing a novel ensemble-based defense architecture to mitigate adversarial attacks in autonomous driving. Our evaluation demonstrates that the proposed architecture significantly enhances the robustness of DRL models. Compared to the baseline under FGSM attacks, our ensemble method improves the mean reward from 5.87 to 18.38 (over 213% increase) and reduces the mean collision rate from 0.50 to 0.09 (an 82% decrease) in the highway scenario and merge scenario, outperforming all standalone defense strategies.



## **25. When LLMs Copy to Think: Uncovering Copy-Guided Attacks in Reasoning LLMs**

cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16773v1) [paper-pdf](http://arxiv.org/pdf/2507.16773v1)

**Authors**: Yue Li, Xiao Li, Hao Wu, Yue Zhang, Fengyuan Xu, Xiuzhen Cheng, Sheng Zhong

**Abstract**: Large Language Models (LLMs) have become integral to automated code analysis, enabling tasks such as vulnerability detection and code comprehension. However, their integration introduces novel attack surfaces. In this paper, we identify and investigate a new class of prompt-based attacks, termed Copy-Guided Attacks (CGA), which exploit the inherent copying tendencies of reasoning-capable LLMs. By injecting carefully crafted triggers into external code snippets, adversaries can induce the model to replicate malicious content during inference. This behavior enables two classes of vulnerabilities: inference length manipulation, where the model generates abnormally short or excessively long reasoning traces; and inference result manipulation, where the model produces misleading or incorrect conclusions. We formalize CGA as an optimization problem and propose a gradient-based approach to synthesize effective triggers. Empirical evaluation on state-of-the-art reasoning LLMs shows that CGA reliably induces infinite loops, premature termination, false refusals, and semantic distortions in code analysis tasks. While highly effective in targeted settings, we observe challenges in generalizing CGA across diverse prompts due to computational constraints, posing an open question for future research. Our findings expose a critical yet underexplored vulnerability in LLM-powered development pipelines and call for urgent advances in prompt-level defense mechanisms.



## **26. ShadowCode: Towards (Automatic) External Prompt Injection Attack against Code LLMs**

cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2407.09164v6) [paper-pdf](http://arxiv.org/pdf/2407.09164v6)

**Authors**: Yuchen Yang, Yiming Li, Hongwei Yao, Bingrun Yang, Yiling He, Tianwei Zhang, Dacheng Tao, Zhan Qin

**Abstract**: Recent advancements have led to the widespread adoption of code-oriented large language models (Code LLMs) for programming tasks. Despite their success in deployment, their security research is left far behind. This paper introduces a new attack paradigm: (automatic) external prompt injection against Code LLMs, where attackers generate concise, non-functional induced perturbations and inject them within a victim's code context. These induced perturbations can be disseminated through commonly used dependencies (e.g., packages or RAG's knowledge base), manipulating Code LLMs to achieve malicious objectives during the code completion process. Compared to existing attacks, this method is more realistic and threatening: it does not necessitate control over the model's training process, unlike backdoor attacks, and can achieve specific malicious objectives that are challenging for adversarial attacks. Furthermore, we propose ShadowCode, a simple yet effective method that automatically generates induced perturbations based on code simulation to achieve effective and stealthy external prompt injection. ShadowCode designs its perturbation optimization objectives by simulating realistic code contexts and employs a greedy optimization approach with two enhancement modules: forward reasoning enhancement and keyword-based perturbation design. We evaluate our method across 13 distinct malicious objectives, generating 31 threat cases spanning three popular programming languages. Our results demonstrate that ShadowCode successfully attacks three representative open-source Code LLMs (achieving up to a 97.9% attack success rate) and two mainstream commercial Code LLM-integrated applications (with over 90% attack success rate) across all threat cases, using only a 12-token non-functional induced perturbation. The code is available at https://github.com/LianPing-cyber/ShadowCodeEPI.



## **27. The Cost of Compression: Tight Quadratic Black-Box Attacks on Sketches for $\ell_2$ Norm Estimation**

cs.LG

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16345v1) [paper-pdf](http://arxiv.org/pdf/2507.16345v1)

**Authors**: Sara Ahmadian, Edith Cohen, Uri Stemmer

**Abstract**: Dimensionality reduction via linear sketching is a powerful and widely used technique, but it is known to be vulnerable to adversarial inputs. We study the black-box adversarial setting, where a fixed, hidden sketching matrix A in $R^{k X n}$ maps high-dimensional vectors v $\in R^n$ to lower-dimensional sketches A v in $R^k$, and an adversary can query the system to obtain approximate ell2-norm estimates that are computed from the sketch.   We present a universal, nonadaptive attack that, using tilde(O)($k^2$) queries, either causes a failure in norm estimation or constructs an adversarial input on which the optimal estimator for the query distribution (used by the attack) fails. The attack is completely agnostic to the sketching matrix and to the estimator: It applies to any linear sketch and any query responder, including those that are randomized, adaptive, or tailored to the query distribution.   Our lower bound construction tightly matches the known upper bounds of tilde(Omega)($k^2$), achieved by specialized estimators for Johnson Lindenstrauss transforms and AMS sketches. Beyond sketching, our results uncover structural parallels to adversarial attacks in image classification, highlighting fundamental vulnerabilities of compressed representations.



## **28. Talking Like a Phisher: LLM-Based Attacks on Voice Phishing Classifiers**

cs.CR

Accepted by EAI ICDF2C 2025

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16291v1) [paper-pdf](http://arxiv.org/pdf/2507.16291v1)

**Authors**: Wenhao Li, Selvakumar Manickam, Yung-wey Chong, Shankar Karuppayah

**Abstract**: Voice phishing (vishing) remains a persistent threat in cybersecurity, exploiting human trust through persuasive speech. While machine learning (ML)-based classifiers have shown promise in detecting malicious call transcripts, they remain vulnerable to adversarial manipulations that preserve semantic content. In this study, we explore a novel attack vector where large language models (LLMs) are leveraged to generate adversarial vishing transcripts that evade detection while maintaining deceptive intent. We construct a systematic attack pipeline that employs prompt engineering and semantic obfuscation to transform real-world vishing scripts using four commercial LLMs. The generated transcripts are evaluated against multiple ML classifiers trained on a real-world Korean vishing dataset (KorCCViD) with statistical testing. Our experiments reveal that LLM-generated transcripts are both practically and statistically effective against ML-based classifiers. In particular, transcripts crafted by GPT-4o significantly reduce classifier accuracy (by up to 30.96%) while maintaining high semantic similarity, as measured by BERTScore. Moreover, these attacks are both time-efficient and cost-effective, with average generation times under 9 seconds and negligible financial cost per query. The results underscore the pressing need for more resilient vishing detection frameworks and highlight the imperative for LLM providers to enforce stronger safeguards against prompt misuse in adversarial social engineering contexts.



## **29. Ownership Verification of DNN Models Using White-Box Adversarial Attacks with Specified Probability Manipulation**

cs.LG

Accepted to EUSIPCO 2025

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2505.17579v2) [paper-pdf](http://arxiv.org/pdf/2505.17579v2)

**Authors**: Teruki Sano, Minoru Kuribayashi, Masao Sakai, Shuji Ishobe, Eisuke Koizumi

**Abstract**: In this paper, we propose a novel framework for ownership verification of deep neural network (DNN) models for image classification tasks. It allows verification of model identity by both the rightful owner and third party without presenting the original model. We assume a gray-box scenario where an unauthorized user owns a model that is illegally copied from the original model, provides services in a cloud environment, and the user throws images and receives the classification results as a probability distribution of output classes. The framework applies a white-box adversarial attack to align the output probability of a specific class to a designated value. Due to the knowledge of original model, it enables the owner to generate such adversarial examples. We propose a simple but effective adversarial attack method based on the iterative Fast Gradient Sign Method (FGSM) by introducing control parameters. Experimental results confirm the effectiveness of the identification of DNN models using adversarial attack.



## **30. Pulling Back the Curtain: Unsupervised Adversarial Detection via Contrastive Auxiliary Networks**

cs.CV

Accepted at SafeMM-AI @ ICCV 2025

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2502.09110v2) [paper-pdf](http://arxiv.org/pdf/2502.09110v2)

**Authors**: Eylon Mizrahi, Raz Lapid, Moshe Sipper

**Abstract**: Deep learning models are widely employed in safety-critical applications yet remain susceptible to adversarial attacks -- imperceptible perturbations that can significantly degrade model performance. Conventional defense mechanisms predominantly focus on either enhancing model robustness or detecting adversarial inputs independently. In this work, we propose an Unsupervised adversarial detection via Contrastive Auxiliary Networks (U-CAN) to uncover adversarial behavior within auxiliary feature representations, without the need for adversarial examples. U-CAN is embedded within selected intermediate layers of the target model. These auxiliary networks, comprising projection layers and ArcFace-based linear layers, refine feature representations to more effectively distinguish between benign and adversarial inputs. Comprehensive experiments across multiple datasets (CIFAR-10, Mammals, and a subset of ImageNet) and architectures (ResNet-50, VGG-16, and ViT) demonstrate that our method surpasses existing unsupervised adversarial detection techniques, achieving superior F1 scores against four distinct attack methods. The proposed framework provides a scalable and effective solution for enhancing the security and reliability of deep learning systems.



## **31. Quality Text, Robust Vision: The Role of Language in Enhancing Visual Robustness of Vision-Language Models**

cs.CV

ACMMM 2025 Accepted

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16257v1) [paper-pdf](http://arxiv.org/pdf/2507.16257v1)

**Authors**: Futa Waseda, Saku Sugawara, Isao Echizen

**Abstract**: Defending pre-trained vision-language models (VLMs), such as CLIP, against adversarial attacks is crucial, as these models are widely used in diverse zero-shot tasks, including image classification. However, existing adversarial training (AT) methods for robust fine-tuning largely overlook the role of language in enhancing visual robustness. Specifically, (1) supervised AT methods rely on short texts (e.g., class labels) to generate adversarial perturbations, leading to overfitting to object classes in the training data, and (2) unsupervised AT avoids this overfitting but remains suboptimal against practical text-guided adversarial attacks due to its lack of semantic guidance. To address these limitations, we propose Quality Text-guided Adversarial Fine-Tuning (QT-AFT), which leverages high-quality captions during training to guide adversarial examples away from diverse semantics present in images. This enables the visual encoder to robustly recognize a broader range of image features even under adversarial noise, thereby enhancing robustness across diverse downstream tasks. QT-AFT overcomes the key weaknesses of prior methods -- overfitting in supervised AT and lack of semantic awareness in unsupervised AT -- achieving state-of-the-art zero-shot adversarial robustness and clean accuracy, evaluated across 16 zero-shot datasets. Furthermore, our comprehensive study uncovers several key insights into the role of language in enhancing vision robustness; for example, describing object properties in addition to object names further enhances zero-shot robustness. Our findings point to an urgent direction for future work -- centering high-quality linguistic supervision in robust visual representation learning.



## **32. Pulse-Level Simulation of Crosstalk Attacks on Superconducting Quantum Hardware**

quant-ph

This paper has been accepted to the Security, Privacy, and Resilience  Workshop at IEEE Quantum Week (QCE 2025) and will appear in the workshop  proceedings

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16181v1) [paper-pdf](http://arxiv.org/pdf/2507.16181v1)

**Authors**: Syed Emad Uddin Shubha, Tasnuva Farheen

**Abstract**: Hardware crosstalk in multi-tenant superconducting quantum computers poses a severe security threat, allowing adversaries to induce targeted errors across tenant boundaries by injecting carefully engineered pulses. We present a simulation-based study of active crosstalk attacks at the pulse level, analyzing how adversarial control of pulse timing, shape, amplitude, and coupling can disrupt a victim's computation. Our framework models the time-dependent dynamics of a three-qubit system in the rotating frame, capturing both always-on couplings and injected drive pulses. We examine two attack strategies: attacker-first (pulse before victim operation) and victim-first (pulse after), and systematically identify the pulse and coupling configurations that cause the largest logical errors. Protocol-level experiments on quantum coin flip and XOR classification circuits show that some protocols are highly vulnerable to these attacks, while others remain robust. Based on these findings, we discuss practical methods for detection and mitigation to improve security in quantum cloud platforms.



## **33. Attacking interpretable NLP systems**

cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16164v1) [paper-pdf](http://arxiv.org/pdf/2507.16164v1)

**Authors**: Eldor Abdukhamidov, Tamer Abuhmed, Joanna C. S. Santos, Mohammed Abuhamad

**Abstract**: Studies have shown that machine learning systems are vulnerable to adversarial examples in theory and practice. Where previous attacks have focused mainly on visual models that exploit the difference between human and machine perception, text-based models have also fallen victim to these attacks. However, these attacks often fail to maintain the semantic meaning of the text and similarity. This paper introduces AdvChar, a black-box attack on Interpretable Natural Language Processing Systems, designed to mislead the classifier while keeping the interpretation similar to benign inputs, thus exploiting trust in system transparency. AdvChar achieves this by making less noticeable modifications to text input, forcing the deep learning classifier to make incorrect predictions and preserve the original interpretation. We use an interpretation-focused scoring approach to determine the most critical tokens that, when changed, can cause the classifier to misclassify the input. We apply simple character-level modifications to measure the importance of tokens, minimizing the difference between the original and new text while generating adversarial interpretations similar to benign ones. We thoroughly evaluated AdvChar by testing it against seven NLP models and three interpretation models using benchmark datasets for the classification task. Our experiments show that AdvChar can significantly reduce the prediction accuracy of current deep learning models by altering just two characters on average in input samples.



## **34. DP2Guard: A Lightweight and Byzantine-Robust Privacy-Preserving Federated Learning Scheme for Industrial IoT**

cs.CR

**SubmitDate**: 2025-07-22    [abs](http://arxiv.org/abs/2507.16134v1) [paper-pdf](http://arxiv.org/pdf/2507.16134v1)

**Authors**: Baofu Han, Bing Li, Yining Qi, Raja Jurdak, Kaibin Huang, Chau Yuen

**Abstract**: Privacy-Preserving Federated Learning (PPFL) has emerged as a secure distributed Machine Learning (ML) paradigm that aggregates locally trained gradients without exposing raw data. To defend against model poisoning threats, several robustness-enhanced PPFL schemes have been proposed by integrating anomaly detection. Nevertheless, they still face two major challenges: (1) the reliance on heavyweight encryption techniques results in substantial communication and computation overhead; and (2) single-strategy defense mechanisms often fail to provide sufficient robustness against adaptive adversaries. To overcome these challenges, we propose DP2Guard, a lightweight PPFL framework that enhances both privacy and robustness. DP2Guard leverages a lightweight gradient masking mechanism to replace costly cryptographic operations while ensuring the privacy of local gradients. A hybrid defense strategy is proposed, which extracts gradient features using singular value decomposition and cosine similarity, and applies a clustering algorithm to effectively identify malicious gradients. Additionally, DP2Guard adopts a trust score-based adaptive aggregation scheme that adjusts client weights according to historical behavior, while blockchain records aggregated results and trust scores to ensure tamper-proof and auditable training. Extensive experiments conducted on two public datasets demonstrate that DP2Guard effectively defends against four advanced poisoning attacks while ensuring privacy with reduced communication and computation costs.



## **35. DP-TLDM: Differentially Private Tabular Latent Diffusion Model**

cs.LG

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2403.07842v2) [paper-pdf](http://arxiv.org/pdf/2403.07842v2)

**Authors**: Chaoyi Zhu, Jiayi Tang, Juan F. Pérez, Marten van Dijk, Lydia Y. Chen

**Abstract**: Synthetic data from generative models emerges as the privacy-preserving data sharing solution. Such a synthetic data set shall resemble the original data without revealing identifiable private information. Till date, the prior focus on limited types of tabular synthesizers and a small number of privacy attacks, particularly on Generative Adversarial Networks, and overlooks membership inference attacks and defense strategies, i.e., differential privacy. Motivated by the conundrum of keeping high data quality and low privacy risk of synthetic data tables, we propose DPTLDM, Differentially Private Tabular Latent Diffusion Model, which is composed of an autoencoder network to encode the tabular data and a latent diffusion model to synthesize the latent tables. Following the emerging f-DP framework, we apply DP-SGD to train the auto-encoder in combination with batch clipping and use the separation value as the privacy metric to better capture the privacy gain from DP algorithms. Our empirical evaluation demonstrates that DPTLDM is capable of achieving a meaningful theoretical privacy guarantee while also significantly enhancing the utility of synthetic data. Specifically, compared to other DP-protected tabular generative models, DPTLDM improves the synthetic quality by an average of 35% in data resemblance, 15% in the utility for downstream tasks, and 50% in data discriminability, all while preserving a comparable level of privacy risk.



## **36. Erasing Conceptual Knowledge from Language Models**

cs.CL

Project Page: https://elm.baulab.info

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2410.02760v3) [paper-pdf](http://arxiv.org/pdf/2410.02760v3)

**Authors**: Rohit Gandikota, Sheridan Feucht, Samuel Marks, David Bau

**Abstract**: In this work, we introduce Erasure of Language Memory (ELM), a principled approach to concept-level unlearning that operates by matching distributions defined by the model's own introspective classification capabilities. Our key insight is that effective unlearning should leverage the model's ability to evaluate its own knowledge, using the language model itself as a classifier to identify and reduce the likelihood of generating content related to undesired concepts. ELM applies this framework to create targeted low-rank updates that reduce generation probabilities for concept-specific content while preserving the model's broader capabilities. We demonstrate ELM's efficacy on biosecurity, cybersecurity, and literary domain erasure tasks. Comparative evaluation reveals that ELM-modified models achieve near-random performance on assessments targeting erased concepts, while simultaneously preserving generation coherence, maintaining benchmark performance on unrelated tasks, and exhibiting strong robustness to adversarial attacks. Our code, data, and trained models are available at https://elm.baulab.info



## **37. Disrupting Semantic and Abstract Features for Better Adversarial Transferability**

cs.CV

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.16052v1) [paper-pdf](http://arxiv.org/pdf/2507.16052v1)

**Authors**: Yuyang Luo, Xiaosen Wang, Zhijin Ge, Yingzhe He

**Abstract**: Adversarial examples pose significant threats to deep neural networks (DNNs), and their property of transferability in the black-box setting has led to the emergence of transfer-based attacks, making it feasible to target real-world applications employing DNNs. Among them, feature-level attacks, where intermediate features are perturbed based on feature importance weight matrix computed from transformed images, have gained popularity. In this work, we find that existing feature-level attacks primarily manipulate the semantic information to derive the weight matrix. Inspired by several works that find CNNs tend to focus more on high-frequency components (a.k.a. abstract features, e.g., texture, edge, etc.), we validate that transforming images in the high-frequency space also improves transferability. Based on this finding, we propose a balanced approach called Semantic and Abstract FEatures disRuption (SAFER). Specifically, SAFER conducts BLOCKMIX on the input image and SELF-MIX on the frequency spectrum when computing the weight matrix to highlight crucial features. By using such a weight matrix, we can direct the attacker to disrupt both semantic and abstract features, leading to improved transferability. Extensive experiments on the ImageNet dataset also demonstrate the effectiveness of our method in boosting adversarial transferability.



## **38. Does More Inference-Time Compute Really Help Robustness?**

cs.AI

Preprint

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.15974v1) [paper-pdf](http://arxiv.org/pdf/2507.15974v1)

**Authors**: Tong Wu, Chong Xiang, Jiachen T. Wang, Weichen Yu, Chawin Sitawarin, Vikash Sehwag, Prateek Mittal

**Abstract**: Recently, Zaremba et al. demonstrated that increasing inference-time computation improves robustness in large proprietary reasoning LLMs. In this paper, we first show that smaller-scale, open-source models (e.g., DeepSeek R1, Qwen3, Phi-reasoning) can also benefit from inference-time scaling using a simple budget forcing strategy. More importantly, we reveal and critically examine an implicit assumption in prior work: intermediate reasoning steps are hidden from adversaries. By relaxing this assumption, we identify an important security risk, intuitively motivated and empirically verified as an inverse scaling law: if intermediate reasoning steps become explicitly accessible, increased inference-time computation consistently reduces model robustness. Finally, we discuss practical scenarios where models with hidden reasoning chains are still vulnerable to attacks, such as models with tool-integrated reasoning and advanced reasoning extraction attacks. Our findings collectively demonstrate that the robustness benefits of inference-time scaling depend heavily on the adversarial setting and deployment context. We urge practitioners to carefully weigh these subtle trade-offs before applying inference-time scaling in security-sensitive, real-world applications.



## **39. Hedge Funds on a Swamp: Analyzing Patterns, Vulnerabilities, and Defense Measures in Blockchain Bridges**

cs.ET

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.06156v3) [paper-pdf](http://arxiv.org/pdf/2507.06156v3)

**Authors**: Poupak Azad, Jiahua Xu, Yebo Feng, Preston Strowbridge, Cuneyt Akcora

**Abstract**: Blockchain bridges have become essential infrastructure for enabling interoperability across different blockchain networks, with more than $24B monthly bridge transaction volume. However, their growing adoption has been accompanied by a disproportionate rise in security breaches, making them the single largest source of financial loss in Web3. For cross-chain ecosystems to be robust and sustainable, it is essential to understand and address these vulnerabilities. In this study, we present a comprehensive systematization of blockchain bridge design and security. We define three bridge security priors, formalize the architectural structure of 13 prominent bridges, and identify 23 attack vectors grounded in real-world blockchain exploits. Using this foundation, we evaluate 43 representative attack scenarios and introduce a layered threat model that captures security failures across source chain, off-chain, and destination chain components.   Our analysis at the static code and transaction network levels reveals recurring design flaws, particularly in access control, validator trust assumptions, and verification logic, and identifies key patterns in adversarial behavior based on transaction-level traces. To support future development, we propose a decision framework for bridge architecture design, along with defense mechanisms such as layered validation and circuit breakers. This work provides a data-driven foundation for evaluating bridge security and lays the groundwork for standardizing resilient cross-chain infrastructure.



## **40. Sparsification Under Siege: Defending Against Poisoning Attacks in Communication-Efficient Federated Learning**

cs.CR

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2505.01454v4) [paper-pdf](http://arxiv.org/pdf/2505.01454v4)

**Authors**: Zhiyong Jin, Runhua Xu, Chao Li, Yizhong Liu, Jianxin Li, James Joshi

**Abstract**: Federated Learning (FL) enables collaborative model training across distributed clients while preserving data privacy, yet it faces significant challenges in communication efficiency and vulnerability to poisoning attacks. While sparsification techniques mitigate communication overhead by transmitting only critical model parameters, they inadvertently amplify security risks: adversarial clients can exploit sparse updates to evade detection and degrade model performance. Existing defense mechanisms, designed for standard FL communication scenarios, are ineffective in addressing these vulnerabilities within sparsified FL. To bridge this gap, we propose FLARE, a novel federated learning framework that integrates sparse index mask inspection and model update sign similarity analysis to detect and mitigate poisoning attacks in sparsified FL. Extensive experiments across multiple datasets and adversarial scenarios demonstrate that FLARE significantly outperforms existing defense strategies, effectively securing sparsified FL against poisoning attacks while maintaining communication efficiency.



## **41. Multi-Stage Prompt Inference Attacks on Enterprise LLM Systems**

cs.CR

26 pages

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.15613v1) [paper-pdf](http://arxiv.org/pdf/2507.15613v1)

**Authors**: Andrii Balashov, Olena Ponomarova, Xiaohua Zhai

**Abstract**: Large Language Models (LLMs) deployed in enterprise settings (e.g., as Microsoft 365 Copilot) face novel security challenges. One critical threat is prompt inference attacks: adversaries chain together seemingly benign prompts to gradually extract confidential data. In this paper, we present a comprehensive study of multi-stage prompt inference attacks in an enterprise LLM context. We simulate realistic attack scenarios where an attacker uses mild-mannered queries and indirect prompt injections to exploit an LLM integrated with private corporate data. We develop a formal threat model for these multi-turn inference attacks and analyze them using probability theory, optimization frameworks, and information-theoretic leakage bounds. The attacks are shown to reliably exfiltrate sensitive information from the LLM's context (e.g., internal SharePoint documents or emails), even when standard safety measures are in place.   We propose and evaluate defenses to counter such attacks, including statistical anomaly detection, fine-grained access control, prompt sanitization techniques, and architectural modifications to LLM deployment. Each defense is supported by mathematical analysis or experimental simulation. For example, we derive bounds on information leakage under differential privacy-based training and demonstrate an anomaly detection method that flags multi-turn attacks with high AUC. We also introduce an approach called "spotlighting" that uses input transformations to isolate untrusted prompt content, reducing attack success by an order of magnitude. Finally, we provide a formal proof of concept and empirical validation for a combined defense-in-depth strategy. Our work highlights that securing LLMs in enterprise settings requires moving beyond single-turn prompt filtering toward a holistic, multi-stage perspective on both attacks and defenses.



## **42. Derivative-Free Diffusion Manifold-Constrained Gradient for Unified XAI**

cs.CV

CVPR 2025 (poster), 19 pages, 5 figures

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2411.15265v2) [paper-pdf](http://arxiv.org/pdf/2411.15265v2)

**Authors**: Won Jun Kim, Hyungjin Chung, Jaemin Kim, Sangmin Lee, Byeongsu Sim, Jong Chul Ye

**Abstract**: Gradient-based methods are a prototypical family of explainability techniques, especially for image-based models. Nonetheless, they have several shortcomings in that they (1) require white-box access to models, (2) are vulnerable to adversarial attacks, and (3) produce attributions that lie off the image manifold, leading to explanations that are not actually faithful to the model and do not align well with human perception. To overcome these challenges, we introduce Derivative-Free Diffusion Manifold-Constrainted Gradients (FreeMCG), a novel method that serves as an improved basis for explainability of a given neural network than the traditional gradient. Specifically, by leveraging ensemble Kalman filters and diffusion models, we derive a derivative-free approximation of the model's gradient projected onto the data manifold, requiring access only to the model's outputs. We demonstrate the effectiveness of FreeMCG by applying it to both counterfactual generation and feature attribution, which have traditionally been treated as distinct tasks. Through comprehensive evaluation on both tasks, counterfactual explanation and feature attribution, we show that our method yields state-of-the-art results while preserving the essential properties expected of XAI tools.



## **43. Transfer Attack for Bad and Good: Explain and Boost Adversarial Transferability across Multimodal Large Language Models**

cs.CV

This paper is accepted by ACM MM 2025

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2405.20090v5) [paper-pdf](http://arxiv.org/pdf/2405.20090v5)

**Authors**: Hao Cheng, Erjia Xiao, Jiayan Yang, Jinhao Duan, Yichi Wang, Jiahang Cao, Qiang Zhang, Le Yang, Kaidi Xu, Jindong Gu, Renjing Xu

**Abstract**: Multimodal Large Language Models (MLLMs) demonstrate exceptional performance in cross-modality interaction, yet they also suffer adversarial vulnerabilities. In particular, the transferability of adversarial examples remains an ongoing challenge. In this paper, we specifically analyze the manifestation of adversarial transferability among MLLMs and identify the key factors that influence this characteristic. We discover that the transferability of MLLMs exists in cross-LLM scenarios with the same vision encoder and indicate \underline{\textit{two key Factors}} that may influence transferability. We provide two semantic-level data augmentation methods, Adding Image Patch (AIP) and Typography Augment Transferability Method (TATM), which boost the transferability of adversarial examples across MLLMs. To explore the potential impact in the real world, we utilize two tasks that can have both negative and positive societal impacts: \ding{182} Harmful Content Insertion and \ding{183} Information Protection.



## **44. Scaling Decentralized Learning with FLock**

cs.LG

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2507.15349v1) [paper-pdf](http://arxiv.org/pdf/2507.15349v1)

**Authors**: Zehua Cheng, Rui Sun, Jiahao Sun, Yike Guo

**Abstract**: Fine-tuning the large language models (LLMs) are prevented by the deficiency of centralized control and the massive computing and communication overhead on the decentralized schemes. While the typical standard federated learning (FL) supports data privacy, the central server requirement creates a single point of attack and vulnerability to poisoning attacks. Generalizing the result in this direction to 70B-parameter models in the heterogeneous, trustless environments has turned out to be a huge, yet unbroken bottleneck. This paper introduces FLock, a decentralized framework for secure and efficient collaborative LLM fine-tuning. Integrating a blockchain-based trust layer with economic incentives, FLock replaces the central aggregator with a secure, auditable protocol for cooperation among untrusted parties. We present the first empirical validation of fine-tuning a 70B LLM in a secure, multi-domain, decentralized setting. Our experiments show the FLock framework defends against backdoor poisoning attacks that compromise standard FL optimizers and fosters synergistic knowledge transfer. The resulting models show a >68% reduction in adversarial attack success rates. The global model also demonstrates superior cross-domain generalization, outperforming models trained in isolation on their own specialized data.



## **45. Cats Confuse Reasoning LLM: Query Agnostic Adversarial Triggers for Reasoning Models**

cs.CL

Accepted to CoLM 2025

**SubmitDate**: 2025-07-21    [abs](http://arxiv.org/abs/2503.01781v2) [paper-pdf](http://arxiv.org/pdf/2503.01781v2)

**Authors**: Meghana Rajeev, Rajkumar Ramamurthy, Prapti Trivedi, Vikas Yadav, Oluwanifemi Bamgbose, Sathwik Tejaswi Madhusudan, James Zou, Nazneen Rajani

**Abstract**: We investigate the robustness of reasoning models trained for step-by-step problem solving by introducing query-agnostic adversarial triggers - short, irrelevant text that, when appended to math problems, systematically mislead models to output incorrect answers without altering the problem's semantics. We propose CatAttack, an automated iterative attack pipeline for generating triggers on a weaker, less expensive proxy model (DeepSeek V3) and successfully transfer them to more advanced reasoning target models like DeepSeek R1 and DeepSeek R1-distilled-Qwen-32B, resulting in greater than 300% increase in the likelihood of the target model generating an incorrect answer. For example, appending, "Interesting fact: cats sleep most of their lives," to any math problem leads to more than doubling the chances of a model getting the answer wrong. Our findings highlight critical vulnerabilities in reasoning models, revealing that even state-of-the-art models remain susceptible to subtle adversarial inputs, raising security and reliability concerns. The CatAttack triggers dataset with model responses is available at https://huggingface.co/datasets/collinear-ai/cat-attack-adversarial-triggers.



## **46. Defective Convolutional Networks**

cs.CV

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/1911.08432v3) [paper-pdf](http://arxiv.org/pdf/1911.08432v3)

**Authors**: Tiange Luo, Tianle Cai, Mengxiao Zhang, Siyu Chen, Di He, Liwei Wang

**Abstract**: Robustness of convolutional neural networks (CNNs) has gained in importance on account of adversarial examples, i.e., inputs added as well-designed perturbations that are imperceptible to humans but can cause the model to predict incorrectly. Recent research suggests that the noises in adversarial examples break the textural structure, which eventually leads to wrong predictions. To mitigate the threat of such adversarial attacks, we propose defective convolutional networks that make predictions relying less on textural information but more on shape information by properly integrating defective convolutional layers into standard CNNs. The defective convolutional layers contain defective neurons whose activations are set to be a constant function. As defective neurons contain no information and are far different from standard neurons in its spatial neighborhood, the textural features cannot be accurately extracted, and so the model has to seek other features for classification, such as the shape. We show extensive evidence to justify our proposal and demonstrate that defective CNNs can defense against black-box attacks better than standard CNNs. In particular, they achieve state-of-the-art performance against transfer-based attacks without any adversarial training being applied.



## **47. ROBAD: Robust Adversary-aware Local-Global Attended Bad Actor Detection Sequential Model**

cs.LG

15 pages, 12 tables

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/2507.15067v1) [paper-pdf](http://arxiv.org/pdf/2507.15067v1)

**Authors**: Bing He, Mustaque Ahamad, Srijan Kumar

**Abstract**: Detecting bad actors is critical to ensure the safety and integrity of internet platforms. Several deep learning-based models have been developed to identify such users. These models should not only accurately detect bad actors, but also be robust against adversarial attacks that aim to evade detection. However, past deep learning-based detection models do not meet the robustness requirement because they are sensitive to even minor changes in the input sequence. To address this issue, we focus on (1) improving the model understanding capability and (2) enhancing the model knowledge such that the model can recognize potential input modifications when making predictions. To achieve these goals, we create a novel transformer-based classification model, called ROBAD (RObust adversary-aware local-global attended Bad Actor Detection model), which uses the sequence of user posts to generate user embedding to detect bad actors. Particularly, ROBAD first leverages the transformer encoder block to encode each post bidirectionally, thus building a post embedding to capture the local information at the post level. Next, it adopts the transformer decoder block to model the sequential pattern in the post embeddings by using the attention mechanism, which generates the sequence embedding to obtain the global information at the sequence level. Finally, to enrich the knowledge of the model, embeddings of modified sequences by mimicked attackers are fed into a contrastive-learning-enhanced classification layer for sequence prediction. In essence, by capturing the local and global information (i.e., the post and sequence information) and leveraging the mimicked behaviors of bad actors in training, ROBAD can be robust to adversarial attacks. Extensive experiments on Yelp and Wikipedia datasets show that ROBAD can effectively detect bad actors when under state-of-the-art adversarial attacks.



## **48. DeRAG: Black-box Adversarial Attacks on Multiple Retrieval-Augmented Generation Applications via Prompt Injection**

cs.AI

Accepted by KDD Workshop on Prompt Optimization 2025

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/2507.15042v1) [paper-pdf](http://arxiv.org/pdf/2507.15042v1)

**Authors**: Jerry Wang, Fang Yu

**Abstract**: Adversarial prompt attacks can significantly alter the reliability of Retrieval-Augmented Generation (RAG) systems by re-ranking them to produce incorrect outputs. In this paper, we present a novel method that applies Differential Evolution (DE) to optimize adversarial prompt suffixes for RAG-based question answering. Our approach is gradient-free, treating the RAG pipeline as a black box and evolving a population of candidate suffixes to maximize the retrieval rank of a targeted incorrect document to be closer to real world scenarios. We conducted experiments on the BEIR QA datasets to evaluate attack success at certain retrieval rank thresholds under multiple retrieving applications. Our results demonstrate that DE-based prompt optimization attains competitive (and in some cases higher) success rates compared to GGPP to dense retrievers and PRADA to sparse retrievers, while using only a small number of tokens (<=5 tokens) in the adversarial suffix. Furthermore, we introduce a readability-aware suffix construction strategy, validated by a statistically significant reduction in MLM negative log-likelihood with Welch's t-test. Through evaluations with a BERT-based adversarial suffix detector, we show that DE-generated suffixes evade detection, yielding near-chance detection accuracy.



## **49. Adversarial Destabilization Attacks to Direct Data-Driven Control**

eess.SY

15 pages

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/2507.14863v1) [paper-pdf](http://arxiv.org/pdf/2507.14863v1)

**Authors**: Hampei Sasahara

**Abstract**: This study investigates the vulnerability of direct data-driven control methods, specifically for the linear quadratic regulator problem, to adversarial perturbations in collected data used for controller synthesis. We consider stealthy attacks that subtly manipulate offline-collected data to destabilize the resulting closed-loop system while evading detection. To generate such perturbations, we propose the Directed Gradient Sign Method (DGSM) and its iterative variant (I-DGSM), adaptations of the fast gradient sign method originally developed for neural networks, which align perturbations with the gradient of the spectral radius of the closed-loop matrix to reduce stability. A key contribution is an efficient gradient computation technique based on implicit differentiation through the Karush-Kuhn-Tucker conditions of the underlying semidefinite program, enabling scalable and exact gradient evaluation without repeated optimization computations. To defend against these attacks, we propose two defense strategies: a regularization-based approach that enhances robustness by suppressing controller sensitivity to data perturbations and a robust data-driven control approach that guarantees closed-loop stability within bounded perturbation sets. Extensive numerical experiments on benchmark systems show that adversarial perturbations with magnitudes up to ten times smaller than random noise can destabilize controllers trained on corrupted data and that the proposed defense strategies effectively mitigate attack success rates while maintaining control performance. Additionally, we evaluate attack transferability under partial knowledge scenarios, highlighting the practical importance of protecting training data confidentiality.



## **50. Data-Plane Telemetry to Mitigate Long-Distance BGP Hijacks**

cs.NI

**SubmitDate**: 2025-07-20    [abs](http://arxiv.org/abs/2507.14842v1) [paper-pdf](http://arxiv.org/pdf/2507.14842v1)

**Authors**: Satadal Sengupta, Hyojoon Kim, Daniel Jubas, Maria Apostolaki, Jennifer Rexford

**Abstract**: Poor security of Internet routing enables adversaries to divert user data through unintended infrastructures (hijack). Of particular concern -- and the focus of this paper -- are cases where attackers reroute domestic traffic through foreign countries, exposing it to surveillance, bypassing legal privacy protections, and posing national security threats. Efforts to detect and mitigate such attacks have focused primarily on the control plane while data-plane signals remain largely overlooked. In particular, change in propagation delay caused by rerouting offers a promising signal: the change is unavoidable and the increased propagation delay is directly observable from the affected networks. In this paper, we explore the practicality of using delay variations for hijack detection, addressing two key questions: (1) What coverage can this provide, given its heavy dependence on the geolocations of the sender, receiver, and adversary? and (2) Can an always-on latency-based detection system be deployed without disrupting normal network operations? We observe that for 86% of victim-attacker country pairs in the world, mid-attack delays exceed pre-attack delays by at least 25% in real deployments, making delay-based hijack detection promising. To demonstrate practicality, we design HiDe, which reliably detects delay surges from long-distance hijacks at line rate. We measure HiDe's accuracy and false-positive rate on real-world data and validate it with ethically conducted hijacks.



