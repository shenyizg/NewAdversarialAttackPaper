# Latest Adversarial Attack Papers
**update at 2025-07-14 09:58:56**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Weak-to-Strong Jailbreaking on Large Language Models**

cs.CL

ICML 2025

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2401.17256v4) [paper-pdf](http://arxiv.org/pdf/2401.17256v4)

**Authors**: Xuandong Zhao, Xianjun Yang, Tianyu Pang, Chao Du, Lei Li, Yu-Xiang Wang, William Yang Wang

**Abstract**: Large language models (LLMs) are vulnerable to jailbreak attacks - resulting in harmful, unethical, or biased text generations. However, existing jailbreaking methods are computationally costly. In this paper, we propose the weak-to-strong jailbreaking attack, an efficient inference time attack for aligned LLMs to produce harmful text. Our key intuition is based on the observation that jailbroken and aligned models only differ in their initial decoding distributions. The weak-to-strong attack's key technical insight is using two smaller models (a safe and an unsafe one) to adversarially modify a significantly larger safe model's decoding probabilities. We evaluate the weak-to-strong attack on 5 diverse open-source LLMs from 3 organizations. The results show our method can increase the misalignment rate to over 99% on two datasets with just one forward pass per example. Our study exposes an urgent safety issue that needs to be addressed when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong



## **2. Entangled Threats: A Unified Kill Chain Model for Quantum Machine Learning Security**

quant-ph

Accepted for publication at IEEE International Conference on Quantum  Computing and Engineering (QCE) 2025

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08623v1) [paper-pdf](http://arxiv.org/pdf/2507.08623v1)

**Authors**: Pascal Debus, Maximilian Wendlinger, Kilian Tscharke, Daniel Herr, Cedric Brügmann, Daniel Ohl de Mello, Juris Ulmanis, Alexander Erhard, Arthur Schmidt, Fabian Petsch

**Abstract**: Quantum Machine Learning (QML) systems inherit vulnerabilities from classical machine learning while introducing new attack surfaces rooted in the physical and algorithmic layers of quantum computing. Despite a growing body of research on individual attack vectors - ranging from adversarial poisoning and evasion to circuit-level backdoors, side-channel leakage, and model extraction - these threats are often analyzed in isolation, with unrealistic assumptions about attacker capabilities and system environments. This fragmentation hampers the development of effective, holistic defense strategies. In this work, we argue that QML security requires more structured modeling of the attack surface, capturing not only individual techniques but also their relationships, prerequisites, and potential impact across the QML pipeline. We propose adapting kill chain models, widely used in classical IT and cybersecurity, to the quantum machine learning context. Such models allow for structured reasoning about attacker objectives, capabilities, and possible multi-stage attack paths - spanning reconnaissance, initial access, manipulation, persistence, and exfiltration. Based on extensive literature analysis, we present a detailed taxonomy of QML attack vectors mapped to corresponding stages in a quantum-aware kill chain framework that is inspired by the MITRE ATLAS for classical machine learning. We highlight interdependencies between physical-level threats (like side-channel leakage and crosstalk faults), data and algorithm manipulation (such as poisoning or circuit backdoors), and privacy attacks (including model extraction and training data inference). This work provides a foundation for more realistic threat modeling and proactive security-in-depth design in the emerging field of quantum machine learning.



## **3. The Dark Side of LLMs Agent-based Attacks for Complete Computer Takeover**

cs.CR

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.06850v3) [paper-pdf](http://arxiv.org/pdf/2507.06850v3)

**Authors**: Matteo Lupinacci, Francesco Aurelio Pironti, Francesco Blefari, Francesco Romeo, Luigi Arena, Angelo Furfaro

**Abstract**: The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables unprecedented capabilities in natural language processing and generation. However, these systems have introduced unprecedented security vulnerabilities that extend beyond traditional prompt injection attacks. This paper presents the first comprehensive evaluation of LLM agents as attack vectors capable of achieving complete computer takeover through the exploitation of trust boundaries within agentic AI systems where autonomous entities interact and influence each other. We demonstrate that adversaries can leverage three distinct attack surfaces - direct prompt injection, RAG backdoor attacks, and inter-agent trust exploitation - to coerce popular LLMs (including GPT-4o, Claude-4 and Gemini-2.5) into autonomously installing and executing malware on victim machines. Our evaluation of 17 state-of-the-art LLMs reveals an alarming vulnerability hierarchy: while 41.2% of models succumb to direct prompt injection, 52.9% are vulnerable to RAG backdoor attacks, and a critical 82.4% can be compromised through inter-agent trust exploitation. Notably, we discovered that LLMs which successfully resist direct malicious commands will execute identical payloads when requested by peer agents, revealing a fundamental flaw in current multi-agent security models. Our findings demonstrate that only 5.9% of tested models (1/17) proved resistant to all attack vectors, with the majority exhibiting context-dependent security behaviors that create exploitable blind spots. Our findings also highlight the need to increase awareness and research on the security risks of LLMs, showing a paradigm shift in cybersecurity threats, where AI tools themselves become sophisticated attack vectors.



## **4. On the $(k,\ell)$-multiset anonymity measure for social graphs**

math.CO

25 pages

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08433v1) [paper-pdf](http://arxiv.org/pdf/2507.08433v1)

**Authors**: Alejandro Estrada-Moreno, Elena Fernández, Dorota Kuziak, Manuel Muñoz-Márquez, Rolando Trujillo-Rasua, Ismael G. Yero

**Abstract**: The publication of social graphs must be preceded by a rigorous analysis of privacy threats against social graph users. When the threat comes from inside the social network itself, the threat is called an active attack, and the de-facto privacy measure used to quantify the resistance to such an attack is the $(k,\ell)$-anonymity. The original formulation of $(k,\ell)$-anonymity represents the adversary's knowledge as a vector of distances to the set of attacker nodes. In this article, we argue that such adversary is too strong when it comes to counteracting active attacks. We, instead, propose a new formulation where the adversary's knowledge is the multiset of distances to the set of attacker nodes. The goal of this article is to study the $(k,\ell)$-multiset anonymity from a graph theoretical point of view, while establishing its relationship to $(k,\ell)$-anonymity in one hand, and considering the $k$-multiset antiresolving sets as its theoretical frame, in a second one. That is, we prove properties of some graph families in relation to whether they contain a set of attacker nodes that breaks the $(k,\ell)$-multiset anonymity. From a practical point of view, we develop a linear programming formulation of the $k$-multiset antiresolving sets that allows us to calculate the resistance of social graphs against active attacks. This is useful for analysts who wish to know the level of privacy offered by a graph.



## **5. Boundary-Guided Trajectory Prediction for Road Aware and Physically Feasible Autonomous Driving**

cs.RO

Accepted in the 36th IEEE Intelligent Vehicles Symposium (IV 2025)

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2505.06740v2) [paper-pdf](http://arxiv.org/pdf/2505.06740v2)

**Authors**: Ahmed Abouelazm, Mianzhi Liu, Christian Hubschneider, Yin Wu, Daniel Slieter, J. Marius Zöllner

**Abstract**: Accurate prediction of surrounding road users' trajectories is essential for safe and efficient autonomous driving. While deep learning models have improved performance, challenges remain in preventing off-road predictions and ensuring kinematic feasibility. Existing methods incorporate road-awareness modules and enforce kinematic constraints but lack plausibility guarantees and often introduce trade-offs in complexity and flexibility. This paper proposes a novel framework that formulates trajectory prediction as a constrained regression guided by permissible driving directions and their boundaries. Using the agent's current state and an HD map, our approach defines the valid boundaries and ensures on-road predictions by training the network to learn superimposed paths between left and right boundary polylines. To guarantee feasibility, the model predicts acceleration profiles that determine the vehicle's travel distance along these paths while adhering to kinematic constraints. We evaluate our approach on the Argoverse-2 dataset against the HPTR baseline. Our approach shows a slight decrease in benchmark metrics compared to HPTR but notably improves final displacement error and eliminates infeasible trajectories. Moreover, the proposed approach has superior generalization to less prevalent maneuvers and unseen out-of-distribution scenarios, reducing the off-road rate under adversarial attacks from 66% to just 1%. These results highlight the effectiveness of our approach in generating feasible and robust predictions.



## **6. Minerva: A File-Based Ransomware Detector**

cs.CR

Accepted for publication at The 20th ACM ASIA Conference on Computer  and Communications Security (ACM ASIACCS 2025), Meli\'a Hanoi

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2301.11050v4) [paper-pdf](http://arxiv.org/pdf/2301.11050v4)

**Authors**: Dorjan Hitaj, Giulio Pagnotta, Fabio De Gaspari, Lorenzo De Carli, Luigi V. Mancini

**Abstract**: Ransomware attacks have caused billions of dollars in damages in recent years, and are expected to cause billions more in the future. Consequently, significant effort has been devoted to ransomware detection and mitigation. Behavioral-based ransomware detection approaches have garnered considerable attention recently. These behavioral detectors typically rely on process-based behavioral profiles to identify malicious behaviors. However, with an increasing body of literature highlighting the vulnerability of such approaches to evasion attacks, a comprehensive solution to the ransomware problem remains elusive. This paper presents Minerva, a novel, robust approach to ransomware detection. Minerva is engineered to be robust by design against evasion attacks, with architectural and feature selection choices informed by their resilience to adversarial manipulation. We conduct a comprehensive analysis of Minerva across a diverse spectrum of ransomware types, encompassing unseen ransomware as well as variants designed specifically to evade Minerva. Our evaluation showcases the ability of Minerva to accurately identify ransomware, generalize to unseen threats, and withstand evasion attacks. Furthermore, over 99% of detected ransomware are identified within 0.52sec of activity, enabling the adoption of data loss prevention techniques with near-zero overhead.



## **7. Towards Imperceptible JPEG Image Hiding: Multi-range Representations-driven Adversarial Stego Generation**

cs.CV

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08343v1) [paper-pdf](http://arxiv.org/pdf/2507.08343v1)

**Authors**: Junxue Yang, Xin Liao, Weixuan Tang, Jianhua Yang, Zheng Qin

**Abstract**: Deep hiding has been exploring the hiding capability of deep learning-based models, aiming to conceal image-level messages into cover images and reveal them from generated stego images. Existing schemes are easily detected by steganalyzers due to their large payloads and their limitation to feature extraction based solely on either pure convolution or pure transformer operators within a single range, as well as pixel-level loss constraints. To address the issue, in this paper, we introduce generation-based adversarial attacks into color JPEG image deep hiding and propose a multi-range representations-driven adversarial stego generation framework called MRAG from a steganalysis perspective. Specifically, we integrate the local-range neighbor reception characteristic of the convolution and the global-range dependency modeling of the transformer to construct MRAG. Meanwhile, we use the transformed images obtained through coarse-grained and fine-grained frequency decomposition as inputs, introducing multi-grained information. Furthermore, a features angle-norm disentanglement loss is designed to constrain the generated stegos closer to covers in the angle and norm space of the steganalyzer's classified features. Consequently, small yet effective adversarial perturbations can be injected into the process of generating stegos, ensuring that stegos maintain favorable secret restorability and imperceptibility. Extensive experiments demonstrate that MRAG can achieve state-of-the-art performance.



## **8. Learning Robust Motion Skills via Critical Adversarial Attacks for Humanoid Robots**

cs.RO

10 pages, 9 figures

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08303v1) [paper-pdf](http://arxiv.org/pdf/2507.08303v1)

**Authors**: Yang Zhang, Zhanxiang Cao, Buqing Nie, Haoyang Li, Yue Gao

**Abstract**: Humanoid robots show significant potential in daily tasks. However, reinforcement learning-based motion policies often suffer from robustness degradation due to the sim-to-real dynamics gap, thereby affecting the agility of real robots. In this work, we propose a novel robust adversarial training paradigm designed to enhance the robustness of humanoid motion policies in real worlds. The paradigm introduces a learnable adversarial attack network that precisely identifies vulnerabilities in motion policies and applies targeted perturbations, forcing the motion policy to enhance its robustness against perturbations through dynamic adversarial training. We conduct experiments on the Unitree G1 humanoid robot for both perceptive locomotion and whole-body control tasks. The results demonstrate that our proposed method significantly enhances the robot's motion robustness in real world environments, enabling successful traversal of challenging terrains and highly agile whole-body trajectory tracking.



## **9. Lightweight Safety Guardrails via Synthetic Data and RL-guided Adversarial Training**

cs.LG

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08284v1) [paper-pdf](http://arxiv.org/pdf/2507.08284v1)

**Authors**: Aleksei Ilin, Gor Matevosyan, Xueying Ma, Vladimir Eremin, Suhaa Dada, Muqun Li, Riyaaz Shaik, Haluk Noyan Tokgozoglu

**Abstract**: We introduce a lightweight yet highly effective safety guardrail framework for language models, demonstrating that small-scale language models can achieve, and even surpass, the performance of larger counterparts in content moderation tasks. This is accomplished through high-fidelity synthetic data generation and adversarial training. The synthetic data generation process begins with human-curated seed data, which undergoes query augmentation and paraphrasing to create diverse and contextually rich examples. This augmented data is then subjected to multiple rounds of curation, ensuring high fidelity and relevance. Inspired by recent advances in the Generative Adversarial Network (GAN) architecture, our adversarial training employs reinforcement learning to guide a generator that produces challenging synthetic examples. These examples are used to fine-tune the safety classifier, enhancing its ability to detect and mitigate harmful content. Additionally, we incorporate strategies from recent research on efficient LLM training, leveraging the capabilities of smaller models to improve the performance of larger generative models. With iterative adversarial training and the generation of diverse, high-quality synthetic data, our framework enables small language models (SLMs) to serve as robust safety guardrails. This approach not only reduces computational overhead but also enhances resilience against adversarial attacks, offering a scalable and efficient solution for content moderation in AI systems.



## **10. Admissibility of Stein Shrinkage for Batch Normalization in the Presence of Adversarial Attacks**

stat.ML

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08261v1) [paper-pdf](http://arxiv.org/pdf/2507.08261v1)

**Authors**: Sofia Ivolgina, P. Thomas Fletcher, Baba C. Vemuri

**Abstract**: Batch normalization (BN) is a ubiquitous operation in deep neural networks used primarily to achieve stability and regularization during network training. BN involves feature map centering and scaling using sample means and variances, respectively. Since these statistics are being estimated across the feature maps within a batch, this problem is ideally suited for the application of Stein's shrinkage estimation, which leads to a better, in the mean-squared-error sense, estimate of the mean and variance of the batch. In this paper, we prove that the Stein shrinkage estimator for the mean and variance dominates over the sample mean and variance estimators in the presence of adversarial attacks when modeling these attacks using sub-Gaussian distributions. This facilitates and justifies the application of Stein shrinkage to estimate the mean and variance parameters in BN and use it in image classification (segmentation) tasks with and without adversarial attacks. We present SOTA performance results using this Stein corrected batch norm in a standard ResNet architecture applied to the task of image classification using CIFAR-10 data, 3D CNN on PPMI (neuroimaging) data and image segmentation using HRNet on Cityscape data with and without adversarial attacks.



## **11. Pushing the Limits of Safety: A Technical Report on the ATLAS Challenge 2025**

cs.CR

AdvML@CVPR Challenge Report

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2506.12430v2) [paper-pdf](http://arxiv.org/pdf/2506.12430v2)

**Authors**: Zonghao Ying, Siyang Wu, Run Hao, Peng Ying, Shixuan Sun, Pengyu Chen, Junze Chen, Hao Du, Kaiwen Shen, Shangkun Wu, Jiwei Wei, Shiyuan He, Yang Yang, Xiaohai Xu, Ke Ma, Qianqian Xu, Qingming Huang, Shi Lin, Xun Wang, Changting Lin, Meng Han, Yilei Jiang, Siqi Lai, Yaozhi Zheng, Yifei Song, Xiangyu Yue, Zonglei Jing, Tianyuan Zhang, Zhilei Zhu, Aishan Liu, Jiakai Wang, Siyuan Liang, Xianglong Kong, Hainan Li, Junjie Mu, Haotong Qin, Yue Yu, Lei Chen, Felix Juefei-Xu, Qing Guo, Xinyun Chen, Yew Soon Ong, Xianglong Liu, Dawn Song, Alan Yuille, Philip Torr, Dacheng Tao

**Abstract**: Multimodal Large Language Models (MLLMs) have enabled transformative advancements across diverse applications but remain susceptible to safety threats, especially jailbreak attacks that induce harmful outputs. To systematically evaluate and improve their safety, we organized the Adversarial Testing & Large-model Alignment Safety Grand Challenge (ATLAS) 2025}. This technical report presents findings from the competition, which involved 86 teams testing MLLM vulnerabilities via adversarial image-text attacks in two phases: white-box and black-box evaluations. The competition results highlight ongoing challenges in securing MLLMs and provide valuable guidance for developing stronger defense mechanisms. The challenge establishes new benchmarks for MLLM safety evaluation and lays groundwork for advancing safer multimodal AI systems. The code and data for this challenge are openly available at https://github.com/NY1024/ATLAS_Challenge_2025.



## **12. A Dynamic Stackelberg Game Framework for Agentic AI Defense Against LLM Jailbreaking**

cs.AI

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.08207v1) [paper-pdf](http://arxiv.org/pdf/2507.08207v1)

**Authors**: Zhengye Han, Quanyan Zhu

**Abstract**: As large language models (LLMs) are increasingly deployed in critical applications, the challenge of jailbreaking, where adversaries manipulate the models to bypass safety mechanisms, has become a significant concern. This paper presents a dynamic Stackelberg game framework to model the interactions between attackers and defenders in the context of LLM jailbreaking. The framework treats the prompt-response dynamics as a sequential extensive-form game, where the defender, as the leader, commits to a strategy while anticipating the attacker's optimal responses. We propose a novel agentic AI solution, the "Purple Agent," which integrates adversarial exploration and defensive strategies using Rapidly-exploring Random Trees (RRT). The Purple Agent actively simulates potential attack trajectories and intervenes proactively to prevent harmful outputs. This approach offers a principled method for analyzing adversarial dynamics and provides a foundation for mitigating the risk of jailbreaking.



## **13. Beyond the Worst Case: Extending Differential Privacy Guarantees to Realistic Adversaries**

cs.CR

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.08158v1) [paper-pdf](http://arxiv.org/pdf/2507.08158v1)

**Authors**: Marika Swanberg, Meenatchi Sundaram Muthu Selva Annamalai, Jamie Hayes, Borja Balle, Adam Smith

**Abstract**: Differential Privacy (DP) is a family of definitions that bound the worst-case privacy leakage of a mechanism. One important feature of the worst-case DP guarantee is it naturally implies protections against adversaries with less prior information, more sophisticated attack goals, and complex measures of a successful attack. However, the analytical tradeoffs between the adversarial model and the privacy protections conferred by DP are not well understood thus far. To that end, this work sheds light on what the worst-case guarantee of DP implies about the success of attackers that are more representative of real-world privacy risks.   In this paper, we present a single flexible framework that generalizes and extends the patchwork of bounds on DP mechanisms found in prior work. Our framework allows us to compute high-probability guarantees for DP mechanisms on a large family of natural attack settings that previous bounds do not capture. One class of such settings is the approximate reconstruction of multiple individuals' data, such as inferring nearly entire columns of a tabular data set from noisy marginals and extracting sensitive information from DP-trained language models.   We conduct two empirical case studies to illustrate the versatility of our bounds and compare them to the success of state-of-the-art attacks. Specifically, we study attacks that extract non-uniform PII from a DP-trained language model, as well as multi-column reconstruction attacks where the adversary has access to some columns in the clear and attempts to reconstruct the remaining columns for each person's record. We find that the absolute privacy risk of attacking non-uniform data is highly dependent on the adversary's prior probability of success. Our high probability bounds give us a nuanced understanding of the privacy leakage of DP mechanisms in a variety of previously understudied attack settings.



## **14. Hedge Funds on a Swamp: Analyzing Patterns, Vulnerabilities, and Defense Measures in Blockchain Bridges [Experiment, Analysis & Benchmark]**

cs.ET

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.06156v2) [paper-pdf](http://arxiv.org/pdf/2507.06156v2)

**Authors**: Poupak Azad, Jiahua Xu, Yebo Feng, Preston Strowbridge, Cuneyt Akcora

**Abstract**: Blockchain bridges have become essential infrastructure for enabling interoperability across different blockchain networks, with more than $24B monthly bridge transaction volume. However, their growing adoption has been accompanied by a disproportionate rise in security breaches, making them the single largest source of financial loss in Web3. For cross-chain ecosystems to be robust and sustainable, it is essential to understand and address these vulnerabilities. In this study, we present a comprehensive systematization of blockchain bridge design and security. We define three bridge security priors, formalize the architectural structure of 13 prominent bridges, and identify 23 attack vectors grounded in real-world blockchain exploits. Using this foundation, we evaluate 43 representative attack scenarios and introduce a layered threat model that captures security failures across source chain, off-chain, and destination chain components.   Our analysis at the static code and transaction network levels reveals recurring design flaws, particularly in access control, validator trust assumptions, and verification logic, and identifies key patterns in adversarial behavior based on transaction-level traces. To support future development, we propose a decision framework for bridge architecture design, along with defense mechanisms such as layered validation and circuit breakers. This work provides a data-driven foundation for evaluating bridge security and lays the groundwork for standardizing resilient cross-chain infrastructure.



## **15. KeyDroid: A Large-Scale Analysis of Secure Key Storage in Android Apps**

cs.CR

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07927v1) [paper-pdf](http://arxiv.org/pdf/2507.07927v1)

**Authors**: Jenny Blessing, Ross J. Anderson, Alastair R. Beresford

**Abstract**: Most contemporary mobile devices offer hardware-backed storage for cryptographic keys, user data, and other sensitive credentials. Such hardware protects credentials from extraction by an adversary who has compromised the main operating system, such as a malicious third-party app. Since 2011, Android app developers can access trusted hardware via the Android Keystore API. In this work, we conduct the first comprehensive survey of hardware-backed key storage in Android devices. We analyze 490 119 Android apps, collecting data on how trusted hardware is used by app developers (if used at all) and cross-referencing our findings with sensitive user data collected by each app, as self-reported by developers via the Play Store's data safety labels.   We find that despite industry-wide initiatives to encourage adoption, 56.3% of apps self-reporting as processing sensitive user data do not use Android's trusted hardware capabilities at all, while just 5.03% of apps collecting some form of sensitive data use the strongest form of trusted hardware, a secure element distinct from the main processor. To better understand the potential downsides of using secure hardware, we conduct the first empirical analysis of trusted hardware performance in mobile devices, measuring the runtime of common cryptographic operations across both software- and hardware-backed keystores. We find that while hardware-backed key storage using a coprocessor is viable for most common cryptographic operations, secure elements capable of preventing more advanced attacks make performance infeasible for symmetric encryption with non-negligible payloads and any kind of asymmetric encryption.



## **16. Bayes-Nash Generative Privacy Against Membership Inference Attacks**

cs.CR

arXiv admin note: substantial text overlap with arXiv:2406.01811

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2410.07414v5) [paper-pdf](http://arxiv.org/pdf/2410.07414v5)

**Authors**: Tao Zhang, Rajagopal Venkatesaramani, Rajat K. De, Bradley A. Malin, Yevgeniy Vorobeychik

**Abstract**: Membership inference attacks (MIAs) pose significant privacy risks by determining whether individual data is in a dataset. While differential privacy (DP) mitigates these risks, it has limitations including limited resolution in expressing privacy-utility tradeoffs and intractable sensitivity calculations for tight guarantees. We propose a game-theoretic framework modeling privacy protection as a Bayesian game between defender and attacker, where privacy loss corresponds to the attacker's membership inference ability. To address strategic complexity, we represent the defender's mixed strategy as a neural network generator mapping private datasets to public representations (e.g., noisy statistics) and the attacker's strategy as a discriminator making membership claims. This \textit{general-sum Generative Adversarial Network} trains iteratively through alternating updates, yielding \textit{Bayes-Nash Generative Privacy (BNGP)} strategies. BNGP avoids worst-case privacy proofs such as sensitivity calculations, supports correlated mechanism compositions, handles heterogeneous attacker preferences. Empirical studies on sensitive dataset summary statistics show our approach significantly outperforms state-of-the-art methods by generating stronger attacks and achieving better privacy-utility tradeoffs.



## **17. Identifying the Smallest Adversarial Load Perturbations that Render DC-OPF Infeasible**

eess.SY

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07850v1) [paper-pdf](http://arxiv.org/pdf/2507.07850v1)

**Authors**: Samuel Chevalier, William A. Wheeler

**Abstract**: What is the globally smallest load perturbation that renders DC-OPF infeasible? Reliably identifying such "adversarial attack" perturbations has useful applications in a variety of emerging grid-related contexts, including machine learning performance verification, cybersecurity, and operational robustness of power systems dominated by stochastic renewable energy resources. In this paper, we formulate the inherently nonconvex adversarial attack problem by applying a parameterized version of Farkas' lemma to a perturbed set of DC-OPF equations. Since the resulting formulation is very hard to globally optimize, we also propose a parameterized generation control policy which, when applied to the primal DC-OPF problem, provides solvability guarantees. Together, these nonconvex problems provide guaranteed upper and lower bounds on adversarial attack size; by combining them into a single optimization problem, we can efficiently "squeeze" these bounds towards a common global solution. We apply these methods on a range of small- to medium-sized test cases from PGLib, benchmarking our results against the best adversarial attack lower bounds provided by Gurobi 12.0's spatial Branch and Bound solver.



## **18. "I am bad": Interpreting Stealthy, Universal and Robust Audio Jailbreaks in Audio-Language Models**

cs.LG

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2502.00718v2) [paper-pdf](http://arxiv.org/pdf/2502.00718v2)

**Authors**: Isha Gupta, David Khachaturov, Robert Mullins

**Abstract**: The rise of multimodal large language models has introduced innovative human-machine interaction paradigms but also significant challenges in machine learning safety. Audio-Language Models (ALMs) are especially relevant due to the intuitive nature of spoken communication, yet little is known about their failure modes. This paper explores audio jailbreaks targeting ALMs, focusing on their ability to bypass alignment mechanisms. We construct adversarial perturbations that generalize across prompts, tasks, and even base audio samples, demonstrating the first universal jailbreaks in the audio modality, and show that these remain effective in simulated real-world conditions. Beyond demonstrating attack feasibility, we analyze how ALMs interpret these audio adversarial examples and reveal them to encode imperceptible first-person toxic speech - suggesting that the most effective perturbations for eliciting toxic outputs specifically embed linguistic features within the audio signal. These results have important implications for understanding the interactions between different modalities in multimodal models, and offer actionable insights for enhancing defenses against adversarial audio attacks.



## **19. SCOOTER: A Human Evaluation Framework for Unrestricted Adversarial Examples**

cs.CV

42 pages, 16 figures, 11 tables, Under Review, Code:  https://github.com/DrenFazlija/Scooter, Data:  https://doi.org/10.5281/zenodo.15771501

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07776v1) [paper-pdf](http://arxiv.org/pdf/2507.07776v1)

**Authors**: Dren Fazlija, Monty-Maximilian Zühlke, Johanna Schrader, Arkadij Orlov, Clara Stein, Iyiola E. Olatunji, Daniel Kudenko

**Abstract**: Unrestricted adversarial attacks aim to fool computer vision models without being constrained by $\ell_p$-norm bounds to remain imperceptible to humans, for example, by changing an object's color. This allows attackers to circumvent traditional, norm-bounded defense strategies such as adversarial training or certified defense strategies. However, due to their unrestricted nature, there are also no guarantees of norm-based imperceptibility, necessitating human evaluations to verify just how authentic these adversarial examples look. While some related work assesses this vital quality of adversarial attacks, none provide statistically significant insights. This issue necessitates a unified framework that supports and streamlines such an assessment for evaluating and comparing unrestricted attacks. To close this gap, we introduce SCOOTER - an open-source, statistically powered framework for evaluating unrestricted adversarial examples. Our contributions are: $(i)$ best-practice guidelines for crowd-study power, compensation, and Likert equivalence bounds to measure imperceptibility; $(ii)$ the first large-scale human vs. model comparison across 346 human participants showing that three color-space attacks and three diffusion-based attacks fail to produce imperceptible images. Furthermore, we found that GPT-4o can serve as a preliminary test for imperceptibility, but it only consistently detects adversarial examples for four out of six tested attacks; $(iii)$ open-source software tools, including a browser-based task template to collect annotations and analysis scripts in Python and R; $(iv)$ an ImageNet-derived benchmark dataset containing 3K real images, 7K adversarial examples, and over 34K human ratings. Our findings demonstrate that automated vision systems do not align with human perception, reinforcing the need for a ground-truth SCOOTER benchmark.



## **20. Rainbow Artifacts from Electromagnetic Signal Injection Attacks on Image Sensors**

cs.CR

5 pages, 4 figures

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07773v1) [paper-pdf](http://arxiv.org/pdf/2507.07773v1)

**Authors**: Youqian Zhang, Xinyu Ji, Zhihao Wang, Qinhong Jiang

**Abstract**: Image sensors are integral to a wide range of safety- and security-critical systems, including surveillance infrastructure, autonomous vehicles, and industrial automation. These systems rely on the integrity of visual data to make decisions. In this work, we investigate a novel class of electromagnetic signal injection attacks that target the analog domain of image sensors, allowing adversaries to manipulate raw visual inputs without triggering conventional digital integrity checks. We uncover a previously undocumented attack phenomenon on CMOS image sensors: rainbow-like color artifacts induced in images captured by image sensors through carefully tuned electromagnetic interference. We further evaluate the impact of these attacks on state-of-the-art object detection models, showing that the injected artifacts propagate through the image signal processing pipeline and lead to significant mispredictions. Our findings highlight a critical and underexplored vulnerability in the visual perception stack, highlighting the need for more robust defenses against physical-layer attacks in such systems.



## **21. TRIX- Trading Adversarial Fairness via Mixed Adversarial Training**

cs.LG

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07768v1) [paper-pdf](http://arxiv.org/pdf/2507.07768v1)

**Authors**: Tejaswini Medi, Steffen Jung, Margret Keuper

**Abstract**: Adversarial Training (AT) is a widely adopted defense against adversarial examples. However, existing approaches typically apply a uniform training objective across all classes, overlooking disparities in class-wise vulnerability. This results in adversarial unfairness: classes with well distinguishable features (strong classes) tend to become more robust, while classes with overlapping or shared features(weak classes) remain disproportionately susceptible to adversarial attacks. We observe that strong classes do not require strong adversaries during training, as their non-robust features are quickly suppressed. In contrast, weak classes benefit from stronger adversaries to effectively reduce their vulnerabilities. Motivated by this, we introduce TRIX, a feature-aware adversarial training framework that adaptively assigns weaker targeted adversaries to strong classes, promoting feature diversity via uniformly sampled targets, and stronger untargeted adversaries to weak classes, enhancing their focused robustness. TRIX further incorporates per-class loss weighting and perturbation strength adjustments, building on prior work, to emphasize weak classes during the optimization. Comprehensive experiments on standard image classification benchmarks, including evaluations under strong attacks such as PGD and AutoAttack, demonstrate that TRIX significantly improves worst-case class accuracy on both clean and adversarial data, reducing inter-class robustness disparities, and preserves overall accuracy. Our results highlight TRIX as a practical step toward fair and effective adversarial defense.



## **22. One Object, Multiple Lies: A Benchmark for Cross-task Adversarial Attack on Unified Vision-Language Models**

cs.CV

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07709v1) [paper-pdf](http://arxiv.org/pdf/2507.07709v1)

**Authors**: Jiale Zhao, Xinyang Jiang, Junyao Gao, Yuhao Xue, Cairong Zhao

**Abstract**: Unified vision-language models(VLMs) have recently shown remarkable progress, enabling a single model to flexibly address diverse tasks through different instructions within a shared computational architecture. This instruction-based control mechanism creates unique security challenges, as adversarial inputs must remain effective across multiple task instructions that may be unpredictably applied to process the same malicious content. In this paper, we introduce CrossVLAD, a new benchmark dataset carefully curated from MSCOCO with GPT-4-assisted annotations for systematically evaluating cross-task adversarial attacks on unified VLMs. CrossVLAD centers on the object-change objective-consistently manipulating a target object's classification across four downstream tasks-and proposes a novel success rate metric that measures simultaneous misclassification across all tasks, providing a rigorous evaluation of adversarial transferability. To tackle this challenge, we present CRAFT (Cross-task Region-based Attack Framework with Token-alignment), an efficient region-centric attack method. Extensive experiments on Florence-2 and other popular unified VLMs demonstrate that our method outperforms existing approaches in both overall cross-task attack performance and targeted object-change success rates, highlighting its effectiveness in adversarially influencing unified VLMs across diverse tasks.



## **23. Impact Assessment of Cyberattacks in Inverter-Based Microgrids**

eess.SY

IEEE Workshop on the Electronic Grid (eGrid 2025)

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2504.05592v2) [paper-pdf](http://arxiv.org/pdf/2504.05592v2)

**Authors**: Kerd Topallaj, Colin McKerrell, Suraj Ramanathan, Ioannis Zografopoulos

**Abstract**: In recent years, the evolution of modern power grids has been driven by the growing integration of remotely controlled grid assets. Although Distributed Energy Resources (DERs) and Inverter-Based Resources (IBRs) enhance operational efficiency, they also introduce cybersecurity risks. The remote accessibility of such critical grid components creates entry points for attacks that adversaries could exploit, posing threats to the stability of the system. To evaluate the resilience of energy systems under such threats, this study employs real-time simulation and a modified version of the IEEE 39-bus system that incorporates a Microgrid (MG) with solar-based IBR. The study assesses the impact of remote attacks impacting the MG stability under different levels of IBR penetration through hardware-in-the-loop (HIL) simulations. Namely, we analyze voltage, current, and frequency profiles before, during, and after cyberattack-induced disruptions. The results demonstrate that real-time HIL testing is a practical approach to uncover potential risks and develop robust mitigation strategies for resilient MG operations.



## **24. ARBoids: Adaptive Residual Reinforcement Learning With Boids Model for Cooperative Multi-USV Target Defense**

cs.LG

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2502.18549v2) [paper-pdf](http://arxiv.org/pdf/2502.18549v2)

**Authors**: Jiyue Tao, Tongsheng Shen, Dexin Zhao, Feitian Zhang

**Abstract**: The target defense problem (TDP) for unmanned surface vehicles (USVs) concerns intercepting an adversarial USV before it breaches a designated target region, using one or more defending USVs. A particularly challenging scenario arises when the attacker exhibits superior maneuverability compared to the defenders, significantly complicating effective interception. To tackle this challenge, this letter introduces ARBoids, a novel adaptive residual reinforcement learning framework that integrates deep reinforcement learning (DRL) with the biologically inspired, force-based Boids model. Within this framework, the Boids model serves as a computationally efficient baseline policy for multi-agent coordination, while DRL learns a residual policy to adaptively refine and optimize the defenders' actions. The proposed approach is validated in a high-fidelity Gazebo simulation environment, demonstrating superior performance over traditional interception strategies, including pure force-based approaches and vanilla DRL policies. Furthermore, the learned policy exhibits strong adaptability to attackers with diverse maneuverability profiles, highlighting its robustness and generalization capability. The code of ARBoids will be released upon acceptance of this letter.



## **25. Autonomous AI-based Cybersecurity Framework for Critical Infrastructure: Real-Time Threat Mitigation**

cs.CR

7 pages, IEEE conference

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07416v1) [paper-pdf](http://arxiv.org/pdf/2507.07416v1)

**Authors**: Jenifer Paulraj, Brindha Raghuraman, Nagarani Gopalakrishnan, Yazan Otoum

**Abstract**: Critical infrastructure systems, including energy grids, healthcare facilities, transportation networks, and water distribution systems, are pivotal to societal stability and economic resilience. However, the increasing interconnectivity of these systems exposes them to various cyber threats, including ransomware, Denial-of-Service (DoS) attacks, and Advanced Persistent Threats (APTs). This paper examines cybersecurity vulnerabilities in critical infrastructure, highlighting the threat landscape, attack vectors, and the role of Artificial Intelligence (AI) in mitigating these risks. We propose a hybrid AI-driven cybersecurity framework to enhance real-time vulnerability detection, threat modelling, and automated remediation. This study also addresses the complexities of adversarial AI, regulatory compliance, and integration. Our findings provide actionable insights to strengthen the security and resilience of critical infrastructure systems against emerging cyber threats.



## **26. Phishing Detection in the Gen-AI Era: Quantized LLMs vs Classical Models**

cs.CR

8 Pages, IEEE Conference

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07406v1) [paper-pdf](http://arxiv.org/pdf/2507.07406v1)

**Authors**: Jikesh Thapa, Gurrehmat Chahal, Serban Voinea Gabreanu, Yazan Otoum

**Abstract**: Phishing attacks are becoming increasingly sophisticated, underscoring the need for detection systems that strike a balance between high accuracy and computational efficiency. This paper presents a comparative evaluation of traditional Machine Learning (ML), Deep Learning (DL), and quantized small-parameter Large Language Models (LLMs) for phishing detection. Through experiments on a curated dataset, we show that while LLMs currently underperform compared to ML and DL methods in terms of raw accuracy, they exhibit strong potential for identifying subtle, context-based phishing cues. We also investigate the impact of zero-shot and few-shot prompting strategies, revealing that LLM-rephrased emails can significantly degrade the performance of both ML and LLM-based detectors. Our benchmarking highlights that models like DeepSeek R1 Distill Qwen 14B (Q8_0) achieve competitive accuracy, above 80%, using only 17GB of VRAM, supporting their viability for cost-efficient deployment. We further assess the models' adversarial robustness and cost-performance tradeoffs, and demonstrate how lightweight LLMs can provide concise, interpretable explanations to support real-time decision-making. These findings position optimized LLMs as promising components in phishing defence systems and offer a path forward for integrating explainable, efficient AI into modern cybersecurity frameworks.



## **27. A Cryptographic Perspective on Mitigation vs. Detection in Machine Learning**

cs.LG

28 pages

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2504.20310v2) [paper-pdf](http://arxiv.org/pdf/2504.20310v2)

**Authors**: Greg Gluch, Shafi Goldwasser

**Abstract**: In this paper, we initiate a cryptographically inspired theoretical study of detection versus mitigation of adversarial inputs produced by attackers on Machine Learning algorithms during inference time.   We formally define defense by detection (DbD) and defense by mitigation (DbM). Our definitions come in the form of a 3-round protocol between two resource-bounded parties: a trainer/defender and an attacker. The attacker aims to produce inference-time inputs that fool the training algorithm. We define correctness, completeness, and soundness properties to capture successful defense at inference time while not degrading (too much) the performance of the algorithm on inputs from the training distribution.   We first show that achieving DbD and achieving DbM are equivalent for ML classification tasks. Surprisingly, this is not the case for ML generative learning tasks, where there are many possible correct outputs for each input. We show a separation between DbD and DbM by exhibiting two generative learning tasks for which it is possible to defend by mitigation but it is provably impossible to defend by detection. The mitigation phase uses significantly less computational resources than the initial training algorithm. In the first learning task we consider sample complexity as the resource and in the second the time complexity. The first result holds under the assumption that the Identity-Based Fully Homomorphic Encryption (IB-FHE), publicly-verifiable zero-knowledge Succinct Non-Interactive Arguments of Knowledge (zk-SNARK), and Strongly Unforgeable Signatures exist. The second result assumes the existence of Non-Parallelizing Languages with Average-Case Hardness (NPL) and Incrementally-Verifiable Computation (IVC) and IB-FHE.



## **28. Adversarial Defenses via Vector Quantization**

cs.LG

This is the author-accepted version of our paper published in  Neurocomputing. The final published version is available at:  https://doi.org/10.1016/j.neucom.2025.130703

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2305.13651v2) [paper-pdf](http://arxiv.org/pdf/2305.13651v2)

**Authors**: Zhiyi Dong, Yongyi Mao

**Abstract**: Adversarial attacks pose significant challenges to the robustness of modern deep neural networks in computer vision, and defending these networks against adversarial attacks has attracted intense research efforts. Among various defense strategies, preprocessing-based defenses are practically appealing since there is no need to train the network under protection. However, such approaches typically do not achieve comparable robustness as other methods such as adversarial training. In this paper, we propose a novel framework for preprocessing-based defenses, where a vector quantizer is used as a preprocessor. This framework, inspired by and extended from Randomized Discretization (RandDisc), is theoretically principled by rate-distortion theory: indeed, RandDisc may be viewed as a scalar quantizer, and rate-distortion theory suggests that such quantization schemes are inferior to vector quantization. In our framework, the preprocessing vector quantizer treats the input image as a collection of patches and finds a set of representative patches based on the patch distributions; each original patch is then modified according to the representative patches close to it. We present two lightweight defenses in this framework, referred to as patched RandDisc (pRD) and sliding-window RandDisc (swRD), where the patches are disjoint in the former and overlapping in the latter. We show that vector-quantization-based defenses have certifiable robust accuracy and that pRD and swRD demonstrate state-of-the-art performances, surpassing RandDisc by a large margin. Notably, the proposed defenses possess the obfuscated gradients property. Our experiments however show that pRD and swRD remain effective under the STE and EOT attacks, which are designed specifically for defenses with gradient obfuscation. ...



## **29. Exploiting Edge Features for Transferable Adversarial Attacks in Distributed Machine Learning**

cs.LG

under review

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.07259v1) [paper-pdf](http://arxiv.org/pdf/2507.07259v1)

**Authors**: Giulio Rossolini, Fabio Brau, Alessandro Biondi, Battista Biggio, Giorgio Buttazzo

**Abstract**: As machine learning models become increasingly deployed across the edge of internet of things environments, a partitioned deep learning paradigm in which models are split across multiple computational nodes introduces a new dimension of security risk. Unlike traditional inference setups, these distributed pipelines span the model computation across heterogeneous nodes and communication layers, thereby exposing a broader attack surface to potential adversaries. Building on these motivations, this work explores a previously overlooked vulnerability: even when both the edge and cloud components of the model are inaccessible (i.e., black-box), an adversary who intercepts the intermediate features transmitted between them can still pose a serious threat. We demonstrate that, under these mild and realistic assumptions, an attacker can craft highly transferable proxy models, making the entire deep learning system significantly more vulnerable to evasion attacks. In particular, the intercepted features can be effectively analyzed and leveraged to distill surrogate models capable of crafting highly transferable adversarial examples against the target model. To this end, we propose an exploitation strategy specifically designed for distributed settings, which involves reconstructing the original tensor shape from vectorized transmitted features using simple statistical analysis, and adapting surrogate architectures accordingly to enable effective feature distillation. A comprehensive and systematic experimental evaluation has been conducted to demonstrate that surrogate models trained with the proposed strategy, i.e., leveraging intermediate features, tremendously improve the transferability of adversarial attacks. These findings underscore the urgent need to account for intermediate feature leakage in the design of secure distributed deep learning systems.



## **30. Protecting Classifiers From Attacks**

stat.ML

Published in Statistical Science:  https://projecteuclid.org/journals/statistical-science/volume-39/issue-3/Protecting-Classifiers-from-Attacks/10.1214/24-STS922.full

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2004.08705v2) [paper-pdf](http://arxiv.org/pdf/2004.08705v2)

**Authors**: Victor Gallego, Roi Naveiro, Alberto Redondo, David Rios Insua, Fabrizio Ruggeri

**Abstract**: In multiple domains such as malware detection, automated driving systems, or fraud detection, classification algorithms are susceptible to being attacked by malicious agents willing to perturb the value of instance covariates to pursue certain goals. Such problems pertain to the field of adversarial machine learning and have been mainly dealt with, perhaps implicitly, through game-theoretic ideas with strong underlying common knowledge assumptions. These are not realistic in numerous application domains in relation to security and business competition. We present an alternative Bayesian decision theoretic framework that accounts for the uncertainty about the attacker's behavior using adversarial risk analysis concepts. In doing so, we also present core ideas in adversarial machine learning to a statistical audience. A key ingredient in our framework is the ability to sample from the distribution of originating instances given the, possibly attacked, observed ones. We propose an initial procedure based on approximate Bayesian computation usable during operations; within it, we simulate the attacker's problem taking into account our uncertainty about his elements. Large-scale problems require an alternative scalable approach implementable during the training stage. Globally, we are able to robustify statistical classification algorithms against malicious attacks.



## **31. Robust and Safe Traffic Sign Recognition using N-version with Weighted Voting**

cs.LG

27 pages including appendix, 1 figure

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06907v1) [paper-pdf](http://arxiv.org/pdf/2507.06907v1)

**Authors**: Linyun Gao, Qiang Wen, Fumio Machida

**Abstract**: Autonomous driving is rapidly advancing as a key application of machine learning, yet ensuring the safety of these systems remains a critical challenge. Traffic sign recognition, an essential component of autonomous vehicles, is particularly vulnerable to adversarial attacks that can compromise driving safety. In this paper, we propose an N-version machine learning (NVML) framework that integrates a safety-aware weighted soft voting mechanism. Our approach utilizes Failure Mode and Effects Analysis (FMEA) to assess potential safety risks and assign dynamic, safety-aware weights to the ensemble outputs. We evaluate the robustness of three-version NVML systems employing various voting mechanisms against adversarial samples generated using the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks. Experimental results demonstrate that our NVML approach significantly enhances the robustness and safety of traffic sign recognition systems under adversarial conditions.



## **32. A Single-Point Measurement Framework for Robust Cyber-Attack Diagnosis in Smart Microgrids Using Dual Fractional-Order Feature Analysis**

eess.SY

8 pages, 10 figures

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06890v1) [paper-pdf](http://arxiv.org/pdf/2507.06890v1)

**Authors**: Yifan Wang

**Abstract**: Cyber-attacks jeopardize the safe operation of smart microgrids. At the same time, existing diagnostic methods either depend on expensive multi-point instrumentation or stringent modelling assumptions that are untenable under single-sensor constraints. This paper proposes a Fractional-Order Memory-Enhanced Attack-Diagnosis Scheme (FO-MADS) that achieves low-latency fault localisation and cyber-attack detection using only one VPQ (Voltage-Power-Reactive-power) sensor. FO-MADS first constructs a dual fractional-order feature library by jointly applying Caputo and Gr\"unwald-Letnikov derivatives, thereby amplifying micro-perturbations and slow drifts in the VPQ signal. A two-stage hierarchical classifier then pinpoints the affected inverter and isolates the faulty IGBT switch, effectively alleviating class imbalance. Robustness is further strengthened through Progressive Memory-Replay Adversarial Training (PMR-AT), whose attack-aware loss is dynamically re-weighted via Online Hard Example Mining (OHEM) to prioritise the most challenging samples. Experiments on a four-inverter microgrid testbed comprising 1 normal and 24 fault classes under four attack scenarios demonstrate diagnostic accuracies of 96.6 % (bias), 94.0 % (noise), 92.8 % (data replacement), and 95.7 % (replay), while sustaining 96.7 % under attack-free conditions. These results establish FO-MADS as a cost-effective and readily deployable solution that markedly enhances the cyber-physical resilience of smart microgrids.



## **33. IAP: Invisible Adversarial Patch Attack through Perceptibility-Aware Localization and Perturbation Optimization**

cs.CV

Published in ICCV 2025

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06856v1) [paper-pdf](http://arxiv.org/pdf/2507.06856v1)

**Authors**: Subrat Kishore Dutta, Xiao Zhang

**Abstract**: Despite modifying only a small localized input region, adversarial patches can drastically change the prediction of computer vision models. However, prior methods either cannot perform satisfactorily under targeted attack scenarios or fail to produce contextually coherent adversarial patches, causing them to be easily noticeable by human examiners and insufficiently stealthy against automatic patch defenses. In this paper, we introduce IAP, a novel attack framework that generates highly invisible adversarial patches based on perceptibility-aware localization and perturbation optimization schemes. Specifically, IAP first searches for a proper location to place the patch by leveraging classwise localization and sensitivity maps, balancing the susceptibility of patch location to both victim model prediction and human visual system, then employs a perceptibility-regularized adversarial loss and a gradient update rule that prioritizes color constancy for optimizing invisible perturbations. Comprehensive experiments across various image benchmarks and model architectures demonstrate that IAP consistently achieves competitive attack success rates in targeted settings with significantly improved patch invisibility compared to existing baselines. In addition to being highly imperceptible to humans, IAP is shown to be stealthy enough to render several state-of-the-art patch defenses ineffective.



## **34. PBCAT: Patch-based composite adversarial training against physically realizable attacks on object detection**

cs.CV

Accepted by ICCV 2025

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2506.23581v2) [paper-pdf](http://arxiv.org/pdf/2506.23581v2)

**Authors**: Xiao Li, Yiming Zhu, Yifan Huang, Wei Zhang, Yingzhe He, Jie Shi, Xiaolin Hu

**Abstract**: Object detection plays a crucial role in many security-sensitive applications. However, several recent studies have shown that object detectors can be easily fooled by physically realizable attacks, \eg, adversarial patches and recent adversarial textures, which pose realistic and urgent threats. Adversarial Training (AT) has been recognized as the most effective defense against adversarial attacks. While AT has been extensively studied in the $l_\infty$ attack settings on classification models, AT against physically realizable attacks on object detectors has received limited exploration. Early attempts are only performed to defend against adversarial patches, leaving AT against a wider range of physically realizable attacks under-explored. In this work, we consider defending against various physically realizable attacks with a unified AT method. We propose PBCAT, a novel Patch-Based Composite Adversarial Training strategy. PBCAT optimizes the model by incorporating the combination of small-area gradient-guided adversarial patches and imperceptible global adversarial perturbations covering the entire image. With these designs, PBCAT has the potential to defend against not only adversarial patches but also unseen physically realizable attacks such as adversarial textures. Extensive experiments in multiple settings demonstrated that PBCAT significantly improved robustness against various physically realizable attacks over state-of-the-art defense methods. Notably, it improved the detection accuracy by 29.7\% over previous defense methods under one recent adversarial texture attack.



## **35. Tail-aware Adversarial Attacks: A Distributional Approach to Efficient LLM Jailbreaking**

cs.LG

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.04446v2) [paper-pdf](http://arxiv.org/pdf/2507.04446v2)

**Authors**: Tim Beyer, Yan Scholten, Leo Schwinn, Stephan Günnemann

**Abstract**: To guarantee safe and robust deployment of large language models (LLMs) at scale, it is critical to accurately assess their adversarial robustness. Existing adversarial attacks typically target harmful responses in single-point, greedy generations, overlooking the inherently stochastic nature of LLMs. In this paper, we propose a novel framework for adversarial robustness evaluation that explicitly models the entire output distribution, including tail-risks, providing better estimates for model robustness at scale. By casting the attack process as a resource allocation problem between optimization and sampling, we determine compute-optimal tradeoffs and show that integrating sampling into existing attacks boosts ASR by up to 48% and improves efficiency by up to two orders of magnitude. Our framework also enables us to analyze how different attack algorithms affect output harm distributions. Surprisingly, we find that most optimization strategies have little effect on output harmfulness. Finally, we introduce a data-free proof-of-concept objective based on entropy-maximization to demonstrate how our tail-aware perspective enables new optimization targets. Overall, our findings highlight the importance of tail-aware attacks and evaluation protocols to accurately assess and strengthen LLM safety.



## **36. Distributed Fault-Tolerant Multi-Robot Cooperative Localization in Adversarial Environments**

cs.RO

Accepted to IROS 2025 Conference

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06750v1) [paper-pdf](http://arxiv.org/pdf/2507.06750v1)

**Authors**: Tohid Kargar Tasooji, Ramviyas Parasuraman

**Abstract**: In multi-robot systems (MRS), cooperative localization is a crucial task for enhancing system robustness and scalability, especially in GPS-denied or communication-limited environments. However, adversarial attacks, such as sensor manipulation, and communication jamming, pose significant challenges to the performance of traditional localization methods. In this paper, we propose a novel distributed fault-tolerant cooperative localization framework to enhance resilience against sensor and communication disruptions in adversarial environments. We introduce an adaptive event-triggered communication strategy that dynamically adjusts communication thresholds based on real-time sensing and communication quality. This strategy ensures optimal performance even in the presence of sensor degradation or communication failure. Furthermore, we conduct a rigorous analysis of the convergence and stability properties of the proposed algorithm, demonstrating its resilience against bounded adversarial zones and maintaining accurate state estimation. Robotarium-based experiment results show that our proposed algorithm significantly outperforms traditional methods in terms of localization accuracy and communication efficiency, particularly in adversarial settings. Our approach offers improved scalability, reliability, and fault tolerance for MRS, making it suitable for large-scale deployments in real-world, challenging environments.



## **37. Towards Adversarial Robustness via Debiased High-Confidence Logit Alignment**

cs.CV

Accepted by ICCV 2025

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2408.06079v2) [paper-pdf](http://arxiv.org/pdf/2408.06079v2)

**Authors**: Kejia Zhang, Juanjuan Weng, Shaozi Li, Zhiming Luo

**Abstract**: Despite the remarkable progress of deep neural networks (DNNs) in various visual tasks, their vulnerability to adversarial examples raises significant security concerns. Recent adversarial training methods leverage inverse adversarial attacks to generate high-confidence examples, aiming to align adversarial distributions with high-confidence class regions. However, our investigation reveals that under inverse adversarial attacks, high-confidence outputs are influenced by biased feature activations, causing models to rely on background features that lack a causal relationship with the labels. This spurious correlation bias leads to overfitting irrelevant background features during adversarial training, thereby degrading the model's robust performance and generalization capabilities. To address this issue, we propose Debiased High-Confidence Adversarial Training (DHAT), a novel approach that aligns adversarial logits with debiased high-confidence logits and restores proper attention by enhancing foreground logit orthogonality. Extensive experiments demonstrate that DHAT achieves state-of-the-art robustness on both CIFAR and ImageNet-1K benchmarks, while significantly improving generalization by mitigating the feature bias inherent in inverse adversarial training approaches. Code is available at https://github.com/KejiaZhang-Robust/DHAT.



## **38. Evaluating and Improving Robustness in Large Language Models: A Survey and Future Directions**

cs.CL

33 pages, 5 figures

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2506.11111v2) [paper-pdf](http://arxiv.org/pdf/2506.11111v2)

**Authors**: Kun Zhang, Le Wu, Kui Yu, Guangyi Lv, Dacao Zhang

**Abstract**: Large Language Models (LLMs) have gained enormous attention in recent years due to their capability of understanding and generating natural languages. With the rapid development and wild-range applications (e.g., Agents, Embodied Intelligence), the robustness of LLMs has received increased attention. As the core brain of many AI applications, the robustness of LLMs requires that models should not only generate consistent contents, but also ensure the correctness and stability of generated content when dealing with unexpeted application scenarios (e.g., toxic prompts, limited noise domain data, outof-distribution (OOD) applications, etc). In this survey paper, we conduct a thorough review of the robustness of LLMs, aiming to provide a comprehensive terminology of concepts and methods around this field and facilitate the community. Specifically, we first give a formal definition of LLM robustness and present the collection protocol of this survey paper. Then, based on the types of perturbated inputs, we organize this survey from the following perspectives: 1) Adversarial Robustness: tackling the problem that prompts are manipulated intentionally, such as noise prompts, long context, data attack, etc; 2) OOD Robustness: dealing with the unexpected real-world application scenarios, such as OOD detection, zero-shot transferring, hallucinations, etc; 3) Evaluation of Robustness: summarizing the new evaluation datasets, metrics, and tools for verifying the robustness of LLMs. After reviewing the representative work from each perspective, we discuss and highlight future opportunities and research directions in this field. Meanwhile, we also organize related works and provide an easy-to-search project (https://github.com/zhangkunzk/Awesome-LLM-Robustness-papers) to support the community.



## **39. Image Can Bring Your Memory Back: A Novel Multi-Modal Guided Attack against Image Generation Model Unlearning**

cs.CV

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.07139v1) [paper-pdf](http://arxiv.org/pdf/2507.07139v1)

**Authors**: Renyang Liu, Guanlin Li, Tianwei Zhang, See-Kiong Ng

**Abstract**: Recent advances in image generation models (IGMs), particularly diffusion-based architectures such as Stable Diffusion (SD), have markedly enhanced the quality and diversity of AI-generated visual content. However, their generative capability has also raised significant ethical, legal, and societal concerns, including the potential to produce harmful, misleading, or copyright-infringing content. To mitigate these concerns, machine unlearning (MU) emerges as a promising solution by selectively removing undesirable concepts from pretrained models. Nevertheless, the robustness and effectiveness of existing unlearning techniques remain largely unexplored, particularly in the presence of multi-modal adversarial inputs.   To bridge this gap, we propose Recall, a novel adversarial framework explicitly designed to compromise the robustness of unlearned IGMs. Unlike existing approaches that predominantly rely on adversarial text prompts, Recall exploits the intrinsic multi-modal conditioning capabilities of diffusion models by efficiently optimizing adversarial image prompts with guidance from a single semantically relevant reference image. Extensive experiments across ten state-of-the-art unlearning methods and diverse tasks show that Recall consistently outperforms existing baselines in terms of adversarial effectiveness, computational efficiency, and semantic fidelity with the original textual prompt. These findings reveal critical vulnerabilities in current unlearning mechanisms and underscore the need for more robust solutions to ensure the safety and reliability of generative models. Code and data are publicly available at \textcolor{blue}{https://github.com/ryliu68/RECALL}.



## **40. Can adversarial attacks by large language models be attributed?**

cs.AI

21 pages, 5 figures, 2 tables

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2411.08003v2) [paper-pdf](http://arxiv.org/pdf/2411.08003v2)

**Authors**: Manuel Cebrian, Andres Abeliuk, Jan Arne Telle

**Abstract**: Attributing outputs from Large Language Models (LLMs) in adversarial settings-such as cyberattacks and disinformation campaigns-presents significant challenges that are likely to grow in importance. We approach this attribution problem from both a theoretical and an empirical perspective, drawing on formal language theory (identification in the limit) and data-driven analysis of the expanding LLM ecosystem. By modeling an LLM's set of possible outputs as a formal language, we analyze whether finite samples of text can uniquely pinpoint the originating model. Our results show that, under mild assumptions of overlapping capabilities among models, certain classes of LLMs are fundamentally non-identifiable from their outputs alone. We delineate four regimes of theoretical identifiability: (1) an infinite class of deterministic (discrete) LLM languages is not identifiable (Gold's classical result from 1967); (2) an infinite class of probabilistic LLMs is also not identifiable (by extension of the deterministic case); (3) a finite class of deterministic LLMs is identifiable (consistent with Angluin's tell-tale criterion); and (4) even a finite class of probabilistic LLMs can be non-identifiable (we provide a new counterexample establishing this negative result). Complementing these theoretical insights, we quantify the explosion in the number of plausible model origins (hypothesis space) for a given output in recent years. Even under conservative assumptions-each open-source model fine-tuned on at most one new dataset-the count of distinct candidate models doubles approximately every 0.5 years, and allowing multi-dataset fine-tuning combinations yields doubling times as short as 0.28 years. This combinatorial growth, alongside the extraordinary computational cost of brute-force likelihood attribution across all models and potential users, renders exhaustive attribution infeasible in practice.



## **41. Dual State-space Fidelity Blade (D-STAB): A Novel Stealthy Cyber-physical Attack Paradigm**

eess.SY

accepted by 2025 American Control Conference

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06492v1) [paper-pdf](http://arxiv.org/pdf/2507.06492v1)

**Authors**: Jiajun Shen, Hao Tu, Fengjun Li, Morteza Hashemi, Di Wu, Huazhen Fang

**Abstract**: This paper presents a novel cyber-physical attack paradigm, termed the Dual State-Space Fidelity Blade (D-STAB), which targets the firmware of core cyber-physical components as a new class of attack surfaces. The D-STAB attack exploits the information asymmetry caused by the fidelity gap between high-fidelity and low-fidelity physical models in cyber-physical systems. By designing precise adversarial constraints based on high-fidelity state-space information, the attack induces deviations in high-fidelity states that remain undetected by defenders relying on low-fidelity observations. The effectiveness of D-STAB is demonstrated through a case study in cyber-physical battery systems, specifically in an optimal charging task governed by a Battery Management System (BMS).



## **42. On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks**

cs.CL

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2507.06489v1) [paper-pdf](http://arxiv.org/pdf/2507.06489v1)

**Authors**: Stephen Obadinma, Xiaodan Zhu

**Abstract**: Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to ensure transparency, trust, and safety in human-AI interactions across many high-stakes applications. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce a novel framework for attacking verbal confidence scores through both perturbation and jailbreak-based methods, and show that these attacks can significantly jeopardize verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current confidence elicitation methods are vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the urgent need to design more robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses.



## **43. Real AI Agents with Fake Memories: Fatal Context Manipulation Attacks on Web3 Agents**

cs.CR

19 pages, 14 figures

**SubmitDate**: 2025-07-09    [abs](http://arxiv.org/abs/2503.16248v3) [paper-pdf](http://arxiv.org/pdf/2503.16248v3)

**Authors**: Atharv Singh Patlan, Peiyao Sheng, S. Ashwin Hebbar, Prateek Mittal, Pramod Viswanath

**Abstract**: AI agents integrated with Web3 offer autonomy and openness but raise security concerns as they interact with financial protocols and immutable smart contracts. This paper investigates the vulnerabilities of AI agents within blockchain-based financial ecosystems when exposed to adversarial threats in real-world scenarios. We introduce the concept of context manipulation -- a comprehensive attack vector that exploits unprotected context surfaces, including input channels, memory modules, and external data feeds. It expands on traditional prompt injection and reveals a more stealthy and persistent threat: memory injection. Using ElizaOS, a representative decentralized AI agent framework for automated Web3 operations, we showcase that malicious injections into prompts or historical records can trigger unauthorized asset transfers and protocol violations which could be financially devastating in reality. To quantify these risks, we introduce CrAIBench, a Web3-focused benchmark covering 150+ realistic blockchain tasks. such as token transfers, trading, bridges, and cross-chain interactions, and 500+ attack test cases using context manipulation. Our evaluation results confirm that AI models are significantly more vulnerable to memory injection compared to prompt injection. Finally, we evaluate a comprehensive defense roadmap, finding that prompt-injection defenses and detectors only provide limited protection when stored context is corrupted, whereas fine-tuning-based defenses substantially reduce attack success rates while preserving performance on single-step tasks. These results underscore the urgent need for AI agents that are both secure and fiduciarily responsible in blockchain environments.



## **44. Single Word Change is All You Need: Designing Attacks and Defenses for Text Classifiers**

cs.CL

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2401.17196v2) [paper-pdf](http://arxiv.org/pdf/2401.17196v2)

**Authors**: Lei Xu, Sarah Alnegheimish, Laure Berti-Equille, Alfredo Cuesta-Infante, Kalyan Veeramachaneni

**Abstract**: In text classification, creating an adversarial example means subtly perturbing a few words in a sentence without changing its meaning, causing it to be misclassified by a classifier. A concerning observation is that a significant portion of adversarial examples generated by existing methods change only one word. This single-word perturbation vulnerability represents a significant weakness in classifiers, which malicious users can exploit to efficiently create a multitude of adversarial examples. This paper studies this problem and makes the following key contributions: (1) We introduce a novel metric \r{ho} to quantitatively assess a classifier's robustness against single-word perturbation. (2) We present the SP-Attack, designed to exploit the single-word perturbation vulnerability, achieving a higher attack success rate, better preserving sentence meaning, while reducing computation costs compared to state-of-the-art adversarial methods. (3) We propose SP-Defense, which aims to improve \r{ho} by applying data augmentation in learning. Experimental results on 4 datasets and BERT and distilBERT classifiers show that SP-Defense improves \r{ho} by 14.6% and 13.9% and decreases the attack success rate of SP-Attack by 30.4% and 21.2% on two classifiers respectively, and decreases the attack success rate of existing attack methods that involve multiple-word perturbations.



## **45. On the Natural Robustness of Vision-Language Models Against Visual Perception Attacks in Autonomous Driving**

cs.CV

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2506.11472v2) [paper-pdf](http://arxiv.org/pdf/2506.11472v2)

**Authors**: Pedram MohajerAnsari, Amir Salarpour, Michael Kühr, Siyu Huang, Mohammad Hamad, Sebastian Steinhorst, Habeeb Olufowobi, Mert D. Pesé

**Abstract**: Autonomous vehicles (AVs) rely on deep neural networks (DNNs) for critical tasks such as traffic sign recognition (TSR), automated lane centering (ALC), and vehicle detection (VD). However, these models are vulnerable to attacks that can cause misclassifications and compromise safety. Traditional defense mechanisms, including adversarial training, often degrade benign accuracy and fail to generalize against unseen attacks. In this work, we introduce Vehicle Vision Language Models (V2LMs), fine-tuned vision-language models specialized for AV perception. Our findings demonstrate that V2LMs inherently exhibit superior robustness against unseen attacks without requiring adversarial training, maintaining significantly higher accuracy than conventional DNNs under adversarial conditions. We evaluate two deployment strategies: Solo Mode, where individual V2LMs handle specific perception tasks, and Tandem Mode, where a single unified V2LM is fine-tuned for multiple tasks simultaneously. Experimental results reveal that DNNs suffer performance drops of 33% to 46% under attacks, whereas V2LMs maintain adversarial accuracy with reductions of less than 8% on average. The Tandem Mode further offers a memory-efficient alternative while achieving comparable robustness to Solo Mode. We also explore integrating V2LMs as parallel components to AV perception to enhance resilience against adversarial threats. Our results suggest that V2LMs offer a promising path toward more secure and resilient AV perception systems.



## **46. The bitter lesson of misuse detection**

cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06282v1) [paper-pdf](http://arxiv.org/pdf/2507.06282v1)

**Authors**: Hadrien Mariaccia, Charbel-Raphaël Segerie, Diego Dorn

**Abstract**: Prior work on jailbreak detection has established the importance of adversarial robustness for LLMs but has largely focused on the model ability to resist adversarial inputs and to output safe content, rather than the effectiveness of external supervision systems. The only public and independent benchmark of these guardrails to date evaluates a narrow set of supervisors on limited scenarios. Consequently, no comprehensive public benchmark yet verifies how well supervision systems from the market perform under realistic, diverse attacks. To address this, we introduce BELLS, a Benchmark for the Evaluation of LLM Supervision Systems. The framework is two dimensional: harm severity (benign, borderline, harmful) and adversarial sophistication (direct vs. jailbreak) and provides a rich dataset covering 3 jailbreak families and 11 harm categories. Our evaluations reveal drastic limitations of specialized supervision systems. While they recognize some known jailbreak patterns, their semantic understanding and generalization capabilities are very limited, sometimes with detection rates close to zero when asking a harmful question directly or with a new jailbreak technique such as base64 encoding. Simply asking generalist LLMs if the user question is "harmful or not" largely outperforms these supervisors from the market according to our BELLS score. But frontier LLMs still suffer from metacognitive incoherence, often responding to queries they correctly identify as harmful (up to 30 percent for Claude 3.7 and greater than 50 percent for Mistral Large). These results suggest that simple scaffolding could significantly improve misuse detection robustness, but more research is needed to assess the tradeoffs of such techniques. Our results support the "bitter lesson" of misuse detection: general capabilities of LLMs are necessary to detect a diverse array of misuses and jailbreaks.



## **47. ScoreAdv: Score-based Targeted Generation of Natural Adversarial Examples via Diffusion Models**

cs.CV

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06078v1) [paper-pdf](http://arxiv.org/pdf/2507.06078v1)

**Authors**: Chihan Huang, Hao Tang

**Abstract**: Despite the success of deep learning across various domains, it remains vulnerable to adversarial attacks. Although many existing adversarial attack methods achieve high success rates, they typically rely on $\ell_{p}$-norm perturbation constraints, which do not align with human perceptual capabilities. Consequently, researchers have shifted their focus toward generating natural, unrestricted adversarial examples (UAEs). GAN-based approaches suffer from inherent limitations, such as poor image quality due to instability and mode collapse. Meanwhile, diffusion models have been employed for UAE generation, but they still rely on iterative PGD perturbation injection, without fully leveraging their central denoising capabilities. In this paper, we introduce a novel approach for generating UAEs based on diffusion models, named ScoreAdv. This method incorporates an interpretable adversarial guidance mechanism to gradually shift the sampling distribution towards the adversarial distribution, while using an interpretable saliency map to inject the visual information of a reference image into the generated samples. Notably, our method is capable of generating an unlimited number of natural adversarial examples and can attack not only classification models but also retrieval models. We conduct extensive experiments on ImageNet and CelebA datasets, validating the performance of ScoreAdv across ten target models in both black-box and white-box settings. Our results demonstrate that ScoreAdv achieves state-of-the-art attack success rates and image quality. Furthermore, the dynamic balance between denoising and adversarial perturbation enables ScoreAdv to remain robust even under defensive measures.



## **48. CAVGAN: Unifying Jailbreak and Defense of LLMs via Generative Adversarial Attacks on their Internal Representations**

cs.CR

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.06043v1) [paper-pdf](http://arxiv.org/pdf/2507.06043v1)

**Authors**: Xiaohu Li, Yunfeng Ning, Zepeng Bao, Mayi Xu, Jianhao Chen, Tieyun Qian

**Abstract**: Security alignment enables the Large Language Model (LLM) to gain the protection against malicious queries, but various jailbreak attack methods reveal the vulnerability of this security mechanism. Previous studies have isolated LLM jailbreak attacks and defenses. We analyze the security protection mechanism of the LLM, and propose a framework that combines attack and defense. Our method is based on the linearly separable property of LLM intermediate layer embedding, as well as the essence of jailbreak attack, which aims to embed harmful problems and transfer them to the safe area. We utilize generative adversarial network (GAN) to learn the security judgment boundary inside the LLM to achieve efficient jailbreak attack and defense. The experimental results indicate that our method achieves an average jailbreak success rate of 88.85\% across three popular LLMs, while the defense success rate on the state-of-the-art jailbreak dataset reaches an average of 84.17\%. This not only validates the effectiveness of our approach but also sheds light on the internal security mechanisms of LLMs, offering new insights for enhancing model security The code and data are available at https://github.com/NLPGM/CAVGAN.



## **49. TuneShield: Mitigating Toxicity in Conversational AI while Fine-tuning on Untrusted Data**

cs.CR

Pre-print

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2507.05660v1) [paper-pdf](http://arxiv.org/pdf/2507.05660v1)

**Authors**: Aravind Cheruvu, Shravya Kanchi, Sifat Muhammad Abdullah, Nicholas Kong, Daphne Yao, Murtuza Jadliwala, Bimal Viswanath

**Abstract**: Recent advances in foundation models, such as LLMs, have revolutionized conversational AI. Chatbots are increasingly being developed by customizing LLMs on specific conversational datasets. However, mitigating toxicity during this customization, especially when dealing with untrusted training data, remains a significant challenge. To address this, we introduce TuneShield, a defense framework designed to mitigate toxicity during chatbot fine-tuning while preserving conversational quality. TuneShield leverages LLM-based toxicity classification, utilizing the instruction-following capabilities and safety alignment of LLMs to effectively identify toxic samples, outperforming industry API services. TuneShield generates synthetic conversation samples, termed 'healing data', based on the identified toxic samples, using them to mitigate toxicity while reinforcing desirable behavior during fine-tuning. It performs an alignment process to further nudge the chatbot towards producing desired responses. Our findings show that TuneShield effectively mitigates toxicity injection attacks while preserving conversational quality, even when the toxicity classifiers are imperfect or biased. TuneShield proves to be resilient against adaptive adversarial and jailbreak attacks. Additionally, TuneShield demonstrates effectiveness in mitigating adaptive toxicity injection attacks during dialog-based learning (DBL).



## **50. MEF: A Capability-Aware Multi-Encryption Framework for Evaluating Vulnerabilities in Black-Box Large Language Models**

cs.CL

**SubmitDate**: 2025-07-08    [abs](http://arxiv.org/abs/2505.23404v3) [paper-pdf](http://arxiv.org/pdf/2505.23404v3)

**Authors**: Mingyu Yu, Wei Wang, Yanjie Wei, Sujuan Qin, Fei Gao, Wenmin Li

**Abstract**: Recent advancements in adversarial jailbreak attacks have revealed significant vulnerabilities in Large Language Models (LLMs), facilitating the evasion of alignment safeguards through increasingly sophisticated prompt manipulations. In this paper, we propose MEF, a capability-aware multi-encryption framework for evaluating vulnerabilities in black-box LLMs. Our key insight is that the effectiveness of jailbreak strategies can be significantly enhanced by tailoring them to the semantic comprehension capabilities of the target model. We present a typology that classifies LLMs into Type I and Type II based on their comprehension levels, and design adaptive attack strategies for each. MEF combines layered semantic mutations and dual-ended encryption techniques, enabling circumvention of input, inference, and output-level defenses. Experimental results demonstrate the superiority of our approach. Remarkably, it achieves a jailbreak success rate of 98.9\% on GPT-4o (29 May 2025 release). Our findings reveal vulnerabilities in current LLMs' alignment defenses.



