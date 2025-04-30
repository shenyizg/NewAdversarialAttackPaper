# Latest Adversarial Attack Papers
**update at 2025-04-30 16:22:34**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. AegisLLM: Scaling Agentic Systems for Self-Reflective Defense in LLM Security**

cs.LG

ICLR 2025 Workshop BuildingTrust

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20965v1) [paper-pdf](http://arxiv.org/pdf/2504.20965v1)

**Authors**: Zikui Cai, Shayan Shabihi, Bang An, Zora Che, Brian R. Bartoldson, Bhavya Kailkhura, Tom Goldstein, Furong Huang

**Abstract**: We introduce AegisLLM, a cooperative multi-agent defense against adversarial attacks and information leakage. In AegisLLM, a structured workflow of autonomous agents - orchestrator, deflector, responder, and evaluator - collaborate to ensure safe and compliant LLM outputs, while self-improving over time through prompt optimization. We show that scaling agentic reasoning system at test-time - both by incorporating additional agent roles and by leveraging automated prompt optimization (such as DSPy)- substantially enhances robustness without compromising model utility. This test-time defense enables real-time adaptability to evolving attacks, without requiring model retraining. Comprehensive evaluations across key threat scenarios, including unlearning and jailbreaking, demonstrate the effectiveness of AegisLLM. On the WMDP unlearning benchmark, AegisLLM achieves near-perfect unlearning with only 20 training examples and fewer than 300 LM calls. For jailbreaking benchmarks, we achieve 51% improvement compared to the base model on StrongReject, with false refusal rates of only 7.9% on PHTest compared to 18-55% for comparable methods. Our results highlight the advantages of adaptive, agentic reasoning over static defenses, establishing AegisLLM as a strong runtime alternative to traditional approaches based on model modifications. Code is available at https://github.com/zikuicai/aegisllm



## **2. NoPain: No-box Point Cloud Attack via Optimal Transport Singular Boundary**

cs.CV

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2503.00063v4) [paper-pdf](http://arxiv.org/pdf/2503.00063v4)

**Authors**: Zezeng Li, Xiaoyu Du, Na Lei, Liming Chen, Weimin Wang

**Abstract**: Adversarial attacks exploit the vulnerability of deep models against adversarial samples. Existing point cloud attackers are tailored to specific models, iteratively optimizing perturbations based on gradients in either a white-box or black-box setting. Despite their promising attack performance, they often struggle to produce transferable adversarial samples due to overfitting the specific parameters of surrogate models. To overcome this issue, we shift our focus to the data distribution itself and introduce a novel approach named NoPain, which employs optimal transport (OT) to identify the inherent singular boundaries of the data manifold for cross-network point cloud attacks. Specifically, we first calculate the OT mapping from noise to the target feature space, then identify singular boundaries by locating non-differentiable positions. Finally, we sample along singular boundaries to generate adversarial point clouds. Once the singular boundaries are determined, NoPain can efficiently produce adversarial samples without the need of iterative updates or guidance from the surrogate classifiers. Extensive experiments demonstrate that the proposed end-to-end method outperforms baseline approaches in terms of both transferability and efficiency, while also maintaining notable advantages even against defense strategies. Code and model are available at https://github.com/cognaclee/nopain



## **3. Quantifying the Noise of Structural Perturbations on Graph Adversarial Attacks**

cs.LG

Ubder Review

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20869v1) [paper-pdf](http://arxiv.org/pdf/2504.20869v1)

**Authors**: Junyuan Fang, Han Yang, Haixian Wen, Jiajing Wu, Zibin Zheng, Chi K. Tse

**Abstract**: Graph neural networks have been widely utilized to solve graph-related tasks because of their strong learning power in utilizing the local information of neighbors. However, recent studies on graph adversarial attacks have proven that current graph neural networks are not robust against malicious attacks. Yet much of the existing work has focused on the optimization objective based on attack performance to obtain (near) optimal perturbations, but paid less attention to the strength quantification of each perturbation such as the injection of a particular node/link, which makes the choice of perturbations a black-box model that lacks interpretability. In this work, we propose the concept of noise to quantify the attack strength of each adversarial link. Furthermore, we propose three attack strategies based on the defined noise and classification margins in terms of single and multiple steps optimization. Extensive experiments conducted on benchmark datasets against three representative graph neural networks demonstrate the effectiveness of the proposed attack strategies. Particularly, we also investigate the preferred patterns of effective adversarial perturbations by analyzing the corresponding properties of the selected perturbation nodes.



## **4. Mitigating the Structural Bias in Graph Adversarial Defenses**

cs.LG

Under Review

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20848v1) [paper-pdf](http://arxiv.org/pdf/2504.20848v1)

**Authors**: Junyuan Fang, Huimin Liu, Han Yang, Jiajing Wu, Zibin Zheng, Chi K. Tse

**Abstract**: In recent years, graph neural networks (GNNs) have shown great potential in addressing various graph structure-related downstream tasks. However, recent studies have found that current GNNs are susceptible to malicious adversarial attacks. Given the inevitable presence of adversarial attacks in the real world, a variety of defense methods have been proposed to counter these attacks and enhance the robustness of GNNs. Despite the commendable performance of these defense methods, we have observed that they tend to exhibit a structural bias in terms of their defense capability on nodes with low degree (i.e., tail nodes), which is similar to the structural bias of traditional GNNs on nodes with low degree in the clean graph. Therefore, in this work, we propose a defense strategy by including hetero-homo augmented graph construction, $k$NN augmented graph construction, and multi-view node-wise attention modules to mitigate the structural bias of GNNs against adversarial attacks. Notably, the hetero-homo augmented graph consists of removing heterophilic links (i.e., links connecting nodes with dissimilar features) globally and adding homophilic links (i.e., links connecting nodes with similar features) for nodes with low degree. To further enhance the defense capability, an attention mechanism is adopted to adaptively combine the representations from the above two kinds of graph views. We conduct extensive experiments to demonstrate the defense and debiasing effect of the proposed strategy on benchmark datasets.



## **5. GaussTrap: Stealthy Poisoning Attacks on 3D Gaussian Splatting for Targeted Scene Confusion**

cs.CV

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20829v1) [paper-pdf](http://arxiv.org/pdf/2504.20829v1)

**Authors**: Jiaxin Hong, Sixu Chen, Shuoyang Sun, Hongyao Yu, Hao Fang, Yuqi Tan, Bin Chen, Shuhan Qi, Jiawei Li

**Abstract**: As 3D Gaussian Splatting (3DGS) emerges as a breakthrough in scene representation and novel view synthesis, its rapid adoption in safety-critical domains (e.g., autonomous systems, AR/VR) urgently demands scrutiny of potential security vulnerabilities. This paper presents the first systematic study of backdoor threats in 3DGS pipelines. We identify that adversaries may implant backdoor views to induce malicious scene confusion during inference, potentially leading to environmental misperception in autonomous navigation or spatial distortion in immersive environments. To uncover this risk, we propose GuassTrap, a novel poisoning attack method targeting 3DGS models. GuassTrap injects malicious views at specific attack viewpoints while preserving high-quality rendering in non-target views, ensuring minimal detectability and maximizing potential harm. Specifically, the proposed method consists of a three-stage pipeline (attack, stabilization, and normal training) to implant stealthy, viewpoint-consistent poisoned renderings in 3DGS, jointly optimizing attack efficacy and perceptual realism to expose security risks in 3D rendering. Extensive experiments on both synthetic and real-world datasets demonstrate that GuassTrap can effectively embed imperceptible yet harmful backdoor views while maintaining high-quality rendering in normal views, validating its robustness, adaptability, and practical applicability.



## **6. Regularized Robustly Reliable Learners and Instance Targeted Attacks**

cs.LG

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2410.10572v3) [paper-pdf](http://arxiv.org/pdf/2410.10572v3)

**Authors**: Avrim Blum, Donya Saless

**Abstract**: Instance-targeted data poisoning attacks, where an adversary corrupts a training set to induce errors on specific test points, have raised significant concerns. Balcan et al (2022) proposed an approach to addressing this challenge by defining a notion of robustly-reliable learners that provide per-instance guarantees of correctness under well-defined assumptions, even in the presence of data poisoning attacks. They then give a generic optimal (but computationally inefficient) robustly reliable learner as well as a computationally efficient algorithm for the case of linear separators over log-concave distributions.   In this work, we address two challenges left open by Balcan et al (2022). The first is that the definition of robustly-reliable learners in Balcan et al (2022) becomes vacuous for highly-flexible hypothesis classes: if there are two classifiers h_0, h_1 \in H both with zero error on the training set such that h_0(x) \neq h_1(x), then a robustly-reliable learner must abstain on x. We address this problem by defining a modified notion of regularized robustly-reliable learners that allows for nontrivial statements in this case. The second is that the generic algorithm of Balcan et al (2022) requires re-running an ERM oracle (essentially, retraining the classifier) on each test point x, which is generally impractical even if ERM can be implemented efficiently. To tackle this problem, we show that at least in certain interesting cases we can design algorithms that can produce their outputs in time sublinear in training time, by using techniques from dynamic algorithm design.



## **7. A Survey on Adversarial Contention Resolution**

cs.DC

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2403.03876v4) [paper-pdf](http://arxiv.org/pdf/2403.03876v4)

**Authors**: Ioana Banicescu, Trisha Chakraborty, Seth Gilbert, Maxwell Young

**Abstract**: Contention resolution addresses the challenge of coordinating access by multiple processes to a shared resource such as memory, disk storage, or a communication channel. Originally spurred by challenges in database systems and bus networks, contention resolution has endured as an important abstraction for resource sharing, despite decades of technological change. Here, we survey the literature on resolving worst-case contention, where the number of processes and the time at which each process may start seeking access to the resource is dictated by an adversary. We also highlight the evolution of contention resolution, where new concerns -- such as security, quality of service, and energy efficiency -- are motivated by modern systems. These efforts have yielded insights into the limits of randomized and deterministic approaches, as well as the impact of different model assumptions such as global clock synchronization, knowledge of the number of processors, feedback from access attempts, and attacks on the availability of the shared resource.



## **8. Data Encryption Battlefield: A Deep Dive into the Dynamic Confrontations in Ransomware Attacks**

cs.CR

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20681v1) [paper-pdf](http://arxiv.org/pdf/2504.20681v1)

**Authors**: Arash Mahboubi, Hamed Aboutorab, Seyit Camtepe, Hang Thanh Bui, Khanh Luong, Keyvan Ansari, Shenlu Wang, Bazara Barry

**Abstract**: In the rapidly evolving landscape of cybersecurity threats, ransomware represents a significant challenge. Attackers increasingly employ sophisticated encryption methods, such as entropy reduction through Base64 encoding, and partial or intermittent encryption to evade traditional detection methods. This study explores the dynamic battle between adversaries who continuously refine encryption strategies and defenders developing advanced countermeasures to protect vulnerable data. We investigate the application of online incremental machine learning algorithms designed to predict file encryption activities despite adversaries evolving obfuscation techniques. Our analysis utilizes an extensive dataset of 32.6 GB, comprising 11,928 files across multiple formats, including Microsoft Word documents (doc), PowerPoint presentations (ppt), Excel spreadsheets (xlsx), image formats (jpg, jpeg, png, tif, gif), PDFs (pdf), audio (mp3), and video (mp4) files. These files were encrypted by 75 distinct ransomware families, facilitating a robust empirical evaluation of machine learning classifiers effectiveness against diverse encryption tactics. Results highlight the Hoeffding Tree algorithms superior incremental learning capability, particularly effective in detecting traditional and AES-Base64 encryption methods employed to lower entropy. Conversely, the Random Forest classifier with warm-start functionality excels at identifying intermittent encryption methods, demonstrating the necessity of tailored machine learning solutions to counter sophisticated ransomware strategies.



## **9. Learning and Generalization with Mixture Data**

stat.ML

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20651v1) [paper-pdf](http://arxiv.org/pdf/2504.20651v1)

**Authors**: Harsh Vardhan, Avishek Ghosh, Arya Mazumdar

**Abstract**: In many, if not most, machine learning applications the training data is naturally heterogeneous (e.g. federated learning, adversarial attacks and domain adaptation in neural net training). Data heterogeneity is identified as one of the major challenges in modern day large-scale learning. A classical way to represent heterogeneous data is via a mixture model. In this paper, we study generalization performance and statistical rates when data is sampled from a mixture distribution. We first characterize the heterogeneity of the mixture in terms of the pairwise total variation distance of the sub-population distributions. Thereafter, as a central theme of this paper, we characterize the range where the mixture may be treated as a single (homogeneous) distribution for learning. In particular, we study the generalization performance under the classical PAC framework and the statistical error rates for parametric (linear regression, mixture of hyperplanes) as well as non-parametric (Lipschitz, convex and H\"older-smooth) regression problems. In order to do this, we obtain Rademacher complexity and (local) Gaussian complexity bounds with mixture data, and apply them to get the generalization and convergence rates respectively. We observe that as the (regression) function classes get more complex, the requirement on the pairwise total variation distance gets stringent, which matches our intuition. We also do a finer analysis for the case of mixed linear regression and provide a tight bound on the generalization error in terms of heterogeneity.



## **10. WILD: a new in-the-Wild Image Linkage Dataset for synthetic image attribution**

cs.MM

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.19595v2) [paper-pdf](http://arxiv.org/pdf/2504.19595v2)

**Authors**: Pietro Bongini, Sara Mandelli, Andrea Montibeller, Mirko Casu, Orazio Pontorno, Claudio Vittorio Ragaglia, Luca Zanchetta, Mattia Aquilina, Taiba Majid Wani, Luca Guarnera, Benedetta Tondi, Giulia Boato, Paolo Bestagini, Irene Amerini, Francesco De Natale, Sebastiano Battiato, Mauro Barni

**Abstract**: Synthetic image source attribution is an open challenge, with an increasing number of image generators being released yearly. The complexity and the sheer number of available generative techniques, as well as the scarcity of high-quality open source datasets of diverse nature for this task, make training and benchmarking synthetic image source attribution models very challenging. WILD is a new in-the-Wild Image Linkage Dataset designed to provide a powerful training and benchmarking tool for synthetic image attribution models. The dataset is built out of a closed set of 10 popular commercial generators, which constitutes the training base of attribution models, and an open set of 10 additional generators, simulating a real-world in-the-wild scenario. Each generator is represented by 1,000 images, for a total of 10,000 images in the closed set and 10,000 images in the open set. Half of the images are post-processed with a wide range of operators. WILD allows benchmarking attribution models in a wide range of tasks, including closed and open set identification and verification, and robust attribution with respect to post-processing and adversarial attacks. Models trained on WILD are expected to benefit from the challenging scenario represented by the dataset itself. Moreover, an assessment of seven baseline methodologies on closed and open set attribution is presented, including robustness tests with respect to post-processing.



## **11. Enhancing Leakage Attacks on Searchable Symmetric Encryption Using LLM-Based Synthetic Data Generation**

cs.CR

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20414v1) [paper-pdf](http://arxiv.org/pdf/2504.20414v1)

**Authors**: Joshua Chiu, Partha Protim Paul, Zahin Wahab

**Abstract**: Searchable Symmetric Encryption (SSE) enables efficient search capabilities over encrypted data, allowing users to maintain privacy while utilizing cloud storage. However, SSE schemes are vulnerable to leakage attacks that exploit access patterns, search frequency, and volume information. Existing studies frequently assume that adversaries possess a substantial fraction of the encrypted dataset to mount effective inference attacks, implying there is a database leakage of such documents, thus, an assumption that may not hold in real-world scenarios. In this work, we investigate the feasibility of enhancing leakage attacks under a more realistic threat model in which adversaries have access to minimal leaked data. We propose a novel approach that leverages large language models (LLMs), specifically GPT-4 variants, to generate synthetic documents that statistically and semantically resemble the real-world dataset of Enron emails. Using the email corpus as a case study, we evaluate the effectiveness of synthetic data generated via random sampling and hierarchical clustering methods on the performance of the SAP (Search Access Pattern) keyword inference attack restricted to token volumes only. Our results demonstrate that, while the choice of LLM has limited effect, increasing dataset size and employing clustering-based generation significantly improve attack accuracy, achieving comparable performance to attacks using larger amounts of real data. We highlight the growing relevance of LLMs in adversarial contexts.



## **12. Inception: Jailbreak the Memory Mechanism of Text-to-Image Generation Systems**

cs.CV

17 pages, 8 figures

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20376v1) [paper-pdf](http://arxiv.org/pdf/2504.20376v1)

**Authors**: Shiqian Zhao, Jiayang Liu, Yiming Li, Runyi Hu, Xiaojun Jia, Wenshu Fan, Xinfeng Li, Jie Zhang, Wei Dong, Tianwei Zhang, Luu Anh Tuan

**Abstract**: Currently, the memory mechanism has been widely and successfully exploited in online text-to-image (T2I) generation systems ($e.g.$, DALL$\cdot$E 3) for alleviating the growing tokenization burden and capturing key information in multi-turn interactions. Despite its practicality, its security analyses have fallen far behind. In this paper, we reveal that this mechanism exacerbates the risk of jailbreak attacks. Different from previous attacks that fuse the unsafe target prompt into one ultimate adversarial prompt, which can be easily detected or may generate non-unsafe images due to under- or over-optimization, we propose Inception, the first multi-turn jailbreak attack against the memory mechanism in real-world text-to-image generation systems. Inception embeds the malice at the inception of the chat session turn by turn, leveraging the mechanism that T2I generation systems retrieve key information in their memory. Specifically, Inception mainly consists of two modules. It first segments the unsafe prompt into chunks, which are subsequently fed to the system in multiple turns, serving as pseudo-gradients for directive optimization. Specifically, we develop a series of segmentation policies that ensure the images generated are semantically consistent with the target prompt. Secondly, after segmentation, to overcome the challenge of the inseparability of minimum unsafe words, we propose recursion, a strategy that makes minimum unsafe words subdivisible. Collectively, segmentation and recursion ensure that all the request prompts are benign but can lead to malicious outcomes. We conduct experiments on the real-world text-to-image generation system ($i.e.$, DALL$\cdot$E 3) to validate the effectiveness of Inception. The results indicate that Inception surpasses the state-of-the-art by a 14\% margin in attack success rate.



## **13. A Cryptographic Perspective on Mitigation vs. Detection in Machine Learning**

cs.LG

29 pages

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.20310v1) [paper-pdf](http://arxiv.org/pdf/2504.20310v1)

**Authors**: Greg Gluch, Shafi Goldwasser

**Abstract**: In this paper, we initiate a cryptographically inspired theoretical study of detection versus mitigation of adversarial inputs produced by attackers of Machine Learning algorithms during inference time.   We formally define defense by detection (DbD) and defense by mitigation (DbM). Our definitions come in the form of a 3-round protocol between two resource-bounded parties: a trainer/defender and an attacker. The attacker aims to produce inference-time inputs that fool the training algorithm. We define correctness, completeness, and soundness properties to capture successful defense at inference time while not degrading (too much) the performance of the algorithm on inputs from the training distribution.   We first show that achieving DbD and achieving DbM are equivalent for ML classification tasks. Surprisingly, this is not the case for ML generative learning tasks, where there are many possible correct outputs that can be generated for each input. We show a separation between DbD and DbM by exhibiting a generative learning task for which is possible to defend by mitigation but is provably impossible to defend by detection under the assumption that the Identity-Based Fully Homomorphic Encryption (IB-FHE), publicly-verifiable zero-knowledge Succinct Non-Interactive Arguments of Knowledge (zk-SNARK) and Strongly Unforgeable Signatures exist. The mitigation phase uses significantly fewer samples than the initial training algorithm.



## **14. The Dark Side of Digital Twins: Adversarial Attacks on AI-Driven Water Forecasting**

cs.LG

7 Pages, 7 Figures

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.20295v1) [paper-pdf](http://arxiv.org/pdf/2504.20295v1)

**Authors**: Mohammadhossein Homaei, Victor Gonzalez Morales, Oscar Mogollon-Gutierrez, Andres Caro

**Abstract**: Digital twins (DTs) are improving water distribution systems by using real-time data, analytics, and prediction models to optimize operations. This paper presents a DT platform designed for a Spanish water supply network, utilizing Long Short-Term Memory (LSTM) networks to predict water consumption. However, machine learning models are vulnerable to adversarial attacks, such as the Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD). These attacks manipulate critical model parameters, injecting subtle distortions that degrade forecasting accuracy. To further exploit these vulnerabilities, we introduce a Learning Automata (LA) and Random LA-based approach that dynamically adjusts perturbations, making adversarial attacks more difficult to detect. Experimental results show that this approach significantly impacts prediction reliability, causing the Mean Absolute Percentage Error (MAPE) to rise from 26% to over 35%. Moreover, adaptive attack strategies amplify this effect, highlighting cybersecurity risks in AI-driven DTs. These findings emphasize the urgent need for robust defenses, including adversarial training, anomaly detection, and secure data pipelines.



## **15. A Case Study on the Use of Representativeness Bias as a Defense Against Adversarial Cyber Threats**

cs.CR

To appear in the 4th Workshop on Active Defense and Deception (ADnD),  co-located with the 10th IEEE European Symposium on Security and Privacy  (EuroS&P 2025)

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.20245v1) [paper-pdf](http://arxiv.org/pdf/2504.20245v1)

**Authors**: Briland Hitaj, Grit Denker, Laura Tinnel, Michael McAnally, Bruce DeBruhl, Nathan Bunting, Alex Fafard, Daniel Aaron, Richard D. Roberts, Joshua Lawson, Greg McCain, Dylan Starink

**Abstract**: Cyberspace is an ever-evolving battleground involving adversaries seeking to circumvent existing safeguards and defenders aiming to stay one step ahead by predicting and mitigating the next threat. Existing mitigation strategies have focused primarily on solutions that consider software or hardware aspects, often ignoring the human factor. This paper takes a first step towards psychology-informed, active defense strategies, where we target biases that human beings are susceptible to under conditions of uncertainty.   Using capture-the-flag events, we create realistic challenges that tap into a particular cognitive bias: representativeness. This study finds that this bias can be triggered to thwart hacking attempts and divert hackers into non-vulnerable attack paths. Participants were exposed to two different challenges designed to exploit representativeness biases. One of the representativeness challenges significantly thwarted attackers away from vulnerable attack vectors and onto non-vulnerable paths, signifying an effective bias-based defense mechanism. This work paves the way towards cyber defense strategies that leverage additional human biases to thwart future, sophisticated adversarial attacks.



## **16. DROP: Poison Dilution via Knowledge Distillation for Federated Learning**

cs.LG

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2502.07011v2) [paper-pdf](http://arxiv.org/pdf/2502.07011v2)

**Authors**: Georgios Syros, Anshuman Suri, Farinaz Koushanfar, Cristina Nita-Rotaru, Alina Oprea

**Abstract**: Federated Learning is vulnerable to adversarial manipulation, where malicious clients can inject poisoned updates to influence the global model's behavior. While existing defense mechanisms have made notable progress, they fail to protect against adversaries that aim to induce targeted backdoors under different learning and attack configurations. To address this limitation, we introduce DROP (Distillation-based Reduction Of Poisoning), a novel defense mechanism that combines clustering and activity-tracking techniques with extraction of benign behavior from clients via knowledge distillation to tackle stealthy adversaries that manipulate low data poisoning rates and diverse malicious client ratios within the federation. Through extensive experimentation, our approach demonstrates superior robustness compared to existing defenses across a wide range of learning configurations. Finally, we evaluate existing defenses and our method under the challenging setting of non-IID client data distribution and highlight the challenges of designing a resilient FL defense in this setting.



## **17. Evaluate-and-Purify: Fortifying Code Language Models Against Adversarial Attacks Using LLM-as-a-Judge**

cs.SE

25 pages, 6 figures

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.19730v1) [paper-pdf](http://arxiv.org/pdf/2504.19730v1)

**Authors**: Wenhan Mu, Ling Xu, Shuren Pei, Le Mi, Huichi Zhou

**Abstract**: The widespread adoption of code language models in software engineering tasks has exposed vulnerabilities to adversarial attacks, especially the identifier substitution attacks. Although existing identifier substitution attackers demonstrate high success rates, they often produce adversarial examples with unnatural code patterns. In this paper, we systematically assess the quality of adversarial examples using LLM-as-a-Judge. Our analysis reveals that over 80% of adversarial examples generated by state-of-the-art identifier substitution attackers (e.g., ALERT) are actually detectable. Based on this insight, we propose EP-Shield, a unified framework for evaluating and purifying identifier substitution attacks via naturalness-aware reasoning. Specifically, we first evaluate the naturalness of code and identify the perturbed adversarial code, then purify it so that the victim model can restore correct prediction. Extensive experiments demonstrate the superiority of EP-Shield over adversarial fine-tuning (up to 83.36% improvement) and its lightweight design 7B parameters) with GPT-4-level performance.



## **18. Fooling the Decoder: An Adversarial Attack on Quantum Error Correction**

quant-ph

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.19651v1) [paper-pdf](http://arxiv.org/pdf/2504.19651v1)

**Authors**: Jerome Lenssen, Alexandru Paler

**Abstract**: Neural network decoders are becoming essential for achieving fault-tolerant quantum computations. However, their internal mechanisms are poorly understood, hindering our ability to ensure their reliability and security against adversarial attacks. Leading machine learning decoders utilize recurrent and transformer models (e.g., AlphaQubit), with reinforcement learning (RL) playing a key role in training advanced transformer models (e.g., DeepSeek R1). In this work, we target a basic RL surface code decoder (DeepQ) to create the first adversarial attack on quantum error correction. By applying state-of-the-art white-box methods, we uncover vulnerabilities in this decoder, demonstrating an attack that reduces the logical qubit lifetime in memory experiments by up to five orders of magnitude. We validate that this attack exploits a genuine weakness, as the decoder exhibits robustness against noise fluctuations, is largely unaffected by substituting the referee decoder, responsible for episode termination, with an MWPM decoder, and demonstrates fault tolerance at checkable code distances. This attack highlights the susceptibility of machine learning-based QEC and underscores the importance of further research into robust QEC methods.



## **19. FCGHunter: Towards Evaluating Robustness of Graph-Based Android Malware Detection**

cs.CR

14 pages, 5 figures

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.19456v1) [paper-pdf](http://arxiv.org/pdf/2504.19456v1)

**Authors**: Shiwen Song, Xiaofei Xie, Ruitao Feng, Qi Guo, Sen Chen

**Abstract**: Graph-based detection methods leveraging Function Call Graphs (FCGs) have shown promise for Android malware detection (AMD) due to their semantic insights. However, the deployment of malware detectors in dynamic and hostile environments raises significant concerns about their robustness. While recent approaches evaluate the robustness of FCG-based detectors using adversarial attacks, their effectiveness is constrained by the vast perturbation space, particularly across diverse models and features.   To address these challenges, we introduce FCGHunter, a novel robustness testing framework for FCG-based AMD systems. Specifically, FCGHunter employs innovative techniques to enhance exploration and exploitation within this huge search space. Initially, it identifies critical areas within the FCG related to malware behaviors to narrow down the perturbation space. We then develop a dependency-aware crossover and mutation method to enhance the validity and diversity of perturbations, generating diverse FCGs. Furthermore, FCGHunter leverages multi-objective feedback to select perturbed FCGs, significantly improving the search process with interpretation-based feature change feedback.   Extensive evaluations across 40 scenarios demonstrate that FCGHunter achieves an average attack success rate of 87.9%, significantly outperforming baselines by at least 44.7%. Notably, FCGHunter achieves a 100% success rate on robust models (e.g., AdaBoost with MalScan), where baselines achieve only 11% or are inapplicable.



## **20. Mitigating Evasion Attacks in Federated Learning-Based Signal Classifiers**

eess.SP

Accepted for publication in IEEE Transactions on Network Science and  Engineering. arXiv admin note: substantial text overlap with arXiv:2301.08866

**SubmitDate**: 2025-04-27    [abs](http://arxiv.org/abs/2306.04872v2) [paper-pdf](http://arxiv.org/pdf/2306.04872v2)

**Authors**: Su Wang, Rajeev Sahay, Adam Piaseczny, Christopher G. Brinton

**Abstract**: Recent interest in leveraging federated learning (FL) for radio signal classification (SC) tasks has shown promise but FL-based SC remains susceptible to model poisoning adversarial attacks. These adversarial attacks mislead the ML model training process, damaging ML models across the network and leading to lower SC performance. In this work, we seek to mitigate model poisoning adversarial attacks on FL-based SC by proposing the Underlying Server Defense of Federated Learning (USD-FL). Unlike existing server-driven defenses, USD-FL does not rely on perfect network information, i.e., knowing the quantity of adversaries, the adversarial attack architecture, or the start time of the adversarial attacks. Our proposed USD-FL methodology consists of deriving logits for devices' ML models on a reserve dataset, comparing pair-wise logits via 1-Wasserstein distance and then determining a time-varying threshold for adversarial detection. As a result, USD-FL effectively mitigates model poisoning attacks introduced in the FL network. Specifically, when baseline server-driven defenses do have perfect network information, USD-FL outperforms them by (i) improving final ML classification accuracies by at least 6%, (ii) reducing false positive adversary detection rates by at least 10%, and (iii) decreasing the total number of misclassified signals by over 8%. Moreover, when baseline defenses do not have perfect network information, we show that USD-FL achieves accuracies of approximately 74.1% and 62.5% in i.i.d. and non-i.i.d. settings, outperforming existing server-driven baselines, which achieve 52.1% and 39.2% in i.i.d. and non-i.i.d. settings, respectively.



## **21. Forging and Removing Latent-Noise Diffusion Watermarks Using a Single Image**

cs.CV

**SubmitDate**: 2025-04-27    [abs](http://arxiv.org/abs/2504.20111v1) [paper-pdf](http://arxiv.org/pdf/2504.20111v1)

**Authors**: Anubhav Jain, Yuya Kobayashi, Naoki Murata, Yuhta Takida, Takashi Shibuya, Yuki Mitsufuji, Niv Cohen, Nasir Memon, Julian Togelius

**Abstract**: Watermarking techniques are vital for protecting intellectual property and preventing fraudulent use of media. Most previous watermarking schemes designed for diffusion models embed a secret key in the initial noise. The resulting pattern is often considered hard to remove and forge into unrelated images. In this paper, we propose a black-box adversarial attack without presuming access to the diffusion model weights. Our attack uses only a single watermarked example and is based on a simple observation: there is a many-to-one mapping between images and initial noises. There are regions in the clean image latent space pertaining to each watermark that get mapped to the same initial noise when inverted. Based on this intuition, we propose an adversarial attack to forge the watermark by introducing perturbations to the images such that we can enter the region of watermarked images. We show that we can also apply a similar approach for watermark removal by learning perturbations to exit this region. We report results on multiple watermarking schemes (Tree-Ring, RingID, WIND, and Gaussian Shading) across two diffusion models (SDv1.4 and SDv2.0). Our results demonstrate the effectiveness of the attack and expose vulnerabilities in the watermarking methods, motivating future research on improving them.



## **22. CapsFake: A Multimodal Capsule Network for Detecting Instruction-Guided Deepfakes**

cs.CV

20 pages

**SubmitDate**: 2025-04-27    [abs](http://arxiv.org/abs/2504.19212v1) [paper-pdf](http://arxiv.org/pdf/2504.19212v1)

**Authors**: Tuan Nguyen, Naseem Khan, Issa Khalil

**Abstract**: The rapid evolution of deepfake technology, particularly in instruction-guided image editing, threatens the integrity of digital images by enabling subtle, context-aware manipulations. Generated conditionally from real images and textual prompts, these edits are often imperceptible to both humans and existing detection systems, revealing significant limitations in current defenses. We propose a novel multimodal capsule network, CapsFake, designed to detect such deepfake image edits by integrating low-level capsules from visual, textual, and frequency-domain modalities. High-level capsules, predicted through a competitive routing mechanism, dynamically aggregate local features to identify manipulated regions with precision. Evaluated on diverse datasets, including MagicBrush, Unsplash Edits, Open Images Edits, and Multi-turn Edits, CapsFake outperforms state-of-the-art methods by up to 20% in detection accuracy. Ablation studies validate its robustness, achieving detection rates above 94% under natural perturbations and 96% against adversarial attacks, with excellent generalization to unseen editing scenarios. This approach establishes a powerful framework for countering sophisticated image manipulations.



## **23. Support is All You Need for Certified VAE Training**

cs.LG

21 pages, 3 figures, ICLR '25

**SubmitDate**: 2025-04-27    [abs](http://arxiv.org/abs/2504.11831v2) [paper-pdf](http://arxiv.org/pdf/2504.11831v2)

**Authors**: Changming Xu, Debangshu Banerjee, Deepak Vasisht, Gagandeep Singh

**Abstract**: Variational Autoencoders (VAEs) have become increasingly popular and deployed in safety-critical applications. In such applications, we want to give certified probabilistic guarantees on performance under adversarial attacks. We propose a novel method, CIVET, for certified training of VAEs. CIVET depends on the key insight that we can bound worst-case VAE error by bounding the error on carefully chosen support sets at the latent layer. We show this point mathematically and present a novel training algorithm utilizing this insight. We show in an extensive evaluation across different datasets (in both the wireless and vision application areas), architectures, and perturbation magnitudes that our method outperforms SOTA methods achieving good standard performance with strong robustness guarantees.



## **24. Graph of Attacks: Improved Black-Box and Interpretable Jailbreaks for LLMs**

cs.CL

19 pages, 1 figure, 6 tables

**SubmitDate**: 2025-04-26    [abs](http://arxiv.org/abs/2504.19019v1) [paper-pdf](http://arxiv.org/pdf/2504.19019v1)

**Authors**: Mohammad Akbar-Tajari, Mohammad Taher Pilehvar, Mohammad Mahmoody

**Abstract**: The challenge of ensuring Large Language Models (LLMs) align with societal standards is of increasing interest, as these models are still prone to adversarial jailbreaks that bypass their safety mechanisms. Identifying these vulnerabilities is crucial for enhancing the robustness of LLMs against such exploits. We propose Graph of ATtacks (GoAT), a method for generating adversarial prompts to test the robustness of LLM alignment using the Graph of Thoughts framework [Besta et al., 2024]. GoAT excels at generating highly effective jailbreak prompts with fewer queries to the victim model than state-of-the-art attacks, achieving up to five times better jailbreak success rate against robust models like Llama. Notably, GoAT creates high-quality, human-readable prompts without requiring access to the targeted model's parameters, making it a black-box attack. Unlike approaches constrained by tree-based reasoning, GoAT's reasoning is based on a more intricate graph structure. By making simultaneous attack paths aware of each other's progress, this dynamic framework allows a deeper integration and refinement of reasoning paths, significantly enhancing the collaborative exploration of adversarial vulnerabilities in LLMs. At a technical level, GoAT starts with a graph structure and iteratively refines it by combining and improving thoughts, enabling synergy between different thought paths. The code for our implementation can be found at: https://github.com/GoAT-pydev/Graph_of_Attacks.



## **25. Unveiling and Mitigating Adversarial Vulnerabilities in Iterative Optimizers**

cs.LG

Under review for publication in the IEEE

**SubmitDate**: 2025-04-26    [abs](http://arxiv.org/abs/2504.19000v1) [paper-pdf](http://arxiv.org/pdf/2504.19000v1)

**Authors**: Elad Sofer, Tomer Shaked, Caroline Chaux, Nir Shlezinger

**Abstract**: Machine learning (ML) models are often sensitive to carefully crafted yet seemingly unnoticeable perturbations. Such adversarial examples are considered to be a property of ML models, often associated with their black-box operation and sensitivity to features learned from data. This work examines the adversarial sensitivity of non-learned decision rules, and particularly of iterative optimizers. Our analysis is inspired by the recent developments in deep unfolding, which cast such optimizers as ML models. We show that non-learned iterative optimizers share the sensitivity to adversarial examples of ML models, and that attacking iterative optimizers effectively alters the optimization objective surface in a manner that modifies the minima sought. We then leverage the ability to cast iteration-limited optimizers as ML models to enhance robustness via adversarial training. For a class of proximal gradient optimizers, we rigorously prove how their learning affects adversarial sensitivity. We numerically back our findings, showing the vulnerability of various optimizers, as well as the robustness induced by unfolding and adversarial training.



## **26. Federated Learning-based Semantic Segmentation for Lane and Object Detection in Autonomous Driving**

eess.SY

This paper has been accepted for publication in Scientific Reports

**SubmitDate**: 2025-04-26    [abs](http://arxiv.org/abs/2504.18939v1) [paper-pdf](http://arxiv.org/pdf/2504.18939v1)

**Authors**: Gharbi Khamis Alshammari, Ahmad Abubakar, Nada M. O. Sid Ahmed, Naif Khalaf Alshammari

**Abstract**: Autonomous Vehicles (AVs) require precise lane and object detection to ensure safe navigation. However, centralized deep learning (DL) approaches for semantic segmentation raise privacy and scalability challenges, particularly when handling sensitive data. This research presents a new federated learning (FL) framework that integrates secure deep Convolutional Neural Networks (CNNs) and Differential Privacy (DP) to address these issues. The core contribution of this work involves: (1) developing a new hybrid UNet-ResNet34 architecture for centralized semantic segmentation to achieve high accuracy and tackle privacy concerns due to centralized training, and (2) implementing the privacy-preserving FL model, distributed across AVs to enhance performance through secure CNNs and DP mechanisms. In the proposed FL framework, the methodology distinguishes itself from the existing approach through the following: (a) ensuring data decentralization through FL to uphold user privacy by eliminating the need for centralized data aggregation, (b) integrating DP mechanisms to secure sensitive model updates against potential adversarial inference attacks, and (c) evaluating the frameworks performance and generalizability using RGB and semantic segmentation datasets derived from the CARLA simulator. Experimental results show significant improvements in accuracy, from 81.5% to 88.7% for the RGB dataset and from 79.3% to 86.9% for the SEG dataset over 20 to 70 Communication Rounds (CRs). Global loss was reduced by over 60%, and minor accuracy trade-offs from DP were observed. This study contributes by offering a scalable, privacy-preserving FL framework tailored for AVs, optimizing communication efficiency while balancing performance and data security.



## **27. Latent Adversarial Training Improves the Representation of Refusal**

cs.CL

**SubmitDate**: 2025-04-26    [abs](http://arxiv.org/abs/2504.18872v1) [paper-pdf](http://arxiv.org/pdf/2504.18872v1)

**Authors**: Alexandra Abbas, Nora Petrova, Helios Ael Lyons, Natalia Perez-Campanero

**Abstract**: Recent work has shown that language models' refusal behavior is primarily encoded in a single direction in their latent space, making it vulnerable to targeted attacks. Although Latent Adversarial Training (LAT) attempts to improve robustness by introducing noise during training, a key question remains: How does this noise-based training affect the underlying representation of refusal behavior? Understanding this encoding is crucial for evaluating LAT's effectiveness and limitations, just as the discovery of linear refusal directions revealed vulnerabilities in traditional supervised safety fine-tuning (SSFT).   Through the analysis of Llama 2 7B, we examine how LAT reorganizes the refusal behavior in the model's latent space compared to SSFT and embedding space adversarial training (AT). By computing activation differences between harmful and harmless instruction pairs and applying Singular Value Decomposition (SVD), we find that LAT significantly alters the refusal representation, concentrating it in the first two SVD components which explain approximately 75 percent of the activation differences variance - significantly higher than in reference models. This concentrated representation leads to more effective and transferable refusal vectors for ablation attacks: LAT models show improved robustness when attacked with vectors from reference models but become more vulnerable to self-generated vectors compared to SSFT and AT. Our findings suggest that LAT's training perturbations enable a more comprehensive representation of refusal behavior, highlighting both its potential strengths and vulnerabilities for improving model safety.



## **28. SynFuzz: Leveraging Fuzzing of Netlist to Detect Synthesis Bugs**

cs.CR

14 pages, 10 figures, 4 tables

**SubmitDate**: 2025-04-26    [abs](http://arxiv.org/abs/2504.18812v1) [paper-pdf](http://arxiv.org/pdf/2504.18812v1)

**Authors**: Raghul Saravanan, Sudipta Paria, Aritra Dasgupta, Venkat Nitin Patnala, Swarup Bhunia, Sai Manoj P D

**Abstract**: In the evolving landscape of integrated circuit (IC) design, the increasing complexity of modern processors and intellectual property (IP) cores has introduced new challenges in ensuring design correctness and security. The recent advancements in hardware fuzzing techniques have shown their efficacy in detecting hardware bugs and vulnerabilities at the RTL abstraction level of hardware. However, they suffer from several limitations, including an inability to address vulnerabilities introduced during synthesis and gate-level transformations. These methods often fail to detect issues arising from library adversaries, where compromised or malicious library components can introduce backdoors or unintended behaviors into the design. In this paper, we present a novel hardware fuzzer, SynFuzz, designed to overcome the limitations of existing hardware fuzzing frameworks. SynFuzz focuses on fuzzing hardware at the gate-level netlist to identify synthesis bugs and vulnerabilities that arise during the transition from RTL to the gate-level. We analyze the intrinsic hardware behaviors using coverage metrics specifically tailored for the gate-level. Furthermore, SynFuzz implements differential fuzzing to uncover bugs associated with EDA libraries. We evaluated SynFuzz on popular open-source processors and IP designs, successfully identifying 7 new synthesis bugs. Additionally, by exploiting the optimization settings of EDA tools, we performed a compromised library mapping attack (CLiMA), creating a malicious version of hardware designs that remains undetectable by traditional verification methods. We also demonstrate how SynFuzz overcomes the limitations of the industry-standard formal verification tool, Cadence Conformal, providing a more robust and comprehensive approach to hardware verification.



## **29. Intelligent Attacks and Defense Methods in Federated Learning-enabled Energy-Efficient Wireless Networks**

cs.LG

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.18519v1) [paper-pdf](http://arxiv.org/pdf/2504.18519v1)

**Authors**: Han Zhang, Hao Zhou, Medhat Elsayed, Majid Bavand, Raimundas Gaigalas, Yigit Ozcan, Melike Erol-Kantarci

**Abstract**: Federated learning (FL) is a promising technique for learning-based functions in wireless networks, thanks to its distributed implementation capability. On the other hand, distributed learning may increase the risk of exposure to malicious attacks where attacks on a local model may spread to other models by parameter exchange. Meanwhile, such attacks can be hard to detect due to the dynamic wireless environment, especially considering local models can be heterogeneous with non-independent and identically distributed (non-IID) data. Therefore, it is critical to evaluate the effect of malicious attacks and develop advanced defense techniques for FL-enabled wireless networks. In this work, we introduce a federated deep reinforcement learning-based cell sleep control scenario that enhances the energy efficiency of the network. We propose multiple intelligent attacks targeting the learning-based approach and we propose defense methods to mitigate such attacks. In particular, we have designed two attack models, generative adversarial network (GAN)-enhanced model poisoning attack and regularization-based model poisoning attack. As a counteraction, we have proposed two defense schemes, autoencoder-based defense, and knowledge distillation (KD)-enabled defense. The autoencoder-based defense method leverages an autoencoder to identify the malicious participants and only aggregate the parameters of benign local models during the global aggregation, while KD-based defense protects the model from attacks by controlling the knowledge transferred between the global model and local models.



## **30. Distributed Multiple Testing with False Discovery Rate Control in the Presence of Byzantines**

eess.SP

Accepted to the 2025 International Symposium on Information Theory  (ISIT)

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2501.13242v2) [paper-pdf](http://arxiv.org/pdf/2501.13242v2)

**Authors**: Daofu Zhang, Mehrdad Pournaderi, Yu Xiang, Pramod Varshney

**Abstract**: This work studies distributed multiple testing with false discovery rate (FDR) control in the presence of Byzantine attacks, where an adversary captures a fraction of the nodes and corrupts their reported p-values. We focus on two baseline attack models: an oracle model with the full knowledge of which hypotheses are true nulls, and a practical attack model that leverages the Benjamini-Hochberg (BH) procedure locally to classify which p-values follow the true null hypotheses. We provide a thorough characterization of how both attack models affect the global FDR, which in turn motivates counter-attack strategies and stronger attack models. Our extensive simulation studies confirm the theoretical results, highlight key design trade-offs under attacks and countermeasures, and provide insights into more sophisticated attacks.



## **31. Adversarial Attacks on LLM-as-a-Judge Systems: Insights from Prompt Injections**

cs.CR

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.18333v1) [paper-pdf](http://arxiv.org/pdf/2504.18333v1)

**Authors**: Narek Maloyan, Dmitry Namiot

**Abstract**: LLM as judge systems used to assess text quality code correctness and argument strength are vulnerable to prompt injection attacks. We introduce a framework that separates content author attacks from system prompt attacks and evaluate five models Gemma 3.27B Gemma 3.4B Llama 3.2 3B GPT 4 and Claude 3 Opus on four tasks with various defenses using fifty prompts per condition. Attacks achieved up to seventy three point eight percent success smaller models proved more vulnerable and transferability ranged from fifty point five to sixty two point six percent. Our results contrast with Universal Prompt Injection and AdvPrompter We recommend multi model committees and comparative scoring and release all code and datasets



## **32. Contrastive Learning and Adversarial Disentanglement for Task-Oriented Semantic Communications**

cs.LG

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2410.22784v2) [paper-pdf](http://arxiv.org/pdf/2410.22784v2)

**Authors**: Omar Erak, Omar Alhussein, Wen Tong

**Abstract**: Task-oriented semantic communication systems have emerged as a promising approach to achieving efficient and intelligent data transmission, where only information relevant to a specific task is communicated. However, existing methods struggle to fully disentangle task-relevant and task-irrelevant information, leading to privacy concerns and subpar performance. To address this, we propose an information-bottleneck method, named CLAD (contrastive learning and adversarial disentanglement). CLAD utilizes contrastive learning to effectively capture task-relevant features while employing adversarial disentanglement to discard task-irrelevant information. Additionally, due to the lack of reliable and reproducible methods to gain insight into the informativeness and minimality of the encoded feature vectors, we introduce a new technique to compute the information retention index (IRI), a comparative metric used as a proxy for the mutual information between the encoded features and the input, reflecting the minimality of the encoded features. The IRI quantifies the minimality and informativeness of the encoded feature vectors across different task-oriented communication techniques. Our extensive experiments demonstrate that CLAD outperforms state-of-the-art baselines in terms of semantic extraction, task performance, privacy preservation, and IRI. CLAD achieves a predictive performance improvement of around 2.5-3%, along with a 77-90% reduction in IRI and a 57-76% decrease in adversarial attribute inference attack accuracy.



## **33. Manipulating Multimodal Agents via Cross-Modal Prompt Injection**

cs.CV

17 pages, 5 figures

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.14348v3) [paper-pdf](http://arxiv.org/pdf/2504.14348v3)

**Authors**: Le Wang, Zonghao Ying, Tianyuan Zhang, Siyuan Liang, Shengshan Hu, Mingchuan Zhang, Aishan Liu, Xianglong Liu

**Abstract**: The emergence of multimodal large language models has redefined the agent paradigm by integrating language and vision modalities with external data sources, enabling agents to better interpret human instructions and execute increasingly complex tasks. However, in this work, we identify a critical yet previously overlooked security vulnerability in multimodal agents: cross-modal prompt injection attacks. To exploit this vulnerability, we propose CrossInject, a novel attack framework in which attackers embed adversarial perturbations across multiple modalities to align with target malicious content, allowing external instructions to hijack the agent's decision-making process and execute unauthorized tasks. Our approach consists of two key components. First, we introduce Visual Latent Alignment, where we optimize adversarial features to the malicious instructions in the visual embedding space based on a text-to-image generative model, ensuring that adversarial images subtly encode cues for malicious task execution. Subsequently, we present Textual Guidance Enhancement, where a large language model is leveraged to infer the black-box defensive system prompt through adversarial meta prompting and generate an malicious textual command that steers the agent's output toward better compliance with attackers' requests. Extensive experiments demonstrate that our method outperforms existing injection attacks, achieving at least a +26.4% increase in attack success rates across diverse tasks. Furthermore, we validate our attack's effectiveness in real-world multimodal autonomous agents, highlighting its potential implications for safety-critical applications.



## **34. Robust Kernel Hypothesis Testing under Data Corruption**

stat.ML

22 pages, 2 figures, 2 algorithms

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2405.19912v3) [paper-pdf](http://arxiv.org/pdf/2405.19912v3)

**Authors**: Antonin Schrab, Ilmun Kim

**Abstract**: We propose a general method for constructing robust permutation tests under data corruption. The proposed tests effectively control the non-asymptotic type I error under data corruption, and we prove their consistency in power under minimal conditions. This contributes to the practical deployment of hypothesis tests for real-world applications with potential adversarial attacks. For the two-sample and independence settings, we show that our kernel robust tests are minimax optimal, in the sense that they are guaranteed to be non-asymptotically powerful against alternatives uniformly separated from the null in the kernel MMD and HSIC metrics at some optimal rate (tight with matching lower bound). We point out that existing differentially private tests can be adapted to be robust to data corruption, and we demonstrate in experiments that our proposed tests achieve much higher power than these private tests. Finally, we provide publicly available implementations and empirically illustrate the practicality of our robust tests.



## **35. Generative AI for Physical-Layer Authentication**

eess.SP

10 pages, 3 figures

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.18175v1) [paper-pdf](http://arxiv.org/pdf/2504.18175v1)

**Authors**: Rui Meng, Xiqi Cheng, Song Gao, Xiaodong Xu, Chen Dong, Guoshun Nan, Xiaofeng Tao, Ping Zhang, Tony Q. S. Quek

**Abstract**: In recent years, Artificial Intelligence (AI)-driven Physical-Layer Authentication (PLA), which focuses on achieving endogenous security and intelligent identity authentication, has attracted considerable interest. When compared with Discriminative AI (DAI), Generative AI (GAI) offers several advantages, such as fingerprint data augmentation, fingerprint denoising and reconstruction, and protection against adversarial attacks. Inspired by these innovations, this paper provides a systematic exploration of GAI's integration into PLA frameworks. We commence with a review of representative authentication techniques, emphasizing PLA's inherent strengths. Following this, we revisit four typical GAI models and contrast the limitations of DAI with the potential of GAI in addressing PLA challenges, including insufficient fingerprint data, environment noises and inferences, perturbations in fingerprint data, and complex tasks. Specifically, we delve into providing GAI-enhance methods for PLA across the data, model, and application layers in detail. Moreover, we present a case study that combines fingerprint extrapolation, generative diffusion models, and cooperative nodes to illustrate the superiority of GAI in bolstering the reliability of PLA compared to DAI. Additionally, we outline potential future research directions for GAI-based PLA.



## **36. Revisiting Locally Differentially Private Protocols: Towards Better Trade-offs in Privacy, Utility, and Attack Resistance**

cs.CR

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2503.01482v2) [paper-pdf](http://arxiv.org/pdf/2503.01482v2)

**Authors**: Héber H. Arcolezi, Sébastien Gambs

**Abstract**: Local Differential Privacy (LDP) offers strong privacy protection, especially in settings in which the server collecting the data is untrusted. However, designing LDP mechanisms that achieve an optimal trade-off between privacy, utility and robustness to adversarial inference attacks remains challenging. In this work, we introduce a general multi-objective optimization framework for refining LDP protocols, enabling the joint optimization of privacy and utility under various adversarial settings. While our framework is flexible to accommodate multiple privacy and security attacks as well as utility metrics, in this paper, we specifically optimize for Attacker Success Rate (ASR) under \emph{data reconstruction attack} as a concrete measure of privacy leakage and Mean Squared Error (MSE) as a measure of utility. More precisely, we systematically revisit these trade-offs by analyzing eight state-of-the-art LDP protocols and proposing refined counterparts that leverage tailored optimization techniques. Experimental results demonstrate that our proposed adaptive mechanisms consistently outperform their non-adaptive counterparts, achieving substantial reductions in ASR while preserving utility, and pushing closer to the ASR-MSE Pareto frontier. By bridging the gap between theoretical guarantees and real-world vulnerabilities, our framework enables modular and context-aware deployment of LDP mechanisms with tunable privacy-utility trade-offs.



## **37. A Parametric Approach to Adversarial Augmentation for Cross-Domain Iris Presentation Attack Detection**

cs.CV

IEEE/CVF Winter Conference on Applications of Computer Vision (WACV),  2025

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2412.07199v2) [paper-pdf](http://arxiv.org/pdf/2412.07199v2)

**Authors**: Debasmita Pal, Redwan Sony, Arun Ross

**Abstract**: Iris-based biometric systems are vulnerable to presentation attacks (PAs), where adversaries present physical artifacts (e.g., printed iris images, textured contact lenses) to defeat the system. This has led to the development of various presentation attack detection (PAD) algorithms, which typically perform well in intra-domain settings. However, they often struggle to generalize effectively in cross-domain scenarios, where training and testing employ different sensors, PA instruments, and datasets. In this work, we use adversarial training samples of both bonafide irides and PAs to improve the cross-domain performance of a PAD classifier. The novelty of our approach lies in leveraging transformation parameters from classical data augmentation schemes (e.g., translation, rotation) to generate adversarial samples. We achieve this through a convolutional autoencoder, ADV-GEN, that inputs original training samples along with a set of geometric and photometric transformations. The transformation parameters act as regularization variables, guiding ADV-GEN to generate adversarial samples in a constrained search space. Experiments conducted on the LivDet-Iris 2017 database, comprising four datasets, and the LivDet-Iris 2020 dataset, demonstrate the efficacy of our proposed method. The code is available at https://github.com/iPRoBe-lab/ADV-GEN-IrisPAD.



## **38. Edge-Based Learning for Improved Classification Under Adversarial Noise**

cs.CV

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.20077v1) [paper-pdf](http://arxiv.org/pdf/2504.20077v1)

**Authors**: Manish Kansana, Keyan Alexander Rahimi, Elias Hossain, Iman Dehzangi, Noorbakhsh Amiri Golilarz

**Abstract**: Adversarial noise introduces small perturbations in images, misleading deep learning models into misclassification and significantly impacting recognition accuracy. In this study, we analyzed the effects of Fast Gradient Sign Method (FGSM) adversarial noise on image classification and investigated whether training on specific image features can improve robustness. We hypothesize that while adversarial noise perturbs various regions of an image, edges may remain relatively stable and provide essential structural information for classification. To test this, we conducted a series of experiments using brain tumor and COVID datasets. Initially, we trained the models on clean images and then introduced subtle adversarial perturbations, which caused deep learning models to significantly misclassify the images. Retraining on a combination of clean and noisy images led to improved performance. To evaluate the robustness of the edge features, we extracted edges from the original/clean images and trained the models exclusively on edge-based representations. When noise was introduced to the images, the edge-based models demonstrated greater resilience to adversarial attacks compared to those trained on the original or clean images. These results suggest that while adversarial noise is able to exploit complex non-edge regions significantly more than edges, the improvement in the accuracy after retraining is marginally more in the original data as compared to the edges. Thus, leveraging edge-based learning can improve the resilience of deep learning models against adversarial perturbations.



## **39. Diffusion-Driven Universal Model Inversion Attack for Face Recognition**

cs.CR

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.18015v1) [paper-pdf](http://arxiv.org/pdf/2504.18015v1)

**Authors**: Hanrui Wang, Shuo Wang, Chun-Shien Lu, Isao Echizen

**Abstract**: Facial recognition technology poses significant privacy risks, as it relies on biometric data that is inherently sensitive and immutable if compromised. To mitigate these concerns, face recognition systems convert raw images into embeddings, traditionally considered privacy-preserving. However, model inversion attacks pose a significant privacy threat by reconstructing these private facial images, making them a crucial tool for evaluating the privacy risks of face recognition systems. Existing methods usually require training individual generators for each target model, a computationally expensive process. In this paper, we propose DiffUMI, a training-free diffusion-driven universal model inversion attack for face recognition systems. DiffUMI is the first approach to apply a diffusion model for unconditional image generation in model inversion. Unlike other methods, DiffUMI is universal, eliminating the need for training target-specific generators. It operates within a fixed framework and pretrained diffusion model while seamlessly adapting to diverse target identities and models. DiffUMI breaches privacy-preserving face recognition systems with state-of-the-art success, demonstrating that an unconditional diffusion model, coupled with optimized adversarial search, enables efficient and high-fidelity facial reconstruction. Additionally, we introduce a novel application of out-of-domain detection (OODD), marking the first use of model inversion to distinguish non-face inputs from face inputs based solely on embeddings.



## **40. Adversarial Attacks to Latent Representations of Distributed Neural Networks in Split Computing**

cs.LG

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2309.17401v4) [paper-pdf](http://arxiv.org/pdf/2309.17401v4)

**Authors**: Milin Zhang, Mohammad Abdi, Jonathan Ashdown, Francesco Restuccia

**Abstract**: Distributed deep neural networks (DNNs) have been shown to reduce the computational burden of mobile devices and decrease the end-to-end inference latency in edge computing scenarios. While distributed DNNs have been studied, to the best of our knowledge, the resilience of distributed DNNs to adversarial action remains an open problem. In this paper, we fill the existing research gap by rigorously analyzing the robustness of distributed DNNs against adversarial action. We cast this problem in the context of information theory and rigorously proved that (i) the compressed latent dimension improves the robustness but also affect task-oriented performance; and (ii) the deeper splitting point enhances the robustness but also increases the computational burden. These two trade-offs provide a novel perspective to design robust distributed DNN. To test our theoretical findings, we perform extensive experimental analysis by considering 6 different DNN architectures, 6 different approaches for distributed DNN and 10 different adversarial attacks using the ImageNet-1K dataset.



## **41. Cluster-Aware Attacks on Graph Watermarks**

cs.CR

15 pages, 16 figures

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17971v1) [paper-pdf](http://arxiv.org/pdf/2504.17971v1)

**Authors**: Alexander Nemecek, Emre Yilmaz, Erman Ayday

**Abstract**: Data from domains such as social networks, healthcare, finance, and cybersecurity can be represented as graph-structured information. Given the sensitive nature of this data and their frequent distribution among collaborators, ensuring secure and attributable sharing is essential. Graph watermarking enables attribution by embedding user-specific signatures into graph-structured data. While prior work has addressed random perturbation attacks, the threat posed by adversaries leveraging structural properties through community detection remains unexplored. In this work, we introduce a cluster-aware threat model in which adversaries apply community-guided modifications to evade detection. We propose two novel attack strategies and evaluate them on real-world social network graphs. Our results show that cluster-aware attacks can reduce attribution accuracy by up to 80% more than random baselines under equivalent perturbation budgets on sparse graphs. To mitigate this threat, we propose a lightweight embedding enhancement that distributes watermark nodes across graph communities. This approach improves attribution accuracy by up to 60% under attack on dense graphs, without increasing runtime or structural distortion. Our findings underscore the importance of cluster-topological awareness in both watermarking design and adversarial modeling.



## **42. ASVspoof 5: Design, Collection and Validation of Resources for Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech**

eess.AS

Database link: https://zenodo.org/records/14498691, Database mirror  link: https://huggingface.co/datasets/jungjee/asvspoof5, ASVspoof 5 Challenge  Workshop Proceeding: https://www.isca-archive.org/asvspoof_2024/index.html

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2502.08857v4) [paper-pdf](http://arxiv.org/pdf/2502.08857v4)

**Authors**: Xin Wang, Héctor Delgado, Hemlata Tak, Jee-weon Jung, Hye-jin Shim, Massimiliano Todisco, Ivan Kukanov, Xuechen Liu, Md Sahidullah, Tomi Kinnunen, Nicholas Evans, Kong Aik Lee, Junichi Yamagishi, Myeonghun Jeong, Ge Zhu, Yongyi Zang, You Zhang, Soumi Maiti, Florian Lux, Nicolas Müller, Wangyou Zhang, Chengzhe Sun, Shuwei Hou, Siwei Lyu, Sébastien Le Maguer, Cheng Gong, Hanjie Guo, Liping Chen, Vishwanath Singh

**Abstract**: ASVspoof 5 is the fifth edition in a series of challenges which promote the study of speech spoofing and deepfake attacks as well as the design of detection solutions. We introduce the ASVspoof 5 database which is generated in a crowdsourced fashion from data collected in diverse acoustic conditions (cf. studio-quality data for earlier ASVspoof databases) and from ~2,000 speakers (cf. ~100 earlier). The database contains attacks generated with 32 different algorithms, also crowdsourced, and optimised to varying degrees using new surrogate detection models. Among them are attacks generated with a mix of legacy and contemporary text-to-speech synthesis and voice conversion models, in addition to adversarial attacks which are incorporated for the first time. ASVspoof 5 protocols comprise seven speaker-disjoint partitions. They include two distinct partitions for the training of different sets of attack models, two more for the development and evaluation of surrogate detection models, and then three additional partitions which comprise the ASVspoof 5 training, development and evaluation sets. An auxiliary set of data collected from an additional 30k speakers can also be used to train speaker encoders for the implementation of attack algorithms. Also described herein is an experimental validation of the new ASVspoof 5 database using a set of automatic speaker verification and spoof/deepfake baseline detectors. With the exception of protocols and tools for the generation of spoofed/deepfake speech, the resources described in this paper, already used by participants of the ASVspoof 5 challenge in 2024, are now all freely available to the community.



## **43. DCT-Shield: A Robust Frequency Domain Defense against Malicious Image Editing**

cs.CV

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17894v1) [paper-pdf](http://arxiv.org/pdf/2504.17894v1)

**Authors**: Aniruddha Bala, Rohit Chowdhury, Rohan Jaiswal, Siddharth Roheda

**Abstract**: Advancements in diffusion models have enabled effortless image editing via text prompts, raising concerns about image security. Attackers with access to user images can exploit these tools for malicious edits. Recent defenses attempt to protect images by adding a limited noise in the pixel space to disrupt the functioning of diffusion-based editing models. However, the adversarial noise added by previous methods is easily noticeable to the human eye. Moreover, most of these methods are not robust to purification techniques like JPEG compression under a feasible pixel budget. We propose a novel optimization approach that introduces adversarial perturbations directly in the frequency domain by modifying the Discrete Cosine Transform (DCT) coefficients of the input image. By leveraging the JPEG pipeline, our method generates adversarial images that effectively prevent malicious image editing. Extensive experiments across a variety of tasks and datasets demonstrate that our approach introduces fewer visual artifacts while maintaining similar levels of edit protection and robustness to noise purification techniques.



## **44. Unsupervised Corpus Poisoning Attacks in Continuous Space for Dense Retrieval**

cs.IR

This paper has been accepted as a full paper at SIGIR 2025 and will  be presented orally

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17884v1) [paper-pdf](http://arxiv.org/pdf/2504.17884v1)

**Authors**: Yongkang Li, Panagiotis Eustratiadis, Simon Lupart, Evangelos Kanoulas

**Abstract**: This paper concerns corpus poisoning attacks in dense information retrieval, where an adversary attempts to compromise the ranking performance of a search algorithm by injecting a small number of maliciously generated documents into the corpus. Our work addresses two limitations in the current literature. First, attacks that perform adversarial gradient-based word substitution search do so in the discrete lexical space, while retrieval itself happens in the continuous embedding space. We thus propose an optimization method that operates in the embedding space directly. Specifically, we train a perturbation model with the objective of maintaining the geometric distance between the original and adversarial document embeddings, while also maximizing the token-level dissimilarity between the original and adversarial documents. Second, it is common for related work to have a strong assumption that the adversary has prior knowledge about the queries. In this paper, we focus on a more challenging variant of the problem where the adversary assumes no prior knowledge about the query distribution (hence, unsupervised). Our core contribution is an adversarial corpus attack that is fast and effective. We present comprehensive experimental results on both in- and out-of-domain datasets, focusing on two related tasks: a top-1 attack and a corpus poisoning attack. We consider attacks under both a white-box and a black-box setting. Notably, our method can generate successful adversarial examples in under two minutes per target document; four times faster compared to the fastest gradient-based word substitution methods in the literature with the same hardware. Furthermore, our adversarial generation method generates text that is more likely to occur under the distribution of natural text (low perplexity), and is therefore more difficult to detect.



## **45. Siren -- Advancing Cybersecurity through Deception and Adaptive Analysis**

cs.CR

14 pages, 5 figures, 13th Computing Conference 2025 - London, United  Kingdom

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2406.06225v2) [paper-pdf](http://arxiv.org/pdf/2406.06225v2)

**Authors**: Samhruth Ananthanarayanan, Girish Kulathumani, Ganesh Narayanan

**Abstract**: Siren represents a pioneering research effort aimed at fortifying cybersecurity through strategic integration of deception, machine learning, and proactive threat analysis. Drawing inspiration from mythical sirens, this project employs sophisticated methods to lure potential threats into controlled environments. The system features a dynamic machine learning model for realtime analysis and classification, ensuring continuous adaptability to emerging cyber threats. The architectural framework includes a link monitoring proxy, a purpose-built machine learning model for dynamic link analysis, and a honeypot enriched with simulated user interactions to intensify threat engagement. Data protection within the honeypot is fortified with probabilistic encryption. Additionally, the incorporation of simulated user activity extends the system's capacity to capture and learn from potential attackers even after user disengagement. Overall, Siren introduces a paradigm shift in cybersecurity, transforming traditional defense mechanisms into proactive systems that actively engage and learn from potential adversaries. The research strives to enhance user protection while yielding valuable insights for ongoing refinement in response to the evolving landscape of cybersecurity threats.



## **46. On the Generalization of Adversarially Trained Quantum Classifiers**

quant-ph

22 pages, 6 figures

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17690v1) [paper-pdf](http://arxiv.org/pdf/2504.17690v1)

**Authors**: Petros Georgiou, Aaron Mark Thomas, Sharu Theresa Jose, Osvaldo Simeone

**Abstract**: Quantum classifiers are vulnerable to adversarial attacks that manipulate their input classical or quantum data. A promising countermeasure is adversarial training, where quantum classifiers are trained by using an attack-aware, adversarial loss function. This work establishes novel bounds on the generalization error of adversarially trained quantum classifiers when tested in the presence of perturbation-constrained adversaries. The bounds quantify the excess generalization error incurred to ensure robustness to adversarial attacks as scaling with the training sample size $m$ as $1/\sqrt{m}$, while yielding insights into the impact of the quantum embedding. For quantum binary classifiers employing \textit{rotation embedding}, we find that, in the presence of adversarial attacks on classical inputs $\mathbf{x}$, the increase in sample complexity due to adversarial training over conventional training vanishes in the limit of high dimensional inputs $\mathbf{x}$. In contrast, when the adversary can directly attack the quantum state $\rho(\mathbf{x})$ encoding the input $\mathbf{x}$, the excess generalization error depends on the choice of embedding only through its Hilbert space dimension. The results are also extended to multi-class classifiers. We validate our theoretical findings with numerical experiments.



## **47. Evaluating the Vulnerability of ML-Based Ethereum Phishing Detectors to Single-Feature Adversarial Perturbations**

cs.CR

24 pages; an extension of a paper that appeared at WISA 2024

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17684v1) [paper-pdf](http://arxiv.org/pdf/2504.17684v1)

**Authors**: Ahod Alghuried, Ali Alkinoon, Abdulaziz Alghamdi, Soohyeon Choi, Manar Mohaisen, David Mohaisen

**Abstract**: This paper explores the vulnerability of machine learning models to simple single-feature adversarial attacks in the context of Ethereum fraudulent transaction detection. Through comprehensive experimentation, we investigate the impact of various adversarial attack strategies on model performance metrics. Our findings, highlighting how prone those techniques are to simple attacks, are alarming, and the inconsistency in the attacks' effect on different algorithms promises ways for attack mitigation. We examine the effectiveness of different mitigation strategies, including adversarial training and enhanced feature selection, in enhancing model robustness and show their effectiveness.



## **48. Regulatory Markets for AI Safety**

cs.CY

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2001.00078v2) [paper-pdf](http://arxiv.org/pdf/2001.00078v2)

**Authors**: Jack Clark, Gillian K. Hadfield

**Abstract**: We propose a new model for regulation to achieve AI safety: global regulatory markets. We first sketch the model in general terms and provide an overview of the costs and benefits of this approach. We then demonstrate how the model might work in practice: responding to the risk of adversarial attacks on AI models employed in commercial drones.



## **49. A Simple DropConnect Approach to Transfer-based Targeted Attack**

cs.LG

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.18594v1) [paper-pdf](http://arxiv.org/pdf/2504.18594v1)

**Authors**: Tongrui Su, Qingbin Li, Shengyu Zhu, Wei Chen, Xueqi Cheng

**Abstract**: We study the problem of transfer-based black-box attack, where adversarial samples generated using a single surrogate model are directly applied to target models. Compared with untargeted attacks, existing methods still have lower Attack Success Rates (ASRs) in the targeted setting, i.e., the obtained adversarial examples often overfit the surrogate model but fail to mislead other models. In this paper, we hypothesize that the pixels or features in these adversarial examples collaborate in a highly dependent manner to maximize the success of an adversarial attack on the surrogate model, which we refer to as perturbation co-adaptation. Then, we propose to Mitigate perturbation Co-adaptation by DropConnect (MCD) to enhance transferability, by creating diverse variants of surrogate model at each optimization iteration. We conduct extensive experiments across various CNN- and Transformer-based models to demonstrate the effectiveness of MCD. In the challenging scenario of transferring from a CNN-based model to Transformer-based models, MCD achieves 13% higher average ASRs compared with state-of-the-art baselines. MCD boosts the performance of self-ensemble methods by bringing in more diversification across the variants while reserving sufficient semantic information for each variant. In addition, MCD attains the highest performance gain when scaling the compute of crafting adversarial examples.



## **50. GRANITE : a Byzantine-Resilient Dynamic Gossip Learning Framework**

cs.LG

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17471v1) [paper-pdf](http://arxiv.org/pdf/2504.17471v1)

**Authors**: Yacine Belal, Mohamed Maouche, Sonia Ben Mokhtar, Anthony Simonet-Boulogne

**Abstract**: Gossip Learning (GL) is a decentralized learning paradigm where users iteratively exchange and aggregate models with a small set of neighboring peers. Recent GL approaches rely on dynamic communication graphs built and maintained using Random Peer Sampling (RPS) protocols. Thanks to graph dynamics, GL can achieve fast convergence even over extremely sparse topologies. However, the robustness of GL over dy- namic graphs to Byzantine (model poisoning) attacks remains unaddressed especially when Byzantine nodes attack the RPS protocol to scale up model poisoning. We address this issue by introducing GRANITE, a framework for robust learning over sparse, dynamic graphs in the presence of a fraction of Byzantine nodes. GRANITE relies on two key components (i) a History-aware Byzantine-resilient Peer Sampling protocol (HaPS), which tracks previously encountered identifiers to reduce adversarial influence over time, and (ii) an Adaptive Probabilistic Threshold (APT), which leverages an estimate of Byzantine presence to set aggregation thresholds with formal guarantees. Empirical results confirm that GRANITE maintains convergence with up to 30% Byzantine nodes, improves learning speed via adaptive filtering of poisoned models and obtains these results in up to 9 times sparser graphs than dictated by current theory.



