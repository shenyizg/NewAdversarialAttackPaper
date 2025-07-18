# Latest Adversarial Attack Papers
**update at 2025-07-18 10:43:36**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Automating Steering for Safe Multimodal Large Language Models**

cs.CL

Working in progress. 22 pages (8+ for main); 25 figures; 1 table

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2507.13255v1) [paper-pdf](http://arxiv.org/pdf/2507.13255v1)

**Authors**: Lyucheng Wu, Mengru Wang, Ziwen Xu, Tri Cao, Nay Oo, Bryan Hooi, Shumin Deng

**Abstract**: Recent progress in Multimodal Large Language Models (MLLMs) has unlocked powerful cross-modal reasoning abilities, but also raised new safety concerns, particularly when faced with adversarial multimodal inputs. To improve the safety of MLLMs during inference, we introduce a modular and adaptive inference-time intervention technology, AutoSteer, without requiring any fine-tuning of the underlying model. AutoSteer incorporates three core components: (1) a novel Safety Awareness Score (SAS) that automatically identifies the most safety-relevant distinctions among the model's internal layers; (2) an adaptive safety prober trained to estimate the likelihood of toxic outputs from intermediate representations; and (3) a lightweight Refusal Head that selectively intervenes to modulate generation when safety risks are detected. Experiments on LLaVA-OV and Chameleon across diverse safety-critical benchmarks demonstrate that AutoSteer significantly reduces the Attack Success Rate (ASR) for textual, visual, and cross-modal threats, while maintaining general abilities. These findings position AutoSteer as a practical, interpretable, and effective framework for safer deployment of multimodal AI systems.



## **2. SHIELD: A Secure and Highly Enhanced Integrated Learning for Robust Deepfake Detection against Adversarial Attacks**

cs.SD

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2507.13170v1) [paper-pdf](http://arxiv.org/pdf/2507.13170v1)

**Authors**: Kutub Uddin, Awais Khan, Muhammad Umar Farooq, Khalid Malik

**Abstract**: Audio plays a crucial role in applications like speaker verification, voice-enabled smart devices, and audio conferencing. However, audio manipulations, such as deepfakes, pose significant risks by enabling the spread of misinformation. Our empirical analysis reveals that existing methods for detecting deepfake audio are often vulnerable to anti-forensic (AF) attacks, particularly those attacked using generative adversarial networks. In this article, we propose a novel collaborative learning method called SHIELD to defend against generative AF attacks. To expose AF signatures, we integrate an auxiliary generative model, called the defense (DF) generative model, which facilitates collaborative learning by combining input and output. Furthermore, we design a triplet model to capture correlations for real and AF attacked audios with real-generated and attacked-generated audios using auxiliary generative models. The proposed SHIELD strengthens the defense against generative AF attacks and achieves robust performance across various generative models. The proposed AF significantly reduces the average detection accuracy from 95.49% to 59.77% for ASVspoof2019, from 99.44% to 38.45% for In-the-Wild, and from 98.41% to 51.18% for HalfTruth for three different generative models. The proposed SHIELD mechanism is robust against AF attacks and achieves an average accuracy of 98.13%, 98.58%, and 99.57% in match, and 98.78%, 98.62%, and 98.85% in mismatch settings for the ASVspoof2019, In-the-Wild, and HalfTruth datasets, respectively.



## **3. Prompt-driven Transferable Adversarial Attack on Person Re-Identification with Attribute-aware Textual Inversion**

cs.CV

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2502.19697v3) [paper-pdf](http://arxiv.org/pdf/2502.19697v3)

**Authors**: Yuan Bian, Min Liu, Yunqi Yi, Xueping Wang, Yaonan Wang

**Abstract**: Person re-identification (re-id) models are vital in security surveillance systems, requiring transferable adversarial attacks to explore the vulnerabilities of them. Recently, vision-language models (VLM) based attacks have shown superior transferability by attacking generalized image and textual features of VLM, but they lack comprehensive feature disruption due to the overemphasis on discriminative semantics in integral representation. In this paper, we introduce the Attribute-aware Prompt Attack (AP-Attack), a novel method that leverages VLM's image-text alignment capability to explicitly disrupt fine-grained semantic features of pedestrian images by destroying attribute-specific textual embeddings. To obtain personalized textual descriptions for individual attributes, textual inversion networks are designed to map pedestrian images to pseudo tokens that represent semantic embeddings, trained in the contrastive learning manner with images and a predefined prompt template that explicitly describes the pedestrian attributes. Inverted benign and adversarial fine-grained textual semantics facilitate attacker in effectively conducting thorough disruptions, enhancing the transferability of adversarial examples. Extensive experiments show that AP-Attack achieves state-of-the-art transferability, significantly outperforming previous methods by 22.9% on mean Drop Rate in cross-model&dataset attack scenarios.



## **4. Adversarial attacks to image classification systems using evolutionary algorithms**

cs.NE

Genetic and Evolutionary Computation Conference (GECCO '25), July  14--18, 2025, Malaga, Spain

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2507.13136v1) [paper-pdf](http://arxiv.org/pdf/2507.13136v1)

**Authors**: Sergio Nesmachnow, Jamal Toutouh

**Abstract**: Image classification currently faces significant security challenges due to adversarial attacks, which consist of intentional alterations designed to deceive classification models based on artificial intelligence. This article explores an approach to generate adversarial attacks against image classifiers using a combination of evolutionary algorithms and generative adversarial networks. The proposed approach explores the latent space of a generative adversarial network with an evolutionary algorithm to find vectors representing adversarial attacks. The approach was evaluated in two case studies corresponding to the classification of handwritten digits and object images. The results showed success rates of up to 35% for handwritten digits, and up to 75% for object images, improving over other search methods and reported results in related works. The applied method proved to be effective in handling data diversity on the target datasets, even in problem instances that presented additional challenges due to the complexity and richness of information.



## **5. Detecting LLM-generated Code with Subtle Modification by Adversarial Training**

cs.SE

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2507.13123v1) [paper-pdf](http://arxiv.org/pdf/2507.13123v1)

**Authors**: Xin Yin, Xinrui Li, Chao Ni, Xiaodan Xu, Xiaohu Yang

**Abstract**: With the rapid development of Large Language Models (LLMs), their powerful code-generation capabilities have been widely applied in tasks like code completion and automated development, demonstrating the value of improving coding efficiency. However, the extensive use of LLM-generated code also raises several new challenges. On the one hand, issues such as the regulation of code provenance, copyright disputes, and code quality have become increasingly concerning. How to effectively detect LLM-generated code and ensure its compliant and responsible use has become a critical and urgent issue. On the other hand, in practical applications, LLM-generated code is often subject to manual modifications, such as variable renaming or structural adjustments. Although some recent studies have proposed training-based and zero-shot methods for detecting LLM-generated code, these approaches show insufficient robustness when facing modified LLM-generated code, and there is a lack of an effective solution. To address the real-world scenario where LLM-generated code may undergo minor modifications, we propose CodeGPTSensor+, an enhanced version of CodeGPTSensor, which employs adversarial training to improve robustness against input perturbations. CodeGPTSensor+ integrates an adversarial sample generation module, Multi-objective Identifier and Structure Transformation (MIST), which systematically generates both high-quality and representative adversarial samples. This module effectively enhances the model's resistance against diverse adversarial attacks. Experimental results on the HMCorp dataset demonstrate that CodeGPTSensor+ significantly improves detection accuracy on the adversarial test set while maintaining high accuracy on the original test set, showcasing superior robustness compared to CodeGPTSensor.



## **6. Slot: Provenance-Driven APT Detection through Graph Reinforcement Learning**

cs.CR

This paper has been accepted to the ACM Conference on Computer and  Communications Security (CCS) 2025

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2410.17910v4) [paper-pdf](http://arxiv.org/pdf/2410.17910v4)

**Authors**: Wei Qiao, Yebo Feng, Teng Li, Zhuo Ma, Yulong Shen, JianFeng Ma, Yang Liu

**Abstract**: Advanced Persistent Threats (APTs) represent sophisticated cyberattacks characterized by their ability to remain undetected within the victim system for extended periods, aiming to exfiltrate sensitive data or disrupt operations. Existing detection approaches often struggle to effectively identify these complex threats, construct the attack chain for defense facilitation, or resist adversarial attacks. To overcome these challenges, we propose Slot, an advanced APT detection approach based on provenance graphs and graph reinforcement learning. Slot excels in uncovering multi-level hidden relationships, such as causal, contextual, and indirect connections, among system behaviors through provenance graph mining. By pioneering the integration of graph reinforcement learning, Slot dynamically adapts to new user activities and evolving attack strategies, enhancing its resilience against adversarial attacks. Additionally, Slot automatically constructs the attack chain according to detected attacks with clustering algorithms, providing precise identification of attack paths and facilitating the development of defense strategies. Evaluations with real-world datasets demonstrate Slot's outstanding accuracy, efficiency, adaptability, and robustness in APT detection, with most metrics surpassing state-of-the-art methods. Additionally, case studies conducted to assess Slot's effectiveness in supporting APT defense further establish it as a practical and reliable tool for cybersecurity protection.



## **7. Semantic Structure-Aware Generative Attacks for Enhanced Adversarial Transferability**

cs.CV

Preprint

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2506.18248v3) [paper-pdf](http://arxiv.org/pdf/2506.18248v3)

**Authors**: Jongoh Jeong, Hunmin Yang, Jaeseok Jeong, Kuk-Jin Yoon

**Abstract**: Generative adversarial attacks train a perturbation generator on a white-box surrogate model and subsequently apply the crafted perturbations to unseen black-box victim models. In contrast to iterative attacks, these methods deliver superior inference-time efficiency, scalability, and transferability; however, up until now, existing studies have not fully exploited the representational capacity of generative models to preserve and harness semantic information. Specifically, the intermediate activations of the generator encode rich semantic features--object boundaries and coarse shapes--that remain under-exploited, thereby limiting the alignment of perturbations with object-salient regions which are critical for adversarial transferability. To remedy this, we introduce a semantic structure-aware attack framework based on the Mean Teacher, which serves as a temporally smoothed feature reference. With this smoothed reference, we further direct semantic consistency between the early-layer activations in the student and those of the semantically rich teacher by feature distillation. By anchoring perturbation synthesis to the semantically salient early intermediate blocks within the generator based on empirical findings, our method guides progressive adversarial perturbation on regions that substantially enhance adversarial transferability. We conduct extensive experiments over diverse models, domains and tasks to demonstrate consistent improvements relative to state-of-the-art generative attacks, comprehensively evaluated using conventional metrics and our newly proposed Accidental Correction Rate (ACR).



## **8. A Bayesian Incentive Mechanism for Poison-Resilient Federated Learning**

cs.LG

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2507.12439v1) [paper-pdf](http://arxiv.org/pdf/2507.12439v1)

**Authors**: Daniel Commey, Rebecca A. Sarpong, Griffith S. Klogo, Winful Bagyl-Bac, Garth V. Crosby

**Abstract**: Federated learning (FL) enables collaborative model training across decentralized clients while preserving data privacy. However, its open-participation nature exposes it to data-poisoning attacks, in which malicious actors submit corrupted model updates to degrade the global model. Existing defenses are often reactive, relying on statistical aggregation rules that can be computationally expensive and that typically assume an honest majority. This paper introduces a proactive, economic defense: a lightweight Bayesian incentive mechanism that makes malicious behavior economically irrational. Each training round is modeled as a Bayesian game of incomplete information in which the server, acting as the principal, uses a small, private validation dataset to verify update quality before issuing payments. The design satisfies Individual Rationality (IR) for benevolent clients, ensuring their participation is profitable, and Incentive Compatibility (IC), making poisoning an economically dominated strategy. Extensive experiments on non-IID partitions of MNIST and FashionMNIST demonstrate robustness: with 50% label-flipping adversaries on MNIST, the mechanism maintains 96.7% accuracy, only 0.3 percentage points lower than in a scenario with 30% label-flipping adversaries. This outcome is 51.7 percentage points better than standard FedAvg, which collapses under the same 50% attack. The mechanism is computationally light, budget-bounded, and readily integrates into existing FL frameworks, offering a practical route to economically robust and sustainable FL ecosystems.



## **9. Trustworthy Tree-based Machine Learning by $MoS_2$ Flash-based Analog CAM with Inherent Soft Boundaries**

cs.LG

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2507.12384v1) [paper-pdf](http://arxiv.org/pdf/2507.12384v1)

**Authors**: Bo Wen, Guoyun Gao, Zhicheng Xu, Ruibin Mao, Xiaojuan Qi, X. Sharon Hu, Xunzhao Yin, Can Li

**Abstract**: The rapid advancement of artificial intelligence has raised concerns regarding its trustworthiness, especially in terms of interpretability and robustness. Tree-based models like Random Forest and XGBoost excel in interpretability and accuracy for tabular data, but scaling them remains computationally expensive due to poor data locality and high data dependence. Previous efforts to accelerate these models with analog content addressable memory (CAM) have struggled, due to the fact that the difficult-to-implement sharp decision boundaries are highly susceptible to device variations, which leads to poor hardware performance and vulnerability to adversarial attacks. This work presents a novel hardware-software co-design approach using $MoS_2$ Flash-based analog CAM with inherent soft boundaries, enabling efficient inference with soft tree-based models. Our soft tree model inference experiments on $MoS_2$ analog CAM arrays show this method achieves exceptional robustness against device variation and adversarial attacks while achieving state-of-the-art accuracy. Specifically, our fabricated analog CAM arrays achieve $96\%$ accuracy on Wisconsin Diagnostic Breast Cancer (WDBC) database, while maintaining decision explainability. Our experimentally calibrated model validated only a $0.6\%$ accuracy drop on the MNIST dataset under $10\%$ device threshold variation, compared to a $45.3\%$ drop for traditional decision trees. This work paves the way for specialized hardware that enhances AI's trustworthiness and efficiency.



## **10. Thought Purity: Defense Paradigm For Chain-of-Thought Attack**

cs.LG

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2507.12314v1) [paper-pdf](http://arxiv.org/pdf/2507.12314v1)

**Authors**: Zihao Xue, Zhen Bi, Long Ma, Zhenlin Hu, Yan Wang, Zhenfang Liu, Qing Sheng, Jie Xiao, Jungang Lou

**Abstract**: While reinforcement learning-trained Large Reasoning Models (LRMs, e.g., Deepseek-R1) demonstrate advanced reasoning capabilities in the evolving Large Language Models (LLMs) domain, their susceptibility to security threats remains a critical vulnerability. This weakness is particularly evident in Chain-of-Thought (CoT) generation processes, where adversarial methods like backdoor prompt attacks can systematically subvert the model's core reasoning mechanisms. The emerging Chain-of-Thought Attack (CoTA) reveals this vulnerability through exploiting prompt controllability, simultaneously degrading both CoT safety and task performance with low-cost interventions. To address this compounded security-performance vulnerability, we propose Thought Purity (TP): a defense paradigm that systematically strengthens resistance to malicious content while preserving operational efficacy. Our solution achieves this through three synergistic components: (1) a safety-optimized data processing pipeline (2) reinforcement learning-enhanced rule constraints (3) adaptive monitoring metrics. Our approach establishes the first comprehensive defense mechanism against CoTA vulnerabilities in reinforcement learning-aligned reasoning systems, significantly advancing the security-functionality equilibrium for next-generation AI architectures.



## **11. Nanopass Back-Translation of Call-Return Trees for Mechanized Secure Compilation Proofs**

cs.PL

ITP'25 final version

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2503.19609v3) [paper-pdf](http://arxiv.org/pdf/2503.19609v3)

**Authors**: Jérémy Thibault, Joseph Lenormand, Catalin Hritcu

**Abstract**: Researchers aim to build secure compilation chains enforcing that if there is no attack a source context can mount against a source program then there is also no attack an adversarial target context can mount against the compiled program. Proving that these compilation chains are secure is, however, challenging, and involves a non-trivial back-translation step: for any attack a target context mounts against the compiled program one has to exhibit a source context mounting the same attack against the source program. We describe a novel back-translation technique, which results in simpler proofs that can be more easily mechanized in a proof assistant. Given a finite set of finite trace prefixes, capturing the interaction recorded during an attack between a target context and the compiled program, we build a call-return tree that we back-translate into a source context producing the same trace prefixes. We use state in the generated source context to record the current location in the call-return tree. The back-translation is done in several small steps, each adding to the tree new information describing how the location should change depending on how the context regains control. To prove this back-translation correct we give semantics to every intermediate call-return tree language, using ghost state to store information and explicitly enforce execution invariants. We prove several small forward simulations, basically seeing the back-translation as a verified nanopass compiler. Thanks to this modular structure, we are able to mechanize this complex back-translation and its correctness proof in the Rocq prover without too much effort.



## **12. Non-Adaptive Adversarial Face Generation**

cs.CV

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2507.12107v1) [paper-pdf](http://arxiv.org/pdf/2507.12107v1)

**Authors**: Sunpill Kim, Seunghun Paik, Chanwoo Hwang, Minsu Kim, Jae Hong Seo

**Abstract**: Adversarial attacks on face recognition systems (FRSs) pose serious security and privacy threats, especially when these systems are used for identity verification. In this paper, we propose a novel method for generating adversarial faces-synthetic facial images that are visually distinct yet recognized as a target identity by the FRS. Unlike iterative optimization-based approaches (e.g., gradient descent or other iterative solvers), our method leverages the structural characteristics of the FRS feature space. We figure out that individuals sharing the same attribute (e.g., gender or race) form an attributed subsphere. By utilizing such subspheres, our method achieves both non-adaptiveness and a remarkably small number of queries. This eliminates the need for relying on transferability and open-source surrogate models, which have been a typical strategy when repeated adaptive queries to commercial FRSs are impossible. Despite requiring only a single non-adaptive query consisting of 100 face images, our method achieves a high success rate of over 93% against AWS's CompareFaces API at its default threshold. Furthermore, unlike many existing attacks that perturb a given image, our method can deliberately produce adversarial faces that impersonate the target identity while exhibiting high-level attributes chosen by the adversary.



## **13. Distributed Resilient State Estimation and Control with Strategically Implemented Security Measures**

eess.SY

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2507.12052v1) [paper-pdf](http://arxiv.org/pdf/2507.12052v1)

**Authors**: Takumi Shinohara, Karl H. Johansson, Henrik Sandberg

**Abstract**: This paper addresses the problem of distributed resilient state estimation and control for linear time-invariant systems in the presence of malicious false data injection sensor attacks and bounded noise. We consider a system operator (defender) capable of deploying cybersecurity measures to counteract the sensor compromises. Although such measures enhance resilience against adversarial attacks, they may incur substantial costs; hence, it is crucial to select countermeasures to balance resilience gains and cost efficiency strategically. We first demonstrate that the system's resilience against attacks is maximized through the appropriate implementation of security measures, implying that no attacker can execute undetectable sensor attacks. Building on this analysis, we propose an algorithm that identifies the optimal security measure. While determining this measure is NP-hard in general, we also derive sufficient conditions under which efficient computation is feasible. Furthermore, we develop a distributed resilient state estimation and control scheme informed by the optimal security measure and establish conditions that guarantee bounded estimation and control errors. Finally, we validate the efficacy of our approach via numerical simulations of a vehicle platooning scenario.



## **14. Watch, Listen, Understand, Mislead: Tri-modal Adversarial Attacks on Short Videos for Content Appropriateness Evaluation**

cs.CV

Accepted as long paper, SVU Workshop at ICCV 2025

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2507.11968v1) [paper-pdf](http://arxiv.org/pdf/2507.11968v1)

**Authors**: Sahid Hossain Mustakim, S M Jishanul Islam, Ummay Maria Muna, Montasir Chowdhury, Mohammed Jawwadul Islam, Sadia Ahmmed, Tashfia Sikder, Syed Tasdid Azam Dhrubo, Swakkhar Shatabda

**Abstract**: Multimodal Large Language Models (MLLMs) are increasingly used for content moderation, yet their robustness in short-form video contexts remains underexplored. Current safety evaluations often rely on unimodal attacks, failing to address combined attack vulnerabilities. In this paper, we introduce a comprehensive framework for evaluating the tri-modal safety of MLLMs. First, we present the Short-Video Multimodal Adversarial (SVMA) dataset, comprising diverse short-form videos with human-guided synthetic adversarial attacks. Second, we propose ChimeraBreak, a novel tri-modal attack strategy that simultaneously challenges visual, auditory, and semantic reasoning pathways. Extensive experiments on state-of-the-art MLLMs reveal significant vulnerabilities with high Attack Success Rates (ASR). Our findings uncover distinct failure modes, showing model biases toward misclassifying benign or policy-violating content. We assess results using LLM-as-a-judge, demonstrating attack reasoning efficacy. Our dataset and findings provide crucial insights for developing more robust and safe MLLMs.



## **15. When and Where do Data Poisons Attack Textual Inversion?**

cs.CR

Accepted to ICCV 2025

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2507.10578v2) [paper-pdf](http://arxiv.org/pdf/2507.10578v2)

**Authors**: Jeremy Styborski, Mingzhi Lyu, Jiayou Lu, Nupur Kapur, Adams Kong

**Abstract**: Poisoning attacks pose significant challenges to the robustness of diffusion models (DMs). In this paper, we systematically analyze when and where poisoning attacks textual inversion (TI), a widely used personalization technique for DMs. We first introduce Semantic Sensitivity Maps, a novel method for visualizing the influence of poisoning on text embeddings. Second, we identify and experimentally verify that DMs exhibit non-uniform learning behavior across timesteps, focusing on lower-noise samples. Poisoning attacks inherit this bias and inject adversarial signals predominantly at lower timesteps. Lastly, we observe that adversarial signals distract learning away from relevant concept regions within training data, corrupting the TI process. Based on these insights, we propose Safe-Zone Training (SZT), a novel defense mechanism comprised of 3 key components: (1) JPEG compression to weaken high-frequency poison signals, (2) restriction to high timesteps during TI training to avoid adversarial signals at lower timesteps, and (3) loss masking to constrain learning to relevant regions. Extensive experiments across multiple poisoning methods demonstrate that SZT greatly enhances the robustness of TI against all poisoning attacks, improving generative quality beyond prior published defenses. Code: www.github.com/JStyborski/Diff_Lab Data: www.github.com/JStyborski/NC10



## **16. LLMs Encode Harmfulness and Refusal Separately**

cs.CL

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2507.11878v1) [paper-pdf](http://arxiv.org/pdf/2507.11878v1)

**Authors**: Jiachen Zhao, Jing Huang, Zhengxuan Wu, David Bau, Weiyan Shi

**Abstract**: LLMs are trained to refuse harmful instructions, but do they truly understand harmfulness beyond just refusing? Prior work has shown that LLMs' refusal behaviors can be mediated by a one-dimensional subspace, i.e., a refusal direction. In this work, we identify a new dimension to analyze safety mechanisms in LLMs, i.e., harmfulness, which is encoded internally as a separate concept from refusal. There exists a harmfulness direction that is distinct from the refusal direction. As causal evidence, steering along the harmfulness direction can lead LLMs to interpret harmless instructions as harmful, but steering along the refusal direction tends to elicit refusal responses directly without reversing the model's judgment on harmfulness. Furthermore, using our identified harmfulness concept, we find that certain jailbreak methods work by reducing the refusal signals without reversing the model's internal belief of harmfulness. We also find that adversarially finetuning models to accept harmful instructions has minimal impact on the model's internal belief of harmfulness. These insights lead to a practical safety application: The model's latent harmfulness representation can serve as an intrinsic safeguard (Latent Guard) for detecting unsafe inputs and reducing over-refusals that is robust to finetuning attacks. For instance, our Latent Guard achieves performance comparable to or better than Llama Guard 3 8B, a dedicated finetuned safeguard model, across different jailbreak methods. Our findings suggest that LLMs' internal understanding of harmfulness is more robust than their refusal decision to diverse input instructions, offering a new perspective to study AI safety



## **17. Some remarks on gradient dominance and LQR policy optimization**

cs.LG

This is a short paper summarizing the first part of the slides  presented at my keynote at the 2025 L4DC (Learning for Dynamics & Control  Conference) in Ann Arbor, Michigan, 05 June 2025. A partial bibliography has  been added

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2507.10452v2) [paper-pdf](http://arxiv.org/pdf/2507.10452v2)

**Authors**: Eduardo D. Sontag

**Abstract**: Solutions of optimization problems, including policy optimization in reinforcement learning, typically rely upon some variant of gradient descent. There has been much recent work in the machine learning, control, and optimization communities applying the Polyak-{\L}ojasiewicz Inequality (PLI) to such problems in order to establish an exponential rate of convergence (a.k.a. ``linear convergence'' in the local-iteration language of numerical analysis) of loss functions to their minima under the gradient flow. Often, as is the case of policy iteration for the continuous-time LQR problem, this rate vanishes for large initial conditions, resulting in a mixed globally linear / locally exponential behavior. This is in sharp contrast with the discrete-time LQR problem, where there is global exponential convergence. That gap between CT and DT behaviors motivates the search for various generalized PLI-like conditions, and this talk will address that topic. Moreover, these generalizations are key to understanding the transient and asymptotic effects of errors in the estimation of the gradient, errors which might arise from adversarial attacks, wrong evaluation by an oracle, early stopping of a simulation, inaccurate and very approximate digital twins, stochastic computations (algorithm ``reproducibility''), or learning by sampling from limited data. We describe an ``input to state stability'' (ISS) analysis of this issue. The second part discusses convergence and PLI-like properties of ``linear feedforward neural networks'' in feedback control. Much of the work described here was done in collaboration with Arthur Castello B. de Oliveira, Leilei Cui, Zhong-Ping Jiang, and Milad Siami.



## **18. A Generative Approach to LLM Harmfulness Detection with Special Red Flag Tokens**

cs.CL

14 pages, 6 figures

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2502.16366v3) [paper-pdf](http://arxiv.org/pdf/2502.16366v3)

**Authors**: Sophie Xhonneux, David Dobre, Mehrnaz Mofakhami, Leo Schwinn, Gauthier Gidel

**Abstract**: Most safety training methods for large language models (LLMs) are based on fine-tuning that forces models to shift from an unsafe answer to refusal when faced with harmful requests. Unfortunately, these drastic distribution shifts generally compromise model capabilities. To avoid that, we propose to expand the model's vocabulary with a special token we call red flag token (<rf>) and propose to train the model to insert this token into its response at any time when harmful content is generated or about to be generated. Our approach offers several advantages: it enables the model to explicitly learn the concept of harmfulness while marginally affecting the generated distribution, thus maintaining the model's utility. It also evaluates each generated answer and provides robustness as good as adversarial training without the need to run attacks during training. Moreover, by encapsulating our safety tuning in a LoRA module, we provide additional defenses against fine-tuning API attacks.



## **19. Robustifying 3D Perception via Least-Squares Graphs for Multi-Agent Object Tracking**

cs.CV

6 pages, 3 figures, 4 tables

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2507.04762v2) [paper-pdf](http://arxiv.org/pdf/2507.04762v2)

**Authors**: Maria Damanaki, Ioulia Kapsali, Nikos Piperigkos, Alexandros Gkillas, Aris S. Lalos

**Abstract**: The critical perception capabilities of EdgeAI systems, such as autonomous vehicles, are required to be resilient against adversarial threats, by enabling accurate identification and localization of multiple objects in the scene over time, mitigating their impact. Single-agent tracking offers resilience to adversarial attacks but lacks situational awareness, underscoring the need for multi-agent cooperation to enhance context understanding and robustness. This paper proposes a novel mitigation framework on 3D LiDAR scene against adversarial noise by tracking objects based on least-squares graph on multi-agent adversarial bounding boxes. Specifically, we employ the least-squares graph tool to reduce the induced positional error of each detection's centroid utilizing overlapped bounding boxes on a fully connected graph via differential coordinates and anchor points. Hence, the multi-vehicle detections are fused and refined mitigating the adversarial impact, and associated with existing tracks in two stages performing tracking to further suppress the adversarial threat. An extensive evaluation study on the real-world V2V4Real dataset demonstrates that the proposed method significantly outperforms both state-of-the-art single and multi-agent tracking frameworks by up to 23.3% under challenging adversarial conditions, operating as a resilient approach without relying on additional defense mechanisms.



## **20. Provable Robustness of (Graph) Neural Networks Against Data Poisoning and Backdoor Attacks**

cs.LG

Published in TMLR. Best Paper Award at the AdvML-Frontiers @ NeurIPS  2024 workshop. Code available at https://github.com/saper0/qpcert

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2407.10867v3) [paper-pdf](http://arxiv.org/pdf/2407.10867v3)

**Authors**: Lukas Gosch, Mahalakshmi Sabanayagam, Debarghya Ghoshdastidar, Stephan Günnemann

**Abstract**: Generalization of machine learning models can be severely compromised by data poisoning, where adversarial changes are applied to the training data. This vulnerability has led to interest in certifying (i.e., proving) that such changes up to a certain magnitude do not affect test predictions. We, for the first time, certify Graph Neural Networks (GNNs) against poisoning attacks, including backdoors, targeting the node features of a given graph. Our certificates are white-box and based upon $(i)$ the neural tangent kernel, which characterizes the training dynamics of sufficiently wide networks; and $(ii)$ a novel reformulation of the bilevel optimization problem describing poisoning as a mixed-integer linear program. Consequently, we leverage our framework to provide fundamental insights into the role of graph structure and its connectivity on the worst-case robustness behavior of convolution-based and PageRank-based GNNs. We note that our framework is more general and constitutes the first approach to derive white-box poisoning certificates for NNs, which can be of independent interest beyond graph-related tasks.



## **21. Real-Time Bayesian Detection of Drift-Evasive GNSS Spoofing in Reinforcement Learning Based UAV Deconfliction**

cs.LG

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2507.11173v1) [paper-pdf](http://arxiv.org/pdf/2507.11173v1)

**Authors**: Deepak Kumar Panda, Weisi Guo

**Abstract**: Autonomous unmanned aerial vehicles (UAVs) rely on global navigation satellite system (GNSS) pseudorange measurements for accurate real-time localization and navigation. However, this dependence exposes them to sophisticated spoofing threats, where adversaries manipulate pseudoranges to deceive UAV receivers. Among these, drift-evasive spoofing attacks subtly perturb measurements, gradually diverting the UAVs trajectory without triggering conventional signal-level anti-spoofing mechanisms. Traditional distributional shift detection techniques often require accumulating a threshold number of samples, causing delays that impede rapid detection and timely response. Consequently, robust temporal-scale detection methods are essential to identify attack onset and enable contingency planning with alternative sensing modalities, improving resilience against stealthy adversarial manipulations. This study explores a Bayesian online change point detection (BOCPD) approach that monitors temporal shifts in value estimates from a reinforcement learning (RL) critic network to detect subtle behavioural deviations in UAV navigation. Experimental results show that this temporal value-based framework outperforms conventional GNSS spoofing detectors, temporal semi-supervised learning frameworks, and the Page-Hinkley test, achieving higher detection accuracy and lower false-positive and false-negative rates for drift-evasive spoofing attacks.



## **22. Multi-Trigger Poisoning Amplifies Backdoor Vulnerabilities in LLMs**

cs.CL

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2507.11112v1) [paper-pdf](http://arxiv.org/pdf/2507.11112v1)

**Authors**: Sanhanat Sivapiromrat, Caiqi Zhang, Marco Basaldella, Nigel Collier

**Abstract**: Recent studies have shown that Large Language Models (LLMs) are vulnerable to data poisoning attacks, where malicious training examples embed hidden behaviours triggered by specific input patterns. However, most existing works assume a phrase and focus on the attack's effectiveness, offering limited understanding of trigger mechanisms and how multiple triggers interact within the model. In this paper, we present a framework for studying poisoning in LLMs. We show that multiple distinct backdoor triggers can coexist within a single model without interfering with each other, enabling adversaries to embed several triggers concurrently. Using multiple triggers with high embedding similarity, we demonstrate that poisoned triggers can achieve robust activation even when tokens are substituted or separated by long token spans. Our findings expose a broader and more persistent vulnerability surface in LLMs. To mitigate this threat, we propose a post hoc recovery method that selectively retrains specific model components based on a layer-wise weight difference analysis. Our method effectively removes the trigger behaviour with minimal parameter updates, presenting a practical and efficient defence against multi-trigger poisoning.



## **23. The Devil behind the mask: An emergent safety vulnerability of Diffusion LLMs**

cs.CL

21 pages, 9 figures, work in progress

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2507.11097v1) [paper-pdf](http://arxiv.org/pdf/2507.11097v1)

**Authors**: Zichen Wen, Jiashu Qu, Dongrui Liu, Zhiyuan Liu, Ruixi Wu, Yicun Yang, Xiangqi Jin, Haoyun Xu, Xuyang Liu, Weijia Li, Chaochao Lu, Jing Shao, Conghui He, Linfeng Zhang

**Abstract**: Diffusion-based large language models (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs, offering faster inference and greater interactivity via parallel decoding and bidirectional modeling. However, despite strong performance in code generation and text infilling, we identify a fundamental safety concern: existing alignment mechanisms fail to safeguard dLLMs against context-aware, masked-input adversarial prompts, exposing novel vulnerabilities. To this end, we present DIJA, the first systematic study and jailbreak attack framework that exploits unique safety weaknesses of dLLMs. Specifically, our proposed DIJA constructs adversarial interleaved mask-text prompts that exploit the text generation mechanisms of dLLMs, i.e., bidirectional modeling and parallel decoding. Bidirectional modeling drives the model to produce contextually consistent outputs for masked spans, even when harmful, while parallel decoding limits model dynamic filtering and rejection sampling of unsafe content. This causes standard alignment mechanisms to fail, enabling harmful completions in alignment-tuned dLLMs, even when harmful behaviors or unsafe instructions are directly exposed in the prompt. Through comprehensive experiments, we demonstrate that DIJA significantly outperforms existing jailbreak methods, exposing a previously overlooked threat surface in dLLM architectures. Notably, our method achieves up to 100% keyword-based ASR on Dream-Instruct, surpassing the strongest prior baseline, ReNeLLM, by up to 78.5% in evaluator-based ASR on JailbreakBench and by 37.7 points in StrongREJECT score, while requiring no rewriting or hiding of harmful content in the jailbreak prompt. Our findings underscore the urgent need for rethinking safety alignment in this emerging class of language models. Code is available at https://github.com/ZichenWen1/DIJA.



## **24. Crafting Imperceptible On-Manifold Adversarial Attacks for Tabular Data**

cs.LG

32 pages

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2507.10998v1) [paper-pdf](http://arxiv.org/pdf/2507.10998v1)

**Authors**: Zhipeng He, Alexander Stevens, Chun Ouyang, Johannes De Smedt, Alistair Barros, Catarina Moreira

**Abstract**: Adversarial attacks on tabular data present fundamental challenges distinct from image or text domains due to the heterogeneous nature of mixed categorical and numerical features. Unlike images where pixel perturbations maintain visual similarity, tabular data lacks intuitive similarity metrics, making it difficult to define imperceptible modifications. Additionally, traditional gradient-based methods prioritise $\ell_p$-norm constraints, often producing adversarial examples that deviate from the original data distributions, making them detectable. We propose a latent space perturbation framework using a mixed-input Variational Autoencoder (VAE) to generate imperceptible adversarial examples. The proposed VAE integrates categorical embeddings and numerical features into a unified latent manifold, enabling perturbations that preserve statistical consistency. We specify In-Distribution Success Rate (IDSR) to measure the proportion of adversarial examples that remain statistically indistinguishable from the input distribution. Evaluation across six publicly available datasets and three model architectures demonstrates that our method achieves substantially lower outlier rates and more consistent performance compared to traditional input-space attacks and other VAE-based methods adapted from image domain approaches. Our comprehensive analysis includes hyperparameter sensitivity, sparsity control mechanisms, and generative architectural comparisons, revealing that VAE-based attacks depend critically on reconstruction quality but offer superior practical utility when sufficient training data is available. This work highlights the importance of on-manifold perturbations for realistic adversarial attacks on tabular data, offering a robust approach for practical deployment. The source code can be accessed through https://github.com/ZhipengHe/VAE-TabAttack.



## **25. Representation Bending for Large Language Model Safety**

cs.LG

Accepted to ACL 2025 (main)

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2504.01550v3) [paper-pdf](http://arxiv.org/pdf/2504.01550v3)

**Authors**: Ashkan Yousefpour, Taeheon Kim, Ryan S. Kwon, Seungbeen Lee, Wonje Jeung, Seungju Han, Alvin Wan, Harrison Ngan, Youngjae Yu, Jonghyun Choi

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools, but their inherent safety risks - ranging from harmful content generation to broader societal harms - pose significant challenges. These risks can be amplified by the recent adversarial attacks, fine-tuning vulnerabilities, and the increasing deployment of LLMs in high-stakes environments. Existing safety-enhancing techniques, such as fine-tuning with human feedback or adversarial training, are still vulnerable as they address specific threats and often fail to generalize across unseen attacks, or require manual system-level defenses. This paper introduces RepBend, a novel approach that fundamentally disrupts the representations underlying harmful behaviors in LLMs, offering a scalable solution to enhance (potentially inherent) safety. RepBend brings the idea of activation steering - simple vector arithmetic for steering model's behavior during inference - to loss-based fine-tuning. Through extensive evaluation, RepBend achieves state-of-the-art performance, outperforming prior methods such as Circuit Breaker, RMU, and NPO, with up to 95% reduction in attack success rates across diverse jailbreak benchmarks, all with negligible reduction in model usability and general capabilities.



## **26. A Survey on Speech Deepfake Detection**

cs.SD

38 pages. This paper has been accepted by ACM Computing Surveys

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2404.13914v2) [paper-pdf](http://arxiv.org/pdf/2404.13914v2)

**Authors**: Menglu Li, Yasaman Ahmadiadli, Xiao-Ping Zhang

**Abstract**: The availability of smart devices leads to an exponential increase in multimedia content. However, advancements in deep learning have also enabled the creation of highly sophisticated Deepfake content, including speech Deepfakes, which pose a serious threat by generating realistic voices and spreading misinformation. To combat this, numerous challenges have been organized to advance speech Deepfake detection techniques. In this survey, we systematically analyze more than 200 papers published up to March 2024. We provide a comprehensive review of each component in the detection pipeline, including model architectures, optimization techniques, generalizability, evaluation metrics, performance comparisons, available datasets, and open source availability. For each aspect, we assess recent progress and discuss ongoing challenges. In addition, we explore emerging topics such as partial Deepfake detection, cross-dataset evaluation, and defences against adversarial attacks, while suggesting promising research directions. This survey not only identifies the current state of the art to establish strong baselines for future experiments but also offers clear guidance for researchers aiming to enhance speech Deepfake detection systems.



## **27. REAL-IoT: Characterizing GNN Intrusion Detection Robustness under Practical Adversarial Attack**

cs.CR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10836v1) [paper-pdf](http://arxiv.org/pdf/2507.10836v1)

**Authors**: Zhonghao Zhan, Huichi Zhou, Hamed Haddadi

**Abstract**: Graph Neural Network (GNN)-based network intrusion detection systems (NIDS) are often evaluated on single datasets, limiting their ability to generalize under distribution drift. Furthermore, their adversarial robustness is typically assessed using synthetic perturbations that lack realism. This measurement gap leads to an overestimation of GNN-based NIDS resilience. To address the limitations, we propose \textbf{REAL-IoT}, a comprehensive framework for robustness evaluation of GNN-based NIDS in IoT environments. Our framework presents a methodology that creates a unified dataset from canonical datasets to assess generalization under drift. In addition, it features a novel intrusion dataset collected from a physical IoT testbed, which captures network traffic and attack scenarios under real-world settings. Furthermore, using REAL-IoT, we explore the usage of Large Language Models (LLMs) to analyze network data and mitigate the impact of adversarial examples by filtering suspicious flows. Our evaluations using REAL-IoT reveal performance drops in GNN models compared to results from standard benchmarks, quantifying their susceptibility to drift and realistic attacks. We also demonstrate the potential of LLM-based filtering to enhance robustness. These findings emphasize the necessity of realistic threat modeling and rigorous measurement practices for developing resilient IoT intrusion detection systems.



## **28. Investigating Adversarial Attacks in Software Analytics via Machine Learning Explainability**

cs.SE

This paper has been accepted for publication in Software Quality  Journal

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2408.04124v2) [paper-pdf](http://arxiv.org/pdf/2408.04124v2)

**Authors**: MD Abdul Awal, Mrigank Rochan, Chanchal K. Roy

**Abstract**: With the recent advancements in machine learning (ML), numerous ML-based approaches have been extensively applied in software analytics tasks to streamline software development and maintenance processes. Nevertheless, studies indicate that despite their potential usefulness, ML models are vulnerable to adversarial attacks, which may result in significant monetary losses in these processes. As a result, the ML models' robustness against adversarial attacks must be assessed before they are deployed in software analytics tasks. Despite several techniques being available for adversarial attacks in software analytics tasks, exploring adversarial attacks using ML explainability is largely unexplored. Therefore, this study aims to investigate the relationship between ML explainability and adversarial attacks to measure the robustness of ML models in software analytics tasks. In addition, unlike most existing attacks that directly perturb input-space, our attack approach focuses on perturbing feature-space. Our extensive experiments, involving six datasets, three ML explainability techniques, and seven ML models, demonstrate that ML explainability can be used to conduct successful adversarial attacks on ML models in software analytics tasks. This is achieved by modifying only the top 1-3 important features identified by ML explainability techniques. Consequently, the ML models under attack fail to accurately predict up to 86.6% of instances that were correctly predicted before adversarial attacks, indicating the models' low robustness against such attacks. Finally, our proposed technique demonstrates promising results compared to four state-of-the-art adversarial attack techniques targeting tabular data.



## **29. BURN: Backdoor Unlearning via Adversarial Boundary Analysis**

cs.CR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10491v1) [paper-pdf](http://arxiv.org/pdf/2507.10491v1)

**Authors**: Yanghao Su, Jie Zhang, Yiming Li, Tianwei Zhang, Qing Guo, Weiming Zhang, Nenghai Yu, Nils Lukas, Wenbo Zhou

**Abstract**: Backdoor unlearning aims to remove backdoor-related information while preserving the model's original functionality. However, existing unlearning methods mainly focus on recovering trigger patterns but fail to restore the correct semantic labels of poison samples. This limitation prevents them from fully eliminating the false correlation between the trigger pattern and the target label. To address this, we leverage boundary adversarial attack techniques, revealing two key observations. First, poison samples exhibit significantly greater distances from decision boundaries compared to clean samples, indicating they require larger adversarial perturbations to change their predictions. Second, while adversarial predicted labels for clean samples are uniformly distributed, those for poison samples tend to revert to their original correct labels. Moreover, the features of poison samples restore to closely resemble those of corresponding clean samples after adding adversarial perturbations. Building upon these insights, we propose Backdoor Unlearning via adversaRial bouNdary analysis (BURN), a novel defense framework that integrates false correlation decoupling, progressive data refinement, and model purification. In the first phase, BURN employs adversarial boundary analysis to detect poisoned samples based on their abnormal adversarial boundary distances, then restores their correct semantic labels for fine-tuning. In the second phase, it employs a feedback mechanism that tracks prediction discrepancies between the original backdoored model and progressively sanitized models, guiding both dataset refinement and model purification. Extensive evaluations across multiple datasets, architectures, and seven diverse backdoor attack types confirm that BURN effectively removes backdoor threats while maintaining the model's original performance.



## **30. Bypassing LLM Guardrails: An Empirical Analysis of Evasion Attacks against Prompt Injection and Jailbreak Detection Systems**

cs.CR

14 pages, 5 figures, 11 tables. To be published in LLMSec 2025

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2504.11168v3) [paper-pdf](http://arxiv.org/pdf/2504.11168v3)

**Authors**: William Hackett, Lewis Birch, Stefan Trawicki, Neeraj Suri, Peter Garraghan

**Abstract**: Large Language Models (LLMs) guardrail systems are designed to protect against prompt injection and jailbreak attacks. However, they remain vulnerable to evasion techniques. We demonstrate two approaches for bypassing LLM prompt injection and jailbreak detection systems via traditional character injection methods and algorithmic Adversarial Machine Learning (AML) evasion techniques. Through testing against six prominent protection systems, including Microsoft's Azure Prompt Shield and Meta's Prompt Guard, we show that both methods can be used to evade detection while maintaining adversarial utility achieving in some instances up to 100% evasion success. Furthermore, we demonstrate that adversaries can enhance Attack Success Rates (ASR) against black-box targets by leveraging word importance ranking computed by offline white-box models. Our findings reveal vulnerabilities within current LLM protection mechanisms and highlight the need for more robust guardrail systems.



## **31. SCOOTER: A Human Evaluation Framework for Unrestricted Adversarial Examples**

cs.CV

42 pages, 16 figures, 11 tables, Under Review, Code:  https://github.com/DrenFazlija/Scooter, Data:  https://doi.org/10.5281/zenodo.15771501

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.07776v2) [paper-pdf](http://arxiv.org/pdf/2507.07776v2)

**Authors**: Dren Fazlija, Monty-Maximilian Zühlke, Johanna Schrader, Arkadij Orlov, Clara Stein, Iyiola E. Olatunji, Daniel Kudenko

**Abstract**: Unrestricted adversarial attacks aim to fool computer vision models without being constrained by $\ell_p$-norm bounds to remain imperceptible to humans, for example, by changing an object's color. This allows attackers to circumvent traditional, norm-bounded defense strategies such as adversarial training or certified defense strategies. However, due to their unrestricted nature, there are also no guarantees of norm-based imperceptibility, necessitating human evaluations to verify just how authentic these adversarial examples look. While some related work assesses this vital quality of adversarial attacks, none provide statistically significant insights. This issue necessitates a unified framework that supports and streamlines such an assessment for evaluating and comparing unrestricted attacks. To close this gap, we introduce SCOOTER - an open-source, statistically powered framework for evaluating unrestricted adversarial examples. Our contributions are: $(i)$ best-practice guidelines for crowd-study power, compensation, and Likert equivalence bounds to measure imperceptibility; $(ii)$ the first large-scale human vs. model comparison across 346 human participants showing that three color-space attacks and three diffusion-based attacks fail to produce imperceptible images. Furthermore, we found that GPT-4o can serve as a preliminary test for imperceptibility, but it only consistently detects adversarial examples for four out of six tested attacks; $(iii)$ open-source software tools, including a browser-based task template to collect annotations and analysis scripts in Python and R; $(iv)$ an ImageNet-derived benchmark dataset containing 3K real images, 7K adversarial examples, and over 34K human ratings. Our findings demonstrate that automated vision systems do not align with human perception, reinforcing the need for a ground-truth SCOOTER benchmark.



## **32. Bridging Robustness and Generalization Against Word Substitution Attacks in NLP via the Growth Bound Matrix Approach**

cs.CL

Accepted to ACL Findings 2025

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10330v1) [paper-pdf](http://arxiv.org/pdf/2507.10330v1)

**Authors**: Mohammed Bouri, Adnane Saoud

**Abstract**: Despite advancements in Natural Language Processing (NLP), models remain vulnerable to adversarial attacks, such as synonym substitutions. While prior work has focused on improving robustness for feed-forward and convolutional architectures, the robustness of recurrent networks and modern state space models (SSMs), such as S4, remains understudied. These architectures pose unique challenges due to their sequential processing and complex parameter dynamics. In this paper, we introduce a novel regularization technique based on Growth Bound Matrices (GBM) to improve NLP model robustness by reducing the impact of input perturbations on model outputs. We focus on computing the GBM for three architectures: Long Short-Term Memory (LSTM), State Space models (S4), and Convolutional Neural Networks (CNN). Our method aims to (1) enhance resilience against word substitution attacks, (2) improve generalization on clean text, and (3) providing the first systematic analysis of SSM (S4) robustness. Extensive experiments across multiple architectures and benchmark datasets demonstrate that our method improves adversarial robustness by up to 8.8% over existing baselines. These results highlight the effectiveness of our approach, outperforming several state-of-the-art methods in adversarial defense. Codes are available at https://github.com/BouriMohammed/GBM



## **33. Kaleidoscopic Background Attack: Disrupting Pose Estimation with Multi-Fold Radial Symmetry Textures**

cs.CV

Accepted at ICCV 2025. Project page is available at  https://wakuwu.github.io/KBA

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10265v1) [paper-pdf](http://arxiv.org/pdf/2507.10265v1)

**Authors**: Xinlong Ding, Hongwei Yu, Jiawei Li, Feifan Li, Yu Shang, Bochao Zou, Huimin Ma, Jiansheng Chen

**Abstract**: Camera pose estimation is a fundamental computer vision task that is essential for applications like visual localization and multi-view stereo reconstruction. In the object-centric scenarios with sparse inputs, the accuracy of pose estimation can be significantly influenced by background textures that occupy major portions of the images across different viewpoints. In light of this, we introduce the Kaleidoscopic Background Attack (KBA), which uses identical segments to form discs with multi-fold radial symmetry. These discs maintain high similarity across different viewpoints, enabling effective attacks on pose estimation models even with natural texture segments. Additionally, a projected orientation consistency loss is proposed to optimize the kaleidoscopic segments, leading to significant enhancement in the attack effectiveness. Experimental results show that optimized adversarial kaleidoscopic backgrounds can effectively attack various camera pose estimation models.



## **34. Transferring Styles for Reduced Texture Bias and Improved Robustness in Semantic Segmentation Networks**

cs.CV

accepted at ECAI 2025

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10239v1) [paper-pdf](http://arxiv.org/pdf/2507.10239v1)

**Authors**: Ben Hamscher, Edgar Heinert, Annika Mütze, Kira Maag, Matthias Rottmann

**Abstract**: Recent research has investigated the shape and texture biases of deep neural networks (DNNs) in image classification which influence their generalization capabilities and robustness. It has been shown that, in comparison to regular DNN training, training with stylized images reduces texture biases in image classification and improves robustness with respect to image corruptions. In an effort to advance this line of research, we examine whether style transfer can likewise deliver these two effects in semantic segmentation. To this end, we perform style transfer with style varying across artificial image areas. Those random areas are formed by a chosen number of Voronoi cells. The resulting style-transferred data is then used to train semantic segmentation DNNs with the objective of reducing their dependence on texture cues while enhancing their reliance on shape-based features. In our experiments, it turns out that in semantic segmentation, style transfer augmentation reduces texture bias and strongly increases robustness with respect to common image corruptions as well as adversarial attacks. These observations hold for convolutional neural networks and transformer architectures on the Cityscapes dataset as well as on PASCAL Context, showing the generality of the proposed method.



## **35. HASSLE: A Self-Supervised Learning Enhanced Hijacking Attack on Vertical Federated Learning**

cs.CR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10162v1) [paper-pdf](http://arxiv.org/pdf/2507.10162v1)

**Authors**: Weiyang He, Chip-Hong Chang

**Abstract**: Vertical Federated Learning (VFL) enables an orchestrating active party to perform a machine learning task by cooperating with passive parties that provide additional task-related features for the same training data entities. While prior research has leveraged the privacy vulnerability of VFL to compromise its integrity through a combination of label inference and backdoor attacks, their effectiveness is constrained by the low label inference precision and suboptimal backdoor injection conditions. To facilitate a more rigorous security evaluation on VFL without these limitations, we propose HASSLE, a hijacking attack framework composed of a gradient-direction-based label inference module and an adversarial embedding generation algorithm enhanced by self-supervised learning. HASSLE accurately identifies private samples associated with a targeted label using only a single known instance of that label. In the two-party scenario, it demonstrates strong performance with an attack success rate (ASR) of over 99% across four datasets, including both image and tabular modalities, and achieves 85% ASR on the more complex CIFAR-100 dataset. Evaluation of HASSLE against 8 potential defenses further highlights its significant threat while providing new insights into building a trustworthy VFL system.



## **36. Explicit Vulnerability Generation with LLMs: An Investigation Beyond Adversarial Attacks**

cs.SE

Accepted to ICSME 2025

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10054v1) [paper-pdf](http://arxiv.org/pdf/2507.10054v1)

**Authors**: Emir Bosnak, Sahand Moslemi, Mayasah Lami, Anil Koyuncu

**Abstract**: Large Language Models (LLMs) are increasingly used as code assistants, yet their behavior when explicitly asked to generate insecure code remains poorly understood. While prior research has focused on unintended vulnerabilities or adversarial prompting techniques, this study examines a more direct threat scenario: open-source LLMs generating vulnerable code when prompted either directly or indirectly. We propose a dual experimental design: (1) Dynamic Prompting, which systematically varies vulnerability type, user persona, and directness across structured templates; and (2) Reverse Prompting, which derives prompts from real vulnerable code samples to assess vulnerability reproduction accuracy. We evaluate three open-source 7B-parameter models (Qwen2, Mistral, and Gemma) using ESBMC static analysis to assess both the presence of vulnerabilities and the correctness of the generated vulnerability type. Results show all models frequently produce vulnerable outputs, with Qwen2 achieving highest correctness rates. User persona significantly affects success, where student personas achieved higher vulnerability rates than professional roles, while direct prompts were marginally more effective. Vulnerability reproduction followed an inverted-U pattern with cyclomatic complexity, peaking at moderate ranges. Our findings expose limitations of safety mechanisms in open-source models, particularly for seemingly benign educational requests.



## **37. 3DGAA: Realistic and Robust 3D Gaussian-based Adversarial Attack for Autonomous Driving**

cs.CV

Submitted to WACV 2026

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.09993v1) [paper-pdf](http://arxiv.org/pdf/2507.09993v1)

**Authors**: Yixun Zhang, Lizhi Wang, Junjun Zhao, Wending Zhao, Feng Zhou, Yonghao Dang, Jianqin Yin

**Abstract**: Camera-based object detection systems play a vital role in autonomous driving, yet they remain vulnerable to adversarial threats in real-world environments. While existing 2D and 3D physical attacks typically optimize texture, they often struggle to balance physical realism and attack robustness. In this work, we propose 3D Gaussian-based Adversarial Attack (3DGAA), a novel adversarial object generation framework that leverages the full 14-dimensional parameterization of 3D Gaussian Splatting (3DGS) to jointly optimize geometry and appearance in physically realizable ways. Unlike prior works that rely on patches or texture, 3DGAA jointly perturbs both geometric attributes (shape, scale, rotation) and appearance attributes (color, opacity) to produce physically realistic and transferable adversarial objects. We further introduce a physical filtering module to preserve geometric fidelity, and a physical augmentation module to simulate complex physical scenarios, thus enhancing attack generalization under real-world conditions. We evaluate 3DGAA on both virtual benchmarks and physical-world setups using miniature vehicle models. Experimental results show that 3DGAA achieves to reduce the detection mAP from 87.21% to 7.38%, significantly outperforming existing 3D physical attacks. Moreover, our method maintains high transferability across different physical conditions, demonstrating a new state-of-the-art in physically realizable adversarial attacks. These results validate 3DGAA as a practical attack framework for evaluating the safety of perception systems in autonomous driving.



## **38. Not all tokens are created equal: Perplexity Attention Weighted Networks for AI generated text detection**

cs.CL

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2501.03940v3) [paper-pdf](http://arxiv.org/pdf/2501.03940v3)

**Authors**: Pablo Miralles-González, Javier Huertas-Tato, Alejandro Martín, David Camacho

**Abstract**: The rapid advancement in large language models (LLMs) has significantly enhanced their ability to generate coherent and contextually relevant text, raising concerns about the misuse of AI-generated content and making it critical to detect it. However, the task remains challenging, particularly in unseen domains or with unfamiliar LLMs. Leveraging LLM next-token distribution outputs offers a theoretically appealing approach for detection, as they encapsulate insights from the models' extensive pre-training on diverse corpora. Despite its promise, zero-shot methods that attempt to operationalize these outputs have met with limited success. We hypothesize that one of the problems is that they use the mean to aggregate next-token distribution metrics across tokens, when some tokens are naturally easier or harder to predict and should be weighted differently. Based on this idea, we propose the Perplexity Attention Weighted Network (PAWN), which uses the last hidden states of the LLM and positions to weight the sum of a series of features based on metrics from the next-token distribution across the sequence length. Although not zero-shot, our method allows us to cache the last hidden states and next-token distribution metrics on disk, greatly reducing the training resource requirements. PAWN shows competitive and even better performance in-distribution than the strongest baselines (fine-tuned LMs) with a fraction of their trainable parameters. Our model also generalizes better to unseen domains and source models, with smaller variability in the decision boundary across distribution shifts. It is also more robust to adversarial attacks, and if the backbone has multilingual capabilities, it presents decent generalization to languages not seen during supervised training, with LLaMA3-1B reaching a mean macro-averaged F1 score of 81.46% in cross-validation with nine languages.



## **39. EVALOOP: Assessing LLM Robustness in Programming from a Self-consistency Perspective**

cs.SE

20 pages, 11 figures

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2505.12185v3) [paper-pdf](http://arxiv.org/pdf/2505.12185v3)

**Authors**: Sen Fang, Weiyuan Ding, Bowen Xu

**Abstract**: Assessing the programming capabilities of Large Language Models (LLMs) is crucial for their effective use in software engineering. Current evaluations, however, predominantly measure the accuracy of generated code on static benchmarks, neglecting the critical aspect of model robustness during programming tasks. While adversarial attacks offer insights on model robustness, their effectiveness is limited and evaluation could be constrained. Current adversarial attack methods for robustness evaluation yield inconsistent results, struggling to provide a unified evaluation across different LLMs. We introduce EVALOOP, a novel assessment framework that evaluate the robustness from a self-consistency perspective, i.e., leveraging the natural duality inherent in popular software engineering tasks, e.g., code generation and code summarization. EVALOOP initiates a self-contained feedback loop: an LLM generates output (e.g., code) from an input (e.g., natural language specification), and then use the generated output as the input to produce a new output (e.g., summarizes that code into a new specification). EVALOOP repeats the process to assess the effectiveness of EVALOOP in each loop. This cyclical strategy intrinsically evaluates robustness without rely on any external attack setups, providing a unified metric to evaluate LLMs' robustness in programming. We evaluate 16 prominent LLMs (e.g., GPT-4.1, O4-mini) on EVALOOP and found that EVALOOP typically induces a 5.01%-19.31% absolute drop in pass@1 performance within ten loops. Intriguingly, robustness does not always align with initial performance (i.e., one-time query); for instance, GPT-3.5-Turbo, despite superior initial code generation compared to DeepSeek-V2, demonstrated lower robustness over repeated evaluation loop.



## **40. AdvGrasp: Adversarial Attacks on Robotic Grasping from a Physical Perspective**

cs.RO

IJCAI'2025

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.09857v1) [paper-pdf](http://arxiv.org/pdf/2507.09857v1)

**Authors**: Xiaofei Wang, Mingliang Han, Tianyu Hao, Cegang Li, Yunbo Zhao, Keke Tang

**Abstract**: Adversarial attacks on robotic grasping provide valuable insights into evaluating and improving the robustness of these systems. Unlike studies that focus solely on neural network predictions while overlooking the physical principles of grasping, this paper introduces AdvGrasp, a framework for adversarial attacks on robotic grasping from a physical perspective. Specifically, AdvGrasp targets two core aspects: lift capability, which evaluates the ability to lift objects against gravity, and grasp stability, which assesses resistance to external disturbances. By deforming the object's shape to increase gravitational torque and reduce stability margin in the wrench space, our method systematically degrades these two key grasping metrics, generating adversarial objects that compromise grasp performance. Extensive experiments across diverse scenarios validate the effectiveness of AdvGrasp, while real-world validations demonstrate its robustness and practical applicability



## **41. Game Theory Meets LLM and Agentic AI: Reimagining Cybersecurity for the Age of Intelligent Threats**

cs.CR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10621v1) [paper-pdf](http://arxiv.org/pdf/2507.10621v1)

**Authors**: Quanyan Zhu

**Abstract**: Protecting cyberspace requires not only advanced tools but also a shift in how we reason about threats, trust, and autonomy. Traditional cybersecurity methods rely on manual responses and brittle heuristics. To build proactive and intelligent defense systems, we need integrated theoretical frameworks and software tools. Game theory provides a rigorous foundation for modeling adversarial behavior, designing strategic defenses, and enabling trust in autonomous systems. Meanwhile, software tools process cyber data, visualize attack surfaces, verify compliance, and suggest mitigations. Yet a disconnect remains between theory and practical implementation.   The rise of Large Language Models (LLMs) and agentic AI offers a new path to bridge this gap. LLM-powered agents can operationalize abstract strategies into real-world decisions. Conversely, game theory can inform the reasoning and coordination of these agents across complex workflows. LLMs also challenge classical game-theoretic assumptions, such as perfect rationality or static payoffs, prompting new models aligned with cognitive and computational realities. This co-evolution promises richer theoretical foundations and novel solution concepts. Agentic AI also reshapes software design: systems must now be modular, adaptive, and trust-aware from the outset.   This chapter explores the intersection of game theory, agentic AI, and cybersecurity. We review key game-theoretic frameworks (e.g., static, dynamic, Bayesian, and signaling games) and solution concepts. We then examine how LLM agents can enhance cyber defense and introduce LLM-driven games that embed reasoning into AI agents. Finally, we explore multi-agent workflows and coordination games, outlining how this convergence fosters secure, intelligent, and adaptive cyber systems.



## **42. Concept Steerers: Leveraging K-Sparse Autoencoders for Test-Time Controllable Generations**

cs.CV

23 pages, 18 figures

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2501.19066v2) [paper-pdf](http://arxiv.org/pdf/2501.19066v2)

**Authors**: Dahye Kim, Deepti Ghadiyaram

**Abstract**: Despite the remarkable progress in text-to-image generative models, they are prone to adversarial attacks and inadvertently generate unsafe, unethical content. Existing approaches often rely on fine-tuning models to remove specific concepts, which is computationally expensive, lacks scalability, and/or compromises generation quality. In this work, we propose a novel framework leveraging k-sparse autoencoders (k-SAEs) to enable efficient and interpretable concept manipulation in diffusion models. Specifically, we first identify interpretable monosemantic concepts in the latent space of text embeddings and leverage them to precisely steer the generation away or towards a given concept (e.g., nudity) or to introduce a new concept (e.g., photographic style) -- all during test time. Through extensive experiments, we demonstrate that our approach is very simple, requires no retraining of the base model nor LoRA adapters, does not compromise the generation quality, and is robust to adversarial prompt manipulations. Our method yields an improvement of $\mathbf{20.01\%}$ in unsafe concept removal, is effective in style manipulation, and is $\mathbf{\sim5}$x faster than the current state-of-the-art. Code is available at: https://github.com/kim-dahye/steerers



## **43. Adversarial Activation Patching: A Framework for Detecting and Mitigating Emergent Deception in Safety-Aligned Transformers**

cs.LG

**SubmitDate**: 2025-07-12    [abs](http://arxiv.org/abs/2507.09406v1) [paper-pdf](http://arxiv.org/pdf/2507.09406v1)

**Authors**: Santhosh Kumar Ravindran

**Abstract**: Large language models (LLMs) aligned for safety through techniques like reinforcement learning from human feedback (RLHF) often exhibit emergent deceptive behaviors, where outputs appear compliant but subtly mislead or omit critical information. This paper introduces adversarial activation patching, a novel mechanistic interpretability framework that leverages activation patching as an adversarial tool to induce, detect, and mitigate such deception in transformer-based models. By sourcing activations from "deceptive" prompts and patching them into safe forward passes at specific layers, we simulate vulnerabilities and quantify deception rates. Through toy neural network simulations across multiple scenarios (e.g., 1000 trials per setup), we demonstrate that adversarial patching increases deceptive outputs to 23.9% from a 0% baseline, with layer-specific variations supporting our hypotheses. We propose six hypotheses, including transferability across models, exacerbation in multimodal settings, and scaling effects. An expanded literature review synthesizes over 20 key works in interpretability, deception, and adversarial attacks. Mitigation strategies, such as activation anomaly detection and robust fine-tuning, are detailed, alongside ethical considerations and future research directions. This work advances AI safety by highlighting patching's dual-use potential and provides a roadmap for empirical studies on large-scale models.



## **44. Single Word Change is All You Need: Using LLMs to Create Synthetic Training Examples for Text Classifiers**

cs.CL

**SubmitDate**: 2025-07-12    [abs](http://arxiv.org/abs/2401.17196v3) [paper-pdf](http://arxiv.org/pdf/2401.17196v3)

**Authors**: Lei Xu, Sarah Alnegheimish, Laure Berti-Equille, Alfredo Cuesta-Infante, Kalyan Veeramachaneni

**Abstract**: In text classification, creating an adversarial example means subtly perturbing a few words in a sentence without changing its meaning, causing it to be misclassified by a classifier. A concerning observation is that a significant portion of adversarial examples generated by existing methods change only one word. This single-word perturbation vulnerability represents a significant weakness in classifiers, which malicious users can exploit to efficiently create a multitude of adversarial examples. This paper studies this problem and makes the following key contributions: (1) We introduce a novel metric $\rho$ to quantitatively assess a classifier's robustness against single-word perturbation. (2) We present the SP-Attack, designed to exploit the single-word perturbation vulnerability, achieving a higher attack success rate, better preserving sentence meaning, while reducing computation costs compared to state-of-the-art adversarial methods. (3) We propose SP-Defense, which aims to improve \r{ho} by applying data augmentation in learning. Experimental results on 4 datasets and BERT and distilBERT classifiers show that SP-Defense improves $\rho$ by 14.6% and 13.9% and decreases the attack success rate of SP-Attack by 30.4% and 21.2% on two classifiers respectively, and decreases the attack success rate of existing attack methods that involve multiple-word perturbations.



## **45. AdRo-FL: Informed and Secure Client Selection for Federated Learning in the Presence of Adversarial Aggregator**

cs.CR

17 pages

**SubmitDate**: 2025-07-12    [abs](http://arxiv.org/abs/2506.17805v2) [paper-pdf](http://arxiv.org/pdf/2506.17805v2)

**Authors**: Md. Kamrul Hossain, Walid Aljoby, Anis Elgabli, Ahmed M. Abdelmoniem, Khaled A. Harras

**Abstract**: Federated Learning (FL) enables collaborative learning without exposing clients' data. While clients only share model updates with the aggregator, studies reveal that aggregators can infer sensitive information from these updates. Secure Aggregation (SA) protects individual updates during transmission; however, recent work demonstrates a critical vulnerability where adversarial aggregators manipulate client selection to bypass SA protections, constituting a Biased Selection Attack (BSA). Although verifiable random selection prevents BSA, it precludes informed client selection essential for FL performance. We propose Adversarial Robust Federated Learning (AdRo-FL), which simultaneously enables: informed client selection based on client utility, and robust defense against BSA maintaining privacy-preserving aggregation. AdRo-FL implements two client selection frameworks tailored for distinct settings. The first framework assumes clients are grouped into clusters based on mutual trust, such as different branches of an organization. The second framework handles distributed clients where no trust relationships exist between them. For the cluster-oriented setting, we propose a novel defense against BSA by (1) enforcing a minimum client selection quota from each cluster, supervised by a cluster-head in every round, and (2) introducing a client utility function to prioritize efficient clients. For the distributed setting, we design a two-phase selection protocol: first, the aggregator selects the top clients based on our utility-driven ranking; then, a verifiable random function (VRF) ensures a BSA-resistant final selection. AdRo-FL also applies quantization to reduce communication overhead and sets strict transmission deadlines to improve energy efficiency. AdRo-FL achieves up to $1.85\times$ faster time-to-accuracy and up to $1.06\times$ higher final accuracy compared to insecure baselines.



## **46. Exploiting Leaderboards for Large-Scale Distribution of Malicious Models**

cs.LG

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08983v1) [paper-pdf](http://arxiv.org/pdf/2507.08983v1)

**Authors**: Anshuman Suri, Harsh Chaudhari, Yuefeng Peng, Ali Naseh, Amir Houmansadr, Alina Oprea

**Abstract**: While poisoning attacks on machine learning models have been extensively studied, the mechanisms by which adversaries can distribute poisoned models at scale remain largely unexplored. In this paper, we shed light on how model leaderboards -- ranked platforms for model discovery and evaluation -- can serve as a powerful channel for adversaries for stealthy large-scale distribution of poisoned models. We present TrojanClimb, a general framework that enables injection of malicious behaviors while maintaining competitive leaderboard performance. We demonstrate its effectiveness across four diverse modalities: text-embedding, text-generation, text-to-speech and text-to-image, showing that adversaries can successfully achieve high leaderboard rankings while embedding arbitrary harmful functionalities, from backdoors to bias injection. Our findings reveal a significant vulnerability in the machine learning ecosystem, highlighting the urgent need to redesign leaderboard evaluation mechanisms to detect and filter malicious (e.g., poisoned) models, while exposing broader security implications for the machine learning community regarding the risks of adopting models from unverified sources.



## **47. VIP: Visual Information Protection through Adversarial Attacks on Vision-Language Models**

eess.IV

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08982v1) [paper-pdf](http://arxiv.org/pdf/2507.08982v1)

**Authors**: Hanene F. Z. Brachemi Meftah, Wassim Hamidouche, Sid Ahmed Fezza, Olivier Déforges

**Abstract**: Recent years have witnessed remarkable progress in developing Vision-Language Models (VLMs) capable of processing both textual and visual inputs. These models have demonstrated impressive performance, leading to their widespread adoption in various applications. However, this widespread raises serious concerns regarding user privacy, particularly when models inadvertently process or expose private visual information. In this work, we frame the preservation of privacy in VLMs as an adversarial attack problem. We propose a novel attack strategy that selectively conceals information within designated Region Of Interests (ROIs) in an image, effectively preventing VLMs from accessing sensitive content while preserving the semantic integrity of the remaining image. Unlike conventional adversarial attacks that often disrupt the entire image, our method maintains high coherence in unmasked areas. Experimental results across three state-of-the-art VLMs namely LLaVA, Instruct-BLIP, and BLIP2-T5 demonstrate up to 98% reduction in detecting targeted ROIs, while maintaining global image semantics intact, as confirmed by high similarity scores between clean and adversarial outputs. We believe that this work contributes to a more privacy conscious use of multimodal models and offers a practical tool for further research, with the source code publicly available at: https://github.com/hbrachemi/Vlm_defense-attack.



## **48. Weak-to-Strong Jailbreaking on Large Language Models**

cs.CL

ICML 2025

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2401.17256v4) [paper-pdf](http://arxiv.org/pdf/2401.17256v4)

**Authors**: Xuandong Zhao, Xianjun Yang, Tianyu Pang, Chao Du, Lei Li, Yu-Xiang Wang, William Yang Wang

**Abstract**: Large language models (LLMs) are vulnerable to jailbreak attacks - resulting in harmful, unethical, or biased text generations. However, existing jailbreaking methods are computationally costly. In this paper, we propose the weak-to-strong jailbreaking attack, an efficient inference time attack for aligned LLMs to produce harmful text. Our key intuition is based on the observation that jailbroken and aligned models only differ in their initial decoding distributions. The weak-to-strong attack's key technical insight is using two smaller models (a safe and an unsafe one) to adversarially modify a significantly larger safe model's decoding probabilities. We evaluate the weak-to-strong attack on 5 diverse open-source LLMs from 3 organizations. The results show our method can increase the misalignment rate to over 99% on two datasets with just one forward pass per example. Our study exposes an urgent safety issue that needs to be addressed when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong



## **49. Entangled Threats: A Unified Kill Chain Model for Quantum Machine Learning Security**

quant-ph

Accepted for publication at IEEE International Conference on Quantum  Computing and Engineering (QCE) 2025

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08623v1) [paper-pdf](http://arxiv.org/pdf/2507.08623v1)

**Authors**: Pascal Debus, Maximilian Wendlinger, Kilian Tscharke, Daniel Herr, Cedric Brügmann, Daniel Ohl de Mello, Juris Ulmanis, Alexander Erhard, Arthur Schmidt, Fabian Petsch

**Abstract**: Quantum Machine Learning (QML) systems inherit vulnerabilities from classical machine learning while introducing new attack surfaces rooted in the physical and algorithmic layers of quantum computing. Despite a growing body of research on individual attack vectors - ranging from adversarial poisoning and evasion to circuit-level backdoors, side-channel leakage, and model extraction - these threats are often analyzed in isolation, with unrealistic assumptions about attacker capabilities and system environments. This fragmentation hampers the development of effective, holistic defense strategies. In this work, we argue that QML security requires more structured modeling of the attack surface, capturing not only individual techniques but also their relationships, prerequisites, and potential impact across the QML pipeline. We propose adapting kill chain models, widely used in classical IT and cybersecurity, to the quantum machine learning context. Such models allow for structured reasoning about attacker objectives, capabilities, and possible multi-stage attack paths - spanning reconnaissance, initial access, manipulation, persistence, and exfiltration. Based on extensive literature analysis, we present a detailed taxonomy of QML attack vectors mapped to corresponding stages in a quantum-aware kill chain framework that is inspired by the MITRE ATLAS for classical machine learning. We highlight interdependencies between physical-level threats (like side-channel leakage and crosstalk faults), data and algorithm manipulation (such as poisoning or circuit backdoors), and privacy attacks (including model extraction and training data inference). This work provides a foundation for more realistic threat modeling and proactive security-in-depth design in the emerging field of quantum machine learning.



## **50. The Dark Side of LLMs Agent-based Attacks for Complete Computer Takeover**

cs.CR

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.06850v3) [paper-pdf](http://arxiv.org/pdf/2507.06850v3)

**Authors**: Matteo Lupinacci, Francesco Aurelio Pironti, Francesco Blefari, Francesco Romeo, Luigi Arena, Angelo Furfaro

**Abstract**: The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables unprecedented capabilities in natural language processing and generation. However, these systems have introduced unprecedented security vulnerabilities that extend beyond traditional prompt injection attacks. This paper presents the first comprehensive evaluation of LLM agents as attack vectors capable of achieving complete computer takeover through the exploitation of trust boundaries within agentic AI systems where autonomous entities interact and influence each other. We demonstrate that adversaries can leverage three distinct attack surfaces - direct prompt injection, RAG backdoor attacks, and inter-agent trust exploitation - to coerce popular LLMs (including GPT-4o, Claude-4 and Gemini-2.5) into autonomously installing and executing malware on victim machines. Our evaluation of 17 state-of-the-art LLMs reveals an alarming vulnerability hierarchy: while 41.2% of models succumb to direct prompt injection, 52.9% are vulnerable to RAG backdoor attacks, and a critical 82.4% can be compromised through inter-agent trust exploitation. Notably, we discovered that LLMs which successfully resist direct malicious commands will execute identical payloads when requested by peer agents, revealing a fundamental flaw in current multi-agent security models. Our findings demonstrate that only 5.9% of tested models (1/17) proved resistant to all attack vectors, with the majority exhibiting context-dependent security behaviors that create exploitable blind spots. Our findings also highlight the need to increase awareness and research on the security risks of LLMs, showing a paradigm shift in cybersecurity threats, where AI tools themselves become sophisticated attack vectors.



