# Latest Adversarial Attack Papers
**update at 2025-10-16 16:22:32**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Provably Invincible Adversarial Attacks on Reinforcement Learning Systems: A Rate-Distortion Information-Theoretic Approach**

cs.LG

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13792v1) [paper-pdf](http://arxiv.org/pdf/2510.13792v1)

**Authors**: Ziqing Lu, Lifeng Lai, Weiyu Xu

**Abstract**: Reinforcement learning (RL) for the Markov Decision Process (MDP) has emerged in many security-related applications, such as autonomous driving, financial decisions, and drone/robot algorithms. In order to improve the robustness/defense of RL systems against adversaries, studying various adversarial attacks on RL systems is very important. Most previous work considered deterministic adversarial attack strategies in MDP, which the recipient (victim) agent can defeat by reversing the deterministic attacks. In this paper, we propose a provably ``invincible'' or ``uncounterable'' type of adversarial attack on RL. The attackers apply a rate-distortion information-theoretic approach to randomly change agents' observations of the transition kernel (or other properties) so that the agent gains zero or very limited information about the ground-truth kernel (or other properties) during the training. We derive an information-theoretic lower bound on the recipient agent's reward regret and show the impact of rate-distortion attacks on state-of-the-art model-based and model-free algorithms. We also extend this notion of an information-theoretic approach to other types of adversarial attack, such as state observation attacks.



## **2. Towards Adversarial Robustness and Uncertainty Quantification in DINOv2-based Few-Shot Anomaly Detection**

cs.CV

10 pages, 5 figures, 3 tables

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13643v1) [paper-pdf](http://arxiv.org/pdf/2510.13643v1)

**Authors**: Akib Mohammed Khan, Bartosz Krawczyk

**Abstract**: Foundation models such as DINOv2 have shown strong performance in few-shot anomaly detection, yet two key questions remain unexamined: (i) how susceptible are these detectors to adversarial perturbations; and (ii) how well do their anomaly scores reflect calibrated uncertainty? Building on AnomalyDINO, a training-free deep nearest-neighbor detector over DINOv2 features, we present one of the first systematic studies of adversarial attacks and uncertainty estimation in this setting. To enable white-box gradient attacks while preserving test-time behavior, we attach a lightweight linear head to frozen DINOv2 features only for crafting perturbations. Using this heuristic, we evaluate the impact of FGSM across the MVTec-AD and VisA datasets and observe consistent drops in F1, AUROC, AP, and G-mean, indicating that imperceptible perturbations can flip nearest-neighbor relations in feature space to induce confident misclassification. Complementing robustness, we probe reliability and find that raw anomaly scores are poorly calibrated, revealing a gap between confidence and correctness that limits safety-critical use. As a simple, strong baseline toward trustworthiness, we apply post-hoc Platt scaling to the anomaly scores for uncertainty estimation. The resulting calibrated posteriors yield significantly higher predictive entropy on adversarially perturbed inputs than on clean ones, enabling a practical flagging mechanism for attack detection while reducing calibration error (ECE). Our findings surface concrete vulnerabilities in DINOv2-based few-shot anomaly detectors and establish an evaluation protocol and baseline for robust, uncertainty-aware anomaly detection. We argue that adversarial robustness and principled uncertainty quantification are not optional add-ons but essential capabilities if anomaly detection systems are to be trustworthy and ready for real-world deployment.



## **3. Selective Adversarial Attacks on LLM Benchmarks**

cs.LG

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13570v1) [paper-pdf](http://arxiv.org/pdf/2510.13570v1)

**Authors**: Ivan Dubrovsky, Anastasia Orlova, Illarion Iov, Nina Gubina, Irena Gureeva, Alexey Zaytsev

**Abstract**: Benchmarking outcomes increasingly govern trust, selection, and deployment of LLMs, yet these evaluations remain vulnerable to semantically equivalent adversarial perturbations. Prior work on adversarial robustness in NLP has emphasized text attacks that affect many models equally, leaving open the question of whether it is possible to selectively degrade or enhance performance while minimally affecting other models. We formalize this problem and study selective adversarial attacks on MMLU - a widely used benchmark designed to measure a language model's broad general knowledge and reasoning ability across different subjects. Using canonical attacks integrated into TextAttack framework, we introduce a protocol for selectivity assessment, develop a custom constraint to increase selectivity of attacks and propose a surrogate-LLM pipeline that generates selective perturbations. Empirically, we find that selective adversarial attacks exist and can materially alter relative rankings, challenging the fairness, reproducibility, and transparency of leaderboard-driven evaluation. Our results motivate perturbation-aware reporting and robustness diagnostics for LLM evaluation and demonstrate that even subtle edits can shift comparative judgments.



## **4. Systematic Literature Review on Vehicular Collaborative Perception - A Computer Vision Perspective**

cs.CV

38 pages, 8 figures

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2504.04631v2) [paper-pdf](http://arxiv.org/pdf/2504.04631v2)

**Authors**: Lei Wan, Jianxin Zhao, Andreas Wiedholz, Manuel Bied, Mateus Martinez de Lucena, Abhishek Dinkar Jagtap, Andreas Festag, Antônio Augusto Fröhlich, Hannan Ejaz Keen, Alexey Vinel

**Abstract**: The effectiveness of autonomous vehicles relies on reliable perception capabilities. Despite significant advancements in artificial intelligence and sensor fusion technologies, current single-vehicle perception systems continue to encounter limitations, notably visual occlusions and limited long-range detection capabilities. Collaborative Perception (CP), enabled by Vehicle-to-Vehicle (V2V) and Vehicle-to-Infrastructure (V2I) communication, has emerged as a promising solution to mitigate these issues and enhance the reliability of autonomous systems. Beyond advancements in communication, the computer vision community is increasingly focusing on improving vehicular perception through collaborative approaches. However, a systematic literature review that thoroughly examines existing work and reduces subjective bias is still lacking. Such a systematic approach helps identify research gaps, recognize common trends across studies, and inform future research directions. In response, this study follows the PRISMA 2020 guidelines and includes 106 peer-reviewed articles. These publications are analyzed based on modalities, collaboration schemes, and key perception tasks. Through a comparative analysis, this review illustrates how different methods address practical issues such as pose errors, temporal latency, communication constraints, domain shifts, heterogeneity, and adversarial attacks. Furthermore, it critically examines evaluation methodologies, highlighting a misalignment between current metrics and CP's fundamental objectives. By delving into all relevant topics in-depth, this review offers valuable insights into challenges, opportunities, and risks, serving as a reference for advancing research in vehicular collaborative perception.



## **5. Towards Quantum Enhanced Adversarial Robustness with Rydberg Reservoir Learnin**

quant-ph

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13473v1) [paper-pdf](http://arxiv.org/pdf/2510.13473v1)

**Authors**: Shehbaz Tariq, Muhammad Talha, Symeon Chatzinotas, Hyundong Shin

**Abstract**: Quantum reservoir computing (QRC) leverages the high-dimensional, nonlinear dynamics inherent in quantum many-body systems for extracting spatiotemporal patterns in sequential and time-series data with minimal training overhead. Although QRC inherits the expressive capabilities associated with quantum encodings, recent studies indicate that quantum classifiers based on variational circuits remain susceptible to adversarial perturbations. In this perspective, we investigate the first systematic evaluation of adversarial robustness in a QRC based learning model. Our reservoir comprises an array of strongly interacting Rydberg atoms governed by a fixed Hamiltonian, which naturally evolves under complex quantum dynamics, producing high-dimensional embeddings. A lightweight multilayer perceptron serves as the trainable readout layer. We utilize the balanced datasets, namely MNIST, Fashion-MNIST, and Kuzushiji-MNIST, as a benchmark for rigorously evaluating the impact of augmenting the quantum reservoir with a Multilayer perceptron (MLP) in white-box adversarial attacks to assess its robustness. We demonstrate that this approach yields significantly higher accuracy than purely classical models across all perturbation strengths tested. This hybrid approach reveals a new source of quantum advantage and



## **6. Generalist++: A Meta-learning Framework for Mitigating Trade-off in Adversarial Training**

cs.LG

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13361v1) [paper-pdf](http://arxiv.org/pdf/2510.13361v1)

**Authors**: Yisen Wang, Yichuan Mo, Hongjun Wang, Junyi Li, Zhouchen Lin

**Abstract**: Despite the rapid progress of neural networks, they remain highly vulnerable to adversarial examples, for which adversarial training (AT) is currently the most effective defense. While AT has been extensively studied, its practical applications expose two major limitations: natural accuracy tends to degrade significantly compared with standard training, and robustness does not transfer well across attacks crafted under different norm constraints. Unlike prior works that attempt to address only one issue within a single network, we propose to partition the overall generalization goal into multiple sub-tasks, each assigned to a dedicated base learner. By specializing in its designated objective, each base learner quickly becomes an expert in its field. In the later stages of training, we interpolate their parameters to form a knowledgeable global learner, while periodically redistributing the global parameters back to the base learners to prevent their optimization trajectories from drifting too far from the shared target. We term this framework Generalist and introduce three variants tailored to different application scenarios. Both theoretical analysis and extensive experiments demonstrate that Generalist achieves lower generalization error and significantly alleviates the trade-off problems compared with baseline methods. Our results suggest that Generalist provides a promising step toward developing fully robust classifiers in the future.



## **7. SafeGuider: Robust and Practical Content Safety Control for Text-to-Image Models**

cs.CR

Accepted by ACM CCS 2025, Code is available at [this https  URL](https://github.com/pgqihere/safeguider)

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.05173v3) [paper-pdf](http://arxiv.org/pdf/2510.05173v3)

**Authors**: Peigui Qi, Kunsheng Tang, Wenbo Zhou, Weiming Zhang, Nenghai Yu, Tianwei Zhang, Qing Guo, Jie Zhang

**Abstract**: Text-to-image models have shown remarkable capabilities in generating high-quality images from natural language descriptions. However, these models are highly vulnerable to adversarial prompts, which can bypass safety measures and produce harmful content. Despite various defensive strategies, achieving robustness against attacks while maintaining practical utility in real-world applications remains a significant challenge. To address this issue, we first conduct an empirical study of the text encoder in the Stable Diffusion (SD) model, which is a widely used and representative text-to-image model. Our findings reveal that the [EOS] token acts as a semantic aggregator, exhibiting distinct distributional patterns between benign and adversarial prompts in its embedding space. Building on this insight, we introduce SafeGuider, a two-step framework designed for robust safety control without compromising generation quality. SafeGuider combines an embedding-level recognition model with a safety-aware feature erasure beam search algorithm. This integration enables the framework to maintain high-quality image generation for benign prompts while ensuring robust defense against both in-domain and out-of-domain attacks. SafeGuider demonstrates exceptional effectiveness in minimizing attack success rates, achieving a maximum rate of only 5.48\% across various attack scenarios. Moreover, instead of refusing to generate or producing black images for unsafe prompts, SafeGuider generates safe and meaningful images, enhancing its practical utility. In addition, SafeGuider is not limited to the SD model and can be effectively applied to other text-to-image models, such as the Flux model, demonstrating its versatility and adaptability across different architectures. We hope that SafeGuider can shed some light on the practical deployment of secure text-to-image systems.



## **8. SAJA: A State-Action Joint Attack Framework on Multi-Agent Deep Reinforcement Learning**

cs.AI

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13262v1) [paper-pdf](http://arxiv.org/pdf/2510.13262v1)

**Authors**: Weiqi Guo, Guanjun Liu, Ziyuan Zhou

**Abstract**: Multi-Agent Deep Reinforcement Learning (MADRL) has shown potential for cooperative and competitive tasks such as autonomous driving and strategic gaming. However, models trained by MADRL are vulnerable to adversarial perturbations on states and actions. Therefore, it is essential to investigate the robustness of MADRL models from an attack perspective. Existing studies focus on either state-only attacks or action-only attacks, but do not consider how to effectively joint them. Simply combining state and action perturbations such as randomly perturbing states and actions does not exploit their potential synergistic effects. In this paper, we propose the State-Action Joint Attack (SAJA) framework that has a good synergistic effects. SAJA consists of two important phases: (1) In the state attack phase, a multi-step gradient ascent method utilizes both the actor network and the critic network to compute an adversarial state, and (2) in the action attack phase, based on the perturbed state, a second gradient ascent uses the critic network to craft the final adversarial action. Additionally, a heuristic regularizer measuring the distance between the perturbed actions and the original clean ones is added into the loss function to enhance the effectiveness of the critic's guidance. We evaluate SAJA in the Multi-Agent Particle Environment (MPE), demonstrating that (1) it outperforms and is more stealthy than state-only or action-only attacks, and (2) existing state or action defense methods cannot defend its attacks.



## **9. Can an Individual Manipulate the Collective Decisions of Multi-Agents?**

cs.CL

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2509.16494v2) [paper-pdf](http://arxiv.org/pdf/2509.16494v2)

**Authors**: Fengyuan Liu, Rui Zhao, Shuo Chen, Guohao Li, Philip Torr, Lei Han, Jindong Gu

**Abstract**: Individual Large Language Models (LLMs) have demonstrated significant capabilities across various domains, such as healthcare and law. Recent studies also show that coordinated multi-agent systems exhibit enhanced decision-making and reasoning abilities through collaboration. However, due to the vulnerabilities of individual LLMs and the difficulty of accessing all agents in a multi-agent system, a key question arises: If attackers only know one agent, could they still generate adversarial samples capable of misleading the collective decision? To explore this question, we formulate it as a game with incomplete information, where attackers know only one target agent and lack knowledge of the other agents in the system. With this formulation, we propose M-Spoiler, a framework that simulates agent interactions within a multi-agent system to generate adversarial samples. These samples are then used to manipulate the target agent in the target system, misleading the system's collaborative decision-making process. More specifically, M-Spoiler introduces a stubborn agent that actively aids in optimizing adversarial samples by simulating potential stubborn responses from agents in the target system. This enhances the effectiveness of the generated adversarial samples in misleading the system. Through extensive experiments across various tasks, our findings confirm the risks posed by the knowledge of an individual agent in multi-agent systems and demonstrate the effectiveness of our framework. We also explore several defense mechanisms, showing that our proposed attack framework remains more potent than baselines, underscoring the need for further research into defensive strategies.



## **10. Model-agnostic Adversarial Attack and Defense for Vision-Language-Action Models**

cs.CV

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13237v1) [paper-pdf](http://arxiv.org/pdf/2510.13237v1)

**Authors**: Haochuan Xu, Yun Sing Koh, Shuhuai Huang, Zirun Zhou, Di Wang, Jun Sakuma, Jingfeng Zhang

**Abstract**: Vision-Language-Action (VLA) models have achieved revolutionary progress in robot learning, enabling robots to execute complex physical robot tasks from natural language instructions. Despite this progress, their adversarial robustness remains underexplored. In this work, we propose both adversarial patch attack and corresponding defense strategies for VLA models. We first introduce the Embedding Disruption Patch Attack (EDPA), a model-agnostic adversarial attack that generates patches directly placeable within the camera's view. In comparison to prior methods, EDPA can be readily applied to different VLA models without requiring prior knowledge of the model architecture, or the controlled robotic manipulator. EDPA constructs these patches by (i) disrupting the semantic alignment between visual and textual latent representations, and (ii) maximizing the discrepancy of latent representations between adversarial and corresponding clean visual inputs. Through the optimization of these objectives, EDPA distorts the VLA's interpretation of visual information, causing the model to repeatedly generate incorrect actions and ultimately result in failure to complete the given robotic task. To counter this, we propose an adversarial fine-tuning scheme for the visual encoder, in which the encoder is optimized to produce similar latent representations for both clean and adversarially perturbed visual inputs. Extensive evaluations on the widely recognized LIBERO robotic simulation benchmark demonstrate that EDPA substantially increases the task failure rate of cutting-edge VLA models, while our proposed defense effectively mitigates this degradation. The codebase is accessible via the homepage at https://edpa-attack.github.io/.



## **11. SHIELD: Classifier-Guided Prompting for Robust and Safer LVLMs**

cs.CL

Preprint

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13190v1) [paper-pdf](http://arxiv.org/pdf/2510.13190v1)

**Authors**: Juan Ren, Mark Dras, Usman Naseem

**Abstract**: Large Vision-Language Models (LVLMs) unlock powerful multimodal reasoning but also expand the attack surface, particularly through adversarial inputs that conceal harmful goals in benign prompts. We propose SHIELD, a lightweight, model-agnostic preprocessing framework that couples fine-grained safety classification with category-specific guidance and explicit actions (Block, Reframe, Forward). Unlike binary moderators, SHIELD composes tailored safety prompts that enforce nuanced refusals or safe redirection without retraining. Across five benchmarks and five representative LVLMs, SHIELD consistently lowers jailbreak and non-following rates while preserving utility. Our method is plug-and-play, incurs negligible overhead, and is easily extendable to new attack types -- serving as a practical safety patch for both weakly and strongly aligned LVLMs.



## **12. Improving Transferability of Adversarial Examples via Bayesian Attacks**

cs.LG

Accepted by TCSVT

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2307.11334v2) [paper-pdf](http://arxiv.org/pdf/2307.11334v2)

**Authors**: Qizhang Li, Yiwen Guo, Xiaochen Yang, Wangmeng Zuo, Hao Chen

**Abstract**: The transferability of adversarial examples allows for the attack on unknown deep neural networks (DNNs), posing a serious threat to many applications and attracting great attention. In this paper, we improve the transferability of adversarial examples by incorporating the Bayesian formulation into both the model parameters and model input, enabling their joint diversification. We demonstrate that combination of Bayesian formulations for both the model input and model parameters yields significant improvements in transferability. By introducing advanced approximations of the posterior distribution over the model input, adversarial transferability achieves further enhancement, surpassing all state-of-the-arts when attacking without model fine-tuning. Additionally, we propose a principled approach to fine-tune model parameters within this Bayesian framework. Extensive experiments demonstrate that our method achieves a new state-of-the-art in transfer-based attacks, significantly improving the average success rate on ImageNet and CIFAR-10. Code at: https://github.com/qizhangli/MoreBayesian-jrnl.



## **13. Privacy-Aware Framework of Robust Malware Detection in Indoor Robots: Hybrid Quantum Computing and Deep Neural Networks**

cs.CR

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2510.13136v1) [paper-pdf](http://arxiv.org/pdf/2510.13136v1)

**Authors**: Tan Le, Van Le, Sachin Shetty

**Abstract**: Indoor robotic systems within Cyber-Physical Systems (CPS) are increasingly exposed to Denial of Service (DoS) attacks that compromise localization, control and telemetry integrity. We propose a privacy-aware malware detection framework for indoor robotic systems, which leverages hybrid quantum computing and deep neural networks to counter DoS threats in CPS, while preserving privacy information. By integrating quantum-enhanced feature encoding with dropout-optimized deep learning, our architecture achieves up to 95.2% detection accuracy under privacy-constrained conditions. The system operates without handcrafted thresholds or persistent beacon data, enabling scalable deployment in adversarial environments. Benchmarking reveals robust generalization, interpretability and resilience against training instability through modular circuit design. This work advances trustworthy AI for secure, autonomous CPS operations.



## **14. RedTeamCUA: Realistic Adversarial Testing of Computer-Use Agents in Hybrid Web-OS Environments**

cs.CL

**SubmitDate**: 2025-10-15    [abs](http://arxiv.org/abs/2505.21936v3) [paper-pdf](http://arxiv.org/pdf/2505.21936v3)

**Authors**: Zeyi Liao, Jaylen Jones, Linxi Jiang, Yuting Ning, Eric Fosler-Lussier, Yu Su, Zhiqiang Lin, Huan Sun

**Abstract**: Computer-use agents (CUAs) promise to automate complex tasks across operating systems (OS) and the web, but remain vulnerable to indirect prompt injection. Current evaluations of this threat either lack support realistic but controlled environments or ignore hybrid web-OS attack scenarios involving both interfaces. To address this, we propose RedTeamCUA, an adversarial testing framework featuring a novel hybrid sandbox that integrates a VM-based OS environment with Docker-based web platforms. Our sandbox supports key features tailored for red teaming, such as flexible adversarial scenario configuration, and a setting that decouples adversarial evaluation from navigational limitations of CUAs by initializing tests directly at the point of an adversarial injection. Using RedTeamCUA, we develop RTC-Bench, a comprehensive benchmark with 864 examples that investigate realistic, hybrid web-OS attack scenarios and fundamental security vulnerabilities. Benchmarking current frontier CUAs identifies significant vulnerabilities: Claude 3.7 Sonnet | CUA demonstrates an ASR of 42.9%, while Operator, the most secure CUA evaluated, still exhibits an ASR of 7.6%. Notably, CUAs often attempt to execute adversarial tasks with an Attempt Rate as high as 92.5%, although failing to complete them due to capability limitations. Nevertheless, we observe concerning high ASRs in realistic end-to-end settings, with the strongest-to-date Claude 4.5 Sonnet | CUA exhibiting the highest ASR of 60%, indicating that CUA threats can already result in tangible risks to users and computer systems. Overall, RedTeamCUA provides an essential framework for advancing realistic, controlled, and systematic analysis of CUA vulnerabilities, highlighting the urgent need for robust defenses to indirect prompt injection prior to real-world deployment.



## **15. SoundnessBench: A Soundness Benchmark for Neural Network Verifiers**

cs.LG

Preprint

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2412.03154v2) [paper-pdf](http://arxiv.org/pdf/2412.03154v2)

**Authors**: Xingjian Zhou, Keyi Shen, Andy Xu, Hongji Xu, Cho-Jui Hsieh, Huan Zhang, Zhouxing Shi

**Abstract**: Neural network (NN) verification aims to formally verify properties of NNs, which is crucial for ensuring the behavior of NN-based models in safety-critical applications. In recent years, the community has developed many NN verifiers and benchmarks to evaluate them. However, existing benchmarks typically lack ground-truth for hard instances where no current verifier can verify the property and no counterexample can be found. This makes it difficult to validate the soundness of a verifier, when it claims verification on such challenging instances that no other verifier can handle. In this work, we develop a new benchmark for NN verification, named "SoundnessBench", specifically for testing the soundness of NN verifiers. SoundnessBench consists of instances with deliberately inserted counterexamples that are hidden from adversarial attacks commonly used to find counterexamples. Thereby, it can identify false verification claims when hidden counterexamples are known to exist. We design a training method to produce NNs with hidden counterexamples and systematically construct our SoundnessBench with instances across various model architectures, activation functions, and input data. We demonstrate that our training effectively produces hidden counterexamples and our SoundnessBench successfully identifies bugs in state-of-the-art NN verifiers. Our code is available at https://github.com/MVP-Harry/SoundnessBench and our benchmark is available at https://huggingface.co/datasets/SoundnessBench/SoundnessBench.



## **16. A Survey of Graph Unlearning**

cs.LG

15 page review paper on graph unlearning

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2310.02164v4) [paper-pdf](http://arxiv.org/pdf/2310.02164v4)

**Authors**: Anwar Said, Ngoc N. Tran, Yuying Zhao, Tyler Derr, Mudassir Shabbir, Waseem Abbas, Xenofon Koutsoukos

**Abstract**: Graph unlearning emerges as a crucial advancement in the pursuit of responsible AI, providing the means to remove sensitive data traces from trained models, thereby upholding the \textit{right to be forgotten}. It is evident that graph machine learning exhibits sensitivity to data privacy and adversarial attacks, necessitating the application of graph unlearning techniques to address these concerns effectively. In this comprehensive survey paper, we present the first systematic review of graph unlearning approaches, encompassing a diverse array of methodologies and offering a detailed taxonomy and up-to-date literature overview to facilitate the understanding of researchers new to this field. To ensure clarity, we provide lucid explanations of the fundamental concepts and evaluation measures used in graph unlearning, catering to a broader audience with varying levels of expertise. Delving into potential applications, we explore the versatility of graph unlearning across various domains, including but not limited to social networks, adversarial settings, recommender systems, and resource-constrained environments like the Internet of Things, illustrating its potential impact in safeguarding data privacy and enhancing AI systems' robustness. Finally, we shed light on promising research directions, encouraging further progress and innovation within the domain of graph unlearning. By laying a solid foundation and fostering continued progress, this survey seeks to inspire researchers to further advance the field of graph unlearning, thereby instilling confidence in the ethical growth of AI systems and reinforcing the responsible application of machine learning techniques in various domains.



## **17. KoALA: KL-L0 Adversarial Detector via Label Agreement**

cs.LG

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12752v1) [paper-pdf](http://arxiv.org/pdf/2510.12752v1)

**Authors**: Siqi Li, Yasser Shoukry

**Abstract**: Deep neural networks are highly susceptible to adversarial attacks, which pose significant risks to security- and safety-critical applications. We present KoALA (KL-L0 Adversarial detection via Label Agreement), a novel, semantics-free adversarial detector that requires no architectural changes or adversarial retraining. KoALA operates on a simple principle: it detects an adversarial attack when class predictions from two complementary similarity metrics disagree. These metrics-KL divergence and an L0-based similarity-are specifically chosen to detect different types of perturbations. The KL divergence metric is sensitive to dense, low-amplitude shifts, while the L0-based similarity is designed for sparse, high-impact changes. We provide a formal proof of correctness for our approach. The only training required is a simple fine-tuning step on a pre-trained image encoder using clean images to ensure the embeddings align well with both metrics. This makes KOALA a lightweight, plug-and-play solution for existing models and various data modalities. Our extensive experiments on ResNet/CIFAR-10 and CLIP/Tiny-ImageNet confirm our theoretical claims. When the theorem's conditions are met, KoALA consistently and effectively detects adversarial examples. On the full test sets, KoALA achieves a precision of 0.94 and a recall of 0.81 on ResNet/CIFAR-10, and a precision of 0.66 and a recall of 0.85 on CLIP/Tiny-ImageNet.



## **18. Towards Robust Artificial Intelligence: Self-Supervised Learning Approach for Out-of-Distribution Detection**

cs.AI

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12713v1) [paper-pdf](http://arxiv.org/pdf/2510.12713v1)

**Authors**: Wissam Salhab, Darine Ameyed, Hamid Mcheick, Fehmi Jaafar

**Abstract**: Robustness in AI systems refers to their ability to maintain reliable and accurate performance under various conditions, including out-of-distribution (OOD) samples, adversarial attacks, and environmental changes. This is crucial in safety-critical systems, such as autonomous vehicles, transportation, or healthcare, where malfunctions could have severe consequences. This paper proposes an approach to improve OOD detection without the need of labeled data, thereby increasing the AI systems' robustness. The proposed approach leverages the principles of self-supervised learning, allowing the model to learn useful representations from unlabeled data. Combined with graph-theoretical techniques, this enables the more efficient identification and categorization of OOD samples. Compared to existing state-of-the-art methods, this approach achieved an Area Under the Receiver Operating Characteristic Curve (AUROC) = 0.99.



## **19. Keep Calm and Avoid Harmful Content: Concept Alignment and Latent Manipulation Towards Safer Answers**

cs.LG

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12672v1) [paper-pdf](http://arxiv.org/pdf/2510.12672v1)

**Authors**: Ruben Belo, Claudia Soares, Marta Guimaraes

**Abstract**: Large Language Models are susceptible to jailbreak attacks that bypass built-in safety guardrails (e.g., by tricking the model with adversarial prompts). We propose Concept Alignment and Concept Manipulation \textbf{CALM}, an inference-time method that suppresses harmful concepts by modifying latent representations of the last layer of the model, without retraining. Leveraging \gls*{cw} technique from Computer Vision combined with orthogonal projection, CALM removes unwanted latent directions associated with harmful content while preserving model performance. Experiments show that CALM reduces harmful outputs and outperforms baseline methods in most metrics, offering a lightweight approach to AI safety with no additional training data or model fine-tuning, while incurring only a small computational overhead at inference.



## **20. Malice in Agentland: Down the Rabbit Hole of Backdoors in the AI Supply Chain**

cs.CR

27 pages

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.05159v2) [paper-pdf](http://arxiv.org/pdf/2510.05159v2)

**Authors**: Léo Boisvert, Abhay Puri, Chandra Kiran Reddy Evuru, Nicolas Chapados, Quentin Cappart, Alexandre Lacoste, Krishnamurthy Dj Dvijotham, Alexandre Drouin

**Abstract**: The practice of fine-tuning AI agents on data from their own interactions--such as web browsing or tool use--, while being a strong general recipe for improving agentic capabilities, also introduces a critical security vulnerability within the AI supply chain. In this work, we show that adversaries can easily poison the data collection pipeline to embed hard-to-detect backdoors that are triggerred by specific target phrases, such that when the agent encounters these triggers, it performs an unsafe or malicious action. We formalize and validate three realistic threat models targeting different layers of the supply chain: 1) direct poisoning of fine-tuning data, where an attacker controls a fraction of the training traces; 2) environmental poisoning, where malicious instructions are injected into webpages scraped or tools called while creating training data; and 3) supply chain poisoning, where a pre-backdoored base model is fine-tuned on clean data to improve its agentic capabilities. Our results are stark: by poisoning as few as 2% of the collected traces, an attacker can embed a backdoor causing an agent to leak confidential user information with over 80% success when a specific trigger is present. This vulnerability holds across all three threat models. Furthermore, we demonstrate that prominent safeguards, including two guardrail models and one weight-based defense, fail to detect or prevent the malicious behavior. These findings highlight an urgent threat to agentic AI development and underscore the critical need for rigorous security vetting of data collection processes and end-to-end model supply chains.



## **21. PEAR: Planner-Executor Agent Robustness Benchmark**

cs.LG

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.07505v2) [paper-pdf](http://arxiv.org/pdf/2510.07505v2)

**Authors**: Shen Dong, Mingxuan Zhang, Pengfei He, Li Ma, Bhavani Thuraisingham, Hui Liu, Yue Xing

**Abstract**: Large Language Model (LLM)-based Multi-Agent Systems (MAS) have emerged as a powerful paradigm for tackling complex, multi-step tasks across diverse domains. However, despite their impressive capabilities, MAS remain susceptible to adversarial manipulation. Existing studies typically examine isolated attack surfaces or specific scenarios, leaving a lack of holistic understanding of MAS vulnerabilities. To bridge this gap, we introduce PEAR, a benchmark for systematically evaluating both the utility and vulnerability of planner-executor MAS. While compatible with various MAS architectures, our benchmark focuses on the planner-executor structure, which is a practical and widely adopted design. Through extensive experiments, we find that (1) a weak planner degrades overall clean task performance more severely than a weak executor; (2) while a memory module is essential for the planner, having a memory module for the executor does not impact the clean task performance; (3) there exists a trade-off between task performance and robustness; and (4) attacks targeting the planner are particularly effective at misleading the system. These findings offer actionable insights for enhancing the robustness of MAS and lay the groundwork for principled defenses in multi-agent settings.



## **22. Proof of Cloud: Data Center Execution Assurance for Confidential VMs**

cs.CR

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12469v1) [paper-pdf](http://arxiv.org/pdf/2510.12469v1)

**Authors**: Filip Rezabek, Moe Mahhouk, Andrew Miller, Stefan Genchev, Quintus Kilbourn, Georg Carle, Jonathan Passerat-Palmbach

**Abstract**: Confidential Virtual Machines (CVMs) protect data in use by running workloads inside hardware-isolated environments. In doing so, they also inherit the limitations of the underlying hardware. Trusted Execution Environments (TEEs), which enforce this isolation, explicitly exclude adversaries with physical access from their threat model. Commercial TEEs, e.g., Intel TDX, thus assume infrastructure providers do not physically exploit hardware and serve as safeguards instead. This creates a tension: tenants must trust provider integrity at the hardware layer, yet existing remote attestation offers no way to verify that CVMs actually run on physically trusted platforms, leaving today's CVM deployments unable to demonstrate that their guarantees align with the TEE vendor's threat model.   We bridge this confidence gap with Data Center Execution Assurance (DCEA), a design generating "Proofs of Cloud". DCEA binds a CVM to its underlying platform using vTPM-anchored measurements, ensuring CVM launch evidence and TPM quotes refer to the same physical chassis.   This takes advantage of the fact that data centers are often identifiable via TPMs. Our approach applies to CVMs accessing vTPMs and running on top of software stacks fully controlled by the cloud provider, as well as single-tenant bare-metal deployments with discrete TPMs. We trust providers for integrity (certificate issuance), but not for the confidentiality of CVM-visible state. DCEA enables remote verification of a CVM's platform origin and integrity, mitigating attacks like replay and attestation proxying. We include a candidate implementation on Google Cloud and Intel TDX that leverages Intel TXT for trusted launch. Our design refines CVMs' threat model and provides a practical path for deploying high-assurance, confidential workloads in minimally trusted environments.



## **23. MS-GAGA: Metric-Selective Guided Adversarial Generation Attack**

cs.CV

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12468v1) [paper-pdf](http://arxiv.org/pdf/2510.12468v1)

**Authors**: Dion J. X. Ho, Gabriel Lee Jun Rong, Niharika Shrivastava, Harshavardhan Abichandani, Pai Chet Ng, Xiaoxiao Miao

**Abstract**: We present MS-GAGA (Metric-Selective Guided Adversarial Generation Attack), a two-stage framework for crafting transferable and visually imperceptible adversarial examples against deepfake detectors in black-box settings. In Stage 1, a dual-stream attack module generates adversarial candidates: MNTD-PGD applies enhanced gradient calculations optimized for small perturbation budgets, while SG-PGD focuses perturbations on visually salient regions. This complementary design expands the adversarial search space and improves transferability across unseen models. In Stage 2, a metric-aware selection module evaluates candidates based on both their success against black-box models and their structural similarity (SSIM) to the original image. By jointly optimizing transferability and imperceptibility, MS-GAGA achieves up to 27% higher misclassification rates on unseen detectors compared to state-of-the-art attacks.



## **24. IP-Augmented Multi-Modal Malicious URL Detection Via Token-Contrastive Representation Enhancement and Multi-Granularity Fusion**

cs.CR

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12395v1) [paper-pdf](http://arxiv.org/pdf/2510.12395v1)

**Authors**: Ye Tian, Yanqiu Yu, Liangliang Song, Zhiquan Liu, Yanbin Wang, Jianguo Sun

**Abstract**: Malicious URL detection remains a critical cybersecurity challenge as adversaries increasingly employ sophisticated evasion techniques including obfuscation, character-level perturbations, and adversarial attacks. Although pre-trained language models (PLMs) like BERT have shown potential for URL analysis tasks, three limitations persist in current implementations: (1) inability to effectively model the non-natural hierarchical structure of URLs, (2) insufficient sensitivity to character-level obfuscation, and (3) lack of mechanisms to incorporate auxiliary network-level signals such as IP addresses-all essential for robust detection. To address these challenges, we propose CURL-IP, an advanced multi-modal detection framework incorporating three key innovations: (1) Token-Contrastive Representation Enhancer, which enhances subword token representations through token-aware contrastive learning to produce more discriminative and isotropic embeddings; (2) Cross-Layer Multi-Scale Aggregator, employing hierarchical aggregation of Transformer outputs via convolutional operations and gated MLPs to capture both local and global semantic patterns across layers; and (3) Blockwise Multi-Modal Coupler that decomposes URL-IP features into localized block units and computes cross-modal attention weights at the block level, enabling fine-grained inter-modal interaction. This architecture enables simultaneous preservation of fine-grained lexical cues, contextual semantics, and integration of network-level signals. Our evaluation on large-scale real-world datasets shows the framework significantly outperforms state-of-the-art baselines across binary and multi-class classification tasks.



## **25. DeepTrust: Multi-Step Classification through Dissimilar Adversarial Representations for Robust Android Malware Detection**

cs.CR

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12310v1) [paper-pdf](http://arxiv.org/pdf/2510.12310v1)

**Authors**: Daniel Pulido-Cortázar, Daniel Gibert, Felip Manyà

**Abstract**: Over the last decade, machine learning has been extensively applied to identify malicious Android applications. However, such approaches remain vulnerable against adversarial examples, i.e., examples that are subtly manipulated to fool a machine learning model into making incorrect predictions. This research presents DeepTrust, a novel metaheuristic that arranges flexible classifiers, like deep neural networks, into an ordered sequence where the final decision is made by a single internal model based on conditions activated in cascade. In the Robust Android Malware Detection competition at the 2025 IEEE Conference SaTML, DeepTrust secured the first place and achieved state-of-the-art results, outperforming the next-best competitor by up to 266% under feature-space evasion attacks. This is accomplished while maintaining the highest detection rate on non-adversarial malware and a false positive rate below 1%. The method's efficacy stems from maximizing the divergence of the learned representations among the internal models. By using classifiers inducing fundamentally dissimilar embeddings of the data, the decision space becomes unpredictable for an attacker. This frustrates the iterative perturbation process inherent to evasion attacks, enhancing system robustness without compromising accuracy on clean examples.



## **26. Train Stochastic Non Linear Coupled ODEs to Classify and Generate**

cond-mat.dis-nn

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12286v1) [paper-pdf](http://arxiv.org/pdf/2510.12286v1)

**Authors**: Stefano Gagliani, Feliciano Giuseppe Pacifico, Lorenzo Chicchi, Duccio Fanelli, Diego Febbe, Lorenzo Buffoni, Raffaele Marino

**Abstract**: A general class of dynamical systems which can be trained to operate in classification and generation modes are introduced. A procedure is proposed to plant asymptotic stationary attractors of the deterministic model. Optimizing the dynamical system amounts to shaping the architecture of inter-nodes connection to steer the evolution towards the assigned equilibrium, as a function of the class to which the item - supplied as an initial condition - belongs to. Under the stochastic perspective, point attractors are turned into probability distributions, made analytically accessible via the linear noise approximation. The addition of noise proves beneficial to oppose adversarial attacks, a property that gets engraved into the trained adjacency matrix and therefore also inherited by the deterministic counterpart of the optimized stochastic model. By providing samples from the target distribution as an input to a feedforward neural network (or even to a dynamical model of the same typology of the adopted for classification purposes), yields a fully generative scheme. Conditional generation is also possible by merging classification and generation modalities. Automatic disentanglement of isolated key features is finally proven.



## **27. L2M-AID: Autonomous Cyber-Physical Defense by Fusing Semantic Reasoning of Large Language Models with Multi-Agent Reinforcement Learning (Preprint)**

cs.AI

This preprint was submitted to IEEE TrustCom 2025. The accepted  version will be published under copyright 2025 IEEE

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.07363v2) [paper-pdf](http://arxiv.org/pdf/2510.07363v2)

**Authors**: Tianxiang Xu, Zhichao Wen, Xinyu Zhao, Jun Wang, Yan Li, Chang Liu

**Abstract**: The increasing integration of Industrial IoT (IIoT) exposes critical cyber-physical systems to sophisticated, multi-stage attacks that elude traditional defenses lacking contextual awareness. This paper introduces L2M-AID, a novel framework for Autonomous Industrial Defense using LLM-empowered, Multi-agent reinforcement learning. L2M-AID orchestrates a team of collaborative agents, each driven by a Large Language Model (LLM), to achieve adaptive and resilient security. The core innovation lies in the deep fusion of two AI paradigms: we leverage an LLM as a semantic bridge to translate vast, unstructured telemetry into a rich, contextual state representation, enabling agents to reason about adversary intent rather than merely matching patterns. This semantically-aware state empowers a Multi-Agent Reinforcement Learning (MARL) algorithm, MAPPO, to learn complex cooperative strategies. The MARL reward function is uniquely engineered to balance security objectives (threat neutralization) with operational imperatives, explicitly penalizing actions that disrupt physical process stability. To validate our approach, we conduct extensive experiments on the benchmark SWaT dataset and a novel synthetic dataset generated based on the MITRE ATT&CK for ICS framework. Results demonstrate that L2M-AID significantly outperforms traditional IDS, deep learning anomaly detectors, and single-agent RL baselines across key metrics, achieving a 97.2% detection rate while reducing false positives by over 80% and improving response times by a factor of four. Crucially, it demonstrates superior performance in maintaining physical process stability, presenting a robust new paradigm for securing critical national infrastructure.



## **28. Unveiling the Vulnerability of Graph-LLMs: An Interpretable Multi-Dimensional Adversarial Attack on TAGs**

cs.LG

12 pages, 4 figures

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2510.12233v1) [paper-pdf](http://arxiv.org/pdf/2510.12233v1)

**Authors**: Bowen Fan, Zhilin Guo, Xunkai Li, Yihan Zhou, Bing Zhou, Zhenjun Li, Rong-Hua Li, Guoren Wang

**Abstract**: Graph Neural Networks (GNNs) have become a pivotal framework for modeling graph-structured data, enabling a wide range of applications from social network analysis to molecular chemistry. By integrating large language models (LLMs), text-attributed graphs (TAGs) enhance node representations with rich textual semantics, significantly boosting the expressive power of graph-based learning. However, this sophisticated synergy introduces critical vulnerabilities, as Graph-LLMs are susceptible to adversarial attacks on both their structural topology and textual attributes. Although specialized attack methods have been designed for each of these aspects, no work has yet unified them into a comprehensive approach. In this work, we propose the Interpretable Multi-Dimensional Graph Attack (IMDGA), a novel human-centric adversarial attack framework designed to orchestrate multi-level perturbations across both graph structure and textual features. IMDGA utilizes three tightly integrated modules to craft attacks that balance interpretability and impact, enabling a deeper understanding of Graph-LLM vulnerabilities. Through rigorous theoretical analysis and comprehensive empirical evaluations on diverse datasets and architectures, IMDGA demonstrates superior interpretability, attack effectiveness, stealthiness, and robustness compared to existing methods. By exposing critical weaknesses in TAG representation learning, this work uncovers a previously underexplored semantic dimension of vulnerability in Graph-LLMs, offering valuable insights for improving their resilience. Our code and resources are publicly available at https://anonymous.4open.science/r/IMDGA-7289.



## **29. How Vulnerable Is My Learned Policy? Universal Adversarial Perturbation Attacks On Modern Behavior Cloning Policies**

cs.LG

**SubmitDate**: 2025-10-14    [abs](http://arxiv.org/abs/2502.03698v3) [paper-pdf](http://arxiv.org/pdf/2502.03698v3)

**Authors**: Akansha Kalra, Basavasagar Patil, Guanhong Tao, Daniel S. Brown

**Abstract**: Learning from Demonstration (LfD) algorithms have shown promising results in robotic manipulation tasks, but their vulnerability to offline universal perturbation attacks remains underexplored. This paper presents a comprehensive study of adversarial attacks on both classic and recently proposed algorithms, including Behavior Cloning (BC), LSTM-GMM, Implicit Behavior Cloning (IBC), Diffusion Policy (DP), and Vector-Quantizied Behavior Transformer (VQ-BET). We study the vulnerability of these methods to universal adversarial perturbations. Our experiments on several simulated robotic manipulation tasks reveal that most of the current methods are highly vulnerable to adversarial perturbations. We also show that these attacks are often transferable across algorithms, architectures, and tasks, raising concerning security vulnerabilities to black-box attacks. To the best of our knowledge, we are the first to present a systematic study of the vulnerabilities of different LfD algorithms to both white-box and black-box attacks. Our findings highlight the vulnerabilities of modern BC algorithms, paving the way for future work in addressing such limitations.



## **30. Robust ML-based Detection of Conventional, LLM-Generated, and Adversarial Phishing Emails Using Advanced Text Preprocessing**

cs.CR

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.11915v1) [paper-pdf](http://arxiv.org/pdf/2510.11915v1)

**Authors**: Deeksha Hareesha Kulal, Chidozie Princewill Arannonu, Afsah Anwar, Nidhi Rastogi, Quamar Niyaz

**Abstract**: Phishing remains a critical cybersecurity threat, especially with the advent of large language models (LLMs) capable of generating highly convincing malicious content. Unlike earlier phishing attempts which are identifiable by grammatical errors, misspellings, incorrect phrasing, and inconsistent formatting, LLM generated emails are grammatically sound, contextually relevant, and linguistically natural. These advancements make phishing emails increasingly difficult to distinguish from legitimate ones, challenging traditional detection mechanisms. Conventional phishing detection systems often fail when faced with emails crafted by LLMs or manipulated using adversarial perturbation techniques. To address this challenge, we propose a robust phishing email detection system featuring an enhanced text preprocessing pipeline. This pipeline includes spelling correction and word splitting to counteract adversarial modifications and improve detection accuracy. Our approach integrates widely adopted natural language processing (NLP) feature extraction techniques and machine learning algorithms. We evaluate our models on publicly available datasets comprising both phishing and legitimate emails, achieving a detection accuracy of 94.26% and F1-score of 84.39% in model deployment setting. To assess robustness, we further evaluate our models using adversarial phishing samples generated by four attack methods in Python TextAttack framework. Additionally, we evaluate models' performance against phishing emails generated by LLMs including ChatGPT and Llama. Results highlight the resilience of models against evolving AI-powered phishing threats.



## **31. A Comprehensive Survey of Website Fingerprinting Attacks and Defenses in Tor: Advances and Open Challenges**

cs.CR

43 pages

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.11804v1) [paper-pdf](http://arxiv.org/pdf/2510.11804v1)

**Authors**: Yuwen Cui, Guangjing Wang, Khanh Vu, Kai Wei, Kehan Shen, Zhengyuan Jiang, Xiao Han, Ning Wang, Zhuo Lu, Yao Liu

**Abstract**: The Tor network provides users with strong anonymity by routing their internet traffic through multiple relays. While Tor encrypts traffic and hides IP addresses, it remains vulnerable to traffic analysis attacks such as the website fingerprinting (WF) attack, achieving increasingly high fingerprinting accuracy even under open-world conditions. In response, researchers have proposed a variety of defenses, ranging from adaptive padding, traffic regularization, and traffic morphing to adversarial perturbation, that seek to obfuscate or reshape traffic traces. However, these defenses often entail trade-offs between privacy, usability, and system performance. Despite extensive research, a comprehensive survey unifying WF datasets, attack methodologies, and defense strategies remains absent. This paper fills that gap by systematically categorizing existing WF research into three key domains: datasets, attack models, and defense mechanisms. We provide an in-depth comparative analysis of techniques, highlight their strengths and limitations under diverse threat models, and discuss emerging challenges such as multi-tab browsing and coarse-grained traffic features. By consolidating prior work and identifying open research directions, this survey serves as a foundation for advancing stronger privacy protection in Tor.



## **32. Adversarial Attacks Leverage Interference Between Features in Superposition**

cs.LG

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.11709v1) [paper-pdf](http://arxiv.org/pdf/2510.11709v1)

**Authors**: Edward Stevinson, Lucas Prieto, Melih Barsbey, Tolga Birdal

**Abstract**: Fundamental questions remain about when and why adversarial examples arise in neural networks, with competing views characterising them either as artifacts of the irregularities in the decision landscape or as products of sensitivity to non-robust input features. In this paper, we instead argue that adversarial vulnerability can stem from efficient information encoding in neural networks. Specifically, we show how superposition - where networks represent more features than they have dimensions - creates arrangements of latent representations that adversaries can exploit. We demonstrate that adversarial perturbations leverage interference between superposed features, making attack patterns predictable from feature arrangements. Our framework provides a mechanistic explanation for two known phenomena: adversarial attack transferability between models with similar training regimes and class-specific vulnerability patterns. In synthetic settings with precisely controlled superposition, we establish that superposition suffices to create adversarial vulnerability. We then demonstrate that these findings persist in a ViT trained on CIFAR-10. These findings reveal adversarial vulnerability can be a byproduct of networks' representational compression, rather than flaws in the learning process or non-robust inputs.



## **33. LLMAtKGE: Large Language Models as Explainable Attackers against Knowledge Graph Embeddings**

cs.CL

13 pages

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.11584v1) [paper-pdf](http://arxiv.org/pdf/2510.11584v1)

**Authors**: Ting Li, Yang Yang, Yipeng Yu, Liang Yao, Guoqing Chao, Ruifeng Xu

**Abstract**: Adversarial attacks on knowledge graph embeddings (KGE) aim to disrupt the model's ability of link prediction by removing or inserting triples. A recent black-box method has attempted to incorporate textual and structural information to enhance attack performance. However, it is unable to generate human-readable explanations, and exhibits poor generalizability. In the past few years, large language models (LLMs) have demonstrated powerful capabilities in text comprehension, generation, and reasoning. In this paper, we propose LLMAtKGE, a novel LLM-based framework that selects attack targets and generates human-readable explanations. To provide the LLM with sufficient factual context under limited input constraints, we design a structured prompting scheme that explicitly formulates the attack as multiple-choice questions while incorporating KG factual evidence. To address the context-window limitation and hesitation issues, we introduce semantics-based and centrality-based filters, which compress the candidate set while preserving high recall of attack-relevant information. Furthermore, to efficiently integrate both semantic and structural information into the filter, we precompute high-order adjacency and fine-tune the LLM with a triple classification task to enhance filtering performance. Experiments on two widely used knowledge graph datasets demonstrate that our attack outperforms the strongest black-box baselines and provides explanations via reasoning, and showing competitive performance compared with white-box methods. Comprehensive ablation and case studies further validate its capability to generate explanations.



## **34. TBRD: TESLA Authenticated UAS Broadcast Remote ID**

cs.CR

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.11343v1) [paper-pdf](http://arxiv.org/pdf/2510.11343v1)

**Authors**: Jason Veara, Manav Jain, Kyle Moy, Aanjhan Ranganathan

**Abstract**: Mysterious sightings of Unmanned Aircraft Systems (UAS) over U.S. military facilities, suburban neighborhoods, and commercial airports have intensified scrutiny of drone activity. To increase accountability, the Federal Aviation Administration (FAA) introduced a Remote ID mandate, requiring unmanned aircraft to broadcast their location, operator's location, and identity in real-time. However, current standards leave authentication mechanisms underspecified, enabling spoofing, relay, and replay attacks that can undermine surveillance efforts and potentially disrupt UAS-to-UAS coordination in future deployments. In this paper, we propose TBRD, a practical system for authenticating Remote ID messages in a manner that aligns with existing standards and UAS capabilities. TBRD leverages the TESLA protocol and mobile device TEEs, and introduces a verification mechanism to build a lightweight, mission-scoped authentication system that is both computationally efficient and requires a low communication footprint. We evaluate the performance of TBRD using both an FAA-requirements compatible proof-of-concept implementation for performance metrics and a simulated 4-drone swarm mission scenario to demonstrate its security guarantees under adversarial conditions. Our system provides a 50\% reduction in authentication overhead compared to digital signatures and a 100x reduction in computation time. Our results demonstrate that TBRD can be integrated into current Remote ID infrastructures to provide a scalable, standards-compliant message authentication for both regulatory and operational use cases.



## **35. Attacks by Content: Automated Fact-checking is an AI Security Issue**

cs.CL

Accepted to EMNLP 2025

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.11238v1) [paper-pdf](http://arxiv.org/pdf/2510.11238v1)

**Authors**: Michael Schlichtkrull

**Abstract**: When AI agents retrieve and reason over external documents, adversaries can manipulate the data they receive to subvert their behaviour. Previous research has studied indirect prompt injection, where the attacker injects malicious instructions. We argue that injection of instructions is not necessary to manipulate agents - attackers could instead supply biased, misleading, or false information. We term this an attack by content. Existing defenses, which focus on detecting hidden commands, are ineffective against attacks by content. To defend themselves and their users, agents must critically evaluate retrieved information, corroborating claims with external evidence and evaluating source trustworthiness. We argue that this is analogous to an existing NLP task, automated fact-checking, which we propose to repurpose as a cognitive self-defense tool for agents.



## **36. Navigating the Dual-Use Nature and Security Implications of Reconfigurable Intelligent Surfaces in Next-Generation Wireless Systems**

eess.SP

This manuscript has been accepted for publication in IEEE  Communications Surveys and Tutorials. It was received on January 17, 2025,  and revised on July 1 and September 16, 2025. This version was accepted on  October 10, 2025

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.11113v1) [paper-pdf](http://arxiv.org/pdf/2510.11113v1)

**Authors**: Hetong Wang, Tiejun Lv, Yashuai Cao, Weicai Li, Jie Zeng, Pingmu Huang, Muhammad Khurram Khan

**Abstract**: Reconfigurable intelligent surface (RIS) technology offers significant promise in enhancing wireless communication systems, but its dual-use potential also introduces substantial security risks. This survey explores the security implications of RIS in next-generation wireless networks. We first highlight the dual-use nature of RIS, demonstrating how its communication-enhancing capabilities can be exploited by adversaries to compromise legitimate users. We identify a new class of security vulnerabilities termed ``passive-active hybrid attacks,'' where RIS, despite passively handling signals, can be reconfigured to actively engage in malicious activities, enabling various RIS-assisted attacks, such as eavesdropping, man-in-the-middle (MITM), replay, reflection jamming, and side-channel attacks. Furthermore, we reveal how adversaries can exploit the openness of wireless channels to introduce adversarial perturbations in artificial intelligence-driven RIS networks, disrupting communication terminals and causing misclassifications or errors in RIS reflection predictions. Despite these risks, RIS technology also plays a critical role in enhancing security and privacy across radio frequency (RF) and visible light communication (VLC) systems. By synthesizing current insights and highlighting emerging threats, we provide actionable insights into cross-layer collaboration, advanced adversarial defenses, and the balance between security and cost. This survey provides a comprehensive overview of RIS technology's security landscape and underscores the urgent need for robust security frameworks in the development of future wireless systems.



## **37. CoDefend: Cross-Modal Collaborative Defense via Diffusion Purification and Prompt Optimization**

cs.CV

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.11096v1) [paper-pdf](http://arxiv.org/pdf/2510.11096v1)

**Authors**: Fengling Zhu, Boshi Liu, Jingyu Hua, Sheng Zhong

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved remarkable success in tasks such as image captioning, visual question answering, and cross-modal reasoning by integrating visual and textual modalities. However, their multimodal nature also exposes them to adversarial threats, where attackers can perturb either modality or both jointly to induce harmful, misleading, or policy violating outputs. Existing defense strategies, such as adversarial training and input purification, face notable limitations: adversarial training typically improves robustness only against known attacks while incurring high computational costs, whereas conventional purification approaches often suffer from degraded image quality and insufficient generalization to complex multimodal tasks.   In this work, we focus on defending the visual modality, which frequently serves as the primary entry point for adversarial manipulation. We propose a supervised diffusion based denoising framework that leverages paired adversarial clean image datasets to fine-tune diffusion models with directional, task specific guidance. Unlike prior unsupervised purification methods such as DiffPure, our approach achieves higher quality reconstructions while significantly improving defense robustness in multimodal tasks. Furthermore, we incorporate prompt optimization as a complementary defense mechanism, enhancing resistance against diverse and unseen attack strategies.   Extensive experiments on image captioning and visual question answering demonstrate that our method not only substantially improves robustness but also exhibits strong transferability to unknown adversarial attacks. These results highlight the effectiveness of supervised diffusion based denoising for multimodal defense, paving the way for more reliable and secure deployment of MLLMs in real world applications.



## **38. TabAttackBench: A Benchmark for Adversarial Attacks on Tabular Data**

cs.LG

71 pages, 21 figures, 11 tables

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2505.21027v2) [paper-pdf](http://arxiv.org/pdf/2505.21027v2)

**Authors**: Zhipeng He, Chun Ouyang, Lijie Wen, Cong Liu, Catarina Moreira

**Abstract**: Adversarial attacks pose a significant threat to machine learning models by inducing incorrect predictions through imperceptible perturbations to input data. While these attacks are well studied in unstructured domains such as images, their behaviour on tabular data remains underexplored due to mixed feature types and complex inter-feature dependencies. This study introduces a comprehensive benchmark that evaluates adversarial attacks on tabular datasets with respect to both effectiveness and imperceptibility. We assess five white-box attack algorithms (FGSM, BIM, PGD, DeepFool, and C\&W) across four representative models (LR, MLP, TabTransformer and FT-Transformer) using eleven datasets spanning finance, energy, and healthcare domains. The benchmark employs four quantitative imperceptibility metrics (proximity, sparsity, deviation, and sensitivity) to characterise perturbation realism. The analysis quantifies the trade-off between these two aspects and reveals consistent differences between attack types, with $\ell_\infty$-based attacks achieving higher success but lower subtlety, and $\ell_2$-based attacks offering more realistic perturbations. The benchmark findings offer actionable insights for designing more imperceptible adversarial attacks, advancing the understanding of adversarial vulnerability in tabular machine learning.



## **39. Adversarial Robustness in One-Stage Learning-to-Defer**

stat.ML

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.10988v1) [paper-pdf](http://arxiv.org/pdf/2510.10988v1)

**Authors**: Yannis Montreuil, Letian Yu, Axel Carlier, Lai Xing Ng, Wei Tsang Ooi

**Abstract**: Learning-to-Defer (L2D) enables hybrid decision-making by routing inputs either to a predictor or to external experts. While promising, L2D is highly vulnerable to adversarial perturbations, which can not only flip predictions but also manipulate deferral decisions. Prior robustness analyses focus solely on two-stage settings, leaving open the end-to-end (one-stage) case where predictor and allocation are trained jointly. We introduce the first framework for adversarial robustness in one-stage L2D, covering both classification and regression. Our approach formalizes attacks, proposes cost-sensitive adversarial surrogate losses, and establishes theoretical guarantees including $\mathcal{H}$, $(\mathcal{R }, \mathcal{F})$, and Bayes consistency. Experiments on benchmark datasets confirm that our methods improve robustness against untargeted and targeted attacks while preserving clean performance.



## **40. Neutral Agent-based Adversarial Policy Learning against Deep Reinforcement Learning in Multi-party Open Systems**

cs.LG

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.10937v1) [paper-pdf](http://arxiv.org/pdf/2510.10937v1)

**Authors**: Qizhou Peng, Yang Zheng, Yu Wen, Yanna Wu, Yingying Du

**Abstract**: Reinforcement learning (RL) has been an important machine learning paradigm for solving long-horizon sequential decision-making problems under uncertainty. By integrating deep neural networks (DNNs) into the RL framework, deep reinforcement learning (DRL) has emerged, which achieved significant success in various domains. However, the integration of DNNs also makes it vulnerable to adversarial attacks. Existing adversarial attack techniques mainly focus on either directly manipulating the environment with which a victim agent interacts or deploying an adversarial agent that interacts with the victim agent to induce abnormal behaviors. While these techniques achieve promising results, their adoption in multi-party open systems remains limited due to two major reasons: impractical assumption of full control over the environment and dependent on interactions with victim agents.   To enable adversarial attacks in multi-party open systems, in this paper, we redesigned an adversarial policy learning approach that can mislead well-trained victim agents without requiring direct interactions with these agents or full control over their environments. Particularly, we propose a neutral agent-based approach across various task scenarios in multi-party open systems. While the neutral agents seemingly are detached from the victim agents, indirectly influence them through the shared environment. We evaluate our proposed method on the SMAC platform based on Starcraft II and the autonomous driving simulation platform Highway-env. The experimental results demonstrate that our method can launch general and effective adversarial attacks in multi-party open systems.



## **41. TabVLA: Targeted Backdoor Attacks on Vision-Language-Action Models**

cs.CR

8 pages, 8 tables, 1 figure. Under review

**SubmitDate**: 2025-10-13    [abs](http://arxiv.org/abs/2510.10932v1) [paper-pdf](http://arxiv.org/pdf/2510.10932v1)

**Authors**: Zonghuan Xu, Xiang Zheng, Xingjun Ma, Yu-Gang Jiang

**Abstract**: With the growing deployment of Vision-Language-Action (VLA) models in real-world embodied AI systems, their increasing vulnerability to backdoor attacks poses a serious safety threat. A backdoored VLA agent can be covertly triggered by a pre-injected backdoor to execute adversarial actions, potentially causing system failures or even physical harm. Although backdoor attacks on VLA models have been explored, prior work has focused only on untargeted attacks, leaving the more practically threatening scenario of targeted manipulation unexamined. In this paper, we study targeted backdoor attacks on VLA models and introduce TabVLA, a novel framework that enables such attacks via black-box fine-tuning. TabVLA explores two deployment-relevant inference-time threat models: input-stream editing and in-scene triggering. It formulates poisoned data generation as an optimization problem to improve attack effectivess. Experiments with OpenVLA-7B on the LIBERO benchmark reveal that the vision channel is the principal attack surface: targeted backdoors succeed with minimal poisoning, remain robust across variations in trigger design, and are degraded only by positional mismatches between fine-tuning and inference triggers. We also investigate a potential detection-based defense against TabVLA, which reconstructs latent visual triggers from the input stream to flag activation-conditioned backdoor samples. Our work highlights the vulnerability of VLA models to targeted backdoor manipulation and underscores the need for more advanced defenses.



## **42. Adversarial Defence without Adversarial Defence: Enhancing Language Model Robustness via Instance-level Principal Component Removal**

cs.CL

This paper was accepted with an A-decision to Transactions of the  Association for Computational Linguistics. This version is the  pre-publication version prior to MIT Press production

**SubmitDate**: 2025-10-12    [abs](http://arxiv.org/abs/2507.21750v3) [paper-pdf](http://arxiv.org/pdf/2507.21750v3)

**Authors**: Yang Wang, Chenghao Xiao, Yizhi Li, Stuart E. Middleton, Noura Al Moubayed, Chenghua Lin

**Abstract**: Pre-trained language models (PLMs) have driven substantial progress in natural language processing but remain vulnerable to adversarial attacks, raising concerns about their robustness in real-world applications. Previous studies have sought to mitigate the impact of adversarial attacks by introducing adversarial perturbations into the training process, either implicitly or explicitly. While both strategies enhance robustness, they often incur high computational costs. In this work, we propose a simple yet effective add-on module that enhances the adversarial robustness of PLMs by removing instance-level principal components, without relying on conventional adversarial defences or perturbing the original training data. Our approach transforms the embedding space to approximate Gaussian properties, thereby reducing its susceptibility to adversarial perturbations while preserving semantic relationships. This transformation aligns embedding distributions in a way that minimises the impact of adversarial noise on decision boundaries, enhancing robustness without requiring adversarial examples or costly training-time augmentation. Evaluations on eight benchmark datasets show that our approach improves adversarial robustness while maintaining comparable before-attack accuracy to baselines, achieving a balanced trade-off between robustness and generalisation.



## **43. On Surjectivity of Neural Networks: Can you elicit any behavior from your model?**

cs.LG

**SubmitDate**: 2025-10-12    [abs](http://arxiv.org/abs/2508.19445v2) [paper-pdf](http://arxiv.org/pdf/2508.19445v2)

**Authors**: Haozhe Jiang, Nika Haghtalab

**Abstract**: Given a trained neural network, can any specified output be generated by some input? Equivalently, does the network correspond to a function that is surjective? In generative models, surjectivity implies that any output, including harmful or undesirable content, can in principle be generated by the networks, raising concerns about model safety and jailbreak vulnerabilities. In this paper, we prove that many fundamental building blocks of modern neural architectures, such as networks with pre-layer normalization and linear-attention modules, are almost always surjective. As corollaries, widely used generative frameworks, including GPT-style transformers and diffusion models with deterministic ODE solvers, admit inverse mappings for arbitrary outputs. By studying surjectivity of these modern and commonly used neural architectures, we contribute a formalism that sheds light on their unavoidable vulnerability to a broad class of adversarial attacks.



## **44. Concept Steerers: Leveraging K-Sparse Autoencoders for Test-Time Controllable Generations**

cs.CV

23 pages, 18 figures

**SubmitDate**: 2025-10-12    [abs](http://arxiv.org/abs/2501.19066v3) [paper-pdf](http://arxiv.org/pdf/2501.19066v3)

**Authors**: Dahye Kim, Deepti Ghadiyaram

**Abstract**: Despite the remarkable progress in text-to-image generative models, they are prone to adversarial attacks and inadvertently generate unsafe, unethical content. Existing approaches often rely on fine-tuning models to remove specific concepts, which is computationally expensive, lacks scalability, and/or compromises generation quality. In this work, we propose a novel framework leveraging k-sparse autoencoders (k-SAEs) to enable efficient and interpretable concept manipulation in diffusion models. Specifically, we first identify interpretable monosemantic concepts in the latent space of text embeddings and leverage them to precisely steer the generation away or towards a given concept (e.g., nudity) or to introduce a new concept (e.g., photographic style) -- all during test time. Through extensive experiments, we demonstrate that our approach is very simple, requires no retraining of the base model nor LoRA adapters, does not compromise the generation quality, and is robust to adversarial prompt manipulations. Our method yields an improvement of $\mathbf{20.01\%}$ in unsafe concept removal, is effective in style manipulation, and is $\mathbf{\sim5}$x faster than the current state-of-the-art. Code is available at: https://github.com/kim-dahye/steerers



## **45. SASER: Stego attacks on open-source LLMs**

cs.CR

**SubmitDate**: 2025-10-12    [abs](http://arxiv.org/abs/2510.10486v1) [paper-pdf](http://arxiv.org/pdf/2510.10486v1)

**Authors**: Ming Tan, Wei Li, Hu Tao, Hailong Ma, Aodi Liu, Qian Chen, Zilong Wang

**Abstract**: Open-source large language models (LLMs) have demonstrated considerable dominance over proprietary LLMs in resolving neural processing tasks, thanks to the collaborative and sharing nature. Although full access to source codes, model parameters, and training data lays the groundwork for transparency, we argue that such a full-access manner is vulnerable to stego attacks, and their ill-effects are not fully understood. In this paper, we conduct a systematic formalization for stego attacks on open-source LLMs by enumerating all possible threat models associated with adversary objectives, knowledge, and capabilities. Therein, the threat posed by adversaries with internal knowledge, who inject payloads and triggers during the model sharing phase, is of practical interest. We go even further and propose the first stego attack on open-source LLMs, dubbed SASER, which wields impacts through identifying targeted parameters, embedding payloads, injecting triggers, and executing payloads sequentially. Particularly, SASER enhances the attack robustness against quantization-based local deployment by de-quantizing the embedded payloads. In addition, to achieve stealthiness, SASER devises the performance-aware importance metric to identify targeted parameters with the least degradation of model performance. Extensive experiments on LlaMA2-7B and ChatGLM3-6B, without quantization, show that the stealth rate of SASER outperforms existing stego attacks (for general DNNs) by up to 98.1%, while achieving the same attack success rate (ASR) of 100%. More importantly, SASER improves ASR on quantized models from 0 to 100% in all settings. We appeal for investigations on countermeasures against SASER in view of the significant attack effectiveness.



## **46. Boosting Adversarial Transferability via Commonality-Oriented Gradient Optimization**

cs.CV

23 pages

**SubmitDate**: 2025-10-12    [abs](http://arxiv.org/abs/2506.06992v2) [paper-pdf](http://arxiv.org/pdf/2506.06992v2)

**Authors**: Yanting Gao, Yepeng Liu, Junming Liu, Qi Zhang, Hongyun Zhang, Duoqian Miao, Cairong Zhao

**Abstract**: Exploring effective and transferable adversarial examples is vital for understanding the characteristics and mechanisms of Vision Transformers (ViTs). However, adversarial examples generated from surrogate models often exhibit weak transferability in black-box settings due to overfitting. Existing methods improve transferability by diversifying perturbation inputs or applying uniform gradient regularization within surrogate models, yet they have not fully leveraged the shared and unique features of surrogate models trained on the same task, leading to suboptimal transfer performance. Therefore, enhancing perturbations of common information shared by surrogate models and suppressing those tied to individual characteristics offers an effective way to improve transferability. Accordingly, we propose a commonality-oriented gradient optimization strategy (COGO) consisting of two components: Commonality Enhancement (CE) and Individuality Suppression (IS). CE perturbs the mid-to-low frequency regions, leveraging the fact that ViTs trained on the same dataset tend to rely more on mid-to-low frequency information for classification. IS employs adaptive thresholds to evaluate the correlation between backpropagated gradients and model individuality, assigning weights to gradients accordingly. Extensive experiments demonstrate that COGO significantly improves the transfer success rates of adversarial attacks, outperforming current state-of-the-art methods.



## **47. Adversarial Attacks on Downstream Weather Forecasting Models: Application to Tropical Cyclone Trajectory Prediction**

cs.LG

**SubmitDate**: 2025-10-11    [abs](http://arxiv.org/abs/2510.10140v1) [paper-pdf](http://arxiv.org/pdf/2510.10140v1)

**Authors**: Yue Deng, Francisco Santos, Pang-Ning Tan, Lifeng Luo

**Abstract**: Deep learning based weather forecasting (DLWF) models leverage past weather observations to generate future forecasts, supporting a wide range of downstream tasks, including tropical cyclone (TC) trajectory prediction. In this paper, we investigate their vulnerability to adversarial attacks, where subtle perturbations to the upstream weather forecasts can alter the downstream TC trajectory predictions. Although research on adversarial attacks in DLWF models has grown recently, generating perturbed upstream forecasts that reliably steer downstream output toward attacker-specified trajectories remains a challenge. First, conventional TC detection systems are opaque, non-differentiable black boxes, making standard gradient-based attacks infeasible. Second, the extreme rarity of TC events leads to severe class imbalance problem, making it difficult to develop efficient attack methods that will produce the attacker's target trajectories. Furthermore, maintaining physical consistency in adversarially generated forecasts presents another significant challenge. To overcome these limitations, we propose Cyc-Attack, a novel method that perturbs the upstream forecasts of DLWF models to generate adversarial trajectories. First, we pre-train a differentiable surrogate model to approximate the TC detector's output, enabling the construction of gradient-based attacks. Cyc-Attack also employs skewness-aware loss function with kernel dilation strategy to address the imbalance problem. Finally, a distance-based gradient weighting scheme and regularization are used to constrain the perturbations and eliminate spurious trajectories to ensure the adversarial forecasts are realistic and not easily detectable.



## **48. A-IPO: Adaptive Intent-driven Preference Optimization**

cs.CL

**SubmitDate**: 2025-10-11    [abs](http://arxiv.org/abs/2510.10077v1) [paper-pdf](http://arxiv.org/pdf/2510.10077v1)

**Authors**: Wenqing Wang, Muhammad Asif Ali, Ali Shoker, Ruohan Yang, Junyang Chen, Ying Sha, Huan Wang

**Abstract**: Human preferences are diverse and dynamic, shaped by regional, cultural, and social factors. Existing alignment methods like Direct Preference Optimization (DPO) and its variants often default to majority views, overlooking minority opinions and failing to capture latent user intentions in prompts.   To address these limitations, we introduce \underline{\textbf{A}}daptive \textbf{\underline{I}}ntent-driven \textbf{\underline{P}}reference \textbf{\underline{O}}ptimization (\textbf{A-IPO}). Specifically,A-IPO introduces an intention module that infers the latent intent behind each user prompt and explicitly incorporates this inferred intent into the reward function, encouraging stronger alignment between the preferred model's responses and the user's underlying intentions. We demonstrate, both theoretically and empirically, that incorporating an intention--response similarity term increases the preference margin (by a positive shift of $\lambda\,\Delta\mathrm{sim}$ in the log-odds), resulting in clearer separation between preferred and dispreferred responses compared to DPO.   For evaluation, we introduce two new benchmarks, Real-pref, Attack-pref along with an extended version of an existing dataset, GlobalOpinionQA-Ext, to assess real-world and adversarial preference alignment.   Through explicit modeling of diverse user intents,A-IPO facilitates pluralistic preference optimization while simultaneously enhancing adversarial robustness in preference alignment. Comprehensive empirical evaluation demonstrates that A-IPO consistently surpasses existing baselines, yielding substantial improvements across key metrics: up to +24.8 win-rate and +45.6 Response-Intention Consistency on Real-pref; up to +38.6 Response Similarity and +52.2 Defense Success Rate on Attack-pref; and up to +54.6 Intention Consistency Score on GlobalOpinionQA-Ext.



## **49. SecureWebArena: A Holistic Security Evaluation Benchmark for LVLM-based Web Agents**

cs.CR

**SubmitDate**: 2025-10-11    [abs](http://arxiv.org/abs/2510.10073v1) [paper-pdf](http://arxiv.org/pdf/2510.10073v1)

**Authors**: Zonghao Ying, Yangguang Shao, Jianle Gan, Gan Xu, Junjie Shen, Wenxin Zhang, Quanchen Zou, Junzheng Shi, Zhenfei Yin, Mingchuan Zhang, Aishan Liu, Xianglong Liu

**Abstract**: Large vision-language model (LVLM)-based web agents are emerging as powerful tools for automating complex online tasks. However, when deployed in real-world environments, they face serious security risks, motivating the design of security evaluation benchmarks. Existing benchmarks provide only partial coverage, typically restricted to narrow scenarios such as user-level prompt manipulation, and thus fail to capture the broad range of agent vulnerabilities. To address this gap, we present \tool{}, the first holistic benchmark for evaluating the security of LVLM-based web agents. \tool{} first introduces a unified evaluation suite comprising six simulated but realistic web environments (\eg, e-commerce platforms, community forums) and includes 2,970 high-quality trajectories spanning diverse tasks and attack settings. The suite defines a structured taxonomy of six attack vectors spanning both user-level and environment-level manipulations. In addition, we introduce a multi-layered evaluation protocol that analyzes agent failures across three critical dimensions: internal reasoning, behavioral trajectory, and task outcome, facilitating a fine-grained risk analysis that goes far beyond simple success metrics. Using this benchmark, we conduct large-scale experiments on 9 representative LVLMs, which fall into three categories: general-purpose, agent-specialized, and GUI-grounded. Our results show that all tested agents are consistently vulnerable to subtle adversarial manipulations and reveal critical trade-offs between model specialization and security. By providing (1) a comprehensive benchmark suite with diverse environments and a multi-layered evaluation pipeline, and (2) empirical insights into the security challenges of modern LVLM-based web agents, \tool{} establishes a foundation for advancing trustworthy web agent deployment.



## **50. GenoArmory: A Unified Evaluation Framework for Adversarial Attacks on Genomic Foundation Models**

cs.LG

**SubmitDate**: 2025-10-11    [abs](http://arxiv.org/abs/2505.10983v2) [paper-pdf](http://arxiv.org/pdf/2505.10983v2)

**Authors**: Haozheng Luo, Chenghao Qiu, Yimin Wang, Shang Wu, Jiahao Yu, Zhenyu Pan, Weian Mao, Haoyang Fang, Hao Xu, Han Liu, Binghui Wang, Yan Chen

**Abstract**: We propose the first unified adversarial attack benchmark for Genomic Foundation Models (GFMs), named GenoArmory. Unlike existing GFM benchmarks, GenoArmory offers the first comprehensive evaluation framework to systematically assess the vulnerability of GFMs to adversarial attacks. Methodologically, we evaluate the adversarial robustness of five state-of-the-art GFMs using four widely adopted attack algorithms and three defense strategies. Importantly, our benchmark provides an accessible and comprehensive framework to analyze GFM vulnerabilities with respect to model architecture, quantization schemes, and training datasets. Additionally, we introduce GenoAdv, a new adversarial sample dataset designed to improve GFM safety. Empirically, classification models exhibit greater robustness to adversarial perturbations compared to generative models, highlighting the impact of task type on model vulnerability. Moreover, adversarial attacks frequently target biologically significant genomic regions, suggesting that these models effectively capture meaningful sequence features.



