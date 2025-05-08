# Latest Adversarial Attack Papers
**update at 2025-05-08 18:51:16**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Fight Fire with Fire: Defending Against Malicious RL Fine-Tuning via Reward Neutralization**

cs.LG

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04578v1) [paper-pdf](http://arxiv.org/pdf/2505.04578v1)

**Authors**: Wenjun Cao

**Abstract**: Reinforcement learning (RL) fine-tuning transforms large language models while creating a vulnerability we experimentally verify: Our experiment shows that malicious RL fine-tuning dismantles safety guardrails with remarkable efficiency, requiring only 50 steps and minimal adversarial prompts, with harmful escalating from 0-2 to 7-9. This attack vector particularly threatens open-source models with parameter-level access. Existing defenses targeting supervised fine-tuning prove ineffective against RL's dynamic feedback mechanisms. We introduce Reward Neutralization, the first defense framework specifically designed against RL fine-tuning attacks, establishing concise rejection patterns that render malicious reward signals ineffective. Our approach trains models to produce minimal-information rejections that attackers cannot exploit, systematically neutralizing attempts to optimize toward harmful outputs. Experiments validate that our approach maintains low harmful scores (no greater than 2) after 200 attack steps, while standard models rapidly deteriorate. This work provides the first constructive proof that robust defense against increasingly accessible RL attacks is achievable, addressing a critical security gap for open-weight models.



## **2. Mitigating Many-Shot Jailbreaking**

cs.LG

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2504.09604v2) [paper-pdf](http://arxiv.org/pdf/2504.09604v2)

**Authors**: Christopher M. Ackerman, Nina Panickssery

**Abstract**: Many-shot jailbreaking (MSJ) is an adversarial technique that exploits the long context windows of modern LLMs to circumvent model safety training by including in the prompt many examples of a "fake" assistant responding inappropriately before the final request. With enough examples, the model's in-context learning abilities override its safety training, and it responds as if it were the "fake" assistant. In this work, we probe the effectiveness of different fine-tuning and input sanitization approaches on mitigating MSJ attacks, alone and in combination. We find incremental mitigation effectiveness for each, and show that the combined techniques significantly reduce the effectiveness of MSJ attacks, while retaining model performance in benign in-context learning and conversational tasks. We suggest that our approach could meaningfully ameliorate this vulnerability if incorporated into model safety post-training.



## **3. Machine Learning Cryptanalysis of a Quantum Random Number Generator**

cs.LG

Published article is at https://ieeexplore.ieee.org/document/8396276.  Related code is at  https://github.com/Nano-Neuro-Research-Lab/Machine-Learning-Cryptanalysis-of-a-Quantum-Random-Number-Generator

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/1905.02342v3) [paper-pdf](http://arxiv.org/pdf/1905.02342v3)

**Authors**: Nhan Duy Truong, Jing Yan Haw, Syed Muhamad Assad, Ping Koy Lam, Omid Kavehei

**Abstract**: Random number generators (RNGs) that are crucial for cryptographic applications have been the subject of adversarial attacks. These attacks exploit environmental information to predict generated random numbers that are supposed to be truly random and unpredictable. Though quantum random number generators (QRNGs) are based on the intrinsic indeterministic nature of quantum properties, the presence of classical noise in the measurement process compromises the integrity of a QRNG. In this paper, we develop a predictive machine learning (ML) analysis to investigate the impact of deterministic classical noise in different stages of an optical continuous variable QRNG. Our ML model successfully detects inherent correlations when the deterministic noise sources are prominent. After appropriate filtering and randomness extraction processes are introduced, our QRNG system, in turn, demonstrates its robustness against ML. We further demonstrate the robustness of our ML approach by applying it to uniformly distributed random numbers from the QRNG and a congruential RNG. Hence, our result shows that ML has potentials in benchmarking the quality of RNG devices.



## **4. Reliable Disentanglement Multi-view Learning Against View Adversarial Attacks**

cs.LG

11 pages, 11 figures, accepted by International Joint Conference on  Artificial Intelligence (IJCAI 2025)

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04046v1) [paper-pdf](http://arxiv.org/pdf/2505.04046v1)

**Authors**: Xuyang Wang, Siyuan Duan, Qizhi Li, Guiduo Duan, Yuan Sun, Dezhong Peng

**Abstract**: Recently, trustworthy multi-view learning has attracted extensive attention because evidence learning can provide reliable uncertainty estimation to enhance the credibility of multi-view predictions. Existing trusted multi-view learning methods implicitly assume that multi-view data is secure. In practice, however, in safety-sensitive applications such as autonomous driving and security monitoring, multi-view data often faces threats from adversarial perturbations, thereby deceiving or disrupting multi-view learning models. This inevitably leads to the adversarial unreliability problem (AUP) in trusted multi-view learning. To overcome this tricky problem, we propose a novel multi-view learning framework, namely Reliable Disentanglement Multi-view Learning (RDML). Specifically, we first propose evidential disentanglement learning to decompose each view into clean and adversarial parts under the guidance of corresponding evidences, which is extracted by a pretrained evidence extractor. Then, we employ the feature recalibration module to mitigate the negative impact of adversarial perturbations and extract potential informative features from them. Finally, to further ignore the irreparable adversarial interferences, a view-level evidential attention mechanism is designed. Extensive experiments on multi-view classification tasks with adversarial attacks show that our RDML outperforms the state-of-the-art multi-view learning methods by a relatively large margin.



## **5. MergeGuard: Efficient Thwarting of Trojan Attacks in Machine Learning Models**

cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.04015v1) [paper-pdf](http://arxiv.org/pdf/2505.04015v1)

**Authors**: Soheil Zibakhsh Shabgahi, Yaman Jandali, Farinaz Koushanfar

**Abstract**: This paper proposes MergeGuard, a novel methodology for mitigation of AI Trojan attacks. Trojan attacks on AI models cause inputs embedded with triggers to be misclassified to an adversary's target class, posing a significant threat to model usability trained by an untrusted third party. The core of MergeGuard is a new post-training methodology for linearizing and merging fully connected layers which we show simultaneously improves model generalizability and performance. Our Proof of Concept evaluation on Transformer models demonstrates that MergeGuard maintains model accuracy while decreasing trojan attack success rate, outperforming commonly used (post-training) Trojan mitigation by fine-tuning methodologies.



## **6. Towards Universal and Black-Box Query-Response Only Attack on LLMs with QROA**

cs.CL

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2406.02044v3) [paper-pdf](http://arxiv.org/pdf/2406.02044v3)

**Authors**: Hussein Jawad, Yassine Chenik, Nicolas J. -B. Brunel

**Abstract**: The rapid adoption of Large Language Models (LLMs) has exposed critical security and ethical vulnerabilities, particularly their susceptibility to adversarial manipulations. This paper introduces QROA, a novel black-box jailbreak method designed to identify adversarial suffixes that can bypass LLM alignment safeguards when appended to a malicious instruction. Unlike existing suffix-based jailbreak approaches, QROA does not require access to the model's logit or any other internal information. It also eliminates reliance on human-crafted templates, operating solely through the standard query-response interface of LLMs. By framing the attack as an optimization bandit problem, QROA employs a surrogate model and token level optimization to efficiently explore suffix variations. Furthermore, we propose QROA-UNV, an extension that identifies universal adversarial suffixes for individual models, enabling one-query jailbreaks across a wide range of instructions. Testing on multiple models demonstrates Attack Success Rate (ASR) greater than 80\%. These findings highlight critical vulnerabilities, emphasize the need for advanced defenses, and contribute to the development of more robust safety evaluations for secure AI deployment. The code is made public on the following link: https://github.com/qroa/QROA



## **7. Model-Targeted Data Poisoning Attacks against ITS Applications with Provable Convergence**

math.OC

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03966v1) [paper-pdf](http://arxiv.org/pdf/2505.03966v1)

**Authors**: Xin Wanga, Feilong Wang, Yuan Hong, R. Tyrrell Rockafellar, Xuegang, Ban

**Abstract**: The growing reliance of intelligent systems on data makes the systems vulnerable to data poisoning attacks. Such attacks could compromise machine learning or deep learning models by disrupting the input data. Previous studies on data poisoning attacks are subject to specific assumptions, and limited attention is given to learning models with general (equality and inequality) constraints or lacking differentiability. Such learning models are common in practice, especially in Intelligent Transportation Systems (ITS) that involve physical or domain knowledge as specific model constraints. Motivated by ITS applications, this paper formulates a model-target data poisoning attack as a bi-level optimization problem with a constrained lower-level problem, aiming to induce the model solution toward a target solution specified by the adversary by modifying the training data incrementally. As the gradient-based methods fail to solve this optimization problem, we propose to study the Lipschitz continuity property of the model solution, enabling us to calculate the semi-derivative, a one-sided directional derivative, of the solution over data. We leverage semi-derivative descent to solve the bi-level optimization problem, and establish the convergence conditions of the method to any attainable target model. The model and solution method are illustrated with a simulation of a poisoning attack on the lane change detection using SVM.



## **8. Sustainable Smart Farm Networks: Enhancing Resilience and Efficiency with Decision Theory-Guided Deep Reinforcement Learning**

cs.LG

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03721v1) [paper-pdf](http://arxiv.org/pdf/2505.03721v1)

**Authors**: Dian Chen, Zelin Wan, Dong Sam Ha, Jin-Hee Cho

**Abstract**: Solar sensor-based monitoring systems have become a crucial agricultural innovation, advancing farm management and animal welfare through integrating sensor technology, Internet-of-Things, and edge and cloud computing. However, the resilience of these systems to cyber-attacks and their adaptability to dynamic and constrained energy supplies remain largely unexplored. To address these challenges, we propose a sustainable smart farm network designed to maintain high-quality animal monitoring under various cyber and adversarial threats, as well as fluctuating energy conditions. Our approach utilizes deep reinforcement learning (DRL) to devise optimal policies that maximize both monitoring effectiveness and energy efficiency. To overcome DRL's inherent challenge of slow convergence, we integrate transfer learning (TL) and decision theory (DT) to accelerate the learning process. By incorporating DT-guided strategies, we optimize monitoring quality and energy sustainability, significantly reducing training time while achieving comparable performance rewards. Our experimental results prove that DT-guided DRL outperforms TL-enhanced DRL models, improving system performance and reducing training runtime by 47.5%.



## **9. Adversarial Robustness of Deep Learning Models for Inland Water Body Segmentation from SAR Images**

eess.IV

21 pages, 15 figures, 2 tables

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.01884v2) [paper-pdf](http://arxiv.org/pdf/2505.01884v2)

**Authors**: Siddharth Kothari, Srinivasan Murali, Sankalp Kothari, Ujjwal Verma, Jaya Sreevalsan-Nair

**Abstract**: Inland water body segmentation from Synthetic Aperture Radar (SAR) images is an important task needed for several applications, such as flood mapping. While SAR sensors capture data in all-weather conditions as high-resolution images, differentiating water and water-like surfaces from SAR images is not straightforward. Inland water bodies, such as large river basins, have complex geometry, which adds to the challenge of segmentation. U-Net is a widely used deep learning model for land-water segmentation of SAR images. In practice, manual annotation is often used to generate the corresponding water masks as ground truth. Manual annotation of the images is prone to label noise owing to data poisoning attacks, especially due to complex geometry. In this work, we simulate manual errors in the form of adversarial attacks on the U-Net model and study the robustness of the model to human errors in annotation. Our results indicate that U-Net can tolerate a certain level of corruption before its performance drops significantly. This finding highlights the crucial role that the quality of manual annotations plays in determining the effectiveness of the segmentation model. The code and the new dataset, along with adversarial examples for robust training, are publicly available. (GitHub link - https://github.com/GVCL/IWSeg-SAR-Poison.git)



## **10. Data-Driven Falsification of Cyber-Physical Systems**

cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03863v1) [paper-pdf](http://arxiv.org/pdf/2505.03863v1)

**Authors**: Atanu Kundu, Sauvik Gon, Rajarshi Ray

**Abstract**: Cyber-Physical Systems (CPS) are abundant in safety-critical domains such as healthcare, avionics, and autonomous vehicles. Formal verification of their operational safety is, therefore, of utmost importance. In this paper, we address the falsification problem, where the focus is on searching for an unsafe execution in the system instead of proving their absence. The contribution of this paper is a framework that (a) connects the falsification of CPS with the falsification of deep neural networks (DNNs) and (b) leverages the inherent interpretability of Decision Trees for faster falsification of CPS. This is achieved by: (1) building a surrogate model of the CPS under test, either as a DNN model or a Decision Tree, (2) application of various DNN falsification tools to falsify CPS, and (3) a novel falsification algorithm guided by the explanations of safety violations of the CPS model extracted from its Decision Tree surrogate. The proposed framework has the potential to exploit a repertoire of \emph{adversarial attack} algorithms designed to falsify robustness properties of DNNs, as well as state-of-the-art falsification algorithms for DNNs. Although the presented methodology is applicable to systems that can be executed/simulated in general, we demonstrate its effectiveness, particularly in CPS. We show that our framework, implemented as a tool \textsc{FlexiFal}, can detect hard-to-find counterexamples in CPS that have linear and non-linear dynamics. Decision tree-guided falsification shows promising results in efficiently finding multiple counterexamples in the ARCH-COMP 2024 falsification benchmarks~\cite{khandait2024arch}.



## **11. ALMA: Aggregated Lipschitz Maximization Attack on Auto-encoders**

cs.LG

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03646v1) [paper-pdf](http://arxiv.org/pdf/2505.03646v1)

**Authors**: Chethan Krishnamurthy Ramanaik, Arjun Roy, Eirini Ntoutsi

**Abstract**: Despite the extensive use of deep autoencoders (AEs) in critical applications, their adversarial robustness remains relatively underexplored compared to classification models. AE robustness is characterized by the Lipschitz bounds of its components. Existing robustness evaluation frameworks based on white-box attacks do not fully exploit the vulnerabilities of intermediate ill-conditioned layers in AEs. In the context of optimizing imperceptible norm-bounded additive perturbations to maximize output damage, existing methods struggle to effectively propagate adversarial loss gradients throughout the network, often converging to less effective perturbations. To address this, we propose a novel layer-conditioning-based adversarial optimization objective that effectively guides the adversarial map toward regions of local Lipschitz bounds by enhancing loss gradient information propagation during attack optimization. We demonstrate through extensive experiments on state-of-the-art AEs that our adversarial objective results in stronger attacks, outperforming existing methods in both universal and sample-specific scenarios. As a defense method against this attack, we introduce an inference-time adversarially trained defense plugin that mitigates the effects of adversarial examples.



## **12. The Adaptive Arms Race: Redefining Robustness in AI Security**

cs.AI

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2312.13435v3) [paper-pdf](http://arxiv.org/pdf/2312.13435v3)

**Authors**: Ilias Tsingenopoulos, Vera Rimmer, Davy Preuveneers, Fabio Pierazzi, Lorenzo Cavallaro, Wouter Joosen

**Abstract**: Despite considerable efforts on making them robust, real-world AI-based systems remain vulnerable to decision based attacks, as definitive proofs of their operational robustness have so far proven intractable. Canonical robustness evaluation relies on adaptive attacks, which leverage complete knowledge of the defense and are tailored to bypass it. This work broadens the notion of adaptivity, which we employ to enhance both attacks and defenses, showing how they can benefit from mutual learning through interaction. We introduce a framework for adaptively optimizing black-box attacks and defenses under the competitive game they form. To assess robustness reliably, it is essential to evaluate against realistic and worst-case attacks. We thus enhance attacks and their evasive arsenal together using RL, apply the same principle to defenses, and evaluate them first independently and then jointly under a multi-agent perspective. We find that active defenses, those that dynamically control system responses, are an essential complement to model hardening against decision-based attacks; that these defenses can be circumvented by adaptive attacks, something that elicits defenses being adaptive too. Our findings, supported by an extensive theoretical and empirical investigation, confirm that adaptive adversaries pose a serious threat to black-box AI-based systems, rekindling the proverbial arms race. Notably, our approach outperforms the state-of-the-art black-box attacks and defenses, while bringing them together to render effective insights into the robustness of real-world deployed ML-based systems.



## **13. Uncovering the Limitations of Model Inversion Evaluation: Benchmarks and Connection to Type-I Adversarial Attacks**

cs.LG

Our dataset and code are available in the Supp

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03519v1) [paper-pdf](http://arxiv.org/pdf/2505.03519v1)

**Authors**: Sy-Tuyen Ho, Koh Jun Hao, Ngoc-Bao Nguyen, Alexander Binder, Ngai-Man Cheung

**Abstract**: Model Inversion (MI) attacks aim to reconstruct information of private training data by exploiting access to machine learning models. The most common evaluation framework for MI attacks/defenses relies on an evaluation model that has been utilized to assess progress across almost all MI attacks and defenses proposed in recent years. In this paper, for the first time, we present an in-depth study of MI evaluation. Firstly, we construct the first comprehensive human-annotated dataset of MI attack samples, based on 28 setups of different MI attacks, defenses, private and public datasets. Secondly, using our dataset, we examine the accuracy of the MI evaluation framework and reveal that it suffers from a significant number of false positives. These findings raise questions about the previously reported success rates of SOTA MI attacks. Thirdly, we analyze the causes of these false positives, design controlled experiments, and discover the surprising effect of Type I adversarial features on MI evaluation, as well as adversarial transferability, highlighting a relationship between two previously distinct research areas. Our findings suggest that the performance of SOTA MI attacks has been overestimated, with the actual privacy leakage being significantly less than previously reported. In conclusion, we highlight critical limitations in the widely used MI evaluation framework and present our methods to mitigate false positive rates. We remark that prior research has shown that Type I adversarial attacks are very challenging, with no existing solution. Therefore, we urge to consider human evaluation as a primary MI evaluation framework rather than merely a supplement as in previous MI research. We also encourage further work on developing more robust and reliable automatic evaluation frameworks.



## **14. BadLingual: A Novel Lingual-Backdoor Attack against Large Language Models**

cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03501v1) [paper-pdf](http://arxiv.org/pdf/2505.03501v1)

**Authors**: Zihan Wang, Hongwei Li, Rui Zhang, Wenbo Jiang, Kangjie Chen, Tianwei Zhang, Qingchuan Zhao, Guowen Xu

**Abstract**: In this paper, we present a new form of backdoor attack against Large Language Models (LLMs): lingual-backdoor attacks. The key novelty of lingual-backdoor attacks is that the language itself serves as the trigger to hijack the infected LLMs to generate inflammatory speech. They enable the precise targeting of a specific language-speaking group, exacerbating racial discrimination by malicious entities. We first implement a baseline lingual-backdoor attack, which is carried out by poisoning a set of training data for specific downstream tasks through translation into the trigger language. However, this baseline attack suffers from poor task generalization and is impractical in real-world settings. To address this challenge, we design BadLingual, a novel task-agnostic lingual-backdoor, capable of triggering any downstream tasks within the chat LLMs, regardless of the specific questions of these tasks. We design a new approach using PPL-constrained Greedy Coordinate Gradient-based Search (PGCG) based adversarial training to expand the decision boundary of lingual-backdoor, thereby enhancing the generalization ability of lingual-backdoor across various tasks. We perform extensive experiments to validate the effectiveness of our proposed attacks. Specifically, the baseline attack achieves an ASR of over 90% on the specified tasks. However, its ASR reaches only 37.61% across six tasks in the task-agnostic scenario. In contrast, BadLingual brings up to 37.35% improvement over the baseline. Our study sheds light on a new perspective of vulnerabilities in LLMs with multilingual capabilities and is expected to promote future research on the potential defenses to enhance the LLMs' robustness



## **15. Mitigating Backdoor Triggered and Targeted Data Poisoning Attacks in Voice Authentication Systems**

cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03455v1) [paper-pdf](http://arxiv.org/pdf/2505.03455v1)

**Authors**: Alireza Mohammadi, Keshav Sood, Dhananjay Thiruvady, Asef Nazari

**Abstract**: Voice authentication systems remain susceptible to two major threats: backdoor triggered attacks and targeted data poisoning attacks. This dual vulnerability is critical because conventional solutions typically address each threat type separately, leaving systems exposed to adversaries who can exploit both attacks simultaneously. We propose a unified defense framework that effectively addresses both BTA and TDPA. Our framework integrates a frequency focused detection mechanism that flags covert pitch boosting and sound masking backdoor attacks in near real time, followed by a convolutional neural network that addresses TDPA. This dual layered defense approach utilizes multidimensional acoustic features to isolate anomalous signals without requiring costly model retraining. In particular, our PBSM detection mechanism can seamlessly integrate into existing voice authentication pipelines and scale effectively for large scale deployments. Experimental results on benchmark datasets and their compression with the state of the art algorithm demonstrate that our PBSM detection mechanism outperforms the state of the art. Our framework reduces attack success rates to as low as five to fifteen percent while maintaining a recall rate of up to ninety five percent in recognizing TDPA.



## **16. Robustness in AI-Generated Detection: Enhancing Resistance to Adversarial Attacks**

cs.CV

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03435v1) [paper-pdf](http://arxiv.org/pdf/2505.03435v1)

**Authors**: Sun Haoxuan, Hong Yan, Zhan Jiahui, Chen Haoxing, Lan Jun, Zhu Huijia, Wang Weiqiang, Zhang Liqing, Zhang Jianfu

**Abstract**: The rapid advancement of generative image technology has introduced significant security concerns, particularly in the domain of face generation detection. This paper investigates the vulnerabilities of current AI-generated face detection systems. Our study reveals that while existing detection methods often achieve high accuracy under standard conditions, they exhibit limited robustness against adversarial attacks. To address these challenges, we propose an approach that integrates adversarial training to mitigate the impact of adversarial examples. Furthermore, we utilize diffusion inversion and reconstruction to further enhance detection robustness. Experimental results demonstrate that minor adversarial perturbations can easily bypass existing detection systems, but our method significantly improves the robustness of these systems. Additionally, we provide an in-depth analysis of adversarial and benign examples, offering insights into the intrinsic characteristics of AI-generated content. All associated code will be made publicly available in a dedicated repository to facilitate further research and verification.



## **17. Attention-aggregated Attack for Boosting the Transferability of Facial Adversarial Examples**

cs.CV

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03383v1) [paper-pdf](http://arxiv.org/pdf/2505.03383v1)

**Authors**: Jian-Wei Li, Wen-Ze Shao

**Abstract**: Adversarial examples have revealed the vulnerability of deep learning models and raised serious concerns about information security. The transfer-based attack is a hot topic in black-box attacks that are practical to real-world scenarios where the training datasets, parameters, and structure of the target model are unknown to the attacker. However, few methods consider the particularity of class-specific deep models for fine-grained vision tasks, such as face recognition (FR), giving rise to unsatisfactory attacking performance. In this work, we first investigate what in a face exactly contributes to the embedding learning of FR models and find that both decisive and auxiliary facial features are specific to each FR model, which is quite different from the biological mechanism of human visual system. Accordingly we then propose a novel attack method named Attention-aggregated Attack (AAA) to enhance the transferability of adversarial examples against FR, which is inspired by the attention divergence and aims to destroy the facial features that are critical for the decision-making of other FR models by imitating their attentions on the clean face images. Extensive experiments conducted on various FR models validate the superiority and robust effectiveness of the proposed method over existing methods.



## **18. A Chaos Driven Metric for Backdoor Attack Detection**

cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03208v1) [paper-pdf](http://arxiv.org/pdf/2505.03208v1)

**Authors**: Hema Karnam Surendrababu, Nithin Nagaraj

**Abstract**: The advancement and adoption of Artificial Intelligence (AI) models across diverse domains have transformed the way we interact with technology. However, it is essential to recognize that while AI models have introduced remarkable advancements, they also present inherent challenges such as their vulnerability to adversarial attacks. The current work proposes a novel defense mechanism against one of the most significant attack vectors of AI models - the backdoor attack via data poisoning of training datasets. In this defense technique, an integrated approach that combines chaos theory with manifold learning is proposed. A novel metric - Precision Matrix Dependency Score (PDS) that is based on the conditional variance of Neurochaos features is formulated. The PDS metric has been successfully evaluated to distinguish poisoned samples from non-poisoned samples across diverse datasets.



## **19. Using Mechanistic Interpretability to Craft Adversarial Attacks against Large Language Models**

cs.LG

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2503.06269v2) [paper-pdf](http://arxiv.org/pdf/2503.06269v2)

**Authors**: Thomas Winninger, Boussad Addad, Katarzyna Kapusta

**Abstract**: Traditional white-box methods for creating adversarial perturbations against LLMs typically rely only on gradient computation from the targeted model, ignoring the internal mechanisms responsible for attack success or failure. Conversely, interpretability studies that analyze these internal mechanisms lack practical applications beyond runtime interventions. We bridge this gap by introducing a novel white-box approach that leverages mechanistic interpretability techniques to craft practical adversarial inputs. Specifically, we first identify acceptance subspaces - sets of feature vectors that do not trigger the model's refusal mechanisms - then use gradient-based optimization to reroute embeddings from refusal subspaces to acceptance subspaces, effectively achieving jailbreaks. This targeted approach significantly reduces computation cost, achieving attack success rates of 80-95\% on state-of-the-art models including Gemma2, Llama3.2, and Qwen2.5 within minutes or even seconds, compared to existing techniques that often fail or require hours of computation. We believe this approach opens a new direction for both attack research and defense development. Furthermore, it showcases a practical application of mechanistic interpretability where other methods are less efficient, which highlights its utility. The code and generated datasets are available at https://github.com/Sckathach/subspace-rerouting.



## **20. PEEK: Phishing Evolution Framework for Phishing Generation and Evolving Pattern Analysis using Large Language Models**

cs.CR

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2411.11389v2) [paper-pdf](http://arxiv.org/pdf/2411.11389v2)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Shuo Wang, Alsharif Abuadbba, Carsten Rudolph

**Abstract**: Phishing remains a pervasive cyber threat, as attackers craft deceptive emails to lure victims into revealing sensitive information. While Artificial Intelligence (AI), in particular, deep learning, has become a key component in defending against phishing attacks, these approaches face critical limitations. The scarcity of publicly available, diverse, and updated data, largely due to privacy concerns, constrains detection effectiveness. As phishing tactics evolve rapidly, models trained on limited, outdated data struggle to detect new, sophisticated deception strategies, leaving systems and people vulnerable to an ever-growing array of attacks. We propose the first Phishing Evolution FramEworK (PEEK) for augmenting phishing email datasets with respect to quality and diversity, and analyzing changing phishing patterns for detection to adapt to updated phishing attacks. Specifically, we integrate large language models (LLMs) into the process of adversarial training to enhance the performance of the generated dataset and leverage persuasion principles in a recurrent framework to facilitate the understanding of changing phishing strategies. PEEK raises the proportion of usable phishing samples from 21.4% to 84.8%, surpassing existing works that rely on prompting and fine-tuning LLMs. The phishing datasets provided by PEEK, with evolving phishing patterns, outperform the other two available LLM-generated phishing email datasets in improving detection robustness. PEEK phishing boosts detectors' accuracy to over 88% and reduces adversarial sensitivity by up to 70%, still maintaining 70% detection accuracy against adversarial attacks.



## **21. Adversarial Sample Generation for Anomaly Detection in Industrial Control Systems**

cs.CR

Accepted in the 1st Workshop on Modeling and Verification for Secure  and Performant Cyber-Physical Systems in conjunction with Cyber-Physical  Systems and Internet-of-Things Week, Irvine, USA, May 6-9, 2025

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03120v1) [paper-pdf](http://arxiv.org/pdf/2505.03120v1)

**Authors**: Abdul Mustafa, Muhammad Talha Khan, Muhammad Azmi Umer, Zaki Masood, Chuadhry Mujeeb Ahmed

**Abstract**: Machine learning (ML)-based intrusion detection systems (IDS) are vulnerable to adversarial attacks. It is crucial for an IDS to learn to recognize adversarial examples before malicious entities exploit them. In this paper, we generated adversarial samples using the Jacobian Saliency Map Attack (JSMA). We validate the generalization and scalability of the adversarial samples to tackle a broad range of real attacks on Industrial Control Systems (ICS). We evaluated the impact by assessing multiple attacks generated using the proposed method. The model trained with adversarial samples detected attacks with 95% accuracy on real-world attack data not used during training. The study was conducted using an operational secure water treatment (SWaT) testbed.



## **22. Adversarial Attacks in Multimodal Systems: A Practitioner's Survey**

cs.LG

Accepted in IEEE COMPSAC 2025

**SubmitDate**: 2025-05-06    [abs](http://arxiv.org/abs/2505.03084v1) [paper-pdf](http://arxiv.org/pdf/2505.03084v1)

**Authors**: Shashank Kapoor, Sanjay Surendranath Girija, Lakshit Arora, Dipen Pradhan, Ankit Shetgaonkar, Aman Raj

**Abstract**: The introduction of multimodal models is a huge step forward in Artificial Intelligence. A single model is trained to understand multiple modalities: text, image, video, and audio. Open-source multimodal models have made these breakthroughs more accessible. However, considering the vast landscape of adversarial attacks across these modalities, these models also inherit vulnerabilities of all the modalities, and ultimately, the adversarial threat amplifies. While broad research is available on possible attacks within or across these modalities, a practitioner-focused view that outlines attack types remains absent in the multimodal world. As more Machine Learning Practitioners adopt, fine-tune, and deploy open-source models in real-world applications, it's crucial that they can view the threat landscape and take the preventive actions necessary. This paper addresses the gap by surveying adversarial attacks targeting all four modalities: text, image, video, and audio. This survey provides a view of the adversarial attack landscape and presents how multimodal adversarial threats have evolved. To the best of our knowledge, this survey is the first comprehensive summarization of the threat landscape in the multimodal world.



## **23. Large Language Models as Robust Data Generators in Software Analytics: Are We There Yet?**

cs.SE

Accepted to the AI Model/Data Track of the Evaluation and Assessment  in Software Engineering (EASE) 2025 Conference

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2411.10565v3) [paper-pdf](http://arxiv.org/pdf/2411.10565v3)

**Authors**: Md. Abdul Awal, Mrigank Rochan, Chanchal K. Roy

**Abstract**: Large Language Model (LLM)-generated data is increasingly used in software analytics, but it is unclear how this data compares to human-written data, particularly when models are exposed to adversarial scenarios. Adversarial attacks can compromise the reliability and security of software systems, so understanding how LLM-generated data performs under these conditions, compared to human-written data, which serves as the benchmark for model performance, can provide valuable insights into whether LLM-generated data offers similar robustness and effectiveness. To address this gap, we systematically evaluate and compare the quality of human-written and LLM-generated data for fine-tuning robust pre-trained models (PTMs) in the context of adversarial attacks. We evaluate the robustness of six widely used PTMs, fine-tuned on human-written and LLM-generated data, before and after adversarial attacks. This evaluation employs nine state-of-the-art (SOTA) adversarial attack techniques across three popular software analytics tasks: clone detection, code summarization, and sentiment analysis in code review discussions. Additionally, we analyze the quality of the generated adversarial examples using eleven similarity metrics. Our findings reveal that while PTMs fine-tuned on LLM-generated data perform competitively with those fine-tuned on human-written data, they exhibit less robustness against adversarial attacks in software analytics tasks. Our study underscores the need for further exploration into enhancing the quality of LLM-generated training data to develop models that are both high-performing and capable of withstanding adversarial attacks in software analytics.



## **24. Adversarial Robustness Analysis of Vision-Language Models in Medical Image Segmentation**

cs.CV

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2505.02971v1) [paper-pdf](http://arxiv.org/pdf/2505.02971v1)

**Authors**: Anjila Budathoki, Manish Dhakal

**Abstract**: Adversarial attacks have been fairly explored for computer vision and vision-language models. However, the avenue of adversarial attack for the vision language segmentation models (VLSMs) is still under-explored, especially for medical image analysis.   Thus, we have investigated the robustness of VLSMs against adversarial attacks for 2D medical images with different modalities with radiology, photography, and endoscopy. The main idea of this project was to assess the robustness of the fine-tuned VLSMs specially in the medical domain setting to address the high risk scenario.   First, we have fine-tuned pre-trained VLSMs for medical image segmentation with adapters.   Then, we have employed adversarial attacks -- projected gradient descent (PGD) and fast gradient sign method (FGSM) -- on that fine-tuned model to determine its robustness against adversaries.   We have reported models' performance decline to analyze the adversaries' impact.   The results exhibit significant drops in the DSC and IoU scores after the introduction of these adversaries. Furthermore, we also explored universal perturbation but were not able to find for the medical images.   \footnote{https://github.com/anjilab/secure-private-ai}



## **25. Constrained Adversarial Learning for Automated Software Testing: a literature review**

cs.SE

36 pages, 4 tables, 2 figures, Discover Applied Sciences journal

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2303.07546v2) [paper-pdf](http://arxiv.org/pdf/2303.07546v2)

**Authors**: João Vitorino, Tiago Dias, Tiago Fonseca, Eva Maia, Isabel Praça

**Abstract**: It is imperative to safeguard computer applications and information systems against the growing number of cyber-attacks. Automated software testing tools can be developed to quickly analyze many lines of code and detect vulnerabilities by generating function-specific testing data. This process draws similarities to the constrained adversarial examples generated by adversarial machine learning methods, so there could be significant benefits to the integration of these methods in testing tools to identify possible attack vectors. Therefore, this literature review is focused on the current state-of-the-art of constrained data generation approaches applied for adversarial learning and software testing, aiming to guide researchers and developers to enhance their software testing tools with adversarial testing methods and improve the resilience and robustness of their information systems. The found approaches were systematized, and the advantages and limitations of those specific for white-box, grey-box, and black-box testing were analyzed, identifying research gaps and opportunities to automate the testing tools with data generated by adversarial attacks.



## **26. Commitment Attacks on Ethereum's Reward Mechanism**

cs.CR

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2407.19479v2) [paper-pdf](http://arxiv.org/pdf/2407.19479v2)

**Authors**: Roozbeh Sarenche, Ertem Nusret Tas, Barnabe Monnot, Caspar Schwarz-Schilling, Bart Preneel

**Abstract**: Validators in permissionless, large-scale blockchains, such as Ethereum, are typically payoff-maximizing, rational actors. Ethereum relies on in-protocol incentives, like rewards for correct and timely votes, to induce honest behavior and secure the blockchain. However, external incentives, such as the block proposer's opportunity to capture maximal extractable value (MEV), may tempt validators to deviate from honest protocol participation.   We show a series of commitment attacks on LMD GHOST, a core part of Ethereum's consensus mechanism. We demonstrate how a single adversarial block proposer can orchestrate long-range chain reorganizations by manipulating Ethereum's reward system for timely votes. These attacks disrupt the intended balance of power between proposers and voters: by leveraging credible threats, the adversarial proposer can coerce voters from previous slots into supporting blocks that conflict with the honest chain, enabling a chain reorganization.   In response, we introduce a novel reward mechanism that restores the voters' role as a check against proposer power. Our proposed mitigation is fairer and more decentralized, not only in the context of these attacks, but also practical for implementation in Ethereum.



## **27. Robustness questions the interpretability of graph neural networks: what to do?**

cs.LG

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2505.02566v1) [paper-pdf](http://arxiv.org/pdf/2505.02566v1)

**Authors**: Kirill Lukyanov, Georgii Sazonov, Serafim Boyarsky, Ilya Makarov

**Abstract**: Graph Neural Networks (GNNs) have become a cornerstone in graph-based data analysis, with applications in diverse domains such as bioinformatics, social networks, and recommendation systems. However, the interplay between model interpretability and robustness remains poorly understood, especially under adversarial scenarios like poisoning and evasion attacks. This paper presents a comprehensive benchmark to systematically analyze the impact of various factors on the interpretability of GNNs, including the influence of robustness-enhancing defense mechanisms.   We evaluate six GNN architectures based on GCN, SAGE, GIN, and GAT across five datasets from two distinct domains, employing four interpretability metrics: Fidelity, Stability, Consistency, and Sparsity. Our study examines how defenses against poisoning and evasion attacks, applied before and during model training, affect interpretability and highlights critical trade-offs between robustness and interpretability. The framework will be published as open source.   The results reveal significant variations in interpretability depending on the chosen defense methods and model architecture characteristics. By establishing a standardized benchmark, this work provides a foundation for developing GNNs that are both robust to adversarial threats and interpretable, facilitating trust in their deployment in sensitive applications.



## **28. Bayesian Robust Aggregation for Federated Learning**

cs.LG

14 pages, 4 figures, 8 tables

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2505.02490v1) [paper-pdf](http://arxiv.org/pdf/2505.02490v1)

**Authors**: Aleksandr Karakulev, Usama Zafar, Salman Toor, Prashant Singh

**Abstract**: Federated Learning enables collaborative training of machine learning models on decentralized data. This scheme, however, is vulnerable to adversarial attacks, when some of the clients submit corrupted model updates. In real-world scenarios, the total number of compromised clients is typically unknown, with the extent of attacks potentially varying over time. To address these challenges, we propose an adaptive approach for robust aggregation of model updates based on Bayesian inference. The mean update is defined by the maximum of the likelihood marginalized over probabilities of each client to be `honest'. As a result, the method shares the simplicity of the classical average estimators (e.g., sample mean or geometric median), being independent of the number of compromised clients. At the same time, it is as effective against attacks as methods specifically tailored to Federated Learning, such as Krum. We compare our approach with other aggregation schemes in federated setting on three benchmark image classification data sets. The proposed method consistently achieves state-of-the-art performance across various attack types with static and varying number of malicious clients.



## **29. Economic Security of Multiple Shared Security Protocols**

cs.CR

21 pages, 6 figures

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2505.03843v1) [paper-pdf](http://arxiv.org/pdf/2505.03843v1)

**Authors**: Abhimanyu Nag, Dhruv Bodani, Abhishek Kumar

**Abstract**: As restaking protocols gain adoption across blockchain ecosystems, there is a need for Actively Validated Services (AVSs) to span multiple Shared Security Providers (SSPs). This leads to stake fragmentation which introduces new complications where an adversary may compromise an AVS by targeting its weakest SSP. In this paper, we formalize the Multiple SSP Problem and analyze two architectures : an isolated fragmented model called Model $\mathbb{M}$ and a shared unified model called Model $\mathbb{S}$, through a convex optimization and game-theoretic lens. We derive utility bounds, attack cost conditions, and market equilibrium that describes protocol security for both models. Our results show that while Model $\mathbb{M}$ offers deployment flexibility, it inherits lowest-cost attack vulnerabilities, whereas Model $\mathbb{S}$ achieves tighter security guarantees through single validator sets and aggregated slashing logic. We conclude with future directions of work including an incentive-compatible stake rebalancing allocation in restaking ecosystems.



## **30. Catastrophic Overfitting, Entropy Gap and Participation Ratio: A Noiseless $l^p$ Norm Solution for Fast Adversarial Training**

cs.LG

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2505.02360v1) [paper-pdf](http://arxiv.org/pdf/2505.02360v1)

**Authors**: Fares B. Mehouachi, Saif Eddin Jabari

**Abstract**: Adversarial training is a cornerstone of robust deep learning, but fast methods like the Fast Gradient Sign Method (FGSM) often suffer from Catastrophic Overfitting (CO), where models become robust to single-step attacks but fail against multi-step variants. While existing solutions rely on noise injection, regularization, or gradient clipping, we propose a novel solution that purely controls the $l^p$ training norm to mitigate CO.   Our study is motivated by the empirical observation that CO is more prevalent under the $l^{\infty}$ norm than the $l^2$ norm. Leveraging this insight, we develop a framework for generalized $l^p$ attack as a fixed point problem and craft $l^p$-FGSM attacks to understand the transition mechanics from $l^2$ to $l^{\infty}$. This leads to our core insight: CO emerges when highly concentrated gradients where information localizes in few dimensions interact with aggressive norm constraints. By quantifying gradient concentration through Participation Ratio and entropy measures, we develop an adaptive $l^p$-FGSM that automatically tunes the training norm based on gradient information. Extensive experiments demonstrate that this approach achieves strong robustness without requiring additional regularization or noise injection, providing a novel and theoretically-principled pathway to mitigate the CO problem.



## **31. Bayes-Nash Generative Privacy Against Membership Inference Attacks**

cs.CR

arXiv admin note: substantial text overlap with arXiv:2406.01811

**SubmitDate**: 2025-05-05    [abs](http://arxiv.org/abs/2410.07414v4) [paper-pdf](http://arxiv.org/pdf/2410.07414v4)

**Authors**: Tao Zhang, Rajagopal Venkatesaramani, Rajat K. De, Bradley A. Malin, Yevgeniy Vorobeychik

**Abstract**: Membership inference attacks (MIAs) expose significant privacy risks by determining whether an individual's data is in a dataset. While differential privacy (DP) mitigates such risks, it has several limitations in achieving an optimal balance between utility and privacy, include limited resolution in expressing this tradeoff in only a few privacy parameters, and intractable sensitivity calculations that may be necessary to provide tight privacy guarantees. We propose a game-theoretic framework that models privacy protection from MIA as a Bayesian game between a defender and an attacker. In this game, a dataset is the defender's private information, with privacy loss to the defender (which is gain to the attacker) captured in terms of the attacker's ability to infer membership of individuals in the dataset. To address the strategic complexity of this game, we represent the mixed strategy of the defender as a neural network generator which maps a private dataset to its public representation (for example, noisy summary statistics), while the mixed strategy of the attacker is captured by a discriminator which makes membership inference claims. We refer to the resulting computational approach as a general-sum Generative Adversarial Network, which is trained iteratively by alternating generator and discriminator updates akin to conventional GANs. We call the defender's data sharing policy thereby obtained Bayes-Nash Generative Privacy (BNGP). The BNGP strategy avoids sensitivity calculations, supports compositions of correlated mechanisms, is robust to the attacker's heterogeneous preferences over true and false positives, and yields provable differential privacy guarantees, albeit in an idealized setting.



## **32. Open Challenges in Multi-Agent Security: Towards Secure Systems of Interacting AI Agents**

cs.CR

**SubmitDate**: 2025-05-04    [abs](http://arxiv.org/abs/2505.02077v1) [paper-pdf](http://arxiv.org/pdf/2505.02077v1)

**Authors**: Christian Schroeder de Witt

**Abstract**: Decentralized AI agents will soon interact across internet platforms, creating security challenges beyond traditional cybersecurity and AI safety frameworks. Free-form protocols are essential for AI's task generalization but enable new threats like secret collusion and coordinated swarm attacks. Network effects can rapidly spread privacy breaches, disinformation, jailbreaks, and data poisoning, while multi-agent dispersion and stealth optimization help adversaries evade oversightcreating novel persistent threats at a systemic level. Despite their critical importance, these security challenges remain understudied, with research fragmented across disparate fields including AI security, multi-agent learning, complex systems, cybersecurity, game theory, distributed systems, and technical AI governance. We introduce \textbf{multi-agent security}, a new field dedicated to securing networks of decentralized AI agents against threats that emerge or amplify through their interactionswhether direct or indirect via shared environmentswith each other, humans, and institutions, and characterize fundamental security-performance trade-offs. Our preliminary work (1) taxonomizes the threat landscape arising from interacting AI agents, (2) surveys security-performance tradeoffs in decentralized AI systems, and (3) proposes a unified research agenda addressing open challenges in designing secure agent systems and interaction environments. By identifying these gaps, we aim to guide research in this critical area to unlock the socioeconomic potential of large-scale agent deployment on the internet, foster public trust, and mitigate national security risks in critical infrastructure and defense contexts.



## **33. Lightweight Defense Against Adversarial Attacks in Time Series Classification**

cs.LG

13 pages, 8 figures. Accepted at RAFDA Workshop, PAKDD 2025  (Springer, EI & Scopus indexed). Code:  https://github.com/Yi126/Lightweight-Defence

**SubmitDate**: 2025-05-04    [abs](http://arxiv.org/abs/2505.02073v1) [paper-pdf](http://arxiv.org/pdf/2505.02073v1)

**Authors**: Yi Han

**Abstract**: As time series classification (TSC) gains prominence, ensuring robust TSC models against adversarial attacks is crucial. While adversarial defense is well-studied in Computer Vision (CV), the TSC field has primarily relied on adversarial training (AT), which is computationally expensive. In this paper, five data augmentation-based defense methods tailored for time series are developed, with the most computationally intensive method among them increasing the computational resources by only 14.07% compared to the original TSC model. Moreover, the deployment process for these methods is straightforward. By leveraging these advantages of our methods, we create two combined methods. One of these methods is an ensemble of all the proposed techniques, which not only provides better defense performance than PGD-based AT but also enhances the generalization ability of TSC models. Moreover, the computational resources required for our ensemble are less than one-third of those required for PGD-based AT. These methods advance robust TSC in data mining. Furthermore, as foundation models are increasingly explored for time series feature learning, our work provides insights into integrating data augmentation-based adversarial defense with large-scale pre-trained models in future research.



## **34. A Comprehensive Analysis of Adversarial Attacks against Spam Filters**

cs.CR

**SubmitDate**: 2025-05-04    [abs](http://arxiv.org/abs/2505.03831v1) [paper-pdf](http://arxiv.org/pdf/2505.03831v1)

**Authors**: Esra Hotoğlu, Sevil Sen, Burcu Can

**Abstract**: Deep learning has revolutionized email filtering, which is critical to protect users from cyber threats such as spam, malware, and phishing. However, the increasing sophistication of adversarial attacks poses a significant challenge to the effectiveness of these filters. This study investigates the impact of adversarial attacks on deep learning-based spam detection systems using real-world datasets. Six prominent deep learning models are evaluated on these datasets, analyzing attacks at the word, character sentence, and AI-generated paragraph-levels. Novel scoring functions, including spam weights and attention weights, are introduced to improve attack effectiveness. This comprehensive analysis sheds light on the vulnerabilities of spam filters and contributes to efforts to improve their security against evolving adversarial threats.



## **35. CAMOUFLAGE: Exploiting Misinformation Detection Systems Through LLM-driven Adversarial Claim Transformation**

cs.CL

**SubmitDate**: 2025-05-03    [abs](http://arxiv.org/abs/2505.01900v1) [paper-pdf](http://arxiv.org/pdf/2505.01900v1)

**Authors**: Mazal Bethany, Nishant Vishwamitra, Cho-Yu Jason Chiang, Peyman Najafirad

**Abstract**: Automated evidence-based misinformation detection systems, which evaluate the veracity of short claims against evidence, lack comprehensive analysis of their adversarial vulnerabilities. Existing black-box text-based adversarial attacks are ill-suited for evidence-based misinformation detection systems, as these attacks primarily focus on token-level substitutions involving gradient or logit-based optimization strategies, which are incapable of fooling the multi-component nature of these detection systems. These systems incorporate both retrieval and claim-evidence comparison modules, which requires attacks to break the retrieval of evidence and/or the comparison module so that it draws incorrect inferences. We present CAMOUFLAGE, an iterative, LLM-driven approach that employs a two-agent system, a Prompt Optimization Agent and an Attacker Agent, to create adversarial claim rewritings that manipulate evidence retrieval and mislead claim-evidence comparison, effectively bypassing the system without altering the meaning of the claim. The Attacker Agent produces semantically equivalent rewrites that attempt to mislead detectors, while the Prompt Optimization Agent analyzes failed attack attempts and refines the prompt of the Attacker to guide subsequent rewrites. This enables larger structural and stylistic transformations of the text rather than token-level substitutions, adapting the magnitude of changes based on previous outcomes. Unlike existing approaches, CAMOUFLAGE optimizes its attack solely based on binary model decisions to guide its rewriting process, eliminating the need for classifier logits or extensive querying. We evaluate CAMOUFLAGE on four systems, including two recent academic systems and two real-world APIs, with an average attack success rate of 46.92\% while preserving textual coherence and semantic equivalence to the original claims.



## **36. PQS-BFL: A Post-Quantum Secure Blockchain-based Federated Learning Framework**

cs.CR

**SubmitDate**: 2025-05-03    [abs](http://arxiv.org/abs/2505.01866v1) [paper-pdf](http://arxiv.org/pdf/2505.01866v1)

**Authors**: Daniel Commey, Garth V. Crosby

**Abstract**: Federated Learning (FL) enables collaborative model training while preserving data privacy, but its classical cryptographic underpinnings are vulnerable to quantum attacks. This vulnerability is particularly critical in sensitive domains like healthcare. This paper introduces PQS-BFL (Post-Quantum Secure Blockchain-based Federated Learning), a framework integrating post-quantum cryptography (PQC) with blockchain verification to secure FL against quantum adversaries. We employ ML-DSA-65 (a FIPS 204 standard candidate, formerly Dilithium) signatures to authenticate model updates and leverage optimized smart contracts for decentralized validation. Extensive evaluations on diverse datasets (MNIST, SVHN, HAR) demonstrate that PQS-BFL achieves efficient cryptographic operations (average PQC sign time: 0.65 ms, verify time: 0.53 ms) with a fixed signature size of 3309 Bytes. Blockchain integration incurs a manageable overhead, with average transaction times around 4.8 s and gas usage per update averaging 1.72 x 10^6 units for PQC configurations. Crucially, the cryptographic overhead relative to transaction time remains minimal (around 0.01-0.02% for PQC with blockchain), confirming that PQC performance is not the bottleneck in blockchain-based FL. The system maintains competitive model accuracy (e.g., over 98.8% for MNIST with PQC) and scales effectively, with round times showing sublinear growth with increasing client numbers. Our open-source implementation and reproducible benchmarks validate the feasibility of deploying long-term, quantum-resistant security in practical FL systems.



## **37. Rogue Cell: Adversarial Attack and Defense in Untrusted O-RAN Setup Exploiting the Traffic Steering xApp**

cs.CR

**SubmitDate**: 2025-05-03    [abs](http://arxiv.org/abs/2505.01816v1) [paper-pdf](http://arxiv.org/pdf/2505.01816v1)

**Authors**: Eran Aizikovich, Dudu Mimran, Edita Grolman, Yuval Elovici, Asaf Shabtai

**Abstract**: The Open Radio Access Network (O-RAN) architecture is revolutionizing cellular networks with its open, multi-vendor design and AI-driven management, aiming to enhance flexibility and reduce costs. Although it has many advantages, O-RAN is not threat-free. While previous studies have mainly examined vulnerabilities arising from O-RAN's intelligent components, this paper is the first to focus on the security challenges and vulnerabilities introduced by transitioning from single-operator to multi-operator RAN architectures. This shift increases the risk of untrusted third-party operators managing different parts of the network. To explore these vulnerabilities and their potential mitigation, we developed an open-access testbed environment that integrates a wireless network simulator with the official O-RAN Software Community (OSC) RAN intelligent component (RIC) cluster. This environment enables realistic, live data collection and serves as a platform for demonstrating APATE (adversarial perturbation against traffic efficiency), an evasion attack in which a malicious cell manipulates its reported key performance indicators (KPIs) and deceives the O-RAN traffic steering to gain unfair allocations of user equipment (UE). To ensure that O-RAN's legitimate activity continues, we introduce MARRS (monitoring adversarial RAN reports), a detection framework based on a long-short term memory (LSTM) autoencoder (AE) that learns contextual features across the network to monitor malicious telemetry (also demonstrated in our testbed). Our evaluation showed that by executing APATE, an attacker can obtain a 248.5% greater UE allocation than it was supposed to in a benign scenario. In addition, the MARRS detection method was also shown to successfully classify malicious cell activity, achieving accuracy of 99.2% and an F1 score of 0.978.



## **38. LeapFrog: The Rowhammer Instruction Skip Attack**

cs.CR

Accepted at EuroS&P 2025 and Hardware.io 2024,

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2404.07878v3) [paper-pdf](http://arxiv.org/pdf/2404.07878v3)

**Authors**: Andrew Adiletta, M. Caner Tol, Kemal Derya, Berk Sunar, Saad Islam

**Abstract**: Since its inception, Rowhammer exploits have rapidly evolved into increasingly sophisticated threats compromising data integrity and the control flow integrity of victim processes. Nevertheless, it remains a challenge for an attacker to identify vulnerable targets (i.e., Rowhammer gadgets), understand the outcome of the attempted fault, and formulate an attack that yields useful results.   In this paper, we present a new type of Rowhammer gadget, called a LeapFrog gadget, which, when present in the victim code, allows an adversary to subvert code execution to bypass a critical piece of code (e.g., authentication check logic, encryption rounds, padding in security protocols). The LeapFrog gadget manifests when the victim code stores the Program Counter (PC) value in the user or kernel stack (e.g., a return address during a function call) which, when tampered with, repositions the return address to a location that bypasses a security-critical code pattern.   This research also presents a systematic process to identify LeapFrog gadgets. This methodology enables the automated detection of susceptible targets and the determination of optimal attack parameters. We first show the attack on a decision tree algorithm to show the potential implications. Secondly, we employ the attack on OpenSSL to bypass the encryption and reveal the plaintext. We then use our tools to scan the Open Quantum Safe library and report on the number of LeapFrog gadgets in the code. Lastly, we demonstrate this new attack vector through a practical demonstration in a client/server TLS handshake scenario, successfully inducing an instruction skip in a client application. Our findings extend the impact of Rowhammer attacks on control flow and contribute to developing more robust defenses against these increasingly sophisticated threats.



## **39. Modeling Behavioral Preferences of Cyber Adversaries Using Inverse Reinforcement Learning**

cs.CR

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.03817v1) [paper-pdf](http://arxiv.org/pdf/2505.03817v1)

**Authors**: Aditya Shinde, Prashant Doshi

**Abstract**: This paper presents a holistic approach to attacker preference modeling from system-level audit logs using inverse reinforcement learning (IRL). Adversary modeling is an important capability in cybersecurity that lets defenders characterize behaviors of potential attackers, which enables attribution to known cyber adversary groups. Existing approaches rely on documenting an ever-evolving set of attacker tools and techniques to track known threat actors. Although attacks evolve constantly, attacker behavioral preferences are intrinsic and less volatile. Our approach learns the behavioral preferences of cyber adversaries from forensics data on their tools and techniques. We model the attacker as an expert decision-making agent with unknown behavioral preferences situated in a computer host. We leverage attack provenance graphs of audit logs to derive a state-action trajectory of the attack. We test our approach on open datasets of audit logs containing real attack data. Our results demonstrate for the first time that low-level forensics data can automatically reveal an adversary's subjective preferences, which serves as an additional dimension to modeling and documenting cyber adversaries. Attackers' preferences tend to be invariant despite their different tools and indicate predispositions that are inherent to the attacker. As such, these inferred preferences can potentially serve as unique behavioral signatures of attackers and improve threat attribution.



## **40. Machine Learning for Cyber-Attack Identification from Traffic Flows**

cs.LG

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01489v1) [paper-pdf](http://arxiv.org/pdf/2505.01489v1)

**Authors**: Yujing Zhou, Marc L. Jacquet, Robel Dawit, Skyler Fabre, Dev Sarawat, Faheem Khan, Madison Newell, Yongxin Liu, Dahai Liu, Hongyun Chen, Jian Wang, Huihui Wang

**Abstract**: This paper presents our simulation of cyber-attacks and detection strategies on the traffic control system in Daytona Beach, FL. using Raspberry Pi virtual machines and the OPNSense firewall, along with traffic dynamics from SUMO and exploitation via the Metasploit framework. We try to answer the research questions: are we able to identify cyber attacks by only analyzing traffic flow patterns. In this research, the cyber attacks are focused particularly when lights are randomly turned all green or red at busy intersections by adversarial attackers. Despite challenges stemming from imbalanced data and overlapping traffic patterns, our best model shows 85\% accuracy when detecting intrusions purely using traffic flow statistics. Key indicators for successful detection included occupancy, jam length, and halting durations.



## **41. Synthesizing Grid Data with Cyber Resilience and Privacy Guarantees**

eess.SY

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2503.14877v2) [paper-pdf](http://arxiv.org/pdf/2503.14877v2)

**Authors**: Shengyang Wu, Vladimir Dvorkin

**Abstract**: Differential privacy (DP) provides a principled approach to synthesizing data (e.g., loads) from real-world power systems while limiting the exposure of sensitive information. However, adversaries may exploit synthetic data to calibrate cyberattacks on the source grids. To control these risks, we propose new DP algorithms for synthesizing data that provide the source grids with both cyber resilience and privacy guarantees. The algorithms incorporate both normal operation and attack optimization models to balance the fidelity of synthesized data and cyber resilience. The resulting post-processing optimization is reformulated as a robust optimization problem, which is compatible with the exponential mechanism of DP to moderate its computational burden.



## **42. Deep Learning-Enabled System Diagnosis in Microgrids: A Feature-Feedback GAN Approach**

eess.SY

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01366v1) [paper-pdf](http://arxiv.org/pdf/2505.01366v1)

**Authors**: Swetha Rani Kasimalla, Kuchan Park, Junho Hong, Young-Jin Kim, HyoJong Lee

**Abstract**: The increasing integration of inverter-based resources (IBRs) and communication networks has brought both modernization and new vulnerabilities to the power system infrastructure. These vulnerabilities expose the system to internal faults and cyber threats, particularly False Data Injection (FDI) attacks, which can closely mimic real fault scenarios. Hence, this work presents a two-stage fault and cyberattack detection framework tailored for inverter-based microgrids. Stage 1 introduces an unsupervised learning model Feature Feedback Generative Adversarial Network (F2GAN), to distinguish between genuine internal faults and cyber-induced anomalies in microgrids. Compared to conventional GAN architectures, F2GAN demonstrates improved system diagnosis and greater adaptability to zero-day attacks through its feature-feedback mechanism. In Stage 2, supervised machine learning techniques, including Support Vector Machines (SVM), k-Nearest Neighbors (KNN), Decision Trees (DT), and Artificial Neural Networks (ANN) are applied to localize and classify faults within inverter switches, distinguishing between single-switch and multi-switch faults. The proposed framework is validated on a simulated microgrid environment, illustrating robust performance in detecting and classifying both physical and cyber-related disturbances in power electronic-dominated systems.



## **43. Constrained Network Adversarial Attacks: Validity, Robustness, and Transferability**

cs.CR

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01328v1) [paper-pdf](http://arxiv.org/pdf/2505.01328v1)

**Authors**: Anass Grini, Oumaima Taheri, Btissam El Khamlichi, Amal El Fallah-Seghrouchni

**Abstract**: While machine learning has significantly advanced Network Intrusion Detection Systems (NIDS), particularly within IoT environments where devices generate large volumes of data and are increasingly susceptible to cyber threats, these models remain vulnerable to adversarial attacks. Our research reveals a critical flaw in existing adversarial attack methodologies: the frequent violation of domain-specific constraints, such as numerical and categorical limits, inherent to IoT and network traffic. This leads to up to 80.3% of adversarial examples being invalid, significantly overstating real-world vulnerabilities. These invalid examples, though effective in fooling models, do not represent feasible attacks within practical IoT deployments. Consequently, relying on these results can mislead resource allocation for defense, inflating the perceived susceptibility of IoT-enabled NIDS models to adversarial manipulation. Furthermore, we demonstrate that simpler surrogate models like Multi-Layer Perceptron (MLP) generate more valid adversarial examples compared to complex architectures such as CNNs and LSTMs. Using the MLP as a surrogate, we analyze the transferability of adversarial severity to other ML/DL models commonly used in IoT contexts. This work underscores the importance of considering both domain constraints and model architecture when evaluating and designing robust ML/DL models for security-critical IoT and network applications.



## **44. Security Metrics for Uncertain Interconnected Systems under Stealthy Data Injection Attacks**

eess.SY

6 pages, 5 figures, accepted to the 10th IFAC Conference on Networked  Systems, Hongkong 2025

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01233v1) [paper-pdf](http://arxiv.org/pdf/2505.01233v1)

**Authors**: Anh Tung Nguyen, Sribalaji C. Anand, André M. H. Teixeira

**Abstract**: This paper quantifies the security of uncertain interconnected systems under stealthy data injection attacks. In particular, we consider a large-scale system composed of a certain subsystem interconnected with an uncertain subsystem, where only the input-output channels are accessible. An adversary is assumed to inject false data to maximize the performance loss of the certain subsystem while remaining undetected. By abstracting the uncertain subsystem as a class of admissible systems satisfying an $\mathcal{L}_2$ gain constraint, the worst-case performance loss is obtained as the solution to a convex semi-definite program depending only on the certain subsystem dynamics and such an $\mathcal{L}_2$ gain constraint. This solution is proved to serve as an upper bound for the actual worst-case performance loss when the model of the entire system is fully certain. The results are demonstrated through numerical simulations of the power transmission grid spanning Sweden and Northern Denmark.



## **45. Bilateral Cognitive Security Games in Networked Control Systems under Stealthy Injection Attacks**

eess.SY

8 pages, 3 figures, conference submission

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01232v1) [paper-pdf](http://arxiv.org/pdf/2505.01232v1)

**Authors**: Anh Tung Nguyen, Quanyan Zhu, André Teixeira

**Abstract**: This paper studies a strategic security problem in networked control systems under stealthy false data injection attacks. The security problem is modeled as a bilateral cognitive security game between a defender and an adversary, each possessing cognitive reasoning abilities. The adversary with an adversarial cognitive ability strategically attacks some interconnections of the system with the aim of disrupting the network performance while remaining stealthy to the defender. Meanwhile, the defender with a defense cognitive ability strategically monitors some nodes to impose the stealthiness constraint with the purpose of minimizing the worst-case disruption caused by the adversary. Within the proposed bilateral cognitive security framework, the preferred cognitive levels of the two strategic agents are formulated in terms of two newly proposed concepts, cognitive mismatch and cognitive resonance. Moreover, we propose a method to compute the policies for the defender and the adversary with arbitrary cognitive abilities. A sufficient condition is established under which an increase in cognitive levels does not alter the policies for the defender and the adversary, ensuring convergence. The obtained results are validated through numerical simulations.



## **46. Secure Cluster-Based Hierarchical Federated Learning in Vehicular Networks**

cs.CR

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01186v1) [paper-pdf](http://arxiv.org/pdf/2505.01186v1)

**Authors**: M. Saeid HaghighiFard, Sinem Coleri

**Abstract**: Hierarchical Federated Learning (HFL) has recently emerged as a promising solution for intelligent decision-making in vehicular networks, helping to address challenges such as limited communication resources, high vehicle mobility, and data heterogeneity. However, HFL remains vulnerable to adversarial and unreliable vehicles, whose misleading updates can significantly compromise the integrity and convergence of the global model. To address these challenges, we propose a novel defense framework that integrates dynamic vehicle selection with robust anomaly detection within a cluster-based HFL architecture, specifically designed to counter Gaussian noise and gradient ascent attacks. The framework performs a comprehensive reliability assessment for each vehicle by evaluating historical accuracy, contribution frequency, and anomaly records. Anomaly detection combines Z-score and cosine similarity analyses on model updates to identify both statistical outliers and directional deviations in model updates. To further refine detection, an adaptive thresholding mechanism is incorporated into the cosine similarity metric, dynamically adjusting the threshold based on the historical accuracy of each vehicle to enforce stricter standards for consistently high-performing vehicles. In addition, a weighted gradient averaging mechanism is implemented, which assigns higher weights to gradient updates from more trustworthy vehicles. To defend against coordinated attacks, a cross-cluster consistency check is applied to identify collaborative attacks in which multiple compromised clusters coordinate misleading updates. Together, these mechanisms form a multi-level defense strategy to filter out malicious contributions effectively. Simulation results show that the proposed algorithm significantly reduces convergence time compared to benchmark methods across both 1-hop and 3-hop topologies.



## **47. Explainable AI Based Diagnosis of Poisoning Attacks in Evolutionary Swarms**

cs.AI

To appear in short form in Genetic and Evolutionary Computation  Conference (GECCO '25 Companion), 2025

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01181v1) [paper-pdf](http://arxiv.org/pdf/2505.01181v1)

**Authors**: Mehrdad Asadi, Roxana Rădulescu, Ann Nowé

**Abstract**: Swarming systems, such as for example multi-drone networks, excel at cooperative tasks like monitoring, surveillance, or disaster assistance in critical environments, where autonomous agents make decentralized decisions in order to fulfill team-level objectives in a robust and efficient manner. Unfortunately, team-level coordinated strategies in the wild are vulnerable to data poisoning attacks, resulting in either inaccurate coordination or adversarial behavior among the agents. To address this challenge, we contribute a framework that investigates the effects of such data poisoning attacks, using explainable AI methods. We model the interaction among agents using evolutionary intelligence, where an optimal coalition strategically emerges to perform coordinated tasks. Then, through a rigorous evaluation, the swarm model is systematically poisoned using data manipulation attacks. We showcase the applicability of explainable AI methods to quantify the effects of poisoning on the team strategy and extract footprint characterizations that enable diagnosing. Our findings indicate that when the model is poisoned above 10%, non-optimal strategies resulting in inefficient cooperation can be identified.



## **48. Harmonizing Intra-coherence and Inter-divergence in Ensemble Attacks for Adversarial Transferability**

cs.LG

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01168v1) [paper-pdf](http://arxiv.org/pdf/2505.01168v1)

**Authors**: Zhaoyang Ma, Zhihao Wu, Wang Lu, Xin Gao, Jinghang Yue, Taolin Zhang, Lipo Wang, Youfang Lin, Jing Wang

**Abstract**: The development of model ensemble attacks has significantly improved the transferability of adversarial examples, but this progress also poses severe threats to the security of deep neural networks. Existing methods, however, face two critical challenges: insufficient capture of shared gradient directions across models and a lack of adaptive weight allocation mechanisms. To address these issues, we propose a novel method Harmonized Ensemble for Adversarial Transferability (HEAT), which introduces domain generalization into adversarial example generation for the first time. HEAT consists of two key modules: Consensus Gradient Direction Synthesizer, which uses Singular Value Decomposition to synthesize shared gradient directions; and Dual-Harmony Weight Orchestrator which dynamically balances intra-domain coherence, stabilizing gradients within individual models, and inter-domain diversity, enhancing transferability across models. Experimental results demonstrate that HEAT significantly outperforms existing methods across various datasets and settings, offering a new perspective and direction for adversarial attack research.



## **49. Active Sybil Attack and Efficient Defense Strategy in IPFS DHT**

cs.CR

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01139v1) [paper-pdf](http://arxiv.org/pdf/2505.01139v1)

**Authors**: V. H. M. Netto, T. Cholez, C. L. Ignat

**Abstract**: The InterPlanetary File System (IPFS) is a decentralized peer-to-peer (P2P) storage that relies on Kademlia, a Distributed Hash Table (DHT) structure commonly used in P2P systems for its proved scalability. However, DHTs are known to be vulnerable to Sybil attacks, in which a single entity controls multiple malicious nodes. Recent studies have shown that IPFS is affected by a passive content eclipse attack, leveraging Sybils, in which adversarial nodes hide received indexed information from other peers, making the content appear unavailable. Fortunately, the latest mitigation strategy coupling an attack detection based on statistical tests and a wider publication strategy upon detection was able to circumvent it.   In this work, we present a new active attack, with malicious nodes responding with semantically correct but intentionally false data, exploiting both an optimized placement of Sybils to stay below the detection threshold and an early trigger of the content discovery termination in Kubo, the main IPFS implementation. Our attack achieves to completely eclipse content on the latest Kubo release. When evaluated against the most recent known mitigation, it successfully denies access to the target content in approximately 80\% of lookup attempts.   To address this vulnerability, we propose a new mitigation called SR-DHT-Store, which enables efficient, Sybil-resistant content publication without relying on attack detection but instead on a systematic and precise use of region-based queries, defined by a dynamically computed XOR distance to the target ID. SR-DHT-Store can be combined with other defense mechanisms resulting in a defense strategy that completely mitigates both passive and active Sybil attacks at a lower overhead, while allowing an incremental deployment.



## **50. Risk Analysis and Design Against Adversarial Actions**

cs.LG

**SubmitDate**: 2025-05-02    [abs](http://arxiv.org/abs/2505.01130v1) [paper-pdf](http://arxiv.org/pdf/2505.01130v1)

**Authors**: Marco C. Campi, Algo Carè, Luis G. Crespo, Simone Garatti, Federico A. Ramponi

**Abstract**: Learning models capable of providing reliable predictions in the face of adversarial actions has become a central focus of the machine learning community in recent years. This challenge arises from observing that data encountered at deployment time often deviate from the conditions under which the model was trained. In this paper, we address deployment-time adversarial actions and propose a versatile, well-principled framework to evaluate the model's robustness against attacks of diverse types and intensities. While we initially focus on Support Vector Regression (SVR), the proposed approach extends naturally to the broad domain of learning via relaxed optimization techniques. Our results enable an assessment of the model vulnerability without requiring additional test data and operate in a distribution-free setup. These results not only provide a tool to enhance trust in the model's applicability but also aid in selecting among competing alternatives. Later in the paper, we show that our findings also offer useful insights for establishing new results within the out-of-distribution framework.



