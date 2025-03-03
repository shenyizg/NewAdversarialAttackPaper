# Latest Adversarial Attack Papers
**update at 2025-03-03 09:52:57**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. The Evolution of Dataset Distillation: Toward Scalable and Generalizable Solutions**

cs.CV

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.05673v2) [paper-pdf](http://arxiv.org/pdf/2502.05673v2)

**Authors**: Ping Liu, Jiawei Du

**Abstract**: Dataset distillation, which condenses large-scale datasets into compact synthetic representations, has emerged as a critical solution for training modern deep learning models efficiently. While prior surveys focus on developments before 2023, this work comprehensively reviews recent advances, emphasizing scalability to large-scale datasets such as ImageNet-1K and ImageNet-21K. We categorize progress into a few key methodologies: trajectory matching, gradient matching, distribution matching, scalable generative approaches, and decoupling optimization mechanisms. As a comprehensive examination of recent dataset distillation advances, this survey highlights breakthrough innovations: the SRe2L framework for efficient and effective condensation, soft label strategies that significantly enhance model accuracy, and lossless distillation techniques that maximize compression while maintaining performance. Beyond these methodological advancements, we address critical challenges, including robustness against adversarial and backdoor attacks, effective handling of non-IID data distributions. Additionally, we explore emerging applications in video and audio processing, multi-modal learning, medical imaging, and scientific computing, highlighting its domain versatility. By offering extensive performance comparisons and actionable research directions, this survey equips researchers and practitioners with practical insights to advance efficient and generalizable dataset distillation, paving the way for future innovations.



## **2. On Adversarial Attacks In Acoustic Drone Localization**

cs.SD

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.20325v1) [paper-pdf](http://arxiv.org/pdf/2502.20325v1)

**Authors**: Tamir Shor, Chaim Baskin, Alex Bronstein

**Abstract**: Multi-rotor aerial autonomous vehicles (MAVs, more widely known as "drones") have been generating increased interest in recent years due to their growing applicability in a vast and diverse range of fields (e.g., agriculture, commercial delivery, search and rescue). The sensitivity of visual-based methods to lighting conditions and occlusions had prompted growing study of navigation reliant on other modalities, such as acoustic sensing. A major concern in using drones in scale for tasks in non-controlled environments is the potential threat of adversarial attacks over their navigational systems, exposing users to mission-critical failures, security breaches, and compromised safety outcomes that can endanger operators and bystanders. While previous work shows impressive progress in acoustic-based drone localization, prior research in adversarial attacks over drone navigation only addresses visual sensing-based systems. In this work, we aim to compensate for this gap by supplying a comprehensive analysis of the effect of PGD adversarial attacks over acoustic drone localization. We furthermore develop an algorithm for adversarial perturbation recovery, capable of markedly diminishing the affect of such attacks in our setting. The code for reproducing all experiments will be released upon publication.



## **3. Logicbreaks: A Framework for Understanding Subversion of Rule-based Inference**

cs.AI

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2407.00075v4) [paper-pdf](http://arxiv.org/pdf/2407.00075v4)

**Authors**: Anton Xue, Avishree Khare, Rajeev Alur, Surbhi Goel, Eric Wong

**Abstract**: We study how to subvert large language models (LLMs) from following prompt-specified rules. We first formalize rule-following as inference in propositional Horn logic, a mathematical system in which rules have the form "if $P$ and $Q$, then $R$" for some propositions $P$, $Q$, and $R$. Next, we prove that although small transformers can faithfully follow such rules, maliciously crafted prompts can still mislead both theoretical constructions and models learned from data. Furthermore, we demonstrate that popular attack algorithms on LLMs find adversarial prompts and induce attention patterns that align with our theory. Our novel logic-based framework provides a foundation for studying LLMs in rule-based settings, enabling a formal analysis of tasks like logical reasoning and jailbreak attacks.



## **4. Adversarial Robustness in Parameter-Space Classifiers**

cs.LG

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.20314v1) [paper-pdf](http://arxiv.org/pdf/2502.20314v1)

**Authors**: Tamir Shor, Ethan Fetaya, Chaim Baskin, Alex Bronstein

**Abstract**: Implicit Neural Representations (INRs) have been recently garnering increasing interest in various research fields, mainly due to their ability to represent large, complex data in a compact and continuous manner. Past work further showed that numerous popular downstream tasks can be performed directly in the INR parameter-space. Doing so can substantially reduce the computational resources required to process the represented data in their native domain. A major difficulty in using modern machine-learning approaches, is their high susceptibility to adversarial attacks, which have been shown to greatly limit the reliability and applicability of such methods in a wide range of settings. In this work, we show that parameter-space models trained for classification are inherently robust to adversarial attacks -- without the need of any robust training. To support our claims, we develop a novel suite of adversarial attacks targeting parameter-space classifiers, and furthermore analyze practical considerations of attacking parameter-space classifiers. Code for reproducing all experiments and implementation of all proposed methods will be released upon publication.



## **5. SecureGaze: Defending Gaze Estimation Against Backdoor Attacks**

cs.CV

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.20306v1) [paper-pdf](http://arxiv.org/pdf/2502.20306v1)

**Authors**: Lingyu Du, Yupei Liu, Jinyuan Jia, Guohao Lan

**Abstract**: Gaze estimation models are widely used in applications such as driver attention monitoring and human-computer interaction. While many methods for gaze estimation exist, they rely heavily on data-hungry deep learning to achieve high performance. This reliance often forces practitioners to harvest training data from unverified public datasets, outsource model training, or rely on pre-trained models. However, such practices expose gaze estimation models to backdoor attacks. In such attacks, adversaries inject backdoor triggers by poisoning the training data, creating a backdoor vulnerability: the model performs normally with benign inputs, but produces manipulated gaze directions when a specific trigger is present. This compromises the security of many gaze-based applications, such as causing the model to fail in tracking the driver's attention. To date, there is no defense that addresses backdoor attacks on gaze estimation models. In response, we introduce SecureGaze, the first solution designed to protect gaze estimation models from such attacks. Unlike classification models, defending gaze estimation poses unique challenges due to its continuous output space and globally activated backdoor behavior. By identifying distinctive characteristics of backdoored gaze estimation models, we develop a novel and effective approach to reverse-engineer the trigger function for reliable backdoor detection. Extensive evaluations in both digital and physical worlds demonstrate that SecureGaze effectively counters a range of backdoor attacks and outperforms seven state-of-the-art defenses adapted from classification models.



## **6. Voting Scheme to Strengthen Localization Security in Randomly Deployed Wireless Sensor Networks**

eess.SP

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.20218v1) [paper-pdf](http://arxiv.org/pdf/2502.20218v1)

**Authors**: Slavisa Tomic, Marko Beko, Dejan Vukobratovic, Srdjan Krco

**Abstract**: This work aspires to provide a trustworthy solution for target localization in adverse environments, where malicious nodes, capable of manipulating distance measurements (i.e., performing spoofing attacks), are present, thus hindering accurate localization. Besides localization, its other goal is to identify (detect) which of the nodes participating in the process are malicious. This problem becomes extremely important with the forthcoming expansion of IoT and smart cities applications, that depend on accurate localization, and the presence of malicious attackers can represent serious security threats if not taken into consideration. This is the case with most existing localization systems which makes them highly vulnerable to spoofing attacks. In addition, existing methods that are intended for adversarial settings consider very specific settings or require additional knowledge about the system model, making them only partially secure. Therefore, this work proposes a novel voting scheme based on clustering and weighted central mass to securely solve the localization problem and detect attackers. The proposed solution has two main phases: 1) Choosing a cluster of suitable points of interest by taking advantage of the problem geometry to assigning votes in order to localize the target, and 2) Attacker detection by exploiting the location estimate and basic statistics. The proposed method is assessed in terms of localization accuracy, success in attacker detection, and computational complexity in different settings. Computer simulations and real-world experiments corroborate the effectiveness of the proposed scheme compared to state-of-the-art methods, showing that it can accomplish an error reduction of $30~\%$ and is capable of achieving almost perfect attacker detection rate when the ratio between attacker intensity and noise standard deviation is significant.



## **7. Modern DDoS Threats and Countermeasures: Insights into Emerging Attacks and Detection Strategies**

cs.CR

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.19996v1) [paper-pdf](http://arxiv.org/pdf/2502.19996v1)

**Authors**: Jincheng Wang, Le Yu, John C. S. Lui, Xiapu Luo

**Abstract**: Distributed Denial of Service (DDoS) attacks persist as significant threats to online services and infrastructure, evolving rapidly in sophistication and eluding traditional detection mechanisms. This evolution demands a comprehensive examination of current trends in DDoS attacks and the efficacy of modern detection strategies. This paper offers an comprehensive survey of emerging DDoS attacks and detection strategies over the past decade. We delve into the diversification of attack targets, extending beyond conventional web services to include newer network protocols and systems, and the adoption of advanced adversarial tactics. Additionally, we review current detection techniques, highlighting essential features that modern systems must integrate to effectively neutralize these evolving threats. Given the technological demands of contemporary network systems, such as high-volume and in-line packet processing capabilities, we also explore how innovative hardware technologies like programmable switches can significantly enhance the development and deployment of robust DDoS detection systems. We conclude by identifying open problems and proposing future directions for DDoS research. In particular, our survey sheds light on the investigation of DDoS attack surfaces for emerging systems, protocols, and adversarial strategies. Moreover, we outlines critical open questions in the development of effective detection systems, e.g., the creation of defense mechanisms independent of control planes.



## **8. Trustworthy AI-Generative Content for Intelligent Network Service: Robustness, Security, and Fairness**

cs.CR

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2405.05930v2) [paper-pdf](http://arxiv.org/pdf/2405.05930v2)

**Authors**: Siyuan Li, Xi Lin, Yaju Liu, Xiang Chen, Jianhua Li

**Abstract**: AI-generated content (AIGC) models, represented by large language models (LLM), have revolutionized content creation. High-speed next-generation communication technology is an ideal platform for providing powerful AIGC network services. At the same time, advanced AIGC techniques can also make future network services more intelligent, especially various online content generation services. However, the significant untrustworthiness concerns of current AIGC models, such as robustness, security, and fairness, greatly affect the credibility of intelligent network services, especially in ensuring secure AIGC services. This paper proposes TrustGAIN, a trustworthy AIGC framework that incorporates robust, secure, and fair network services. We first discuss the robustness to adversarial attacks faced by AIGC models in network systems and the corresponding protection issues. Subsequently, we emphasize the importance of avoiding unsafe and illegal services and ensuring the fairness of the AIGC network services. Then as a case study, we propose a novel sentiment analysis-based detection method to guide the robust detection of unsafe content in network services. We conduct our experiments on fake news, malicious code, and unsafe review datasets to represent LLM application scenarios. Our results indicate that TrustGAIN is an exploration of future networks that can support trustworthy AIGC network services.



## **9. The Hidden Risks of Large Reasoning Models: A Safety Assessment of R1**

cs.CY

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.12659v3) [paper-pdf](http://arxiv.org/pdf/2502.12659v3)

**Authors**: Kaiwen Zhou, Chengzhi Liu, Xuandong Zhao, Shreedhar Jangam, Jayanth Srinivasa, Gaowen Liu, Dawn Song, Xin Eric Wang

**Abstract**: The rapid development of large reasoning models, such as OpenAI-o3 and DeepSeek-R1, has led to significant improvements in complex reasoning over non-reasoning large language models~(LLMs). However, their enhanced capabilities, combined with the open-source access of models like DeepSeek-R1, raise serious safety concerns, particularly regarding their potential for misuse. In this work, we present a comprehensive safety assessment of these reasoning models, leveraging established safety benchmarks to evaluate their compliance with safety regulations. Furthermore, we investigate their susceptibility to adversarial attacks, such as jailbreaking and prompt injection, to assess their robustness in real-world applications. Through our multi-faceted analysis, we uncover four key findings: (1) There is a significant safety gap between the open-source R1 models and the o3-mini model, on both safety benchmark and attack, suggesting more safety effort on R1 is needed. (2) The distilled reasoning model shows poorer safety performance compared to its safety-aligned base models. (3) The stronger the model's reasoning ability, the greater the potential harm it may cause when answering unsafe questions. (4) The thinking process in R1 models pose greater safety concerns than their final answers. Our study provides insights into the security implications of reasoning models and highlights the need for further advancements in R1 models' safety to close the gap.



## **10. Foot-In-The-Door: A Multi-turn Jailbreak for LLMs**

cs.CL

19 pages, 8 figures

**SubmitDate**: 2025-02-28    [abs](http://arxiv.org/abs/2502.19820v2) [paper-pdf](http://arxiv.org/pdf/2502.19820v2)

**Authors**: Zixuan Weng, Xiaolong Jin, Jinyuan Jia, Xiangyu Zhang

**Abstract**: Ensuring AI safety is crucial as large language models become increasingly integrated into real-world applications. A key challenge is jailbreak, where adversarial prompts bypass built-in safeguards to elicit harmful disallowed outputs. Inspired by psychological foot-in-the-door principles, we introduce FITD,a novel multi-turn jailbreak method that leverages the phenomenon where minor initial commitments lower resistance to more significant or more unethical transgressions. Our approach progressively escalates the malicious intent of user queries through intermediate bridge prompts and aligns the model's response by itself to induce toxic responses. Extensive experimental results on two jailbreak benchmarks demonstrate that FITD achieves an average attack success rate of 94% across seven widely used models, outperforming existing state-of-the-art methods. Additionally, we provide an in-depth analysis of LLM self-corruption, highlighting vulnerabilities in current alignment strategies and emphasizing the risks inherent in multi-turn interactions. The code is available at https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak.



## **11. Hyperparameters in Score-Based Membership Inference Attacks**

cs.LG

This work has been accepted for publication in the 3rd IEEE  Conference on Secure and Trustworthy Machine Learning (SaTML'25). The final  version will be available on IEEE Xplore

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.06374v2) [paper-pdf](http://arxiv.org/pdf/2502.06374v2)

**Authors**: Gauri Pradhan, Joonas Jälkö, Marlon Tobaben, Antti Honkela

**Abstract**: Membership Inference Attacks (MIAs) have emerged as a valuable framework for evaluating privacy leakage by machine learning models. Score-based MIAs are distinguished, in particular, by their ability to exploit the confidence scores that the model generates for particular inputs. Existing score-based MIAs implicitly assume that the adversary has access to the target model's hyperparameters, which can be used to train the shadow models for the attack. In this work, we demonstrate that the knowledge of target hyperparameters is not a prerequisite for MIA in the transfer learning setting. Based on this, we propose a novel approach to select the hyperparameters for training the shadow models for MIA when the attacker has no prior knowledge about them by matching the output distributions of target and shadow models. We demonstrate that using the new approach yields hyperparameters that lead to an attack near indistinguishable in performance from an attack that uses target hyperparameters to train the shadow models. Furthermore, we study the empirical privacy risk of unaccounted use of training data for hyperparameter optimization (HPO) in differentially private (DP) transfer learning. We find no statistically significant evidence that performing HPO using training data would increase vulnerability to MIA.



## **12. Snowball Adversarial Attack on Traffic Sign Classification**

cs.CV

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.19757v1) [paper-pdf](http://arxiv.org/pdf/2502.19757v1)

**Authors**: Anthony Etim, Jakub Szefer

**Abstract**: Adversarial attacks on machine learning models often rely on small, imperceptible perturbations to mislead classifiers. Such strategy focuses on minimizing the visual perturbation for humans so they are not confused, and also maximizing the misclassification for machine learning algorithms. An orthogonal strategy for adversarial attacks is to create perturbations that are clearly visible but do not confuse humans, yet still maximize misclassification for machine learning algorithms. This work follows the later strategy, and demonstrates instance of it through the Snowball Adversarial Attack in the context of traffic sign recognition. The attack leverages the human brain's superior ability to recognize objects despite various occlusions, while machine learning algorithms are easily confused. The evaluation shows that the Snowball Adversarial Attack is robust across various images and is able to confuse state-of-the-art traffic sign recognition algorithm. The findings reveal that Snowball Adversarial Attack can significantly degrade model performance with minimal effort, raising important concerns about the vulnerabilities of deep neural networks and highlighting the necessity for improved defenses for image recognition machine learning models.



## **13. HALO: Robust Out-of-Distribution Detection via Joint Optimisation**

cs.LG

SaTML 2025

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.19755v1) [paper-pdf](http://arxiv.org/pdf/2502.19755v1)

**Authors**: Hugo Lyons Keenan, Sarah Erfani, Christopher Leckie

**Abstract**: Effective out-of-distribution (OOD) detection is crucial for the safe deployment of machine learning models in real-world scenarios. However, recent work has shown that OOD detection methods are vulnerable to adversarial attacks, potentially leading to critical failures in high-stakes applications. This discovery has motivated work on robust OOD detection methods that are capable of maintaining performance under various attack settings. Prior approaches have made progress on this problem but face a number of limitations: often only exhibiting robustness to attacks on OOD data or failing to maintain strong clean performance. In this work, we adapt an existing robust classification framework, TRADES, extending it to the problem of robust OOD detection and discovering a novel objective function. Recognising the critical importance of a strong clean/robust trade-off for OOD detection, we introduce an additional loss term which boosts classification and detection performance. Our approach, called HALO (Helper-based AdversariaL OOD detection), surpasses existing methods and achieves state-of-the-art performance across a number of datasets and attack settings. Extensive experiments demonstrate an average AUROC improvement of 3.15 in clean settings and 7.07 under adversarial attacks when compared to the next best method. Furthermore, HALO exhibits resistance to transferred attacks, offers tuneable performance through hyperparameter selection, and is compatible with existing OOD detection frameworks out-of-the-box, leaving open the possibility of future performance gains. Code is available at: https://github.com/hugo0076/HALO



## **14. SAP-DIFF: Semantic Adversarial Patch Generation for Black-Box Face Recognition Models via Diffusion Models**

cs.CV

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.19710v1) [paper-pdf](http://arxiv.org/pdf/2502.19710v1)

**Authors**: Mingsi Wang, Shuaiyin Yao, Chang Yue, Lijie Zhang, Guozhu Meng

**Abstract**: Given the need to evaluate the robustness of face recognition (FR) models, many efforts have focused on adversarial patch attacks that mislead FR models by introducing localized perturbations. Impersonation attacks are a significant threat because adversarial perturbations allow attackers to disguise themselves as legitimate users. This can lead to severe consequences, including data breaches, system damage, and misuse of resources. However, research on such attacks in FR remains limited. Existing adversarial patch generation methods exhibit limited efficacy in impersonation attacks due to (1) the need for high attacker capabilities, (2) low attack success rates, and (3) excessive query requirements. To address these challenges, we propose a novel method SAP-DIFF that leverages diffusion models to generate adversarial patches via semantic perturbations in the latent space rather than direct pixel manipulation. We introduce an attention disruption mechanism to generate features unrelated to the original face, facilitating the creation of adversarial samples and a directional loss function to guide perturbations toward the target identity feature space, thereby enhancing attack effectiveness and efficiency. Extensive experiments on popular FR models and datasets demonstrate that our method outperforms state-of-the-art approaches, achieving an average attack success rate improvement of 45.66% (all exceeding 40%), and a reduction in the number of queries by about 40% compared to the SOTA approach



## **15. LiD-FL: Towards List-Decodable Federated Learning**

cs.LG

26 pages, 5 figures

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2408.04963v3) [paper-pdf](http://arxiv.org/pdf/2408.04963v3)

**Authors**: Hong Liu, Liren Shan, Han Bao, Ronghui You, Yuhao Yi, Jiancheng Lv

**Abstract**: Federated learning is often used in environments with many unverified participants. Therefore, federated learning under adversarial attacks receives significant attention. This paper proposes an algorithmic framework for list-decodable federated learning, where a central server maintains a list of models, with at least one guaranteed to perform well. The framework has no strict restriction on the fraction of honest workers, extending the applicability of Byzantine federated learning to the scenario with more than half adversaries. Under proper assumptions on the loss function, we prove a convergence theorem for our method. Experimental results, including image classification tasks with both convex and non-convex losses, demonstrate that the proposed algorithm can withstand the malicious majority under various attacks.



## **16. Prompt-driven Transferable Adversarial Attack on Person Re-Identification with Attribute-aware Textual Inversion**

cs.CV

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.19697v1) [paper-pdf](http://arxiv.org/pdf/2502.19697v1)

**Authors**: Yuan Bian, Min Liu, Yunqi Yi, Xueping Wang, Yaonan Wang

**Abstract**: Person re-identification (re-id) models are vital in security surveillance systems, requiring transferable adversarial attacks to explore the vulnerabilities of them. Recently, vision-language models (VLM) based attacks have shown superior transferability by attacking generalized image and textual features of VLM, but they lack comprehensive feature disruption due to the overemphasis on discriminative semantics in integral representation. In this paper, we introduce the Attribute-aware Prompt Attack (AP-Attack), a novel method that leverages VLM's image-text alignment capability to explicitly disrupt fine-grained semantic features of pedestrian images by destroying attribute-specific textual embeddings. To obtain personalized textual descriptions for individual attributes, textual inversion networks are designed to map pedestrian images to pseudo tokens that represent semantic embeddings, trained in the contrastive learning manner with images and a predefined prompt template that explicitly describes the pedestrian attributes. Inverted benign and adversarial fine-grained textual semantics facilitate attacker in effectively conducting thorough disruptions, enhancing the transferability of adversarial examples. Extensive experiments show that AP-Attack achieves state-of-the-art transferability, significantly outperforming previous methods by 22.9% on mean Drop Rate in cross-model&dataset attack scenarios.



## **17. MAA: Meticulous Adversarial Attack against Vision-Language Pre-trained Models**

cs.CV

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.08079v2) [paper-pdf](http://arxiv.org/pdf/2502.08079v2)

**Authors**: Peng-Fei Zhang, Guangdong Bai, Zi Huang

**Abstract**: Current adversarial attacks for evaluating the robustness of vision-language pre-trained (VLP) models in multi-modal tasks suffer from limited transferability, where attacks crafted for a specific model often struggle to generalize effectively across different models, limiting their utility in assessing robustness more broadly. This is mainly attributed to the over-reliance on model-specific features and regions, particularly in the image modality. In this paper, we propose an elegant yet highly effective method termed Meticulous Adversarial Attack (MAA) to fully exploit model-independent characteristics and vulnerabilities of individual samples, achieving enhanced generalizability and reduced model dependence. MAA emphasizes fine-grained optimization of adversarial images by developing a novel resizing and sliding crop (RScrop) technique, incorporating a multi-granularity similarity disruption (MGSD) strategy. Extensive experiments across diverse VLP models, multiple benchmark datasets, and a variety of downstream tasks demonstrate that MAA significantly enhances the effectiveness and transferability of adversarial attacks. A large cohort of performance studies is conducted to generate insights into the effectiveness of various model configurations, guiding future advancements in this domain.



## **18. Improving Adversarial Transferability in MLLMs via Dynamic Vision-Language Alignment Attack**

cs.CV

arXiv admin note: text overlap with arXiv:2403.09766

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2502.19672v1) [paper-pdf](http://arxiv.org/pdf/2502.19672v1)

**Authors**: Chenhe Gu, Jindong Gu, Andong Hua, Yao Qin

**Abstract**: Multimodal Large Language Models (MLLMs), built upon LLMs, have recently gained attention for their capabilities in image recognition and understanding. However, while MLLMs are vulnerable to adversarial attacks, the transferability of these attacks across different models remains limited, especially under targeted attack setting. Existing methods primarily focus on vision-specific perturbations but struggle with the complex nature of vision-language modality alignment. In this work, we introduce the Dynamic Vision-Language Alignment (DynVLA) Attack, a novel approach that injects dynamic perturbations into the vision-language connector to enhance generalization across diverse vision-language alignment of different models. Our experimental results show that DynVLA significantly improves the transferability of adversarial examples across various MLLMs, including BLIP2, InstructBLIP, MiniGPT4, LLaVA, and closed-source models such as Gemini.



## **19. Algebraic Adversarial Attacks on Integrated Gradients**

cs.LG

To appear in the proceedings of the International Conference on  Machine Learning and Cybernetics (ICMLC)

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2407.16233v2) [paper-pdf](http://arxiv.org/pdf/2407.16233v2)

**Authors**: Lachlan Simpson, Federico Costanza, Kyle Millar, Adriel Cheng, Cheng-Chew Lim, Hong Gunn Chew

**Abstract**: Adversarial attacks on explainability models have drastic consequences when explanations are used to understand the reasoning of neural networks in safety critical systems. Path methods are one such class of attribution methods susceptible to adversarial attacks. Adversarial learning is typically phrased as a constrained optimisation problem. In this work, we propose algebraic adversarial examples and study the conditions under which one can generate adversarial examples for integrated gradients. Algebraic adversarial examples provide a mathematically tractable approach to adversarial examples.



## **20. Fast Preemption: Forward-Backward Cascade Learning for Efficient and Transferable Proactive Adversarial Defense**

cs.CR

**SubmitDate**: 2025-02-27    [abs](http://arxiv.org/abs/2407.15524v6) [paper-pdf](http://arxiv.org/pdf/2407.15524v6)

**Authors**: Hanrui Wang, Ching-Chun Chang, Chun-Shien Lu, Isao Echizen

**Abstract**: Deep learning has advanced significantly but remains vulnerable to adversarial attacks, compromising its reliability. While conventional defenses typically mitigate perturbations post-attack, few studies explore proactive strategies for preemptive protection. This paper proposes Fast Preemption, a novel defense that neutralizes adversarial effects before third-party attacks. By employing distinct models for input labeling and feature extraction, Fast Preemption enables an efficient, transferable preemptive defense with state-of-the-art robustness across diverse systems. To further enhance efficiency, we introduce a forward-backward cascade learning algorithm that generates protective perturbations, leveraging forward propagation for rapid convergence and iterative backward propagation to mitigate overfitting. Executing in just three iterations, Fast Preemption surpasses existing training-time, test-time, and preemptive defenses. Additionally, we propose the first effective white-box adaptive reversion attack to evaluate the reversibility of preemptive defenses, demonstrating that our approach remains secure unless the backbone model, algorithm, and settings are entirely compromised. This work establishes a new paradigm for proactive adversarial defense.



## **21. EMPRA: Embedding Perturbation Rank Attack against Neural Ranking Models**

cs.IR

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2412.16382v2) [paper-pdf](http://arxiv.org/pdf/2412.16382v2)

**Authors**: Amin Bigdeli, Negar Arabzadeh, Ebrahim Bagheri, Charles L. A. Clarke

**Abstract**: Recent research has shown that neural information retrieval techniques may be susceptible to adversarial attacks. Adversarial attacks seek to manipulate the ranking of documents, with the intention of exposing users to targeted content. In this paper, we introduce the Embedding Perturbation Rank Attack (EMPRA) method, a novel approach designed to perform adversarial attacks on black-box Neural Ranking Models (NRMs). EMPRA manipulates sentence-level embeddings, guiding them towards pertinent context related to the query while preserving semantic integrity. This process generates adversarial texts that seamlessly integrate with the original content and remain imperceptible to humans. Our extensive evaluation conducted on the widely-used MS MARCO V1 passage collection demonstrate the effectiveness of EMPRA against a wide range of state-of-the-art baselines in promoting a specific set of target documents within a given ranked results. Specifically, EMPRA successfully achieves a re-ranking of almost 96% of target documents originally ranked between 51-100 to rank within the top 10. Furthermore, EMPRA does not depend on surrogate models for adversarial text generation, enhancing its robustness against different NRMs in realistic settings.



## **22. Prevailing against Adversarial Noncentral Disturbances: Exact Recovery of Linear Systems with the $l_1$-norm Estimator**

math.OC

8 pages, 2 figures

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2410.03218v4) [paper-pdf](http://arxiv.org/pdf/2410.03218v4)

**Authors**: Jihun Kim, Javad Lavaei

**Abstract**: This paper studies the linear system identification problem in the general case where the disturbance is sub-Gaussian, correlated, and possibly adversarial. First, we consider the case with noncentral (nonzero-mean) disturbances for which the ordinary least-squares (OLS) method fails to correctly identify the system. We prove that the $l_1$-norm estimator accurately identifies the system under the condition that each disturbance has equal probabilities of being positive or negative. This condition restricts the sign of each disturbance but allows its magnitude to be arbitrary. Second, we consider the case where each disturbance is adversarial with the model that the attack times happen occasionally but the distributions of the attack values are arbitrary. We show that when the probability of having an attack at a given time is less than 0.5 and each attack spans the entire space in expectation, the $l_1$-norm estimator prevails against any adversarial noncentral disturbances and the exact recovery is achieved within a finite time. These results pave the way to effectively defend against arbitrarily large noncentral attacks in safety-critical systems.



## **23. No, of course I can! Refusal Mechanisms Can Be Exploited Using Harmless Fine-Tuning Data**

cs.CR

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.19537v1) [paper-pdf](http://arxiv.org/pdf/2502.19537v1)

**Authors**: Joshua Kazdan, Lisa Yu, Rylan Schaeffer, Chris Cundy, Sanmi Koyejo, Dvijotham Krishnamurthy

**Abstract**: Leading language model (LM) providers like OpenAI and Google offer fine-tuning APIs that allow customers to adapt LMs for specific use cases. To prevent misuse, these LM providers implement filtering mechanisms to block harmful fine-tuning data. Consequently, adversaries seeking to produce unsafe LMs via these APIs must craft adversarial training data that are not identifiably harmful. We make three contributions in this context: 1. We show that many existing attacks that use harmless data to create unsafe LMs rely on eliminating model refusals in the first few tokens of their responses. 2. We show that such prior attacks can be blocked by a simple defense that pre-fills the first few tokens from an aligned model before letting the fine-tuned model fill in the rest. 3. We describe a new data-poisoning attack, ``No, Of course I Can Execute'' (NOICE), which exploits an LM's formulaic refusal mechanism to elicit harmful responses. By training an LM to refuse benign requests on the basis of safety before fulfilling those requests regardless, we are able to jailbreak several open-source models and a closed-source model (GPT-4o). We show an attack success rate (ASR) of 57% against GPT-4o; our attack earned a Bug Bounty from OpenAI. Against open-source models protected by simple defenses, we improve ASRs by an average of 3.25 times compared to the best performing previous attacks that use only harmless data. NOICE demonstrates the exploitability of repetitive refusal mechanisms and broadens understanding of the threats closed-source models face from harmless data.



## **24. Unveiling Wireless Users' Locations via Modulation Classification-based Passive Attack**

cs.IT

7 pages, 4 figures, submitted to IEEE for possible publication

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.19341v1) [paper-pdf](http://arxiv.org/pdf/2502.19341v1)

**Authors**: Ali Hanif, Abdulrahman Katranji, Nour Kouzayha, Muhammad Mahboob Ur Rahman, Tareq Y. Al-Naffouri

**Abstract**: The broadcast nature of the wireless medium and openness of wireless standards, e.g., 3GPP releases 16-20, invite adversaries to launch various active and passive attacks on cellular and other wireless networks. This work identifies one such loose end of wireless standards and presents a novel passive attack method enabling an eavesdropper (Eve) to localize a line of sight wireless user (Bob) who is communicating with a base station or WiFi access point (Alice). The proposed attack involves two phases. In the first phase, Eve performs modulation classification by intercepting the downlink channel between Alice and Bob. This enables Eve to utilize the publicly available modulation and coding scheme (MCS) tables to do pesudo-ranging, i.e., the Eve determines the ring within which Bob is located, which drastically reduces the search space. In the second phase, Eve sniffs the uplink channel, and employs multiple strategies to further refine Bob's location within the ring. Towards the end, we present our thoughts on how this attack can be extended to non-line-of-sight scenarios, and how this attack could act as a scaffolding to construct a malicious digital twin map.



## **25. Extreme vulnerability to intruder attacks destabilizes network dynamics**

nlin.AO

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.08552v2) [paper-pdf](http://arxiv.org/pdf/2502.08552v2)

**Authors**: Amirhossein Nazerian, Sahand Tangerami, Malbor Asllani, David Phillips, Hernan Makse, Francesco Sorrentino

**Abstract**: Consensus, synchronization, formation control, and power grid balance are all examples of virtuous dynamical states that may arise in networks. Here we focus on how such states can be destabilized from a fundamental perspective; namely, we address the question of how one or a few intruder agents within an otherwise functioning network may compromise its dynamics. We show that a single adversarial node coupled via adversarial couplings to one or more other nodes is sufficient to destabilize the entire network, which we prove to be more efficient than targeting multiple nodes. Then, we show that concentrating the attack on a single low-indegree node induces the greatest instability, challenging the common assumption that hubs are the most critical nodes. This leads to a new characterization of the vulnerability of a node, which contrasts with previous work, and identifies low-indegree nodes (as opposed to the hubs) as the most vulnerable components of a network. Our results are derived for linear systems but hold true for nonlinear networks, including those described by the Kuramoto model. Finally, we derive scaling laws showing that larger networks are less susceptible, on average, to single-node attacks. Overall, these findings highlight an intrinsic vulnerability of technological systems such as autonomous networks, sensor networks, power grids, and the internet of things, with implications also to the realm of complex social and biological networks.



## **26. On the Byzantine Fault Tolerance of signSGD with Majority Vote**

cs.LG

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.19170v1) [paper-pdf](http://arxiv.org/pdf/2502.19170v1)

**Authors**: Emanuele Mengoli, Luzius Moll, Virgilio Strozzi, El-Mahdi El-Mhamdi

**Abstract**: In distributed learning, sign-based compression algorithms such as signSGD with majority vote provide a lightweight alternative to SGD with an additional advantage: fault tolerance (almost) for free. However, for signSGD with majority vote, this fault tolerance has been shown to cover only the case of weaker adversaries, i.e., ones that are not omniscient or cannot collude to base their attack on common knowledge and strategy. In this work, we close this gap and provide new insights into how signSGD with majority vote can be resilient against omniscient and colluding adversaries, which craft an attack after communicating with other adversaries, thus having better information to perform the most damaging attack based on a common optimal strategy. Our core contribution is in providing a proof that begins by defining the omniscience framework and the strongest possible damage against signSGD with majority vote without imposing any restrictions on the attacker. Thanks to the filtering effect of the sign-based method, we upper-bound the space of attacks to the optimal strategy for maximizing damage by an attacker. Hence, we derive an explicit probabilistic bound in terms of incorrect aggregation without resorting to unknown constants, providing a convergence bound on signSGD with majority vote in the presence of Byzantine attackers, along with a precise convergence rate. Our findings are supported by experiments on the MNIST dataset in a distributed learning environment with adversaries of varying strength.



## **27. XSS Adversarial Attacks Based on Deep Reinforcement Learning: A Replication and Extension Study**

cs.SE

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.19095v1) [paper-pdf](http://arxiv.org/pdf/2502.19095v1)

**Authors**: Samuele Pasini, Gianluca Maragliano, Jinhan Kim, Paolo Tonella

**Abstract**: Cross-site scripting (XSS) poses a significant threat to web application security. While Deep Learning (DL) has shown remarkable success in detecting XSS attacks, it remains vulnerable to adversarial attacks due to the discontinuous nature of its input-output mapping. These adversarial attacks employ mutation-based strategies for different components of XSS attack vectors, allowing adversarial agents to iteratively select mutations to evade detection. Our work replicates a state-of-the-art XSS adversarial attack, highlighting threats to validity in the reference work and extending it toward a more effective evaluation strategy. Moreover, we introduce an XSS Oracle to mitigate these threats. The experimental results show that our approach achieves an escape rate above 96% when the threats to validity of the replicated technique are addressed.



## **28. Beyond Surface-Level Patterns: An Essence-Driven Defense Framework Against Jailbreak Attacks in LLMs**

cs.CR

15 pages, 12 figures

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.19041v1) [paper-pdf](http://arxiv.org/pdf/2502.19041v1)

**Authors**: Shiyu Xiang, Ansen Zhang, Yanfei Cao, Yang Fan, Ronghao Chen

**Abstract**: Although Aligned Large Language Models (LLMs) are trained to refuse harmful requests, they remain vulnerable to jailbreak attacks. Unfortunately, existing methods often focus on surface-level patterns, overlooking the deeper attack essences. As a result, defenses fail when attack prompts change, even though the underlying "attack essence" remains the same. To address this issue, we introduce EDDF, an \textbf{E}ssence-\textbf{D}riven \textbf{D}efense \textbf{F}ramework Against Jailbreak Attacks in LLMs. EDDF is a plug-and-play input-filtering method and operates in two stages: 1) offline essence database construction, and 2) online adversarial query detection. The key idea behind EDDF is to extract the "attack essence" from a diverse set of known attack instances and store it in an offline vector database. Experimental results demonstrate that EDDF significantly outperforms existing methods by reducing the Attack Success Rate by at least 20\%, underscoring its superior robustness against jailbreak attacks.



## **29. Robust Over-the-Air Computation with Type-Based Multiple Access**

eess.SP

Paper submitted to 33rd European Signal Processing Conference  (EUSIPCO 2025)

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.19014v1) [paper-pdf](http://arxiv.org/pdf/2502.19014v1)

**Authors**: Marc Martinez-Gost, Ana Pérez-Neira, Miguel Ángel Lagunas

**Abstract**: This paper utilizes the properties of type-based multiple access (TBMA) to investigate its effectiveness as a robust approach for over-the-air computation (AirComp) in the presence of Byzantine attacks, this is, adversarial strategies where malicious nodes intentionally distort their transmissions to corrupt the aggregated result. Unlike classical direct aggregation (DA) AirComp, which aggregates data in the amplitude of the signals and are highly vulnerable to attacks, TBMA distributes data over multiple radio resources, enabling the receiver to construct a histogram representation of the transmitted data. This structure allows the integration of classical robust estimators and supports the computation of diverse functions beyond the arithmetic mean, which is not feasible with DA. Through extensive simulations, we demonstrate that robust TBMA significantly outperforms DA, maintaining high accuracy even under adversarial conditions, and showcases its applicability in federated learning (FEEL) scenarios. Additionally, TBMA reduces channel state information (CSI) requirements, lowers energy consumption, and enhances resiliency by leveraging the diversity of the transmitted data. These results establish TBMA as a scalable and robust solution for AirComp, paving the way for secure and efficient aggregation in next-generation networks.



## **30. Learning atomic forces from uncertainty-calibrated adversarial attacks**

physics.comp-ph

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.18314v2) [paper-pdf](http://arxiv.org/pdf/2502.18314v2)

**Authors**: Henrique Musseli Cezar, Tilmann Bodenstein, Henrik Andersen Sveinsson, Morten Ledum, Simen Reine, Sigbjørn Løland Bore

**Abstract**: Adversarial approaches, which intentionally challenge machine learning models by generating difficult examples, are increasingly being adopted to improve machine learning interatomic potentials (MLIPs). While already providing great practical value, little is known about the actual prediction errors of MLIPs on adversarial structures and whether these errors can be controlled. We propose the Calibrated Adversarial Geometry Optimization (CAGO) algorithm to discover adversarial structures with user-assigned errors. Through uncertainty calibration, the estimated uncertainty of MLIPs is unified with real errors. By performing geometry optimization for calibrated uncertainty, we reach adversarial structures with the user-assigned target MLIP prediction error. Integrating with active learning pipelines, we benchmark CAGO, demonstrating stable MLIPs that systematically converge structural, dynamical, and thermodynamical properties for liquid water and water adsorption in a metal-organic framework within only hundreds of training structures, where previously many thousands were typically required.



## **31. Towards Label-Only Membership Inference Attack against Pre-trained Large Language Models**

cs.CR

Accepted by USENIX Security 2025

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.18943v1) [paper-pdf](http://arxiv.org/pdf/2502.18943v1)

**Authors**: Yu He, Boheng Li, Liu Liu, Zhongjie Ba, Wei Dong, Yiming Li, Zhan Qin, Kui Ren, Chun Chen

**Abstract**: Membership Inference Attacks (MIAs) aim to predict whether a data sample belongs to the model's training set or not. Although prior research has extensively explored MIAs in Large Language Models (LLMs), they typically require accessing to complete output logits (\ie, \textit{logits-based attacks}), which are usually not available in practice. In this paper, we study the vulnerability of pre-trained LLMs to MIAs in the \textit{label-only setting}, where the adversary can only access generated tokens (text). We first reveal that existing label-only MIAs have minor effects in attacking pre-trained LLMs, although they are highly effective in inferring fine-tuning datasets used for personalized LLMs. We find that their failure stems from two main reasons, including better generalization and overly coarse perturbation. Specifically, due to the extensive pre-training corpora and exposing each sample only a few times, LLMs exhibit minimal robustness differences between members and non-members. This makes token-level perturbations too coarse to capture such differences.   To alleviate these problems, we propose \textbf{PETAL}: a label-only membership inference attack based on \textbf{PE}r-\textbf{T}oken sem\textbf{A}ntic simi\textbf{L}arity. Specifically, PETAL leverages token-level semantic similarity to approximate output probabilities and subsequently calculate the perplexity. It finally exposes membership based on the common assumption that members are `better' memorized and have smaller perplexity. We conduct extensive experiments on the WikiMIA benchmark and the more challenging MIMIR benchmark. Empirically, our PETAL performs better than the extensions of existing label-only attacks against personalized LLMs and even on par with other advanced logit-based attacks across all metrics on five prevalent open-source LLMs.



## **32. Adversarial Universal Stickers: Universal Perturbation Attacks on Traffic Sign using Stickers**

cs.CV

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2502.18724v1) [paper-pdf](http://arxiv.org/pdf/2502.18724v1)

**Authors**: Anthony Etim, Jakub Szefer

**Abstract**: Adversarial attacks on deep learning models have proliferated in recent years. In many cases, a different adversarial perturbation is required to be added to each image to cause the deep learning model to misclassify it. This is ineffective as each image has to be modified in a different way. Meanwhile, research on universal perturbations focuses on designing a single perturbation that can be applied to all images in a data set, and cause a deep learning model to misclassify the images. This work advances the field of universal perturbations by exploring universal perturbations in the context of traffic signs and autonomous vehicle systems. This work introduces a novel method for generating universal perturbations that visually look like simple black and white stickers, and using them to cause incorrect street sign predictions. Unlike traditional adversarial perturbations, the adversarial universal stickers are designed to be applicable to any street sign: same sticker, or stickers, can be applied in same location to any street sign and cause it to be misclassified. Further, to enable safe experimentation with adversarial images and street signs, this work presents a virtual setting that leverages Street View images of street signs, rather than the need to physically modify street signs, to test the attacks. The experiments in the virtual setting demonstrate that these stickers can consistently mislead deep learning models used commonly in street sign recognition, and achieve high attack success rates on dataset of US traffic signs. The findings highlight the practical security risks posed by simple stickers applied to traffic signs, and the ease with which adversaries can generate adversarial universal stickers that can be applied to many street signs.



## **33. Time Traveling to Defend Against Adversarial Example Attacks in Image Classification**

cs.CR

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2410.08338v2) [paper-pdf](http://arxiv.org/pdf/2410.08338v2)

**Authors**: Anthony Etim, Jakub Szefer

**Abstract**: Adversarial example attacks have emerged as a critical threat to machine learning. Adversarial attacks in image classification abuse various, minor modifications to the image that confuse the image classification neural network -- while the image still remains recognizable to humans. One important domain where the attacks have been applied is in the automotive setting with traffic sign classification. Researchers have demonstrated that adding stickers, shining light, or adding shadows are all different means to make machine learning inference algorithms mis-classify the traffic signs. This can cause potentially dangerous situations as a stop sign is recognized as a speed limit sign causing vehicles to ignore it and potentially leading to accidents. To address these attacks, this work focuses on enhancing defenses against such adversarial attacks. This work shifts the advantage to the user by introducing the idea of leveraging historical images and majority voting. While the attacker modifies a traffic sign that is currently being processed by the victim's machine learning inference, the victim can gain advantage by examining past images of the same traffic sign. This work introduces the notion of ''time traveling'' and uses historical Street View images accessible to anybody to perform inference on different, past versions of the same traffic sign. In the evaluation, the proposed defense has 100% effectiveness against latest adversarial example attack on traffic sign classification algorithm.



## **34. Fall Leaf Adversarial Attack on Traffic Sign Classification**

cs.CV

**SubmitDate**: 2025-02-26    [abs](http://arxiv.org/abs/2411.18776v2) [paper-pdf](http://arxiv.org/pdf/2411.18776v2)

**Authors**: Anthony Etim, Jakub Szefer

**Abstract**: Adversarial input image perturbation attacks have emerged as a significant threat to machine learning algorithms, particularly in image classification setting. These attacks involve subtle perturbations to input images that cause neural networks to misclassify the input images, even though the images remain easily recognizable to humans. One critical area where adversarial attacks have been demonstrated is in automotive systems where traffic sign classification and recognition is critical, and where misclassified images can cause autonomous systems to take wrong actions. This work presents a new class of adversarial attacks. Unlike existing work that has focused on adversarial perturbations that leverage human-made artifacts to cause the perturbations, such as adding stickers, paint, or shining flashlights at traffic signs, this work leverages nature-made artifacts: tree leaves. By leveraging nature-made artifacts, the new class of attacks has plausible deniability: a fall leaf stuck to a street sign could come from a near-by tree, rather than be placed there by an malicious human attacker. To evaluate the new class of the adversarial input image perturbation attacks, this work analyses how fall leaves can cause misclassification in street signs. The work evaluates various leaves from different species of trees, and considers various parameters such as size, color due to tree leaf type, and rotation. The work demonstrates high success rate for misclassification. The work also explores the correlation between successful attacks and how they affect the edge detection, which is critical in many image classification algorithms.



## **35. Toward Breaking Watermarks in Distortion-free Large Language Models**

cs.CR

5 pages, AAAI'25 Workshop on Preventing and Detecting LLM Generated  Misinformation

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.18608v1) [paper-pdf](http://arxiv.org/pdf/2502.18608v1)

**Authors**: Shayleen Reynolds, Saheed Obitayo, Niccolò Dalmasso, Dung Daniel T. Ngo, Vamsi K. Potluru, Manuela Veloso

**Abstract**: In recent years, LLM watermarking has emerged as an attractive safeguard against AI-generated content, with promising applications in many real-world domains. However, there are growing concerns that the current LLM watermarking schemes are vulnerable to expert adversaries wishing to reverse-engineer the watermarking mechanisms. Prior work in "breaking" or "stealing" LLM watermarks mainly focuses on the distribution-modifying algorithm of Kirchenbauer et al. (2023), which perturbs the logit vector before sampling. In this work, we focus on reverse-engineering the other prominent LLM watermarking scheme, distortion-free watermarking (Kuditipudi et al. 2024), which preserves the underlying token distribution by using a hidden watermarking key sequence. We demonstrate that, even under a more sophisticated watermarking scheme, it is possible to "compromise" the LLM and carry out a "spoofing" attack. Specifically, we propose a mixed integer linear programming framework that accurately estimates the secret key used for watermarking using only a few samples of the watermarked dataset. Our initial findings challenge the current theoretical claims on the robustness and usability of existing LLM watermarking techniques.



## **36. Topic-FlipRAG: Topic-Orientated Adversarial Opinion Manipulation Attacks to Retrieval-Augmented Generation Models**

cs.CL

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.01386v2) [paper-pdf](http://arxiv.org/pdf/2502.01386v2)

**Authors**: Yuyang Gong, Zhuo Chen, Miaokun Chen, Fengchang Yu, Wei Lu, Xiaofeng Wang, Xiaozhong Liu, Jiawei Liu

**Abstract**: Retrieval-Augmented Generation (RAG) systems based on Large Language Models (LLMs) have become essential for tasks such as question answering and content generation. However, their increasing impact on public opinion and information dissemination has made them a critical focus for security research due to inherent vulnerabilities. Previous studies have predominantly addressed attacks targeting factual or single-query manipulations. In this paper, we address a more practical scenario: topic-oriented adversarial opinion manipulation attacks on RAG models, where LLMs are required to reason and synthesize multiple perspectives, rendering them particularly susceptible to systematic knowledge poisoning. Specifically, we propose Topic-FlipRAG, a two-stage manipulation attack pipeline that strategically crafts adversarial perturbations to influence opinions across related queries. This approach combines traditional adversarial ranking attack techniques and leverages the extensive internal relevant knowledge and reasoning capabilities of LLMs to execute semantic-level perturbations. Experiments show that the proposed attacks effectively shift the opinion of the model's outputs on specific topics, significantly impacting user information perception. Current mitigation methods cannot effectively defend against such attacks, highlighting the necessity for enhanced safeguards for RAG systems, and offering crucial insights for LLM security research.



## **37. CLIPure: Purification in Latent Space via CLIP for Adversarially Robust Zero-Shot Classification**

cs.CV

accepted by ICLR 2025

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.18176v1) [paper-pdf](http://arxiv.org/pdf/2502.18176v1)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: In this paper, we aim to build an adversarially robust zero-shot image classifier. We ground our work on CLIP, a vision-language pre-trained encoder model that can perform zero-shot classification by matching an image with text prompts ``a photo of a <class-name>.''. Purification is the path we choose since it does not require adversarial training on specific attack types and thus can cope with any foreseen attacks. We then formulate purification risk as the KL divergence between the joint distributions of the purification process of denoising the adversarial samples and the attack process of adding perturbations to benign samples, through bidirectional Stochastic Differential Equations (SDEs). The final derived results inspire us to explore purification in the multi-modal latent space of CLIP. We propose two variants for our CLIPure approach: CLIPure-Diff which models the likelihood of images' latent vectors with the DiffusionPrior module in DaLLE-2 (modeling the generation process of CLIP's latent vectors), and CLIPure-Cos which models the likelihood with the cosine similarity between the embeddings of an image and ``a photo of a.''. As far as we know, CLIPure is the first purification method in multi-modal latent space and CLIPure-Cos is the first purification method that is not based on generative models, which substantially improves defense efficiency. We conducted extensive experiments on CIFAR-10, ImageNet, and 13 datasets that previous CLIP-based defense methods used for evaluating zero-shot classification robustness. Results show that CLIPure boosts the SOTA robustness by a large margin, e.g., from 71.7% to 91.1% on CIFAR10, from 59.6% to 72.6% on ImageNet, and 108% relative improvements of average robustness on the 13 datasets over previous SOTA. The code is available at https://github.com/TMLResearchGroup-CAS/CLIPure.



## **38. Exploring the Robustness and Transferability of Patch-Based Adversarial Attacks in Quantized Neural Networks**

cs.CR

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2411.15246v2) [paper-pdf](http://arxiv.org/pdf/2411.15246v2)

**Authors**: Amira Guesmi, Bassem Ouni, Muhammad Shafique

**Abstract**: Quantized neural networks (QNNs) are increasingly used for efficient deployment of deep learning models on resource-constrained platforms, such as mobile devices and edge computing systems. While quantization reduces model size and computational demands, its impact on adversarial robustness-especially against patch-based attacks-remains inadequately addressed. Patch-based attacks, characterized by localized, high-visibility perturbations, pose significant security risks due to their transferability and resilience. In this study, we systematically evaluate the vulnerability of QNNs to patch-based adversarial attacks across various quantization levels and architectures, focusing on factors that contribute to the robustness of these attacks. Through experiments analyzing feature representations, quantization strength, gradient alignment, and spatial sensitivity, we find that patch attacks consistently achieve high success rates across bitwidths and architectures, demonstrating significant transferability even in heavily quantized models. Contrary to the expectation that quantization might enhance adversarial defenses, our results show that QNNs remain highly susceptible to patch attacks due to the persistence of distinct, localized features within quantized representations. These findings underscore the need for quantization-aware defenses that address the specific challenges posed by patch-based attacks. Our work contributes to a deeper understanding of adversarial robustness in QNNs and aims to guide future research in developing secure, quantization-compatible defenses for real-world applications.



## **39. Towards Robust and Secure Embodied AI: A Survey on Vulnerabilities and Attacks**

cs.CR

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.13175v2) [paper-pdf](http://arxiv.org/pdf/2502.13175v2)

**Authors**: Wenpeng Xing, Minghao Li, Mohan Li, Meng Han

**Abstract**: Embodied AI systems, including robots and autonomous vehicles, are increasingly integrated into real-world applications, where they encounter a range of vulnerabilities stemming from both environmental and system-level factors. These vulnerabilities manifest through sensor spoofing, adversarial attacks, and failures in task and motion planning, posing significant challenges to robustness and safety. Despite the growing body of research, existing reviews rarely focus specifically on the unique safety and security challenges of embodied AI systems. Most prior work either addresses general AI vulnerabilities or focuses on isolated aspects, lacking a dedicated and unified framework tailored to embodied AI. This survey fills this critical gap by: (1) categorizing vulnerabilities specific to embodied AI into exogenous (e.g., physical attacks, cybersecurity threats) and endogenous (e.g., sensor failures, software flaws) origins; (2) systematically analyzing adversarial attack paradigms unique to embodied AI, with a focus on their impact on perception, decision-making, and embodied interaction; (3) investigating attack vectors targeting large vision-language models (LVLMs) and large language models (LLMs) within embodied systems, such as jailbreak attacks and instruction misinterpretation; (4) evaluating robustness challenges in algorithms for embodied perception, decision-making, and task planning; and (5) proposing targeted strategies to enhance the safety and reliability of embodied AI systems. By integrating these dimensions, we provide a comprehensive framework for understanding the interplay between vulnerabilities and safety in embodied AI.



## **40. CausalDiff: Causality-Inspired Disentanglement via Diffusion Model for Adversarial Defense**

cs.CV

accepted by NeurIPS 2024

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2410.23091v7) [paper-pdf](http://arxiv.org/pdf/2410.23091v7)

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Quanrun Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: Despite ongoing efforts to defend neural classifiers from adversarial attacks, they remain vulnerable, especially to unseen attacks. In contrast, humans are difficult to be cheated by subtle manipulations, since we make judgments only based on essential factors. Inspired by this observation, we attempt to model label generation with essential label-causative factors and incorporate label-non-causative factors to assist data generation. For an adversarial example, we aim to discriminate the perturbations as non-causative factors and make predictions only based on the label-causative factors. Concretely, we propose a casual diffusion model (CausalDiff) that adapts diffusion models for conditional data generation and disentangles the two types of casual factors by learning towards a novel casual information bottleneck objective. Empirically, CausalDiff has significantly outperformed state-of-the-art defense methods on various unseen attacks, achieving an average robustness of 86.39% (+4.01%) on CIFAR-10, 56.25% (+3.13%) on CIFAR-100, and 82.62% (+4.93%) on GTSRB (German Traffic Sign Recognition Benchmark). The code is available at https://github.com/CAS-AISafetyBasicResearchGroup/CausalDiff.



## **41. Towards Certification of Uncertainty Calibration under Adversarial Attacks**

cs.LG

10 pages main paper, appendix included Published at: International  Conference on Learning Representations (ICLR) 2025

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2405.13922v3) [paper-pdf](http://arxiv.org/pdf/2405.13922v3)

**Authors**: Cornelius Emde, Francesco Pinto, Thomas Lukasiewicz, Philip H. S. Torr, Adel Bibi

**Abstract**: Since neural classifiers are known to be sensitive to adversarial perturbations that alter their accuracy, \textit{certification methods} have been developed to provide provable guarantees on the insensitivity of their predictions to such perturbations. Furthermore, in safety-critical applications, the frequentist interpretation of the confidence of a classifier (also known as model calibration) can be of utmost importance. This property can be measured via the Brier score or the expected calibration error. We show that attacks can significantly harm calibration, and thus propose certified calibration as worst-case bounds on calibration under adversarial perturbations. Specifically, we produce analytic bounds for the Brier score and approximate bounds via the solution of a mixed-integer program on the expected calibration error. Finally, we propose novel calibration attacks and demonstrate how they can improve model calibration through \textit{adversarial calibration training}.



## **42. Model-Free Adversarial Purification via Coarse-To-Fine Tensor Network Representation**

cs.LG

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.17972v1) [paper-pdf](http://arxiv.org/pdf/2502.17972v1)

**Authors**: Guang Lin, Duc Thien Nguyen, Zerui Tao, Konstantinos Slavakis, Toshihisa Tanaka, Qibin Zhao

**Abstract**: Deep neural networks are known to be vulnerable to well-designed adversarial attacks. Although numerous defense strategies have been proposed, many are tailored to the specific attacks or tasks and often fail to generalize across diverse scenarios. In this paper, we propose Tensor Network Purification (TNP), a novel model-free adversarial purification method by a specially designed tensor network decomposition algorithm. TNP depends neither on the pre-trained generative model nor the specific dataset, resulting in strong robustness across diverse adversarial scenarios. To this end, the key challenge lies in relaxing Gaussian-noise assumptions of classical decompositions and accommodating the unknown distribution of adversarial perturbations. Unlike the low-rank representation of classical decompositions, TNP aims to reconstruct the unobserved clean examples from an adversarial example. Specifically, TNP leverages progressive downsampling and introduces a novel adversarial optimization objective to address the challenge of minimizing reconstruction error but without inadvertently restoring adversarial perturbations. Extensive experiments conducted on CIFAR-10, CIFAR-100, and ImageNet demonstrate that our method generalizes effectively across various norm threats, attack types, and tasks, providing a versatile and promising adversarial purification technique.



## **43. LiSA: Leveraging Link Recommender to Attack Graph Neural Networks via Subgraph Injection**

cs.LG

PAKDD 2025

**SubmitDate**: 2025-02-25    [abs](http://arxiv.org/abs/2502.09271v3) [paper-pdf](http://arxiv.org/pdf/2502.09271v3)

**Authors**: Wenlun Zhang, Enyan Dai, Kentaro Yoshioka

**Abstract**: Graph Neural Networks (GNNs) have demonstrated remarkable proficiency in modeling data with graph structures, yet recent research reveals their susceptibility to adversarial attacks. Traditional attack methodologies, which rely on manipulating the original graph or adding links to artificially created nodes, often prove impractical in real-world settings. This paper introduces a novel adversarial scenario involving the injection of an isolated subgraph to deceive both the link recommender and the node classifier within a GNN system. Specifically, the link recommender is mislead to propose links between targeted victim nodes and the subgraph, encouraging users to unintentionally establish connections and that would degrade the node classification accuracy, thereby facilitating a successful attack. To address this, we present the LiSA framework, which employs a dual surrogate model and bi-level optimization to simultaneously meet two adversarial objectives. Extensive experiments on real-world datasets demonstrate the effectiveness of our method.



## **44. Relationship between Uncertainty in DNNs and Adversarial Attacks**

cs.LG

review

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2409.13232v2) [paper-pdf](http://arxiv.org/pdf/2409.13232v2)

**Authors**: Mabel Ogonna, Abigail Adeniran, Adewale Adeyemo

**Abstract**: Deep Neural Networks (DNNs) have achieved state of the art results and even outperformed human accuracy in many challenging tasks, leading to DNNs adoption in a variety of fields including natural language processing, pattern recognition, prediction, and control optimization. However, DNNs are accompanied by uncertainty about their results, causing them to predict an outcome that is either incorrect or outside of a certain level of confidence. These uncertainties stem from model or data constraints, which could be exacerbated by adversarial attacks. Adversarial attacks aim to provide perturbed input to DNNs, causing the DNN to make incorrect predictions or increase model uncertainty. In this review, we explore the relationship between DNN uncertainty and adversarial attacks, emphasizing how adversarial attacks might raise DNN uncertainty.



## **45. The Geometry of Refusal in Large Language Models: Concept Cones and Representational Independence**

cs.LG

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17420v1) [paper-pdf](http://arxiv.org/pdf/2502.17420v1)

**Authors**: Tom Wollschläger, Jannes Elstner, Simon Geisler, Vincent Cohen-Addad, Stephan Günnemann, Johannes Gasteiger

**Abstract**: The safety alignment of large language models (LLMs) can be circumvented through adversarially crafted inputs, yet the mechanisms by which these attacks bypass safety barriers remain poorly understood. Prior work suggests that a single refusal direction in the model's activation space determines whether an LLM refuses a request. In this study, we propose a novel gradient-based approach to representation engineering and use it to identify refusal directions. Contrary to prior work, we uncover multiple independent directions and even multi-dimensional concept cones that mediate refusal. Moreover, we show that orthogonality alone does not imply independence under intervention, motivating the notion of representational independence that accounts for both linear and non-linear effects. Using this framework, we identify mechanistically independent refusal directions. We show that refusal mechanisms in LLMs are governed by complex spatial structures and identify functionally independent directions, confirming that multiple distinct mechanisms drive refusal behavior. Our gradient-based approach uncovers these mechanisms and can further serve as a foundation for future work on understanding LLMs.



## **46. Emoti-Attack: Zero-Perturbation Adversarial Attacks on NLP Systems via Emoji Sequences**

cs.AI

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17392v1) [paper-pdf](http://arxiv.org/pdf/2502.17392v1)

**Authors**: Yangshijie Zhang

**Abstract**: Deep neural networks (DNNs) have achieved remarkable success in the field of natural language processing (NLP), leading to widely recognized applications such as ChatGPT. However, the vulnerability of these models to adversarial attacks remains a significant concern. Unlike continuous domains like images, text exists in a discrete space, making even minor alterations at the sentence, word, or character level easily perceptible to humans. This inherent discreteness also complicates the use of conventional optimization techniques, as text is non-differentiable. Previous research on adversarial attacks in text has focused on character-level, word-level, sentence-level, and multi-level approaches, all of which suffer from inefficiency or perceptibility issues due to the need for multiple queries or significant semantic shifts.   In this work, we introduce a novel adversarial attack method, Emoji-Attack, which leverages the manipulation of emojis to create subtle, yet effective, perturbations. Unlike character- and word-level strategies, Emoji-Attack targets emojis as a distinct layer of attack, resulting in less noticeable changes with minimal disruption to the text. This approach has been largely unexplored in previous research, which typically focuses on emoji insertion as an extension of character-level attacks. Our experiments demonstrate that Emoji-Attack achieves strong attack performance on both large and small models, making it a promising technique for enhancing adversarial robustness in NLP systems.



## **47. On the Vulnerability of Concept Erasure in Diffusion Models**

cs.LG

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17537v1) [paper-pdf](http://arxiv.org/pdf/2502.17537v1)

**Authors**: Lucas Beerens, Alex D. Richardson, Kaicheng Zhang, Dongdong Chen

**Abstract**: The proliferation of text-to-image diffusion models has raised significant privacy and security concerns, particularly regarding the generation of copyrighted or harmful images. To address these issues, research on machine unlearning has developed various concept erasure methods, which aim to remove the effect of unwanted data through post-hoc training. However, we show these erasure techniques are vulnerable, where images of supposedly erased concepts can still be generated using adversarially crafted prompts. We introduce RECORD, a coordinate-descent-based algorithm that discovers prompts capable of eliciting the generation of erased content. We demonstrate that RECORD significantly beats the attack success rate of current state-of-the-art attack methods. Furthermore, our findings reveal that models subjected to concept erasure are more susceptible to adversarial attacks than previously anticipated, highlighting the urgency for more robust unlearning approaches. We open source all our code at https://github.com/LucasBeerens/RECORD



## **48. Order Fairness Evaluation of DAG-based ledgers**

cs.CR

19 pages with 9 pages dedicated to references and appendices, 23  figures, 13 of which are in the appendices

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17270v1) [paper-pdf](http://arxiv.org/pdf/2502.17270v1)

**Authors**: Erwan Mahe, Sara Tucci-Piergiovanni

**Abstract**: Order fairness in distributed ledgers refers to properties that relate the order in which transactions are sent or received to the order in which they are eventually finalized, i.e., totally ordered. The study of such properties is relatively new and has been especially stimulated by the rise of Maximal Extractable Value (MEV) attacks in blockchain environments. Indeed, in many classical blockchain protocols, leaders are responsible for selecting the transactions to be included in blocks, which creates a clear vulnerability and opportunity for transaction order manipulation.   Unlike blockchains, DAG-based ledgers allow participants in the network to independently propose blocks, which are then arranged as vertices of a directed acyclic graph. Interestingly, leaders in DAG-based ledgers are elected only after the fact, once transactions are already part of the graph, to determine their total order. In other words, transactions are not chosen by single leaders; instead, they are collectively validated by the nodes, and leaders are only elected to establish an ordering. This approach intuitively reduces the risk of transaction manipulation and enhances fairness.   In this paper, we aim to quantify the capability of DAG-based ledgers to achieve order fairness. To this end, we define new variants of order fairness adapted to DAG-based ledgers and evaluate the impact of an adversary capable of compromising a limited number of nodes (below the one-third threshold) to reorder transactions. We analyze how often our order fairness properties are violated under different network conditions and parameterizations of the DAG algorithm, depending on the adversary's power.   Our study shows that DAG-based ledgers are still vulnerable to reordering attacks, as an adversary can coordinate a minority of Byzantine nodes to manipulate the DAG's structure.



## **49. REINFORCE Adversarial Attacks on Large Language Models: An Adaptive, Distributional, and Semantic Objective**

cs.LG

30 pages, 6 figures, 15 tables

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17254v1) [paper-pdf](http://arxiv.org/pdf/2502.17254v1)

**Authors**: Simon Geisler, Tom Wollschläger, M. H. I. Abdalla, Vincent Cohen-Addad, Johannes Gasteiger, Stephan Günnemann

**Abstract**: To circumvent the alignment of large language models (LLMs), current optimization-based adversarial attacks usually craft adversarial prompts by maximizing the likelihood of a so-called affirmative response. An affirmative response is a manually designed start of a harmful answer to an inappropriate request. While it is often easy to craft prompts that yield a substantial likelihood for the affirmative response, the attacked model frequently does not complete the response in a harmful manner. Moreover, the affirmative objective is usually not adapted to model-specific preferences and essentially ignores the fact that LLMs output a distribution over responses. If low attack success under such an objective is taken as a measure of robustness, the true robustness might be grossly overestimated. To alleviate these flaws, we propose an adaptive and semantic optimization problem over the population of responses. We derive a generally applicable objective via the REINFORCE policy-gradient formalism and demonstrate its efficacy with the state-of-the-art jailbreak algorithms Greedy Coordinate Gradient (GCG) and Projected Gradient Descent (PGD). For example, our objective doubles the attack success rate (ASR) on Llama3 and increases the ASR from 2% to 50% with circuit breaker defense.



## **50. Adversarial Training for Defense Against Label Poisoning Attacks**

cs.LG

Accepted at the International Conference on Learning Representations  (ICLR 2025)

**SubmitDate**: 2025-02-24    [abs](http://arxiv.org/abs/2502.17121v1) [paper-pdf](http://arxiv.org/pdf/2502.17121v1)

**Authors**: Melis Ilayda Bal, Volkan Cevher, Michael Muehlebach

**Abstract**: As machine learning models grow in complexity and increasingly rely on publicly sourced data, such as the human-annotated labels used in training large language models, they become more vulnerable to label poisoning attacks. These attacks, in which adversaries subtly alter the labels within a training dataset, can severely degrade model performance, posing significant risks in critical applications. In this paper, we propose FLORAL, a novel adversarial training defense strategy based on support vector machines (SVMs) to counter these threats. Utilizing a bilevel optimization framework, we cast the training process as a non-zero-sum Stackelberg game between an attacker, who strategically poisons critical training labels, and the model, which seeks to recover from such attacks. Our approach accommodates various model architectures and employs a projected gradient descent algorithm with kernel SVMs for adversarial training. We provide a theoretical analysis of our algorithm's convergence properties and empirically evaluate FLORAL's effectiveness across diverse classification tasks. Compared to robust baselines and foundation models such as RoBERTa, FLORAL consistently achieves higher robust accuracy under increasing attacker budgets. These results underscore the potential of FLORAL to enhance the resilience of machine learning models against label poisoning threats, thereby ensuring robust classification in adversarial settings.



