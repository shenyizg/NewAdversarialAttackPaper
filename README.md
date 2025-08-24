# Latest Adversarial Attack Papers
**update at 2025-08-24 10:00:57**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Distributed Detection of Adversarial Attacks in Multi-Agent Reinforcement Learning with Continuous Action Space**

cs.LG

Accepted for publication at ECAI 2025

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15764v1) [paper-pdf](http://arxiv.org/pdf/2508.15764v1)

**Authors**: Kiarash Kazari, Ezzeldin Shereen, György Dán

**Abstract**: We address the problem of detecting adversarial attacks against cooperative multi-agent reinforcement learning with continuous action space. We propose a decentralized detector that relies solely on the local observations of the agents and makes use of a statistical characterization of the normal behavior of observable agents. The proposed detector utilizes deep neural networks to approximate the normal behavior of agents as parametric multivariate Gaussian distributions. Based on the predicted density functions, we define a normality score and provide a characterization of its mean and variance. This characterization allows us to employ a two-sided CUSUM procedure for detecting deviations of the normality score from its mean, serving as a detector of anomalous behavior in real-time. We evaluate our scheme on various multi-agent PettingZoo benchmarks against different state-of-the-art attack methods, and our results demonstrate the effectiveness of our method in detecting impactful adversarial attacks. Particularly, it outperforms the discrete counterpart by achieving AUC-ROC scores of over 0.95 against the most impactful attacks in all evaluated environments.



## **2. Let's Measure Information Step-by-Step: LLM-Based Evaluation Beyond Vibes**

cs.LG

Add AUC results, pre-reg conformance, theory section clarification.  12 pages

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.05469v2) [paper-pdf](http://arxiv.org/pdf/2508.05469v2)

**Authors**: Zachary Robertson, Sanmi Koyejo

**Abstract**: We study evaluation of AI systems without ground truth by exploiting a link between strategic gaming and information loss. We analyze which information-theoretic mechanisms resist adversarial manipulation, extending finite-sample bounds to show that bounded f-divergences (e.g., total variation distance) maintain polynomial guarantees under attacks while unbounded measures (e.g., KL divergence) degrade exponentially. To implement these mechanisms, we model the overseer as an agent and characterize incentive-compatible scoring rules as f-mutual information objectives. Under adversarial attacks, TVD-MI maintains effectiveness (area under curve 0.70-0.77) while traditional judge queries are near change (AUC $\approx$ 0.50), demonstrating that querying the same LLM for information relationships rather than quality judgments provides both theoretical and practical robustness. The mechanisms decompose pairwise evaluations into reliable item-level quality scores without ground truth, addressing a key limitation of traditional peer prediction. We release preregistration and code.



## **3. Towards a 3D Transfer-based Black-box Attack via Critical Feature Guidance**

cs.CV

11 pages, 6 figures

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15650v1) [paper-pdf](http://arxiv.org/pdf/2508.15650v1)

**Authors**: Shuchao Pang, Zhenghan Chen, Shen Zhang, Liming Lu, Siyuan Liang, Anan Du, Yongbin Zhou

**Abstract**: Deep neural networks for 3D point clouds have been demonstrated to be vulnerable to adversarial examples. Previous 3D adversarial attack methods often exploit certain information about the target models, such as model parameters or outputs, to generate adversarial point clouds. However, in realistic scenarios, it is challenging to obtain any information about the target models under conditions of absolute security. Therefore, we focus on transfer-based attacks, where generating adversarial point clouds does not require any information about the target models. Based on our observation that the critical features used for point cloud classification are consistent across different DNN architectures, we propose CFG, a novel transfer-based black-box attack method that improves the transferability of adversarial point clouds via the proposed Critical Feature Guidance. Specifically, our method regularizes the search of adversarial point clouds by computing the importance of the extracted features, prioritizing the corruption of critical features that are likely to be adopted by diverse architectures. Further, we explicitly constrain the maximum deviation extent of the generated adversarial point clouds in the loss function to ensure their imperceptibility. Extensive experiments conducted on the ModelNet40 and ScanObjectNN benchmark datasets demonstrate that the proposed CFG outperforms the state-of-the-art attack methods by a large margin.



## **4. Vulnerabilities in AI-generated Image Detection: The Challenge of Adversarial Attacks**

cs.CV

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2407.20836v4) [paper-pdf](http://arxiv.org/pdf/2407.20836v4)

**Authors**: Yunfeng Diao, Naixin Zhai, Changtao Miao, Zitong Yu, Xingxing Wei, Xun Yang, Meng Wang

**Abstract**: Recent advancements in image synthesis, particularly with the advent of GAN and Diffusion models, have amplified public concerns regarding the dissemination of disinformation. To address such concerns, numerous AI-generated Image (AIGI) Detectors have been proposed and achieved promising performance in identifying fake images. However, there still lacks a systematic understanding of the adversarial robustness of AIGI detectors. In this paper, we examine the vulnerability of state-of-the-art AIGI detectors against adversarial attack under white-box and black-box settings, which has been rarely investigated so far. To this end, we propose a new method to attack AIGI detectors. First, inspired by the obvious difference between real images and fake images in the frequency domain, we add perturbations under the frequency domain to push the image away from its original frequency distribution. Second, we explore the full posterior distribution of the surrogate model to further narrow this gap between heterogeneous AIGI detectors, e.g. transferring adversarial examples across CNNs and ViTs. This is achieved by introducing a novel post-train Bayesian strategy that turns a single surrogate into a Bayesian one, capable of simulating diverse victim models using one pre-trained surrogate, without the need for re-training. We name our method as Frequency-based Post-train Bayesian Attack, or FPBA. Through FPBA, we show that adversarial attack is truly a real threat to AIGI detectors, because FPBA can deliver successful black-box attacks across models, generators, defense methods, and even evade cross-generator detection, which is a crucial real-world detection scenario. The code will be shared upon acceptance.



## **5. Any-to-any Speaker Attribute Perturbation for Asynchronous Voice Anonymization**

cs.SD

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15565v1) [paper-pdf](http://arxiv.org/pdf/2508.15565v1)

**Authors**: Liping Chen, Chenyang Guo, Rui Wang, Kong Aik Lee, Zhenhua Ling

**Abstract**: Speaker attribute perturbation offers a feasible approach to asynchronous voice anonymization by employing adversarially perturbed speech as anonymized output. In order to enhance the identity unlinkability among anonymized utterances from the same original speaker, the targeted attack training strategy is usually applied to anonymize the utterances to a common designated speaker. However, this strategy may violate the privacy of the designated speaker who is an actual speaker. To mitigate this risk, this paper proposes an any-to-any training strategy. It is accomplished by defining a batch mean loss to anonymize the utterances from various speakers within a training mini-batch to a common pseudo-speaker, which is approximated as the average speaker in the mini-batch. Based on this, a speaker-adversarial speech generation model is proposed, incorporating the supervision from both the untargeted attack and the any-to-any strategies. The speaker attribute perturbations are generated and incorporated into the original speech to produce its anonymized version. The effectiveness of the proposed model was justified in asynchronous voice anonymization through experiments conducted on the VoxCeleb datasets. Additional experiments were carried out to explore the potential limitations of speaker-adversarial speech in voice privacy protection. With them, we aim to provide insights for future research on its protective efficacy against black-box speaker extractors \textcolor{black}{and adaptive attacks, as well as} generalization to out-of-domain datasets \textcolor{black}{and stability}. Audio samples and open-source code are published in https://github.com/VoicePrivacy/any-to-any-speaker-attribute-perturbation.



## **6. BadFU: Backdoor Federated Learning through Adversarial Machine Unlearning**

cs.CR

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15541v1) [paper-pdf](http://arxiv.org/pdf/2508.15541v1)

**Authors**: Bingguang Lu, Hongsheng Hu, Yuantian Miao, Shaleeza Sohail, Chaoxiang He, Shuo Wang, Xiao Chen

**Abstract**: Federated learning (FL) has been widely adopted as a decentralized training paradigm that enables multiple clients to collaboratively learn a shared model without exposing their local data. As concerns over data privacy and regulatory compliance grow, machine unlearning, which aims to remove the influence of specific data from trained models, has become increasingly important in the federated setting to meet legal, ethical, or user-driven demands. However, integrating unlearning into FL introduces new challenges and raises largely unexplored security risks. In particular, adversaries may exploit the unlearning process to compromise the integrity of the global model. In this paper, we present the first backdoor attack in the context of federated unlearning, demonstrating that an adversary can inject backdoors into the global model through seemingly legitimate unlearning requests. Specifically, we propose BadFU, an attack strategy where a malicious client uses both backdoor and camouflage samples to train the global model normally during the federated training process. Once the client requests unlearning of the camouflage samples, the global model transitions into a backdoored state. Extensive experiments under various FL frameworks and unlearning strategies validate the effectiveness of BadFU, revealing a critical vulnerability in current federated unlearning practices and underscoring the urgent need for more secure and robust federated unlearning mechanisms.



## **7. On Evaluating the Adversarial Robustness of Foundation Models for Multimodal Entity Linking**

cs.IR

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15481v1) [paper-pdf](http://arxiv.org/pdf/2508.15481v1)

**Authors**: Fang Wang, Yongjie Wang, Zonghao Yang, Minghao Hu, Xiaoying Bai

**Abstract**: The explosive growth of multimodal data has driven the rapid development of multimodal entity linking (MEL) models. However, existing studies have not systematically investigated the impact of visual adversarial attacks on MEL models. We conduct the first comprehensive evaluation of the robustness of mainstream MEL models under different adversarial attack scenarios, covering two core tasks: Image-to-Text (I2T) and Image+Text-to-Text (IT2T). Experimental results show that current MEL models generally lack sufficient robustness against visual perturbations. Interestingly, contextual semantic information in input can partially mitigate the impact of adversarial perturbations. Based on this insight, we propose an LLM and Retrieval-Augmented Entity Linking (LLM-RetLink), which significantly improves the model's anti-interference ability through a two-stage process: first, extracting initial entity descriptions using large vision models (LVMs), and then dynamically generating candidate descriptive sentences via web-based retrieval. Experiments on five datasets demonstrate that LLM-RetLink improves the accuracy of MEL by 0.4%-35.7%, especially showing significant advantages under adversarial conditions. This research highlights a previously unexplored facet of MEL robustness, constructs and releases the first MEL adversarial example dataset, and sets the stage for future work aimed at strengthening the resilience of multimodal systems in adversarial environments.



## **8. Mini-Batch Robustness Verification of Deep Neural Networks**

cs.LG

30 pages, 12 figures, conference OOPSLA 2025

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15454v1) [paper-pdf](http://arxiv.org/pdf/2508.15454v1)

**Authors**: Saar Tzour-Shaday, Dana Drachsler Cohen

**Abstract**: Neural network image classifiers are ubiquitous in many safety-critical applications. However, they are susceptible to adversarial attacks. To understand their robustness to attacks, many local robustness verifiers have been proposed to analyze $\epsilon$-balls of inputs. Yet, existing verifiers introduce a long analysis time or lose too much precision, making them less effective for a large set of inputs. In this work, we propose a new approach to local robustness: group local robustness verification. The key idea is to leverage the similarity of the network computations of certain $\epsilon$-balls to reduce the overall analysis time. We propose BaVerLy, a sound and complete verifier that boosts the local robustness verification of a set of $\epsilon$-balls by dynamically constructing and verifying mini-batches. BaVerLy adaptively identifies successful mini-batch sizes, accordingly constructs mini-batches of $\epsilon$-balls that have similar network computations, and verifies them jointly. If a mini-batch is verified, all $\epsilon$-balls are proven robust. Otherwise, one $\epsilon$-ball is suspected as not being robust, guiding the refinement. In the latter case, BaVerLy leverages the analysis results to expedite the analysis of that $\epsilon$-ball as well as the other $\epsilon$-balls in the batch. We evaluate BaVerLy on fully connected and convolutional networks for MNIST and CIFAR-10. Results show that BaVerLy scales the common one by one verification by 2.3x on average and up to 4.1x, in which case it reduces the total analysis time from 24 hours to 6 hours.



## **9. Tensor Train Decomposition for Adversarial Attacks on Computer Vision Models**

math.NA

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2312.12556v2) [paper-pdf](http://arxiv.org/pdf/2312.12556v2)

**Authors**: Andrei Chertkov, Ivan Oseledets

**Abstract**: Deep neural networks (DNNs) are widely used today, but they are vulnerable to adversarial attacks. To develop effective methods of defense, it is important to understand the potential weak spots of DNNs. Often attacks are organized taking into account the architecture of models (white-box approach) and based on gradient methods, but for real-world DNNs this approach in most cases is impossible. At the same time, several gradient-free optimization algorithms are used to attack black-box models. However, classical methods are often ineffective in the multidimensional case. To organize black-box attacks for computer vision models, in this work, we propose the use of an optimizer based on the low-rank tensor train (TT) format, which has gained popularity in various practical multidimensional applications in recent years. Combined with the attribution of the target image, which is built by the auxiliary (white-box) model, the TT-based optimization method makes it possible to organize an effective black-box attack by small perturbation of pixels in the target image. The superiority of the proposed approach over three popular baselines is demonstrated for seven modern DNNs on the ImageNet dataset.



## **10. Setup Once, Secure Always: A Single-Setup Secure Federated Learning Aggregation Protocol with Forward and Backward Secrecy for Dynamic Users**

cs.CR

17 pages, 12 Figures

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2502.08989v4) [paper-pdf](http://arxiv.org/pdf/2502.08989v4)

**Authors**: Nazatul Haque Sultan, Yan Bo, Yansong Gao, Seyit Camtepe, Arash Mahboubi, Hang Thanh Bui, Aufeef Chauhan, Hamed Aboutorab, Michael Bewong, Dineshkumar Singh, Praveen Gauravaram, Rafiqul Islam, Sharif Abuadbba

**Abstract**: Federated Learning (FL) enables multiple users to collaboratively train a machine learning model without sharing raw data, making it suitable for privacy-sensitive applications. However, local model or weight updates can still leak sensitive information. Secure aggregation protocols mitigate this risk by ensuring that only the aggregated updates are revealed. Among these, single-setup protocols, where key generation and exchange occur only once, are the most efficient due to reduced communication and computation overhead. However, existing single-setup protocols often lack support for dynamic user participation and do not provide strong privacy guarantees such as forward and backward secrecy. \par In this paper, we present a novel secure aggregation protocol that requires only a single setup for the entire FL training. Our protocol supports dynamic user participation, tolerates dropouts, and achieves both forward and backward secrecy. It leverages lightweight symmetric homomorphic encryption with a key negation technique to mask updates efficiently, eliminating the need for user-to-user communication. To defend against model inconsistency attacks, we introduce a low-overhead verification mechanism using message authentication codes (MACs). We provide formal security proofs under both semi-honest and malicious adversarial models and implement a full prototype. Experimental results show that our protocol reduces user-side computation by up to $99\%$ compared to state-of-the-art protocols like e-SeaFL (ACSAC'24), while maintaining competitive model accuracy. These features make our protocol highly practical for real-world FL deployments, especially on resource-constrained devices.



## **11. Adversarial Attacks against Neural Ranking Models via In-Context Learning**

cs.IR

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15283v1) [paper-pdf](http://arxiv.org/pdf/2508.15283v1)

**Authors**: Amin Bigdeli, Negar Arabzadeh, Ebrahim Bagheri, Charles L. A. Clarke

**Abstract**: While neural ranking models (NRMs) have shown high effectiveness, they remain susceptible to adversarial manipulation. In this work, we introduce Few-Shot Adversarial Prompting (FSAP), a novel black-box attack framework that leverages the in-context learning capabilities of Large Language Models (LLMs) to generate high-ranking adversarial documents. Unlike previous approaches that rely on token-level perturbations or manual rewriting of existing documents, FSAP formulates adversarial attacks entirely through few-shot prompting, requiring no gradient access or internal model instrumentation. By conditioning the LLM on a small support set of previously observed harmful examples, FSAP synthesizes grammatically fluent and topically coherent documents that subtly embed false or misleading information and rank competitively against authentic content. We instantiate FSAP in two modes: FSAP-IntraQ, which leverages harmful examples from the same query to enhance topic fidelity, and FSAP-InterQ, which enables broader generalization by transferring adversarial patterns across unrelated queries. Our experiments on the TREC 2020 and 2021 Health Misinformation Tracks, using four diverse neural ranking models, reveal that FSAP-generated documents consistently outrank credible, factually accurate documents. Furthermore, our analysis demonstrates that these adversarial outputs exhibit strong stance alignment and low detectability, posing a realistic and scalable threat to neural retrieval systems. FSAP also effectively generalizes across both proprietary and open-source LLMs.



## **12. SafeLLM: Unlearning Harmful Outputs from Large Language Models against Jailbreak Attacks**

cs.LG

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15182v1) [paper-pdf](http://arxiv.org/pdf/2508.15182v1)

**Authors**: Xiangman Li, Xiaodong Wu, Qi Li, Jianbing Ni, Rongxing Lu

**Abstract**: Jailbreak attacks pose a serious threat to the safety of Large Language Models (LLMs) by crafting adversarial prompts that bypass alignment mechanisms, causing the models to produce harmful, restricted, or biased content. In this paper, we propose SafeLLM, a novel unlearning-based defense framework that unlearn the harmful knowledge from LLMs while preserving linguistic fluency and general capabilities. SafeLLM employs a three-stage pipeline: (1) dynamic unsafe output detection using a hybrid approach that integrates external classifiers with model-internal evaluations; (2) token-level harmful content tracing through feedforward network (FFN) activations to localize harmful knowledge; and (3) constrained optimization to suppress unsafe behavior without degrading overall model quality. SafeLLM achieves targeted and irreversible forgetting by identifying and neutralizing FFN substructures responsible for harmful generation pathways. Extensive experiments on prominent LLMs (Vicuna, LLaMA, and GPT-J) across multiple jailbreak benchmarks show that SafeLLM substantially reduces attack success rates while maintaining high general-purpose performance. Compared to standard defense methods such as supervised fine-tuning and direct preference optimization, SafeLLM offers stronger safety guarantees, more precise control over harmful behavior, and greater robustness to unseen attacks. Moreover, SafeLLM maintains the general performance after the harmful knowledge unlearned. These results highlight unlearning as a promising direction for scalable and effective LLM safety.



## **13. MoEcho: Exploiting Side-Channel Attacks to Compromise User Privacy in Mixture-of-Experts LLMs**

cs.CR

This paper will appear in CCS 2025

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2508.15036v1) [paper-pdf](http://arxiv.org/pdf/2508.15036v1)

**Authors**: Ruyi Ding, Tianhong Xu, Xinyi Shen, Aidong Adam Ding, Yunsi Fei

**Abstract**: The transformer architecture has become a cornerstone of modern AI, fueling remarkable progress across applications in natural language processing, computer vision, and multimodal learning. As these models continue to scale explosively for performance, implementation efficiency remains a critical challenge. Mixture of Experts (MoE) architectures, selectively activating specialized subnetworks (experts), offer a unique balance between model accuracy and computational cost. However, the adaptive routing in MoE architectures, where input tokens are dynamically directed to specialized experts based on their semantic meaning inadvertently opens up a new attack surface for privacy breaches. These input-dependent activation patterns leave distinctive temporal and spatial traces in hardware execution, which adversaries could exploit to deduce sensitive user data. In this work, we propose MoEcho, discovering a side channel analysis based attack surface that compromises user privacy on MoE based systems. Specifically, in MoEcho, we introduce four novel architectural side channels on different computing platforms, including Cache Occupancy Channels and Pageout+Reload on CPUs, and Performance Counter and TLB Evict+Reload on GPUs, respectively. Exploiting these vulnerabilities, we propose four attacks that effectively breach user privacy in large language models (LLMs) and vision language models (VLMs) based on MoE architectures: Prompt Inference Attack, Response Reconstruction Attack, Visual Inference Attack, and Visual Reconstruction Attack. MoEcho is the first runtime architecture level security analysis of the popular MoE structure common in modern transformers, highlighting a serious security and privacy threat and calling for effective and timely safeguards when harnessing MoE based models for developing efficient large scale AI services.



## **14. A Systematic Survey of Model Extraction Attacks and Defenses: State-of-the-Art and Perspectives**

cs.CR

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2508.15031v1) [paper-pdf](http://arxiv.org/pdf/2508.15031v1)

**Authors**: Kaixiang Zhao, Lincan Li, Kaize Ding, Neil Zhenqiang Gong, Yue Zhao, Yushun Dong

**Abstract**: Machine learning (ML) models have significantly grown in complexity and utility, driving advances across multiple domains. However, substantial computational resources and specialized expertise have historically restricted their wide adoption. Machine-Learning-as-a-Service (MLaaS) platforms have addressed these barriers by providing scalable, convenient, and affordable access to sophisticated ML models through user-friendly APIs. While this accessibility promotes widespread use of advanced ML capabilities, it also introduces vulnerabilities exploited through Model Extraction Attacks (MEAs). Recent studies have demonstrated that adversaries can systematically replicate a target model's functionality by interacting with publicly exposed interfaces, posing threats to intellectual property, privacy, and system security. In this paper, we offer a comprehensive survey of MEAs and corresponding defense strategies. We propose a novel taxonomy that classifies MEAs according to attack mechanisms, defense approaches, and computing environments. Our analysis covers various attack techniques, evaluates their effectiveness, and highlights challenges faced by existing defenses, particularly the critical trade-off between preserving model utility and ensuring security. We further assess MEAs within different computing paradigms and discuss their technical, ethical, legal, and societal implications, along with promising directions for future research. This systematic survey aims to serve as a valuable reference for researchers, practitioners, and policymakers engaged in AI security and privacy. Additionally, we maintain an online repository continuously updated with related literature at https://github.com/kzhao5/ModelExtractionPapers.



## **15. TAIGen: Training-Free Adversarial Image Generation via Diffusion Models**

cs.CV

Accepted at ICCVW-CV4BIOM 2025

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2508.15020v1) [paper-pdf](http://arxiv.org/pdf/2508.15020v1)

**Authors**: Susim Roy, Anubhooti Jain, Mayank Vatsa, Richa Singh

**Abstract**: Adversarial attacks from generative models often produce low-quality images and require substantial computational resources. Diffusion models, though capable of high-quality generation, typically need hundreds of sampling steps for adversarial generation. This paper introduces TAIGen, a training-free black-box method for efficient adversarial image generation. TAIGen produces adversarial examples using only 3-20 sampling steps from unconditional diffusion models. Our key finding is that perturbations injected during the mixing step interval achieve comparable attack effectiveness without processing all timesteps. We develop a selective RGB channel strategy that applies attention maps to the red channel while using GradCAM-guided perturbations on green and blue channels. This design preserves image structure while maximizing misclassification in target models. TAIGen maintains visual quality with PSNR above 30 dB across all tested datasets. On ImageNet with VGGNet as source, TAIGen achieves 70.6% success against ResNet, 80.8% against MNASNet, and 97.8% against ShuffleNet. The method generates adversarial examples 10x faster than existing diffusion-based attacks. Our method achieves the lowest robust accuracy, indicating it is the most impactful attack as the defense mechanism is least successful in purifying the images generated by TAIGen.



## **16. Security Concerns for Large Language Models: A Survey**

cs.CR

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2505.18889v4) [paper-pdf](http://arxiv.org/pdf/2505.18889v4)

**Authors**: Miles Q. Li, Benjamin C. M. Fung

**Abstract**: Large Language Models (LLMs) such as ChatGPT and its competitors have caused a revolution in natural language processing, but their capabilities also introduce new security vulnerabilities. This survey provides a comprehensive overview of these emerging concerns, categorizing threats into several key areas: prompt injection and jailbreaking; adversarial attacks, including input perturbations and data poisoning; misuse by malicious actors to generate disinformation, phishing emails, and malware; and the worrisome risks inherent in autonomous LLM agents. Recently, a significant focus is increasingly being placed on the latter, exploring goal misalignment, emergent deception, self-preservation instincts, and the potential for LLMs to develop and pursue covert, misaligned objectives, a behavior known as scheming, which may even persist through safety training. We summarize recent academic and industrial studies from 2022 to 2025 that exemplify each threat, analyze proposed defenses and their limitations, and identify open challenges in securing LLM-based applications. We conclude by emphasizing the importance of advancing robust, multi-layered security strategies to ensure LLMs are safe and beneficial.



## **17. Universal and Transferable Adversarial Attack on Large Language Models Using Exponentiated Gradient Descent**

cs.LG

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2508.14853v1) [paper-pdf](http://arxiv.org/pdf/2508.14853v1)

**Authors**: Sajib Biswas, Mao Nishino, Samuel Jacob Chacko, Xiuwen Liu

**Abstract**: As large language models (LLMs) are increasingly deployed in critical applications, ensuring their robustness and safety alignment remains a major challenge. Despite the overall success of alignment techniques such as reinforcement learning from human feedback (RLHF) on typical prompts, LLMs remain vulnerable to jailbreak attacks enabled by crafted adversarial triggers appended to user prompts. Most existing jailbreak methods either rely on inefficient searches over discrete token spaces or direct optimization of continuous embeddings. While continuous embeddings can be given directly to selected open-source models as input, doing so is not feasible for proprietary models. On the other hand, projecting these embeddings back into valid discrete tokens introduces additional complexity and often reduces attack effectiveness. We propose an intrinsic optimization method which directly optimizes relaxed one-hot encodings of the adversarial suffix tokens using exponentiated gradient descent coupled with Bregman projection, ensuring that the optimized one-hot encoding of each token always remains within the probability simplex. We provide theoretical proof of convergence for our proposed method and implement an efficient algorithm that effectively jailbreaks several widely used LLMs. Our method achieves higher success rates and faster convergence compared to three state-of-the-art baselines, evaluated on five open-source LLMs and four adversarial behavior datasets curated for evaluating jailbreak methods. In addition to individual prompt attacks, we also generate universal adversarial suffixes effective across multiple prompts and demonstrate transferability of optimized suffixes to different LLMs.



## **18. Fragile, Robust, and Antifragile: A Perspective from Parameter Responses in Reinforcement Learning Under Stress**

cs.LG

Withdrawn pending a review of attribution and overlap with Pravin et  al., Artificial Intelligence (2024), DOI: 10.1016/j.artint.2023.104060.  Further dissemination is paused while we determine appropriate next steps

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2506.23036v2) [paper-pdf](http://arxiv.org/pdf/2506.23036v2)

**Authors**: Zain ul Abdeen, Ming Jin

**Abstract**: This paper explores Reinforcement learning (RL) policy robustness by systematically analyzing network parameters under internal and external stresses. Inspired by synaptic plasticity in neuroscience, synaptic filtering introduces internal stress by selectively perturbing parameters, while adversarial attacks apply external stress through modified agent observations. This dual approach enables the classification of parameters as fragile, robust, or antifragile, based on their influence on policy performance in clean and adversarial settings. Parameter scores are defined to quantify these characteristics, and the framework is validated on PPO-trained agents in Mujoco continuous control environments. The results highlight the presence of antifragile parameters that enhance policy performance under stress, demonstrating the potential of targeted filtering techniques to improve RL policy adaptability. These insights provide a foundation for future advancements in the design of robust and antifragile RL systems.



## **19. Distributional Adversarial Attacks and Training in Deep Hedging**

math.OC

Preprint. Under review

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2508.14757v1) [paper-pdf](http://arxiv.org/pdf/2508.14757v1)

**Authors**: Guangyi He, Tobias Sutter, Lukas Gonon

**Abstract**: In this paper, we study the robustness of classical deep hedging strategies under distributional shifts by leveraging the concept of adversarial attacks. We first demonstrate that standard deep hedging models are highly vulnerable to small perturbations in the input distribution, resulting in significant performance degradation. Motivated by this, we propose an adversarial training framework tailored to increase the robustness of deep hedging strategies. Our approach extends pointwise adversarial attacks to the distributional setting and introduces a computationally tractable reformulation of the adversarial optimization problem over a Wasserstein ball. This enables the efficient training of hedging strategies that are resilient to distributional perturbations. Through extensive numerical experiments, we show that adversarially trained deep hedging strategies consistently outperform their classical counterparts in terms of out-of-sample performance and resilience to model misspecification. Our findings establish a practical and effective framework for robust deep hedging under realistic market uncertainties.



## **20. Foe for Fraud: Transferable Adversarial Attacks in Credit Card Fraud Detection**

cs.CR

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2508.14699v1) [paper-pdf](http://arxiv.org/pdf/2508.14699v1)

**Authors**: Jan Lum Fok, Qingwen Zeng, Shiping Chen, Oscar Fawkes, Huaming Chen

**Abstract**: Credit card fraud detection (CCFD) is a critical application of Machine Learning (ML) in the financial sector, where accurately identifying fraudulent transactions is essential for mitigating financial losses. ML models have demonstrated their effectiveness in fraud detection task, in particular with the tabular dataset. While adversarial attacks have been extensively studied in computer vision and deep learning, their impacts on the ML models, particularly those trained on CCFD tabular datasets, remains largely unexplored. These latent vulnerabilities pose significant threats to the security and stability of the financial industry, especially in high-value transactions where losses could be substantial. To address this gap, in this paper, we present a holistic framework that investigate the robustness of CCFD ML model against adversarial perturbations under different circumstances. Specifically, the gradient-based attack methods are incorporated into the tabular credit card transaction data in both black- and white-box adversarial attacks settings. Our findings confirm that tabular data is also susceptible to subtle perturbations, highlighting the need for heightened awareness among financial technology practitioners regarding ML model security and trustworthiness. Furthermore, the experiments by transferring adversarial samples from gradient-based attack method to non-gradient-based models also verify our findings. Our results demonstrate that such attacks remain effective, emphasizing the necessity of developing robust defenses for CCFD algorithms.



## **21. Dark Miner: Defend against undesirable generation for text-to-image diffusion models**

cs.CV

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2409.17682v3) [paper-pdf](http://arxiv.org/pdf/2409.17682v3)

**Authors**: Zheling Meng, Bo Peng, Xiaochuan Jin, Yue Jiang, Wei Wang, Jing Dong, Tieniu Tan

**Abstract**: Text-to-image diffusion models have been demonstrated with undesired generation due to unfiltered large-scale training data, such as sexual images and copyrights, necessitating the erasure of undesired concepts. Most existing methods focus on modifying the generation probabilities conditioned on the texts containing target concepts. However, they fail to guarantee the desired generation of texts unseen in the training phase, especially for the adversarial texts from malicious attacks. In this paper, we analyze the erasure task and point out that existing methods cannot guarantee the minimization of the total probabilities of undesired generation. To tackle this problem, we propose Dark Miner. It entails a recurring three-stage process that comprises mining, verifying, and circumventing. This method greedily mines embeddings with maximum generation probabilities of target concepts and more effectively reduces their generation. In the experiments, we evaluate its performance on the inappropriateness, object, and style concepts. Compared with the previous methods, our method achieves better erasure and defense results, especially under multiple adversarial attacks, while preserving the native generation capability of the models. Our code will be available on GitHub.



## **22. When Good Sounds Go Adversarial: Jailbreaking Audio-Language Models with Benign Inputs**

cs.SD

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2508.03365v2) [paper-pdf](http://arxiv.org/pdf/2508.03365v2)

**Authors**: Bodam Kim, Hiskias Dingeto, Taeyoun Kwon, Dasol Choi, DongGeon Lee, Haon Park, JaeHoon Lee, Jongho Shin

**Abstract**: As large language models become increasingly integrated into daily life, audio has emerged as a key interface for human-AI interaction. However, this convenience also introduces new vulnerabilities, making audio a potential attack surface for adversaries. Our research introduces WhisperInject, a two-stage adversarial audio attack framework that can manipulate state-of-the-art audio language models to generate harmful content. Our method uses imperceptible perturbations in audio inputs that remain benign to human listeners. The first stage uses a novel reward-based optimization method, Reinforcement Learning with Projected Gradient Descent (RL-PGD), to guide the target model to circumvent its own safety protocols and generate harmful native responses. This native harmful response then serves as the target for Stage 2, Payload Injection, where we use Projected Gradient Descent (PGD) to optimize subtle perturbations that are embedded into benign audio carriers, such as weather queries or greeting messages. Validated under the rigorous StrongREJECT, LlamaGuard, as well as Human Evaluation safety evaluation framework, our experiments demonstrate a success rate exceeding 86% across Qwen2.5-Omni-3B, Qwen2.5-Omni-7B, and Phi-4-Multimodal. Our work demonstrates a new class of practical, audio-native threats, moving beyond theoretical exploits to reveal a feasible and covert method for manipulating AI behavior.



## **23. Adversarial control of synchronization in complex oscillator networks**

nlin.AO

10 pages, 4 figures

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2506.02403v2) [paper-pdf](http://arxiv.org/pdf/2506.02403v2)

**Authors**: Yasutoshi Nagahama, Kosuke Miyazato, Kazuhiro Takemoto

**Abstract**: This study investigates adversarial attacks, a concept from deep learning, designed to control synchronization dynamics through strategically crafted weak perturbations. We propose a gradient-based optimization method that identifies small phase perturbations to dramatically enhance or suppress collective synchronization in Kuramoto oscillator networks. Our approach formulates synchronization control as an adversarial optimization problem, computing gradients of the order parameter with respect to oscillator phases to determine optimal perturbation directions. Results demonstrate that extremely small phase perturbations applied to network oscillators can achieve significant synchronization control across diverse network architectures. Our analysis reveals that synchronization enhancement is achievable across various network sizes, while synchronization suppression becomes particularly effective in larger networks, with effectiveness scaling favorably with network size. The method is systematically validated on canonical model networks including scale-free and small-world topologies, and real-world networks representing power grids and brain connectivity patterns. This adversarial framework represents a novel paradigm for synchronization management by introducing deep learning concepts to networked dynamical systems.



## **24. Backdooring Self-Supervised Contrastive Learning by Noisy Alignment**

cs.CV

Accepted by ICCV 2025

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.14015v1) [paper-pdf](http://arxiv.org/pdf/2508.14015v1)

**Authors**: Tuo Chen, Jie Gui, Minjing Dong, Ju Jia, Lanting Fang, Jian Liu

**Abstract**: Self-supervised contrastive learning (CL) effectively learns transferable representations from unlabeled data containing images or image-text pairs but suffers vulnerability to data poisoning backdoor attacks (DPCLs). An adversary can inject poisoned images into pretraining datasets, causing compromised CL encoders to exhibit targeted misbehavior in downstream tasks. Existing DPCLs, however, achieve limited efficacy due to their dependence on fragile implicit co-occurrence between backdoor and target object and inadequate suppression of discriminative features in backdoored images. We propose Noisy Alignment (NA), a DPCL method that explicitly suppresses noise components in poisoned images. Inspired by powerful training-controllable CL attacks, we identify and extract the critical objective of noisy alignment, adapting it effectively into data-poisoning scenarios. Our method implements noisy alignment by strategically manipulating contrastive learning's random cropping mechanism, formulating this process as an image layout optimization problem with theoretically derived optimal parameters. The resulting method is simple yet effective, achieving state-of-the-art performance compared to existing DPCLs, while maintaining clean-data accuracy. Furthermore, Noisy Alignment demonstrates robustness against common backdoor defenses. Codes can be found at https://github.com/jsrdcht/Noisy-Alignment.



## **25. FedUP: Efficient Pruning-based Federated Unlearning for Model Poisoning Attacks**

cs.LG

15 pages, 5 figures, 7 tables

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.13853v1) [paper-pdf](http://arxiv.org/pdf/2508.13853v1)

**Authors**: Nicolò Romandini, Cristian Borcea, Rebecca Montanari, Luca Foschini

**Abstract**: Federated Learning (FL) can be vulnerable to attacks, such as model poisoning, where adversaries send malicious local weights to compromise the global model. Federated Unlearning (FU) is emerging as a solution to address such vulnerabilities by selectively removing the influence of detected malicious contributors on the global model without complete retraining. However, unlike typical FU scenarios where clients are trusted and cooperative, applying FU with malicious and possibly colluding clients is challenging because their collaboration in unlearning their data cannot be assumed. This work presents FedUP, a lightweight FU algorithm designed to efficiently mitigate malicious clients' influence by pruning specific connections within the attacked model. Our approach achieves efficiency by relying only on clients' weights from the last training round before unlearning to identify which connections to inhibit. Isolating malicious influence is non-trivial due to overlapping updates from benign and malicious clients. FedUP addresses this by carefully selecting and zeroing the highest magnitude weights that diverge the most between the latest updates from benign and malicious clients while preserving benign information. FedUP is evaluated under a strong adversarial threat model, where up to 50%-1 of the clients could be malicious and have full knowledge of the aggregation process. We demonstrate the effectiveness, robustness, and efficiency of our solution through experiments across IID and Non-IID data, under label-flipping and backdoor attacks, and by comparing it with state-of-the-art (SOTA) FU solutions. In all scenarios, FedUP reduces malicious influence, lowering accuracy on malicious data to match that of a model retrained from scratch while preserving performance on benign data. FedUP achieves effective unlearning while consistently being faster and saving storage compared to the SOTA.



## **26. Timestep-Compressed Attack on Spiking Neural Networks through Timestep-Level Backpropagation**

cs.CV

8 pages

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.13812v1) [paper-pdf](http://arxiv.org/pdf/2508.13812v1)

**Authors**: Donghwa Kang, Doohyun Kim, Sang-Ki Ko, Jinkyu Lee, Hyeongboo Baek, Brent ByungHoon Kang

**Abstract**: State-of-the-art (SOTA) gradient-based adversarial attacks on spiking neural networks (SNNs), which largely rely on extending FGSM and PGD frameworks, face a critical limitation: substantial attack latency from multi-timestep processing, rendering them infeasible for practical real-time applications. This inefficiency stems from their design as direct extensions of ANN paradigms, which fail to exploit key SNN properties. In this paper, we propose the timestep-compressed attack (TCA), a novel framework that significantly reduces attack latency. TCA introduces two components founded on key insights into SNN behavior. First, timestep-level backpropagation (TLBP) is based on our finding that global temporal information in backpropagation to generate perturbations is not critical for an attack's success, enabling per-timestep evaluation for early stopping. Second, adversarial membrane potential reuse (A-MPR) is motivated by the observation that initial timesteps are inefficiently spent accumulating membrane potential, a warm-up phase that can be pre-calculated and reused. Our experiments on VGG-11 and ResNet-17 with the CIFAR-10/100 and CIFAR10-DVS datasets show that TCA significantly reduces the required attack latency by up to 56.6% and 57.1% compared to SOTA methods in white-box and black-box settings, respectively, while maintaining a comparable attack success rate.



## **27. Enhancing Targeted Adversarial Attacks on Large Vision-Language Models through Intermediate Projector Guidance**

cs.CV

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.13739v1) [paper-pdf](http://arxiv.org/pdf/2508.13739v1)

**Authors**: Yiming Cao, Yanjie Li, Kaisheng Liang, Yuni Lai, Bin Xiao

**Abstract**: Targeted adversarial attacks are essential for proactively identifying security flaws in Vision-Language Models before real-world deployment. However, current methods perturb images to maximize global similarity with the target text or reference image at the encoder level, collapsing rich visual semantics into a single global vector. This limits attack granularity, hindering fine-grained manipulations such as modifying a car while preserving its background. Furthermore, these methods largely overlook the projector module, a critical semantic bridge between the visual encoder and the language model in VLMs, thereby failing to disrupt the full vision-language alignment pipeline within VLMs and limiting attack effectiveness. To address these issues, we propose the Intermediate Projector Guided Attack (IPGA), the first method to attack using the intermediate stage of the projector module, specifically the widely adopted Q-Former, which transforms global image embeddings into fine-grained visual features. This enables more precise control over adversarial perturbations by operating on semantically meaningful visual tokens rather than a single global representation. Specifically, IPGA leverages the Q-Former pretrained solely on the first vision-language alignment stage, without LLM fine-tuning, which improves both attack effectiveness and transferability across diverse VLMs. Furthermore, we propose Residual Query Alignment (RQA) to preserve unrelated visual content, thereby yielding more controlled and precise adversarial manipulations. Extensive experiments show that our attack method consistently outperforms existing methods in both standard global image captioning tasks and fine-grained visual question-answering tasks in black-box environment. Additionally, IPGA successfully transfers to multiple commercial VLMs, including Google Gemini and OpenAI GPT.



## **28. The AI Risk Spectrum: From Dangerous Capabilities to Existential Threats**

cs.CY

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.13700v1) [paper-pdf](http://arxiv.org/pdf/2508.13700v1)

**Authors**: Markov Grey, Charbel-Raphaël Segerie

**Abstract**: As AI systems become more capable, integrated, and widespread, understanding the associated risks becomes increasingly important. This paper maps the full spectrum of AI risks, from current harms affecting individual users to existential threats that could endanger humanity's survival. We organize these risks into three main causal categories. Misuse risks, which occur when people deliberately use AI for harmful purposes - creating bioweapons, launching cyberattacks, adversarial AI attacks or deploying lethal autonomous weapons. Misalignment risks happen when AI systems pursue outcomes that conflict with human values, irrespective of developer intentions. This includes risks arising through specification gaming (reward hacking), scheming and power-seeking tendencies in pursuit of long-term strategic goals. Systemic risks, which arise when AI integrates into complex social systems in ways that gradually undermine human agency - concentrating power, accelerating political and economic disempowerment, creating overdependence that leads to human enfeeblement, or irreversibly locking in current values curtailing future moral progress. Beyond these core categories, we identify risk amplifiers - competitive pressures, accidents, corporate indifference, and coordination failures - that make all risks more likely and severe. Throughout, we connect today's existing risks and empirically observable AI behaviors to plausible future outcomes, demonstrating how existing trends could escalate to catastrophic outcomes. Our goal is to help readers understand the complete landscape of AI risks. Good futures are possible, but they don't happen by default. Navigating these challenges will require unprecedented coordination, but an extraordinary future awaits if we do.



## **29. Mask and Restore: Blind Backdoor Defense at Test Time with Masked Autoencoder**

cs.LG

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2303.15564v3) [paper-pdf](http://arxiv.org/pdf/2303.15564v3)

**Authors**: Tao Sun, Lu Pang, Weimin Lyu, Chao Chen, Haibin Ling

**Abstract**: Deep neural networks are vulnerable to backdoor attacks, where an adversary manipulates the model behavior through overlaying images with special triggers. Existing backdoor defense methods often require accessing a few validation data and model parameters, which is impractical in many real-world applications, e.g., when the model is provided as a cloud service. In this paper, we address the practical task of blind backdoor defense at test time, in particular for local attacks and black-box models. The true label of every test image needs to be recovered on the fly from a suspicious model regardless of image benignity. We consider test-time image purification that incapacitates local triggers while keeping semantic contents intact. Due to diverse trigger patterns and sizes, the heuristic trigger search can be unscalable. We circumvent such barrier by leveraging the strong reconstruction power of generative models, and propose Blind Defense with Masked AutoEncoder (BDMAE). BDMAE detects possible local triggers using image structural similarity and label consistency between the test image and MAE restorations. The detection results are then refined by considering trigger topology. Finally, we fuse MAE restorations adaptively into a purified image for making prediction. Extensive experiments under different backdoor settings validate its effectiveness and generalizability.



## **30. Robust Federated Learning under Adversarial Attacks via Loss-Based Client Clustering**

cs.LG

16 pages, 5 figures

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.12672v2) [paper-pdf](http://arxiv.org/pdf/2508.12672v2)

**Authors**: Emmanouil Kritharakis, Dusan Jakovetic, Antonios Makris, Konstantinos Tserpes

**Abstract**: Federated Learning (FL) enables collaborative model training across multiple clients without sharing private data. We consider FL scenarios wherein FL clients are subject to adversarial (Byzantine) attacks, while the FL server is trusted (honest) and has a trustworthy side dataset. This may correspond to, e.g., cases where the server possesses trusted data prior to federation, or to the presence of a trusted client that temporarily assumes the server role. Our approach requires only two honest participants, i.e., the server and one client, to function effectively, without prior knowledge of the number of malicious clients. Theoretical analysis demonstrates bounded optimality gaps even under strong Byzantine attacks. Experimental results show that our algorithm significantly outperforms standard and robust FL baselines such as Mean, Trimmed Mean, Median, Krum, and Multi-Krum under various attack strategies including label flipping, sign flipping, and Gaussian noise addition across MNIST, FMNIST, and CIFAR-10 benchmarks using the Flower framework.



## **31. CCFC: Core & Core-Full-Core Dual-Track Defense for LLM Jailbreak Protection**

cs.CR

11 pages, 1 figure

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.14128v1) [paper-pdf](http://arxiv.org/pdf/2508.14128v1)

**Authors**: Jiaming Hu, Haoyu Wang, Debarghya Mukherjee, Ioannis Ch. Paschalidis

**Abstract**: Jailbreak attacks pose a serious challenge to the safe deployment of large language models (LLMs). We introduce CCFC (Core & Core-Full-Core), a dual-track, prompt-level defense framework designed to mitigate LLMs' vulnerabilities from prompt injection and structure-aware jailbreak attacks. CCFC operates by first isolating the semantic core of a user query via few-shot prompting, and then evaluating the query using two complementary tracks: a core-only track to ignore adversarial distractions (e.g., toxic suffixes or prefix injections), and a core-full-core (CFC) track to disrupt the structural patterns exploited by gradient-based or edit-based attacks. The final response is selected based on a safety consistency check across both tracks, ensuring robustness without compromising on response quality. We demonstrate that CCFC cuts attack success rates by 50-75% versus state-of-the-art defenses against strong adversaries (e.g., DeepInception, GCG), without sacrificing fidelity on benign queries. Our method consistently outperforms state-of-the-art prompt-level defenses, offering a practical and effective solution for safer LLM deployment.



## **32. Boosting Adversarial Transferability for Hyperspectral Image Classification Using 3D Structure-invariant Transformation and Weighted Intermediate Feature Divergence**

cs.CV

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2506.10459v2) [paper-pdf](http://arxiv.org/pdf/2506.10459v2)

**Authors**: Chun Liu, Bingqian Zhu, Tao Xu, Zheng Zheng, Zheng Li, Wei Yang, Zhigang Han, Jiayao Wang

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to adversarial attacks, which pose security challenges to hyperspectral image (HSI) classification based on DNNs. Numerous adversarial attack methods have been designed in the domain of natural images. However, different from natural images, HSIs contains high-dimensional rich spectral information, which presents new challenges for generating adversarial examples. Based on the specific characteristics of HSIs, this paper proposes a novel method to enhance the transferability of the adversarial examples for HSI classification using 3D structure-invariant transformation and weighted intermediate feature divergence. While keeping the HSIs structure invariant, the proposed method divides the image into blocks in both spatial and spectral dimensions. Then, various transformations are applied on each block to increase input diversity and mitigate the overfitting to substitute models. Moreover, a weighted intermediate feature divergence loss is also designed by leveraging the differences between the intermediate features of original and adversarial examples. It constrains the perturbation direction by enlarging the feature maps of the original examples, and assigns different weights to different feature channels to destroy the features that have a greater impact on HSI classification. Extensive experiments demonstrate that the adversarial examples generated by the proposed method achieve more effective adversarial transferability on three public HSI datasets. Furthermore, the method maintains robust attack performance even under defense strategies.



## **33. A robust and composable device-independent protocol for oblivious transfer using (fully) untrusted quantum devices in the bounded storage model**

quant-ph

Major improvement in the main result (security against non-IID  devices)

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2404.11283v3) [paper-pdf](http://arxiv.org/pdf/2404.11283v3)

**Authors**: Rishabh Batra, Sayantan Chakraborty, Rahul Jain, Upendra Kapshikar

**Abstract**: We present a robust and composable device-independent (DI) quantum protocol between two parties for oblivious transfer (OT) using Magic Square devices in the bounded storage model in which the (honest and cheating) devices and parties have no long-term quantum memory. After a fixed constant (real-world) time interval, referred to as DELAY, the quantum states decohere completely. The adversary (cheating party), with full control over the devices, is allowed joint (non-IID) quantum operations on the devices, and there are no time and space complexity bounds placed on its powers. The running time of the honest parties is polylog({\lambda}) (where {\lambda} is the security parameter). Our protocol has negligible (in {\lambda}) correctness and security errors and can be implemented in the NISQ (Noisy Intermediate Scale Quantum) era. By robustness, we mean that our protocol is correct even when devices are slightly off (by a small constant) from their ideal specification. This is an important property since small manufacturing errors in the real-world devices are inevitable. Our protocol is sequentially composable and, hence, can be used as a building block to construct larger protocols (including DI bit-commitment and DI secure multi-party computation) while still preserving correctness and security guarantees.   None of the known DI protocols for OT in the literature are robust and secure against joint quantum attacks. This was a major open question in device-independent two-party distrustful cryptography, which we resolve.   We prove a parallel repetition theorem for a certain class of entangled games with a hybrid (quantum-classical) strategy to show the security of our protocol. The hybrid strategy helps to incorporate DELAY in our protocol. This parallel repetition theorem is a main technical contribution of our work.



## **34. A Risk Manager for Intrusion Tolerant Systems: Enhancing HAL 9000 with New Scoring and Data Sources**

cs.CR

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.13364v1) [paper-pdf](http://arxiv.org/pdf/2508.13364v1)

**Authors**: Tadeu Freitas, Carlos Novo, Inês Dutra, João Soares, Manuel Correia, Benham Shariati, Rolando Martins

**Abstract**: Intrusion Tolerant Systems (ITSs) have become increasingly critical due to the rise of multi-domain adversaries exploiting diverse attack surfaces. ITS architectures aim to tolerate intrusions, ensuring system compromise is prevented or mitigated even with adversary presence. Existing ITS solutions often employ Risk Managers leveraging public security intelligence to adjust system defenses dynamically against emerging threats. However, these approaches rely heavily on databases like NVD and ExploitDB, which require manual analysis for newly discovered vulnerabilities. This dependency limits the system's responsiveness to rapidly evolving threats. HAL 9000, an ITS Risk Manager introduced in our prior work, addressed these challenges through machine learning. By analyzing descriptions of known vulnerabilities, HAL 9000 predicts and assesses new vulnerabilities automatically. To calculate the risk of a system, it also incorporates the Exploitability Probability Scoring system to estimate the likelihood of exploitation within 30 days, enhancing proactive defense capabilities.   Despite its success, HAL 9000's reliance on NVD and ExploitDB knowledge is a limitation, considering the availability of other sources of information. This extended work introduces a custom-built scraper that continuously mines diverse threat sources, including security advisories, research forums, and real-time exploit proofs-of-concept. This significantly expands HAL 9000's intelligence base, enabling earlier detection and assessment of unverified vulnerabilities. Our evaluation demonstrates that integrating scraper-derived intelligence with HAL 9000's risk management framework substantially improves its ability to address emerging threats. This paper details the scraper's integration into the architecture, its role in providing additional information on new threats, and the effects on HAL 9000's management.



## **35. Augmented Adversarial Trigger Learning**

cs.LG

Findings of the Association for Computational Linguistics: NAACL 2025

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2503.12339v3) [paper-pdf](http://arxiv.org/pdf/2503.12339v3)

**Authors**: Zhe Wang, Yanjun Qi

**Abstract**: Gradient optimization-based adversarial attack methods automate the learning of adversarial triggers to generate jailbreak prompts or leak system prompts. In this work, we take a closer look at the optimization objective of adversarial trigger learning and propose ATLA: Adversarial Trigger Learning with Augmented objectives. ATLA improves the negative log-likelihood loss used by previous studies into a weighted loss formulation that encourages the learned adversarial triggers to optimize more towards response format tokens. This enables ATLA to learn an adversarial trigger from just one query-response pair and the learned trigger generalizes well to other similar queries. We further design a variation to augment trigger optimization with an auxiliary loss that suppresses evasive responses. We showcase how to use ATLA to learn adversarial suffixes jailbreaking LLMs and to extract hidden system prompts. Empirically we demonstrate that ATLA consistently outperforms current state-of-the-art techniques, achieving nearly 100% success in attacking while requiring 80% fewer queries. ATLA learned jailbreak suffixes demonstrate high generalization to unseen queries and transfer well to new LLMs. We released our code https://github.com/QData/ALTA_Augmented_Adversarial_Trigger_Learning



## **36. DAASH: A Meta-Attack Framework for Synthesizing Effective and Stealthy Adversarial Examples**

cs.CV

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.13309v1) [paper-pdf](http://arxiv.org/pdf/2508.13309v1)

**Authors**: Abdullah Al Nomaan Nafi, Habibur Rahaman, Zafaryab Haider, Tanzim Mahfuz, Fnu Suya, Swarup Bhunia, Prabuddha Chakraborty

**Abstract**: Numerous techniques have been proposed for generating adversarial examples in white-box settings under strict Lp-norm constraints. However, such norm-bounded examples often fail to align well with human perception, and only recently have a few methods begun specifically exploring perceptually aligned adversarial examples. Moreover, it remains unclear whether insights from Lp-constrained attacks can be effectively leveraged to improve perceptual efficacy. In this paper, we introduce DAASH, a fully differentiable meta-attack framework that generates effective and perceptually aligned adversarial examples by strategically composing existing Lp-based attack methods. DAASH operates in a multi-stage fashion: at each stage, it aggregates candidate adversarial examples from multiple base attacks using learned, adaptive weights and propagates the result to the next stage. A novel meta-loss function guides this process by jointly minimizing misclassification loss and perceptual distortion, enabling the framework to dynamically modulate the contribution of each base attack throughout the stages. We evaluate DAASH on adversarially trained models across CIFAR-10, CIFAR-100, and ImageNet. Despite relying solely on Lp-constrained based methods, DAASH significantly outperforms state-of-the-art perceptual attacks such as AdvAD -- achieving higher attack success rates (e.g., 20.63\% improvement) and superior visual quality, as measured by SSIM, LPIPS, and FID (improvements $\approx$ of 11, 0.015, and 5.7, respectively). Furthermore, DAASH generalizes well to unseen defenses, making it a practical and strong baseline for evaluating robustness without requiring handcrafted adaptive attacks for each new defense.



## **37. RoTO: Robust Topology Obfuscation Against Tomography Inference Attacks**

cs.NI

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.12852v1) [paper-pdf](http://arxiv.org/pdf/2508.12852v1)

**Authors**: Chengze Du, Heng Xu, Zhiwei Yu, Ying Zhou, Zili Meng, Jialong Li

**Abstract**: Tomography inference attacks aim to reconstruct network topology by analyzing end-to-end probe delays. Existing defenses mitigate these attacks by manipulating probe delays to mislead inference, but rely on two strong assumptions: (i) probe packets can be perfectly detected and altered, and (ii) attackers use known, fixed inference algorithms. These assumptions often break in practice, leading to degraded defense performance under detection errors or adaptive adversaries. We present RoTO, a robust topology obfuscation scheme that eliminates both assumptions by modeling uncertainty in attacker-observed delays through a distributional formulation. RoTO casts the defense objective as a min-max optimization problem that maximizes expected topological distortion across this uncertainty set, without relying on perfect probe control or specific attacker models. To approximate attacker behavior, RoTO leverages graph neural networks for inference simulation and adversarial training. We also derive an upper bound on attacker success probability, and demonstrate that our approach enhances topology obfuscation performance through the optimization of this upper bound. Experimental results show that RoTO outperforms existing defense methods, achieving average improvements of 34% in structural similarity and 42.6% in link distance while maintaining strong robustness and concealment capabilities.



## **38. Deep Positive-Negative Prototypes for Adversarially Robust Discriminative Prototypical Learning**

cs.LG

This version substantially revises the manuscript, including a new  title and updated experimental results

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2504.03782v2) [paper-pdf](http://arxiv.org/pdf/2504.03782v2)

**Authors**: Ramin Zarei Sabzevar, Hamed Mohammadzadeh, Tahmineh Tavakoli, Ahad Harati

**Abstract**: Despite the advantages of discriminative prototype-based methods, their role in adversarial robustness remains underexplored. Meanwhile, current adversarial training methods predominantly focus on robustness against adversarial attacks without explicitly leveraging geometric structures in the latent space, usually resulting in reduced accuracy on the original clean data. We propose a novel framework named Adversarially trained Deep Positive-Negative Prototypes (Adv-DPNP), which integrates discriminative prototype-based learning with adversarial training. Adv-DPNP uses unified class prototypes that serve as both classifier weights and robust anchors in the latent space. Moreover, a novel dual-branch training mechanism maintains stable prototypes by updating them exclusively with clean data, while the feature extractor is trained on both clean and adversarial inputs to increase invariance to adversarial perturbations. In addition, we use a composite loss that combines positive-prototype alignment, negative-prototype repulsion, and consistency regularization to further enhance discrimination, adversarial robustness, and clean accuracy. Extensive experiments on standard benchmarks (CIFAR-10/100 and SVHN) confirm that Adv-DPNP improves clean accuracy over state-of-the-art defenses and baseline methods, while maintaining competitive or superior robustness under a suite of widely used attacks, including FGSM, PGD, C\&W, and AutoAttack. We also evaluate robustness to common corruptions on CIFAR-10-C, where Adv-DPNP achieves the highest average accuracy across severities and corruption types. Additionally, we provide an in-depth analysis of the discriminative quality of the learned feature representations, highlighting the effectiveness of Adv-DPNP in maintaining compactness and clear separation in the latent space.



## **39. Boosting Active Defense Persistence: A Two-Stage Defense Framework Combining Interruption and Poisoning Against Deepfake**

cs.CV

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.07795v2) [paper-pdf](http://arxiv.org/pdf/2508.07795v2)

**Authors**: Hongrui Zheng, Yuezun Li, Liejun Wang, Yunfeng Diao, Zhiqing Guo

**Abstract**: Active defense strategies have been developed to counter the threat of deepfake technology. However, a primary challenge is their lack of persistence, as their effectiveness is often short-lived. Attackers can bypass these defenses by simply collecting protected samples and retraining their models. This means that static defenses inevitably fail when attackers retrain their models, which severely limits practical use. We argue that an effective defense not only distorts forged content but also blocks the model's ability to adapt, which occurs when attackers retrain their models on protected images. To achieve this, we propose an innovative Two-Stage Defense Framework (TSDF). Benefiting from the intensity separation mechanism designed in this paper, the framework uses dual-function adversarial perturbations to perform two roles. First, it can directly distort the forged results. Second, it acts as a poisoning vehicle that disrupts the data preparation process essential for an attacker's retraining pipeline. By poisoning the data source, TSDF aims to prevent the attacker's model from adapting to the defensive perturbations, thus ensuring the defense remains effective long-term. Comprehensive experiments show that the performance of traditional interruption methods degrades sharply when it is subjected to adversarial retraining. However, our framework shows a strong dual defense capability, which can improve the persistence of active defense. Our code will be available at https://github.com/vpsg-research/TSDF.



## **40. Concealment of Intent: A Game-Theoretic Analysis**

cs.CL

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2505.20841v2) [paper-pdf](http://arxiv.org/pdf/2505.20841v2)

**Authors**: Xinbo Wu, Abhishek Umrawal, Lav R. Varshney

**Abstract**: As large language models (LLMs) grow more capable, concerns about their safe deployment have also grown. Although alignment mechanisms have been introduced to deter misuse, they remain vulnerable to carefully designed adversarial prompts. In this work, we present a scalable attack strategy: intent-hiding adversarial prompting, which conceals malicious intent through the composition of skills. We develop a game-theoretic framework to model the interaction between such attacks and defense systems that apply both prompt and response filtering. Our analysis identifies equilibrium points and reveals structural advantages for the attacker. To counter these threats, we propose and analyze a defense mechanism tailored to intent-hiding attacks. Empirically, we validate the attack's effectiveness on multiple real-world LLMs across a range of malicious behaviors, demonstrating clear advantages over existing adversarial prompting techniques.



## **41. Quantifying Loss Aversion in Cyber Adversaries via LLM Analysis**

cs.CR

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.13240v1) [paper-pdf](http://arxiv.org/pdf/2508.13240v1)

**Authors**: Soham Hans, Nikolos Gurney, Stacy Marsella, Sofia Hirschmann

**Abstract**: Understanding and quantifying human cognitive biases from empirical data has long posed a formidable challenge, particularly in cybersecurity, where defending against unknown adversaries is paramount. Traditional cyber defense strategies have largely focused on fortification, while some approaches attempt to anticipate attacker strategies by mapping them to cognitive vulnerabilities, yet they fall short in dynamically interpreting attacks in progress. In recognition of this gap, IARPA's ReSCIND program seeks to infer, defend against, and even exploit attacker cognitive traits. In this paper, we present a novel methodology that leverages large language models (LLMs) to extract quantifiable insights into the cognitive bias of loss aversion from hacker behavior. Our data are collected from an experiment in which hackers were recruited to attack a controlled demonstration network. We process the hacker generated notes using LLMs using it to segment the various actions and correlate the actions to predefined persistence mechanisms used by hackers. By correlating the implementation of these mechanisms with various operational triggers, our analysis provides new insights into how loss aversion manifests in hacker decision-making. The results demonstrate that LLMs can effectively dissect and interpret nuanced behavioral patterns, thereby offering a transformative approach to enhancing cyber defense strategies through real-time, behavior-based analysis.



## **42. Heuristic-Induced Multimodal Risk Distribution Jailbreak Attack for Multimodal Large Language Models**

cs.CR

ICCV 2025

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2412.05934v3) [paper-pdf](http://arxiv.org/pdf/2412.05934v3)

**Authors**: Ma Teng, Jia Xiaojun, Duan Ranjie, Li Xinfeng, Huang Yihao, Jia Xiaoshuang, Chu Zhixuan, Ren Wenqi

**Abstract**: With the rapid advancement of multimodal large language models (MLLMs), concerns regarding their security have increasingly captured the attention of both academia and industry. Although MLLMs are vulnerable to jailbreak attacks, designing effective jailbreak attacks poses unique challenges, especially given the highly constrained adversarial capabilities in real-world deployment scenarios. Previous works concentrate risks into a single modality, resulting in limited jailbreak performance. In this paper, we propose a heuristic-induced multimodal risk distribution jailbreak attack method, called HIMRD, which is black-box and consists of two elements: multimodal risk distribution strategy and heuristic-induced search strategy. The multimodal risk distribution strategy is used to distribute harmful semantics into multiple modalities to effectively circumvent the single-modality protection mechanisms of MLLMs. The heuristic-induced search strategy identifies two types of prompts: the understanding-enhancing prompt, which helps MLLMs reconstruct the malicious prompt, and the inducing prompt, which increases the likelihood of affirmative outputs over refusals, enabling a successful jailbreak attack. HIMRD achieves an average attack success rate (ASR) of 90% across seven open-source MLLMs and an average ASR of around 68% in three closed-source MLLMs. HIMRD reveals cross-modal security vulnerabilities in current MLLMs and underscores the imperative for developing defensive strategies to mitigate such emerging risks. Code is available at https://github.com/MaTengSYSU/HIMRD-jailbreak.



## **43. Reducing False Positives with Active Behavioral Analysis for Cloud Security**

cs.CR

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.12584v1) [paper-pdf](http://arxiv.org/pdf/2508.12584v1)

**Authors**: Dikshant, Verma

**Abstract**: Rule-based cloud security posture management (CSPM) solutions are known to produce a lot of false positives based on the limited contextual understanding and dependence on static heuristics testing. This paper introduces a validation-driven methodology that integrates active behavioral testing in cloud security posture management solution(s) to evaluate the exploitability of policy violations in real time. The proposed system employs lightweight and automated probes, built from open-source tools, validation scripts, and penetration testing test cases, to simulate adversarial attacks on misconfigured or vulnerable cloud assets without any impact to the cloud services or environment. For instance, cloud services may be flagged as publicly exposed and vulnerable despite being protected by access control layers, or secure policies, resulting in non-actionable alerts that consumes analysts time during manual validation. Through controlled experimentation in a reproducible AWS setup, we evaluated the reduction in false positive rates across various misconfiguration and vulnerable alerts. Our findings indicate an average reduction of 93\% in false positives. Furthermore, the framework demonstrates low latency performance. These results demonstrate a scalable method to improve detection accuracy and analyst productivity in large cloud environments. While our evaluation focuses on AWS, the architecture is modular and extensible to multi-cloud setups.



## **44. Adversarial Attacks on VQA-NLE: Exposing and Alleviating Inconsistencies in Visual Question Answering Explanations**

cs.CV

**SubmitDate**: 2025-08-17    [abs](http://arxiv.org/abs/2508.12430v1) [paper-pdf](http://arxiv.org/pdf/2508.12430v1)

**Authors**: Yahsin Yeh, Yilun Wu, Bokai Ruan, Honghan Shuai

**Abstract**: Natural language explanations in visual question answering (VQA-NLE) aim to make black-box models more transparent by elucidating their decision-making processes. However, we find that existing VQA-NLE systems can produce inconsistent explanations and reach conclusions without genuinely understanding the underlying context, exposing weaknesses in either their inference pipeline or explanation-generation mechanism. To highlight these vulnerabilities, we not only leverage an existing adversarial strategy to perturb questions but also propose a novel strategy that minimally alters images to induce contradictory or spurious outputs. We further introduce a mitigation method that leverages external knowledge to alleviate these inconsistencies, thereby bolstering model robustness. Extensive evaluations on two standard benchmarks and two widely used VQA-NLE models underscore the effectiveness of our attacks and the potential of knowledge-based defenses, ultimately revealing pressing security and reliability concerns in current VQA-NLE systems.



## **45. Cascading and Proxy Membership Inference Attacks**

cs.CR

Accepted by NDSS 2026

**SubmitDate**: 2025-08-17    [abs](http://arxiv.org/abs/2507.21412v2) [paper-pdf](http://arxiv.org/pdf/2507.21412v2)

**Authors**: Yuntao Du, Jiacheng Li, Yuetian Chen, Kaiyuan Zhang, Zhizhen Yuan, Hanshen Xiao, Bruno Ribeiro, Ninghui Li

**Abstract**: A Membership Inference Attack (MIA) assesses how much a trained machine learning model reveals about its training data by determining whether specific query instances were included in the dataset. We classify existing MIAs into adaptive or non-adaptive, depending on whether the adversary is allowed to train shadow models on membership queries. In the adaptive setting, where the adversary can train shadow models after accessing query instances, we highlight the importance of exploiting membership dependencies between instances and propose an attack-agnostic framework called Cascading Membership Inference Attack (CMIA), which incorporates membership dependencies via conditional shadow training to boost membership inference performance.   In the non-adaptive setting, where the adversary is restricted to training shadow models before obtaining membership queries, we introduce Proxy Membership Inference Attack (PMIA). PMIA employs a proxy selection strategy that identifies samples with similar behaviors to the query instance and uses their behaviors in shadow models to perform a membership posterior odds test for membership inference. We provide theoretical analyses for both attacks, and extensive experimental results demonstrate that CMIA and PMIA substantially outperform existing MIAs in both settings, particularly in the low false-positive regime, which is crucial for evaluating privacy risks.



## **46. ViT-EnsembleAttack: Augmenting Ensemble Models for Stronger Adversarial Transferability in Vision Transformers**

cs.CV

**SubmitDate**: 2025-08-17    [abs](http://arxiv.org/abs/2508.12384v1) [paper-pdf](http://arxiv.org/pdf/2508.12384v1)

**Authors**: Hanwen Cao, Haobo Lu, Xiaosen Wang, Kun He

**Abstract**: Ensemble-based attacks have been proven to be effective in enhancing adversarial transferability by aggregating the outputs of models with various architectures. However, existing research primarily focuses on refining ensemble weights or optimizing the ensemble path, overlooking the exploration of ensemble models to enhance the transferability of adversarial attacks. To address this gap, we propose applying adversarial augmentation to the surrogate models, aiming to boost overall generalization of ensemble models and reduce the risk of adversarial overfitting. Meanwhile, observing that ensemble Vision Transformers (ViTs) gain less attention, we propose ViT-EnsembleAttack based on the idea of model adversarial augmentation, the first ensemble-based attack method tailored for ViTs to the best of our knowledge. Our approach generates augmented models for each surrogate ViT using three strategies: Multi-head dropping, Attention score scaling, and MLP feature mixing, with the associated parameters optimized by Bayesian optimization. These adversarially augmented models are ensembled to generate adversarial examples. Furthermore, we introduce Automatic Reweighting and Step Size Enlargement modules to boost transferability. Extensive experiments demonstrate that ViT-EnsembleAttack significantly enhances the adversarial transferability of ensemble-based attacks on ViTs, outperforming existing methods by a substantial margin. Code is available at https://github.com/Trustworthy-AI-Group/TransferAttack.



## **47. Jamming Identification with Differential Transformer for Low-Altitude Wireless Networks**

eess.SP

**SubmitDate**: 2025-08-17    [abs](http://arxiv.org/abs/2508.12320v1) [paper-pdf](http://arxiv.org/pdf/2508.12320v1)

**Authors**: Pengyu Wang, Zhaocheng Wang, Tianqi Mao, Weijie Yuan, Haijun Zhang, George K. Karagiannidis

**Abstract**: Wireless jamming identification, which detects and classifies electromagnetic jamming from non-cooperative devices, is crucial for emerging low-altitude wireless networks consisting of many drone terminals that are highly susceptible to electromagnetic jamming. However, jamming identification schemes adopting deep learning (DL) are vulnerable to attacks involving carefully crafted adversarial samples, resulting in inevitable robustness degradation. To address this issue, we propose a differential transformer framework for wireless jamming identification. Firstly, we introduce a differential transformer network in order to distinguish jamming signals, which overcomes the attention noise when compared with its traditional counterpart by performing self-attention operations in a differential manner. Secondly, we propose a randomized masking training strategy to improve network robustness, which leverages the patch partitioning mechanism inherent to transformer architectures in order to create parallel feature extraction branches. Each branch operates on a distinct, randomly masked subset of patches, which fundamentally constrains the propagation of adversarial perturbations across the network. Additionally, the ensemble effect generated by fusing predictions from these diverse branches demonstrates superior resilience against adversarial attacks. Finally, we introduce a novel consistent training framework that significantly enhances adversarial robustness through dualbranch regularization. Simulation results demonstrate that our proposed methodology is superior to existing methods in boosting robustness to adversarial samples.



## **48. Adjustable AprilTags For Identity Secured Tasks**

cs.CR

**SubmitDate**: 2025-08-17    [abs](http://arxiv.org/abs/2508.12304v1) [paper-pdf](http://arxiv.org/pdf/2508.12304v1)

**Authors**: Hao Li

**Abstract**: Special tags such as AprilTags that facilitate image processing and pattern recognition are useful in practical applications. In close and private environments, identity security is unlikely to be an issue because all involved AprilTags can be completely regulated. However, in open and public environments, identity security is no longer an issue that can be neglected. To handle potential harm caused by adversarial attacks, this note advocates utilization of adjustable AprilTags instead of fixed ones.



## **49. ForensicsSAM: Toward Robust and Unified Image Forgery Detection and Localization Resisting to Adversarial Attack**

cs.CV

**SubmitDate**: 2025-08-17    [abs](http://arxiv.org/abs/2508.07402v2) [paper-pdf](http://arxiv.org/pdf/2508.07402v2)

**Authors**: Rongxuan Peng, Shunquan Tan, Chenqi Kong, Anwei Luo, Alex C. Kot, Jiwu Huang

**Abstract**: Parameter-efficient fine-tuning (PEFT) has emerged as a popular strategy for adapting large vision foundation models, such as the Segment Anything Model (SAM) and LLaVA, to downstream tasks like image forgery detection and localization (IFDL). However, existing PEFT-based approaches overlook their vulnerability to adversarial attacks. In this paper, we show that highly transferable adversarial images can be crafted solely via the upstream model, without accessing the downstream model or training data, significantly degrading the IFDL performance. To address this, we propose ForensicsSAM, a unified IFDL framework with built-in adversarial robustness. Our design is guided by three key ideas: (1) To compensate for the lack of forgery-relevant knowledge in the frozen image encoder, we inject forgery experts into each transformer block to enhance its ability to capture forgery artifacts. These forgery experts are always activated and shared across any input images. (2) To detect adversarial images, we design an light-weight adversary detector that learns to capture structured, task-specific artifact in RGB domain, enabling reliable discrimination across various attack methods. (3) To resist adversarial attacks, we inject adversary experts into the global attention layers and MLP modules to progressively correct feature shifts induced by adversarial noise. These adversary experts are adaptively activated by the adversary detector, thereby avoiding unnecessary interference with clean images. Extensive experiments across multiple benchmarks demonstrate that ForensicsSAM achieves superior resistance to various adversarial attack methods, while also delivering state-of-the-art performance in image-level forgery detection and pixel-level forgery localization. The resource is available at https://github.com/siriusPRX/ForensicsSAM.



## **50. TriQDef: Disrupting Semantic and Gradient Alignment to Prevent Adversarial Patch Transferability in Quantized Neural Networks**

cs.CV

**SubmitDate**: 2025-08-16    [abs](http://arxiv.org/abs/2508.12132v1) [paper-pdf](http://arxiv.org/pdf/2508.12132v1)

**Authors**: Amira Guesmi, Bassem Ouni, Muhammad Shafique

**Abstract**: Quantized Neural Networks (QNNs) are increasingly deployed in edge and resource-constrained environments due to their efficiency in computation and memory usage. While shown to distort the gradient landscape and weaken conventional pixel-level attacks, it provides limited robustness against patch-based adversarial attacks-localized, high-saliency perturbations that remain surprisingly transferable across bit-widths. Existing defenses either overfit to fixed quantization settings or fail to address this cross-bit generalization vulnerability. We introduce \textbf{TriQDef}, a tri-level quantization-aware defense framework designed to disrupt the transferability of patch-based adversarial attacks across QNNs. TriQDef consists of: (1) a Feature Disalignment Penalty (FDP) that enforces semantic inconsistency by penalizing perceptual similarity in intermediate representations; (2) a Gradient Perceptual Dissonance Penalty (GPDP) that explicitly misaligns input gradients across bit-widths by minimizing structural and directional agreement via Edge IoU and HOG Cosine metrics; and (3) a Joint Quantization-Aware Training Protocol that unifies these penalties within a shared-weight training scheme across multiple quantization levels. Extensive experiments on CIFAR-10 and ImageNet demonstrate that TriQDef reduces Attack Success Rates (ASR) by over 40\% on unseen patch and quantization combinations, while preserving high clean accuracy. Our findings underscore the importance of disrupting both semantic and perceptual gradient alignment to mitigate patch transferability in QNNs.



