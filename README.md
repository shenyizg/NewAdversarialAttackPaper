# Latest Adversarial Attack Papers
**update at 2025-04-28 17:21:36**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Intelligent Attacks and Defense Methods in Federated Learning-enabled Energy-Efficient Wireless Networks**

cs.LG

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.18519v1) [paper-pdf](http://arxiv.org/pdf/2504.18519v1)

**Authors**: Han Zhang, Hao Zhou, Medhat Elsayed, Majid Bavand, Raimundas Gaigalas, Yigit Ozcan, Melike Erol-Kantarci

**Abstract**: Federated learning (FL) is a promising technique for learning-based functions in wireless networks, thanks to its distributed implementation capability. On the other hand, distributed learning may increase the risk of exposure to malicious attacks where attacks on a local model may spread to other models by parameter exchange. Meanwhile, such attacks can be hard to detect due to the dynamic wireless environment, especially considering local models can be heterogeneous with non-independent and identically distributed (non-IID) data. Therefore, it is critical to evaluate the effect of malicious attacks and develop advanced defense techniques for FL-enabled wireless networks. In this work, we introduce a federated deep reinforcement learning-based cell sleep control scenario that enhances the energy efficiency of the network. We propose multiple intelligent attacks targeting the learning-based approach and we propose defense methods to mitigate such attacks. In particular, we have designed two attack models, generative adversarial network (GAN)-enhanced model poisoning attack and regularization-based model poisoning attack. As a counteraction, we have proposed two defense schemes, autoencoder-based defense, and knowledge distillation (KD)-enabled defense. The autoencoder-based defense method leverages an autoencoder to identify the malicious participants and only aggregate the parameters of benign local models during the global aggregation, while KD-based defense protects the model from attacks by controlling the knowledge transferred between the global model and local models.



## **2. Distributed Multiple Testing with False Discovery Rate Control in the Presence of Byzantines**

eess.SP

Accepted to the 2025 International Symposium on Information Theory  (ISIT)

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2501.13242v2) [paper-pdf](http://arxiv.org/pdf/2501.13242v2)

**Authors**: Daofu Zhang, Mehrdad Pournaderi, Yu Xiang, Pramod Varshney

**Abstract**: This work studies distributed multiple testing with false discovery rate (FDR) control in the presence of Byzantine attacks, where an adversary captures a fraction of the nodes and corrupts their reported p-values. We focus on two baseline attack models: an oracle model with the full knowledge of which hypotheses are true nulls, and a practical attack model that leverages the Benjamini-Hochberg (BH) procedure locally to classify which p-values follow the true null hypotheses. We provide a thorough characterization of how both attack models affect the global FDR, which in turn motivates counter-attack strategies and stronger attack models. Our extensive simulation studies confirm the theoretical results, highlight key design trade-offs under attacks and countermeasures, and provide insights into more sophisticated attacks.



## **3. Adversarial Attacks on LLM-as-a-Judge Systems: Insights from Prompt Injections**

cs.CR

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.18333v1) [paper-pdf](http://arxiv.org/pdf/2504.18333v1)

**Authors**: Narek Maloyan, Dmitry Namiot

**Abstract**: LLM as judge systems used to assess text quality code correctness and argument strength are vulnerable to prompt injection attacks. We introduce a framework that separates content author attacks from system prompt attacks and evaluate five models Gemma 3.27B Gemma 3.4B Llama 3.2 3B GPT 4 and Claude 3 Opus on four tasks with various defenses using fifty prompts per condition. Attacks achieved up to seventy three point eight percent success smaller models proved more vulnerable and transferability ranged from fifty point five to sixty two point six percent. Our results contrast with Universal Prompt Injection and AdvPrompter We recommend multi model committees and comparative scoring and release all code and datasets



## **4. Contrastive Learning and Adversarial Disentanglement for Task-Oriented Semantic Communications**

cs.LG

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2410.22784v2) [paper-pdf](http://arxiv.org/pdf/2410.22784v2)

**Authors**: Omar Erak, Omar Alhussein, Wen Tong

**Abstract**: Task-oriented semantic communication systems have emerged as a promising approach to achieving efficient and intelligent data transmission, where only information relevant to a specific task is communicated. However, existing methods struggle to fully disentangle task-relevant and task-irrelevant information, leading to privacy concerns and subpar performance. To address this, we propose an information-bottleneck method, named CLAD (contrastive learning and adversarial disentanglement). CLAD utilizes contrastive learning to effectively capture task-relevant features while employing adversarial disentanglement to discard task-irrelevant information. Additionally, due to the lack of reliable and reproducible methods to gain insight into the informativeness and minimality of the encoded feature vectors, we introduce a new technique to compute the information retention index (IRI), a comparative metric used as a proxy for the mutual information between the encoded features and the input, reflecting the minimality of the encoded features. The IRI quantifies the minimality and informativeness of the encoded feature vectors across different task-oriented communication techniques. Our extensive experiments demonstrate that CLAD outperforms state-of-the-art baselines in terms of semantic extraction, task performance, privacy preservation, and IRI. CLAD achieves a predictive performance improvement of around 2.5-3%, along with a 77-90% reduction in IRI and a 57-76% decrease in adversarial attribute inference attack accuracy.



## **5. Manipulating Multimodal Agents via Cross-Modal Prompt Injection**

cs.CV

17 pages, 5 figures

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.14348v3) [paper-pdf](http://arxiv.org/pdf/2504.14348v3)

**Authors**: Le Wang, Zonghao Ying, Tianyuan Zhang, Siyuan Liang, Shengshan Hu, Mingchuan Zhang, Aishan Liu, Xianglong Liu

**Abstract**: The emergence of multimodal large language models has redefined the agent paradigm by integrating language and vision modalities with external data sources, enabling agents to better interpret human instructions and execute increasingly complex tasks. However, in this work, we identify a critical yet previously overlooked security vulnerability in multimodal agents: cross-modal prompt injection attacks. To exploit this vulnerability, we propose CrossInject, a novel attack framework in which attackers embed adversarial perturbations across multiple modalities to align with target malicious content, allowing external instructions to hijack the agent's decision-making process and execute unauthorized tasks. Our approach consists of two key components. First, we introduce Visual Latent Alignment, where we optimize adversarial features to the malicious instructions in the visual embedding space based on a text-to-image generative model, ensuring that adversarial images subtly encode cues for malicious task execution. Subsequently, we present Textual Guidance Enhancement, where a large language model is leveraged to infer the black-box defensive system prompt through adversarial meta prompting and generate an malicious textual command that steers the agent's output toward better compliance with attackers' requests. Extensive experiments demonstrate that our method outperforms existing injection attacks, achieving at least a +26.4% increase in attack success rates across diverse tasks. Furthermore, we validate our attack's effectiveness in real-world multimodal autonomous agents, highlighting its potential implications for safety-critical applications.



## **6. Robust Kernel Hypothesis Testing under Data Corruption**

stat.ML

22 pages, 2 figures, 2 algorithms

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2405.19912v3) [paper-pdf](http://arxiv.org/pdf/2405.19912v3)

**Authors**: Antonin Schrab, Ilmun Kim

**Abstract**: We propose a general method for constructing robust permutation tests under data corruption. The proposed tests effectively control the non-asymptotic type I error under data corruption, and we prove their consistency in power under minimal conditions. This contributes to the practical deployment of hypothesis tests for real-world applications with potential adversarial attacks. For the two-sample and independence settings, we show that our kernel robust tests are minimax optimal, in the sense that they are guaranteed to be non-asymptotically powerful against alternatives uniformly separated from the null in the kernel MMD and HSIC metrics at some optimal rate (tight with matching lower bound). We point out that existing differentially private tests can be adapted to be robust to data corruption, and we demonstrate in experiments that our proposed tests achieve much higher power than these private tests. Finally, we provide publicly available implementations and empirically illustrate the practicality of our robust tests.



## **7. Generative AI for Physical-Layer Authentication**

eess.SP

10 pages, 3 figures

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.18175v1) [paper-pdf](http://arxiv.org/pdf/2504.18175v1)

**Authors**: Rui Meng, Xiqi Cheng, Song Gao, Xiaodong Xu, Chen Dong, Guoshun Nan, Xiaofeng Tao, Ping Zhang, Tony Q. S. Quek

**Abstract**: In recent years, Artificial Intelligence (AI)-driven Physical-Layer Authentication (PLA), which focuses on achieving endogenous security and intelligent identity authentication, has attracted considerable interest. When compared with Discriminative AI (DAI), Generative AI (GAI) offers several advantages, such as fingerprint data augmentation, fingerprint denoising and reconstruction, and protection against adversarial attacks. Inspired by these innovations, this paper provides a systematic exploration of GAI's integration into PLA frameworks. We commence with a review of representative authentication techniques, emphasizing PLA's inherent strengths. Following this, we revisit four typical GAI models and contrast the limitations of DAI with the potential of GAI in addressing PLA challenges, including insufficient fingerprint data, environment noises and inferences, perturbations in fingerprint data, and complex tasks. Specifically, we delve into providing GAI-enhance methods for PLA across the data, model, and application layers in detail. Moreover, we present a case study that combines fingerprint extrapolation, generative diffusion models, and cooperative nodes to illustrate the superiority of GAI in bolstering the reliability of PLA compared to DAI. Additionally, we outline potential future research directions for GAI-based PLA.



## **8. Revisiting Locally Differentially Private Protocols: Towards Better Trade-offs in Privacy, Utility, and Attack Resistance**

cs.CR

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2503.01482v2) [paper-pdf](http://arxiv.org/pdf/2503.01482v2)

**Authors**: Héber H. Arcolezi, Sébastien Gambs

**Abstract**: Local Differential Privacy (LDP) offers strong privacy protection, especially in settings in which the server collecting the data is untrusted. However, designing LDP mechanisms that achieve an optimal trade-off between privacy, utility and robustness to adversarial inference attacks remains challenging. In this work, we introduce a general multi-objective optimization framework for refining LDP protocols, enabling the joint optimization of privacy and utility under various adversarial settings. While our framework is flexible to accommodate multiple privacy and security attacks as well as utility metrics, in this paper, we specifically optimize for Attacker Success Rate (ASR) under \emph{data reconstruction attack} as a concrete measure of privacy leakage and Mean Squared Error (MSE) as a measure of utility. More precisely, we systematically revisit these trade-offs by analyzing eight state-of-the-art LDP protocols and proposing refined counterparts that leverage tailored optimization techniques. Experimental results demonstrate that our proposed adaptive mechanisms consistently outperform their non-adaptive counterparts, achieving substantial reductions in ASR while preserving utility, and pushing closer to the ASR-MSE Pareto frontier. By bridging the gap between theoretical guarantees and real-world vulnerabilities, our framework enables modular and context-aware deployment of LDP mechanisms with tunable privacy-utility trade-offs.



## **9. A Parametric Approach to Adversarial Augmentation for Cross-Domain Iris Presentation Attack Detection**

cs.CV

IEEE/CVF Winter Conference on Applications of Computer Vision (WACV),  2025

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2412.07199v2) [paper-pdf](http://arxiv.org/pdf/2412.07199v2)

**Authors**: Debasmita Pal, Redwan Sony, Arun Ross

**Abstract**: Iris-based biometric systems are vulnerable to presentation attacks (PAs), where adversaries present physical artifacts (e.g., printed iris images, textured contact lenses) to defeat the system. This has led to the development of various presentation attack detection (PAD) algorithms, which typically perform well in intra-domain settings. However, they often struggle to generalize effectively in cross-domain scenarios, where training and testing employ different sensors, PA instruments, and datasets. In this work, we use adversarial training samples of both bonafide irides and PAs to improve the cross-domain performance of a PAD classifier. The novelty of our approach lies in leveraging transformation parameters from classical data augmentation schemes (e.g., translation, rotation) to generate adversarial samples. We achieve this through a convolutional autoencoder, ADV-GEN, that inputs original training samples along with a set of geometric and photometric transformations. The transformation parameters act as regularization variables, guiding ADV-GEN to generate adversarial samples in a constrained search space. Experiments conducted on the LivDet-Iris 2017 database, comprising four datasets, and the LivDet-Iris 2020 dataset, demonstrate the efficacy of our proposed method. The code is available at https://github.com/iPRoBe-lab/ADV-GEN-IrisPAD.



## **10. Diffusion-Driven Universal Model Inversion Attack for Face Recognition**

cs.CR

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.18015v1) [paper-pdf](http://arxiv.org/pdf/2504.18015v1)

**Authors**: Hanrui Wang, Shuo Wang, Chun-Shien Lu, Isao Echizen

**Abstract**: Facial recognition technology poses significant privacy risks, as it relies on biometric data that is inherently sensitive and immutable if compromised. To mitigate these concerns, face recognition systems convert raw images into embeddings, traditionally considered privacy-preserving. However, model inversion attacks pose a significant privacy threat by reconstructing these private facial images, making them a crucial tool for evaluating the privacy risks of face recognition systems. Existing methods usually require training individual generators for each target model, a computationally expensive process. In this paper, we propose DiffUMI, a training-free diffusion-driven universal model inversion attack for face recognition systems. DiffUMI is the first approach to apply a diffusion model for unconditional image generation in model inversion. Unlike other methods, DiffUMI is universal, eliminating the need for training target-specific generators. It operates within a fixed framework and pretrained diffusion model while seamlessly adapting to diverse target identities and models. DiffUMI breaches privacy-preserving face recognition systems with state-of-the-art success, demonstrating that an unconditional diffusion model, coupled with optimized adversarial search, enables efficient and high-fidelity facial reconstruction. Additionally, we introduce a novel application of out-of-domain detection (OODD), marking the first use of model inversion to distinguish non-face inputs from face inputs based solely on embeddings.



## **11. Adversarial Attacks to Latent Representations of Distributed Neural Networks in Split Computing**

cs.LG

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2309.17401v4) [paper-pdf](http://arxiv.org/pdf/2309.17401v4)

**Authors**: Milin Zhang, Mohammad Abdi, Jonathan Ashdown, Francesco Restuccia

**Abstract**: Distributed deep neural networks (DNNs) have been shown to reduce the computational burden of mobile devices and decrease the end-to-end inference latency in edge computing scenarios. While distributed DNNs have been studied, to the best of our knowledge, the resilience of distributed DNNs to adversarial action remains an open problem. In this paper, we fill the existing research gap by rigorously analyzing the robustness of distributed DNNs against adversarial action. We cast this problem in the context of information theory and rigorously proved that (i) the compressed latent dimension improves the robustness but also affect task-oriented performance; and (ii) the deeper splitting point enhances the robustness but also increases the computational burden. These two trade-offs provide a novel perspective to design robust distributed DNN. To test our theoretical findings, we perform extensive experimental analysis by considering 6 different DNN architectures, 6 different approaches for distributed DNN and 10 different adversarial attacks using the ImageNet-1K dataset.



## **12. Cluster-Aware Attacks on Graph Watermarks**

cs.CR

15 pages, 16 figures

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17971v1) [paper-pdf](http://arxiv.org/pdf/2504.17971v1)

**Authors**: Alexander Nemecek, Emre Yilmaz, Erman Ayday

**Abstract**: Data from domains such as social networks, healthcare, finance, and cybersecurity can be represented as graph-structured information. Given the sensitive nature of this data and their frequent distribution among collaborators, ensuring secure and attributable sharing is essential. Graph watermarking enables attribution by embedding user-specific signatures into graph-structured data. While prior work has addressed random perturbation attacks, the threat posed by adversaries leveraging structural properties through community detection remains unexplored. In this work, we introduce a cluster-aware threat model in which adversaries apply community-guided modifications to evade detection. We propose two novel attack strategies and evaluate them on real-world social network graphs. Our results show that cluster-aware attacks can reduce attribution accuracy by up to 80% more than random baselines under equivalent perturbation budgets on sparse graphs. To mitigate this threat, we propose a lightweight embedding enhancement that distributes watermark nodes across graph communities. This approach improves attribution accuracy by up to 60% under attack on dense graphs, without increasing runtime or structural distortion. Our findings underscore the importance of cluster-topological awareness in both watermarking design and adversarial modeling.



## **13. ASVspoof 5: Design, Collection and Validation of Resources for Spoofing, Deepfake, and Adversarial Attack Detection Using Crowdsourced Speech**

eess.AS

Database link: https://zenodo.org/records/14498691, Database mirror  link: https://huggingface.co/datasets/jungjee/asvspoof5, ASVspoof 5 Challenge  Workshop Proceeding: https://www.isca-archive.org/asvspoof_2024/index.html

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2502.08857v4) [paper-pdf](http://arxiv.org/pdf/2502.08857v4)

**Authors**: Xin Wang, Héctor Delgado, Hemlata Tak, Jee-weon Jung, Hye-jin Shim, Massimiliano Todisco, Ivan Kukanov, Xuechen Liu, Md Sahidullah, Tomi Kinnunen, Nicholas Evans, Kong Aik Lee, Junichi Yamagishi, Myeonghun Jeong, Ge Zhu, Yongyi Zang, You Zhang, Soumi Maiti, Florian Lux, Nicolas Müller, Wangyou Zhang, Chengzhe Sun, Shuwei Hou, Siwei Lyu, Sébastien Le Maguer, Cheng Gong, Hanjie Guo, Liping Chen, Vishwanath Singh

**Abstract**: ASVspoof 5 is the fifth edition in a series of challenges which promote the study of speech spoofing and deepfake attacks as well as the design of detection solutions. We introduce the ASVspoof 5 database which is generated in a crowdsourced fashion from data collected in diverse acoustic conditions (cf. studio-quality data for earlier ASVspoof databases) and from ~2,000 speakers (cf. ~100 earlier). The database contains attacks generated with 32 different algorithms, also crowdsourced, and optimised to varying degrees using new surrogate detection models. Among them are attacks generated with a mix of legacy and contemporary text-to-speech synthesis and voice conversion models, in addition to adversarial attacks which are incorporated for the first time. ASVspoof 5 protocols comprise seven speaker-disjoint partitions. They include two distinct partitions for the training of different sets of attack models, two more for the development and evaluation of surrogate detection models, and then three additional partitions which comprise the ASVspoof 5 training, development and evaluation sets. An auxiliary set of data collected from an additional 30k speakers can also be used to train speaker encoders for the implementation of attack algorithms. Also described herein is an experimental validation of the new ASVspoof 5 database using a set of automatic speaker verification and spoof/deepfake baseline detectors. With the exception of protocols and tools for the generation of spoofed/deepfake speech, the resources described in this paper, already used by participants of the ASVspoof 5 challenge in 2024, are now all freely available to the community.



## **14. DCT-Shield: A Robust Frequency Domain Defense against Malicious Image Editing**

cs.CV

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17894v1) [paper-pdf](http://arxiv.org/pdf/2504.17894v1)

**Authors**: Aniruddha Bala, Rohit Chowdhury, Rohan Jaiswal, Siddharth Roheda

**Abstract**: Advancements in diffusion models have enabled effortless image editing via text prompts, raising concerns about image security. Attackers with access to user images can exploit these tools for malicious edits. Recent defenses attempt to protect images by adding a limited noise in the pixel space to disrupt the functioning of diffusion-based editing models. However, the adversarial noise added by previous methods is easily noticeable to the human eye. Moreover, most of these methods are not robust to purification techniques like JPEG compression under a feasible pixel budget. We propose a novel optimization approach that introduces adversarial perturbations directly in the frequency domain by modifying the Discrete Cosine Transform (DCT) coefficients of the input image. By leveraging the JPEG pipeline, our method generates adversarial images that effectively prevent malicious image editing. Extensive experiments across a variety of tasks and datasets demonstrate that our approach introduces fewer visual artifacts while maintaining similar levels of edit protection and robustness to noise purification techniques.



## **15. Unsupervised Corpus Poisoning Attacks in Continuous Space for Dense Retrieval**

cs.IR

This paper has been accepted as a full paper at SIGIR 2025 and will  be presented orally

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17884v1) [paper-pdf](http://arxiv.org/pdf/2504.17884v1)

**Authors**: Yongkang Li, Panagiotis Eustratiadis, Simon Lupart, Evangelos Kanoulas

**Abstract**: This paper concerns corpus poisoning attacks in dense information retrieval, where an adversary attempts to compromise the ranking performance of a search algorithm by injecting a small number of maliciously generated documents into the corpus. Our work addresses two limitations in the current literature. First, attacks that perform adversarial gradient-based word substitution search do so in the discrete lexical space, while retrieval itself happens in the continuous embedding space. We thus propose an optimization method that operates in the embedding space directly. Specifically, we train a perturbation model with the objective of maintaining the geometric distance between the original and adversarial document embeddings, while also maximizing the token-level dissimilarity between the original and adversarial documents. Second, it is common for related work to have a strong assumption that the adversary has prior knowledge about the queries. In this paper, we focus on a more challenging variant of the problem where the adversary assumes no prior knowledge about the query distribution (hence, unsupervised). Our core contribution is an adversarial corpus attack that is fast and effective. We present comprehensive experimental results on both in- and out-of-domain datasets, focusing on two related tasks: a top-1 attack and a corpus poisoning attack. We consider attacks under both a white-box and a black-box setting. Notably, our method can generate successful adversarial examples in under two minutes per target document; four times faster compared to the fastest gradient-based word substitution methods in the literature with the same hardware. Furthermore, our adversarial generation method generates text that is more likely to occur under the distribution of natural text (low perplexity), and is therefore more difficult to detect.



## **16. Siren -- Advancing Cybersecurity through Deception and Adaptive Analysis**

cs.CR

14 pages, 5 figures, 13th Computing Conference 2025 - London, United  Kingdom

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2406.06225v2) [paper-pdf](http://arxiv.org/pdf/2406.06225v2)

**Authors**: Samhruth Ananthanarayanan, Girish Kulathumani, Ganesh Narayanan

**Abstract**: Siren represents a pioneering research effort aimed at fortifying cybersecurity through strategic integration of deception, machine learning, and proactive threat analysis. Drawing inspiration from mythical sirens, this project employs sophisticated methods to lure potential threats into controlled environments. The system features a dynamic machine learning model for realtime analysis and classification, ensuring continuous adaptability to emerging cyber threats. The architectural framework includes a link monitoring proxy, a purpose-built machine learning model for dynamic link analysis, and a honeypot enriched with simulated user interactions to intensify threat engagement. Data protection within the honeypot is fortified with probabilistic encryption. Additionally, the incorporation of simulated user activity extends the system's capacity to capture and learn from potential attackers even after user disengagement. Overall, Siren introduces a paradigm shift in cybersecurity, transforming traditional defense mechanisms into proactive systems that actively engage and learn from potential adversaries. The research strives to enhance user protection while yielding valuable insights for ongoing refinement in response to the evolving landscape of cybersecurity threats.



## **17. On the Generalization of Adversarially Trained Quantum Classifiers**

quant-ph

22 pages, 6 figures

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17690v1) [paper-pdf](http://arxiv.org/pdf/2504.17690v1)

**Authors**: Petros Georgiou, Aaron Mark Thomas, Sharu Theresa Jose, Osvaldo Simeone

**Abstract**: Quantum classifiers are vulnerable to adversarial attacks that manipulate their input classical or quantum data. A promising countermeasure is adversarial training, where quantum classifiers are trained by using an attack-aware, adversarial loss function. This work establishes novel bounds on the generalization error of adversarially trained quantum classifiers when tested in the presence of perturbation-constrained adversaries. The bounds quantify the excess generalization error incurred to ensure robustness to adversarial attacks as scaling with the training sample size $m$ as $1/\sqrt{m}$, while yielding insights into the impact of the quantum embedding. For quantum binary classifiers employing \textit{rotation embedding}, we find that, in the presence of adversarial attacks on classical inputs $\mathbf{x}$, the increase in sample complexity due to adversarial training over conventional training vanishes in the limit of high dimensional inputs $\mathbf{x}$. In contrast, when the adversary can directly attack the quantum state $\rho(\mathbf{x})$ encoding the input $\mathbf{x}$, the excess generalization error depends on the choice of embedding only through its Hilbert space dimension. The results are also extended to multi-class classifiers. We validate our theoretical findings with numerical experiments.



## **18. Evaluating the Vulnerability of ML-Based Ethereum Phishing Detectors to Single-Feature Adversarial Perturbations**

cs.CR

24 pages; an extension of a paper that appeared at WISA 2024

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17684v1) [paper-pdf](http://arxiv.org/pdf/2504.17684v1)

**Authors**: Ahod Alghuried, Ali Alkinoon, Abdulaziz Alghamdi, Soohyeon Choi, Manar Mohaisen, David Mohaisen

**Abstract**: This paper explores the vulnerability of machine learning models to simple single-feature adversarial attacks in the context of Ethereum fraudulent transaction detection. Through comprehensive experimentation, we investigate the impact of various adversarial attack strategies on model performance metrics. Our findings, highlighting how prone those techniques are to simple attacks, are alarming, and the inconsistency in the attacks' effect on different algorithms promises ways for attack mitigation. We examine the effectiveness of different mitigation strategies, including adversarial training and enhanced feature selection, in enhancing model robustness and show their effectiveness.



## **19. Regulatory Markets for AI Safety**

cs.CY

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2001.00078v2) [paper-pdf](http://arxiv.org/pdf/2001.00078v2)

**Authors**: Jack Clark, Gillian K. Hadfield

**Abstract**: We propose a new model for regulation to achieve AI safety: global regulatory markets. We first sketch the model in general terms and provide an overview of the costs and benefits of this approach. We then demonstrate how the model might work in practice: responding to the risk of adversarial attacks on AI models employed in commercial drones.



## **20. GRANITE : a Byzantine-Resilient Dynamic Gossip Learning Framework**

cs.LG

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17471v1) [paper-pdf](http://arxiv.org/pdf/2504.17471v1)

**Authors**: Yacine Belal, Mohamed Maouche, Sonia Ben Mokhtar, Anthony Simonet-Boulogne

**Abstract**: Gossip Learning (GL) is a decentralized learning paradigm where users iteratively exchange and aggregate models with a small set of neighboring peers. Recent GL approaches rely on dynamic communication graphs built and maintained using Random Peer Sampling (RPS) protocols. Thanks to graph dynamics, GL can achieve fast convergence even over extremely sparse topologies. However, the robustness of GL over dy- namic graphs to Byzantine (model poisoning) attacks remains unaddressed especially when Byzantine nodes attack the RPS protocol to scale up model poisoning. We address this issue by introducing GRANITE, a framework for robust learning over sparse, dynamic graphs in the presence of a fraction of Byzantine nodes. GRANITE relies on two key components (i) a History-aware Byzantine-resilient Peer Sampling protocol (HaPS), which tracks previously encountered identifiers to reduce adversarial influence over time, and (ii) an Adaptive Probabilistic Threshold (APT), which leverages an estimate of Byzantine presence to set aggregation thresholds with formal guarantees. Empirical results confirm that GRANITE maintains convergence with up to 30% Byzantine nodes, improves learning speed via adaptive filtering of poisoned models and obtains these results in up to 9 times sparser graphs than dictated by current theory.



## **21. Evaluating Time Series Models for Urban Wastewater Management: Predictive Performance, Model Complexity and Resilience**

cs.LG

6 pages, 6 figures, accepted at 10th International Conference on  Smart and Sustainable Technologies (SpliTech) 2025, GitHub:  https://github.com/calgo-lab/resilient-timeseries-evaluation

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17461v1) [paper-pdf](http://arxiv.org/pdf/2504.17461v1)

**Authors**: Vipin Singh, Tianheng Ling, Teodor Chiaburu, Felix Biessmann

**Abstract**: Climate change increases the frequency of extreme rainfall, placing a significant strain on urban infrastructures, especially Combined Sewer Systems (CSS). Overflows from overburdened CSS release untreated wastewater into surface waters, posing environmental and public health risks. Although traditional physics-based models are effective, they are costly to maintain and difficult to adapt to evolving system dynamics. Machine Learning (ML) approaches offer cost-efficient alternatives with greater adaptability. To systematically assess the potential of ML for modeling urban infrastructure systems, we propose a protocol for evaluating Neural Network architectures for CSS time series forecasting with respect to predictive performance, model complexity, and robustness to perturbations. In addition, we assess model performance on peak events and critical fluctuations, as these are the key regimes for urban wastewater management. To investigate the feasibility of lightweight models suitable for IoT deployment, we compare global models, which have access to all information, with local models, which rely solely on nearby sensor readings. Additionally, to explore the security risks posed by network outages or adversarial attacks on urban infrastructure, we introduce error models that assess the resilience of models. Our results demonstrate that while global models achieve higher predictive performance, local models provide sufficient resilience in decentralized scenarios, ensuring robust modeling of urban infrastructure. Furthermore, models with longer native forecast horizons exhibit greater robustness to data perturbations. These findings contribute to the development of interpretable and reliable ML solutions for sustainable urban wastewater management. The implementation is available in our GitHub repository.



## **22. Unveiling Hidden Vulnerabilities in Digital Human Generation via Adversarial Attacks**

cs.CV

14 pages, 7 figures

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17457v1) [paper-pdf](http://arxiv.org/pdf/2504.17457v1)

**Authors**: Zhiying Li, Yeying Jin, Fan Shen, Zhi Liu, Weibin Chen, Pengju Zhang, Xiaomei Zhang, Boyu Chen, Michael Shen, Kejian Wu, Zhaoxin Fan, Jin Dong

**Abstract**: Expressive human pose and shape estimation (EHPS) is crucial for digital human generation, especially in applications like live streaming. While existing research primarily focuses on reducing estimation errors, it largely neglects robustness and security aspects, leaving these systems vulnerable to adversarial attacks. To address this significant challenge, we propose the \textbf{Tangible Attack (TBA)}, a novel framework designed to generate adversarial examples capable of effectively compromising any digital human generation model. Our approach introduces a \textbf{Dual Heterogeneous Noise Generator (DHNG)}, which leverages Variational Autoencoders (VAE) and ControlNet to produce diverse, targeted noise tailored to the original image features. Additionally, we design a custom \textbf{adversarial loss function} to optimize the noise, ensuring both high controllability and potent disruption. By iteratively refining the adversarial sample through multi-gradient signals from both the noise and the state-of-the-art EHPS model, TBA substantially improves the effectiveness of adversarial attacks. Extensive experiments demonstrate TBA's superiority, achieving a remarkable 41.0\% increase in estimation error, with an average improvement of approximately 17.0\%. These findings expose significant security vulnerabilities in current EHPS models and highlight the need for stronger defenses in digital human generation systems.



## **23. Fine-Tuning Adversarially-Robust Transformers for Single-Image Dehazing**

cs.CV

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17829v1) [paper-pdf](http://arxiv.org/pdf/2504.17829v1)

**Authors**: Vlad Vasilescu, Ana Neacsu, Daniela Faur

**Abstract**: Single-image dehazing is an important topic in remote sensing applications, enhancing the quality of acquired images and increasing object detection precision. However, the reliability of such structures has not been sufficiently analyzed, which poses them to the risk of imperceptible perturbations that can significantly hinder their performance. In this work, we show that state-of-the-art image-to-image dehazing transformers are susceptible to adversarial noise, with even 1 pixel change being able to decrease the PSNR by as much as 2.8 dB. Next, we propose two lightweight fine-tuning strategies aimed at increasing the robustness of pre-trained transformers. Our methods results in comparable clean performance, while significantly increasing the protection against adversarial data. We further present their applicability in two remote sensing scenarios, showcasing their robust behavior for out-of-distribution data. The source code for adversarial fine-tuning and attack algorithms can be found at github.com/Vladimirescu/RobustDehazing.



## **24. Analysis and Mitigation of Data injection Attacks against Data-Driven Control**

eess.SY

Under review for publication

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17347v1) [paper-pdf](http://arxiv.org/pdf/2504.17347v1)

**Authors**: Sribalaji C. Anand

**Abstract**: This paper investigates the impact of false data injection attacks on data-driven control systems. Specifically, we consider an adversary injecting false data into the sensor channels during the learning phase. When the operator seeks to learn a stable state-feedback controller, we propose an attack strategy capable of misleading the operator into learning an unstable feedback gain. We also investigate the effects of constant-bias injection attacks on data-driven linear quadratic regulation (LQR). Finally, we explore potential mitigation strategies and support our findings with numerical examples.



## **25. JailbreakLens: Interpreting Jailbreak Mechanism in the Lens of Representation and Circuit**

cs.CR

17 pages, 11 figures

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2411.11114v2) [paper-pdf](http://arxiv.org/pdf/2411.11114v2)

**Authors**: Zeqing He, Zhibo Wang, Zhixuan Chu, Huiyu Xu, Wenhui Zhang, Qinglong Wang, Rui Zheng

**Abstract**: Despite the outstanding performance of Large language Models (LLMs) in diverse tasks, they are vulnerable to jailbreak attacks, wherein adversarial prompts are crafted to bypass their security mechanisms and elicit unexpected responses. Although jailbreak attacks are prevalent, the understanding of their underlying mechanisms remains limited. Recent studies have explained typical jailbreaking behavior (e.g., the degree to which the model refuses to respond) of LLMs by analyzing representation shifts in their latent space caused by jailbreak prompts or identifying key neurons that contribute to the success of jailbreak attacks. However, these studies neither explore diverse jailbreak patterns nor provide a fine-grained explanation from the failure of circuit to the changes of representational, leaving significant gaps in uncovering the jailbreak mechanism. In this paper, we propose JailbreakLens, an interpretation framework that analyzes jailbreak mechanisms from both representation (which reveals how jailbreaks alter the model's harmfulness perception) and circuit perspectives~(which uncovers the causes of these deceptions by identifying key circuits contributing to the vulnerability), tracking their evolution throughout the entire response generation process. We then conduct an in-depth evaluation of jailbreak behavior on five mainstream LLMs under seven jailbreak strategies. Our evaluation reveals that jailbreak prompts amplify components that reinforce affirmative responses while suppressing those that produce refusal. This manipulation shifts model representations toward safe clusters to deceive the LLM, leading it to provide detailed responses instead of refusals. Notably, we find a strong and consistent correlation between representation deception and activation shift of key circuits across diverse jailbreak methods and multiple LLMs.



## **26. Enhancing Variational Autoencoders with Smooth Robust Latent Encoding**

cs.LG

Under review

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17219v1) [paper-pdf](http://arxiv.org/pdf/2504.17219v1)

**Authors**: Hyomin Lee, Minseon Kim, Sangwon Jang, Jongheon Jeong, Sung Ju Hwang

**Abstract**: Variational Autoencoders (VAEs) have played a key role in scaling up diffusion-based generative models, as in Stable Diffusion, yet questions regarding their robustness remain largely underexplored. Although adversarial training has been an established technique for enhancing robustness in predictive models, it has been overlooked for generative models due to concerns about potential fidelity degradation by the nature of trade-offs between performance and robustness. In this work, we challenge this presumption, introducing Smooth Robust Latent VAE (SRL-VAE), a novel adversarial training framework that boosts both generation quality and robustness. In contrast to conventional adversarial training, which focuses on robustness only, our approach smooths the latent space via adversarial perturbations, promoting more generalizable representations while regularizing with originality representation to sustain original fidelity. Applied as a post-training step on pre-trained VAEs, SRL-VAE improves image robustness and fidelity with minimal computational overhead. Experiments show that SRL-VAE improves both generation quality, in image reconstruction and text-guided image editing, and robustness, against Nightshade attacks and image editing attacks. These results establish a new paradigm, showing that adversarial training, once thought to be detrimental to generative models, can instead enhance both fidelity and robustness.



## **27. CheatAgent: Attacking LLM-Empowered Recommender Systems via LLM Agent**

cs.CR

Accepted by KDD 2024;

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.13192v2) [paper-pdf](http://arxiv.org/pdf/2504.13192v2)

**Authors**: Liang-bo Ning, Shijie Wang, Wenqi Fan, Qing Li, Xin Xu, Hao Chen, Feiran Huang

**Abstract**: Recently, Large Language Model (LLM)-empowered recommender systems (RecSys) have brought significant advances in personalized user experience and have attracted considerable attention. Despite the impressive progress, the research question regarding the safety vulnerability of LLM-empowered RecSys still remains largely under-investigated. Given the security and privacy concerns, it is more practical to focus on attacking the black-box RecSys, where attackers can only observe the system's inputs and outputs. However, traditional attack approaches employing reinforcement learning (RL) agents are not effective for attacking LLM-empowered RecSys due to the limited capabilities in processing complex textual inputs, planning, and reasoning. On the other hand, LLMs provide unprecedented opportunities to serve as attack agents to attack RecSys because of their impressive capability in simulating human-like decision-making processes. Therefore, in this paper, we propose a novel attack framework called CheatAgent by harnessing the human-like capabilities of LLMs, where an LLM-based agent is developed to attack LLM-Empowered RecSys. Specifically, our method first identifies the insertion position for maximum impact with minimal input modification. After that, the LLM agent is designed to generate adversarial perturbations to insert at target positions. To further improve the quality of generated perturbations, we utilize the prompt tuning technique to improve attacking strategies via feedback from the victim RecSys iteratively. Extensive experiments across three real-world datasets demonstrate the effectiveness of our proposed attacking method.



## **28. Range and Topology Mutation Based Wireless Agility**

cs.ET

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17164v1) [paper-pdf](http://arxiv.org/pdf/2504.17164v1)

**Authors**: Qi Duan, Ehab Al-Shae, Jiang Xie

**Abstract**: In this paper, we present formal foundations for two wireless agility techniques: (1) Random Range Mutation (RNM) that allows for periodic changes of AP coverage range randomly, and (2) Ran- dom Topology Mutation (RTM) that allows for random motion and placement of APs in the wireless infrastructure. The goal of these techniques is to proactively defend against targeted attacks (e.g., DoS and eavesdropping) by forcing the wireless clients to change their AP association randomly. We apply Satisfiability Modulo The- ories (SMT) and Answer Set Programming (ASP) based constraint solving methods that allow for optimizing wireless AP mutation while maintaining service requirements including coverage, secu- rity and energy properties under incomplete information about the adversary strategies. Our evaluation validates the feasibility, scalability, and effectiveness of the formal methods based technical approaches.



## **29. Trading Devil: Robust backdoor attack via Stochastic investment models and Bayesian approach**

cs.CR

(Last update!, a constructive comment from arxiv led to this latest  update ) Stochastic investment models and a Bayesian approach to better  modeling of uncertainty : adversarial machine learning or Stochastic market.  arXiv admin note: substantial text overlap with arXiv:2402.05967 (see this  link to the paper by : Orson Mengara)

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2406.10719v5) [paper-pdf](http://arxiv.org/pdf/2406.10719v5)

**Authors**: Orson Mengara

**Abstract**: With the growing use of voice-activated systems and speech recognition technologies, the danger of backdoor attacks on audio data has grown significantly. This research looks at a specific type of attack, known as a Stochastic investment-based backdoor attack (MarketBack), in which adversaries strategically manipulate the stylistic properties of audio to fool speech recognition systems. The security and integrity of machine learning models are seriously threatened by backdoor attacks, in order to maintain the reliability of audio applications and systems, the identification of such attacks becomes crucial in the context of audio data. Experimental results demonstrated that MarketBack is feasible to achieve an average attack success rate close to 100% in seven victim models when poisoning less than 1% of the training data.



## **30. On Minimizing Adversarial Counterfactual Error in Adversarial RL**

cs.LG

Presented at ICLR 2025

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2406.04724v4) [paper-pdf](http://arxiv.org/pdf/2406.04724v4)

**Authors**: Roman Belaire, Arunesh Sinha, Pradeep Varakantham

**Abstract**: Deep Reinforcement Learning (DRL) policies are highly susceptible to adversarial noise in observations, which poses significant risks in safety-critical scenarios. The challenge inherent to adversarial perturbations is that by altering the information observed by the agent, the state becomes only partially observable. Existing approaches address this by either enforcing consistent actions across nearby states or maximizing the worst-case value within adversarially perturbed observations. However, the former suffers from performance degradation when attacks succeed, while the latter tends to be overly conservative, leading to suboptimal performance in benign settings. We hypothesize that these limitations stem from their failing to account for partial observability directly. To this end, we introduce a novel objective called Adversarial Counterfactual Error (ACoE), defined on the beliefs about the true state and balancing value optimization with robustness. To make ACoE scalable in model-free settings, we propose the theoretically-grounded surrogate objective Cumulative-ACoE (C-ACoE). Our empirical evaluations on standard benchmarks (MuJoCo, Atari, and Highway) demonstrate that our method significantly outperforms current state-of-the-art approaches for addressing adversarial RL challenges, offering a promising direction for improving robustness in DRL under adversarial conditions. Our code is available at https://github.com/romanbelaire/acoe-robust-rl.



## **31. BadVideo: Stealthy Backdoor Attack against Text-to-Video Generation**

cs.CV

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.16907v1) [paper-pdf](http://arxiv.org/pdf/2504.16907v1)

**Authors**: Ruotong Wang, Mingli Zhu, Jiarong Ou, Rui Chen, Xin Tao, Pengfei Wan, Baoyuan Wu

**Abstract**: Text-to-video (T2V) generative models have rapidly advanced and found widespread applications across fields like entertainment, education, and marketing. However, the adversarial vulnerabilities of these models remain rarely explored. We observe that in T2V generation tasks, the generated videos often contain substantial redundant information not explicitly specified in the text prompts, such as environmental elements, secondary objects, and additional details, providing opportunities for malicious attackers to embed hidden harmful content. Exploiting this inherent redundancy, we introduce BadVideo, the first backdoor attack framework tailored for T2V generation. Our attack focuses on designing target adversarial outputs through two key strategies: (1) Spatio-Temporal Composition, which combines different spatiotemporal features to encode malicious information; (2) Dynamic Element Transformation, which introduces transformations in redundant elements over time to convey malicious information. Based on these strategies, the attacker's malicious target seamlessly integrates with the user's textual instructions, providing high stealthiness. Moreover, by exploiting the temporal dimension of videos, our attack successfully evades traditional content moderation systems that primarily analyze spatial information within individual frames. Extensive experiments demonstrate that BadVideo achieves high attack success rates while preserving original semantics and maintaining excellent performance on clean inputs. Overall, our work reveals the adversarial vulnerability of T2V models, calling attention to potential risks and misuse. Our project page is at https://wrt2000.github.io/BadVideo2025/.



## **32. aiXamine: Simplified LLM Safety and Security**

cs.CR

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.14985v2) [paper-pdf](http://arxiv.org/pdf/2504.14985v2)

**Authors**: Fatih Deniz, Dorde Popovic, Yazan Boshmaf, Euisuh Jeong, Minhaj Ahmad, Sanjay Chawla, Issa Khalil

**Abstract**: Evaluating Large Language Models (LLMs) for safety and security remains a complex task, often requiring users to navigate a fragmented landscape of ad hoc benchmarks, datasets, metrics, and reporting formats. To address this challenge, we present aiXamine, a comprehensive black-box evaluation platform for LLM safety and security. aiXamine integrates over 40 tests (i.e., benchmarks) organized into eight key services targeting specific dimensions of safety and security: adversarial robustness, code security, fairness and bias, hallucination, model and data privacy, out-of-distribution (OOD) robustness, over-refusal, and safety alignment. The platform aggregates the evaluation results into a single detailed report per model, providing a detailed breakdown of model performance, test examples, and rich visualizations. We used aiXamine to assess over 50 publicly available and proprietary LLMs, conducting over 2K examinations. Our findings reveal notable vulnerabilities in leading models, including susceptibility to adversarial attacks in OpenAI's GPT-4o, biased outputs in xAI's Grok-3, and privacy weaknesses in Google's Gemini 2.0. Additionally, we observe that open-source models can match or exceed proprietary models in specific services such as safety alignment, fairness and bias, and OOD robustness. Finally, we identify trade-offs between distillation strategies, model size, training methods, and architectural choices.



## **33. A robust and composable device-independent protocol for oblivious transfer using (fully) untrusted quantum devices in the bounded storage model**

quant-ph

Major improvement in the main result (security against non-IID  devices)

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2404.11283v2) [paper-pdf](http://arxiv.org/pdf/2404.11283v2)

**Authors**: Rishabh Batra, Sayantan Chakraborty, Rahul Jain, Upendra Kapshikar

**Abstract**: We present a robust and composable device-independent (DI) quantum protocol between two parties for oblivious transfer (OT) using Magic Square devices in the bounded storage model in which the (honest and cheating) devices and parties have no long-term quantum memory. After a fixed constant (real-world) time interval, referred to as DELAY, the quantum states decohere completely. The adversary (cheating party), with full control over the devices, is allowed joint (non-IID) quantum operations on the devices, and there are no time and space complexity bounds placed on its powers. The running time of the honest parties is polylog({\lambda}) (where {\lambda} is the security parameter). Our protocol has negligible (in {\lambda}) correctness and security errors and can be implemented in the NISQ (Noisy Intermediate Scale Quantum) era. By robustness, we mean that our protocol is correct even when devices are slightly off (by a small constant) from their ideal specification. This is an important property since small manufacturing errors in the real-world devices are inevitable. Our protocol is sequentially composable and, hence, can be used as a building block to construct larger protocols (including DI bit-commitment and DI secure multi-party computation) while still preserving correctness and security guarantees.   None of the known DI protocols for OT in the literature are robust and secure against joint quantum attacks. This was a major open question in device-independent two-party distrustful cryptography, which we resolve.   We prove a parallel repetition theorem for a certain class of entangled games with a hybrid (quantum-classical) strategy to show the security of our protocol. The hybrid strategy helps to incorporate DELAY in our protocol. This parallel repetition theorem is a main technical contribution of our work.



## **34. Exploring Adversarial Transferability between Kolmogorov-arnold Networks**

cs.CV

After the submission of the paper, we realized that the study still  has room for expansion. In order to make the research findings more profound  and comprehensive, we have decided to withdraw the paper so that we can  conduct further research and expansion

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2503.06276v2) [paper-pdf](http://arxiv.org/pdf/2503.06276v2)

**Authors**: Songping Wang, Xinquan Yue, Yueming Lyu, Caifeng Shan

**Abstract**: Kolmogorov-Arnold Networks (KANs) have emerged as a transformative model paradigm, significantly impacting various fields. However, their adversarial robustness remains less underexplored, especially across different KAN architectures. To explore this critical safety issue, we conduct an analysis and find that due to overfitting to the specific basis functions of KANs, they possess poor adversarial transferability among different KANs. To tackle this challenge, we propose AdvKAN, the first transfer attack method for KANs. AdvKAN integrates two key components: 1) a Breakthrough-Defense Surrogate Model (BDSM), which employs a breakthrough-defense training strategy to mitigate overfitting to the specific structures of KANs. 2) a Global-Local Interaction (GLI) technique, which promotes sufficient interaction between adversarial gradients of hierarchical levels, further smoothing out loss surfaces of KANs. Both of them work together to enhance the strength of transfer attack among different KANs. Extensive experimental results on various KANs and datasets demonstrate the effectiveness of AdvKAN, which possesses notably superior attack capabilities and deeply reveals the vulnerabilities of KANs. Code will be released upon acceptance.



## **35. Fast Adversarial Training with Weak-to-Strong Spatial-Temporal Consistency in the Frequency Domain on Videos**

cs.CV

After the submission of the paper, we realized that the study still  has room for expansion. In order to make the research findings more profound  and comprehensive, we have decided to withdraw the paper so that we can  conduct further research and expansion

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.14921v2) [paper-pdf](http://arxiv.org/pdf/2504.14921v2)

**Authors**: Songping Wang, Hanqing Liu, Yueming Lyu, Xiantao Hu, Ziwen He, Wei Wang, Caifeng Shan, Liang Wang

**Abstract**: Adversarial Training (AT) has been shown to significantly enhance adversarial robustness via a min-max optimization approach. However, its effectiveness in video recognition tasks is hampered by two main challenges. First, fast adversarial training for video models remains largely unexplored, which severely impedes its practical applications. Specifically, most video adversarial training methods are computationally costly, with long training times and high expenses. Second, existing methods struggle with the trade-off between clean accuracy and adversarial robustness. To address these challenges, we introduce Video Fast Adversarial Training with Weak-to-Strong consistency (VFAT-WS), the first fast adversarial training method for video data. Specifically, VFAT-WS incorporates the following key designs: First, it integrates a straightforward yet effective temporal frequency augmentation (TF-AUG), and its spatial-temporal enhanced form STF-AUG, along with a single-step PGD attack to boost training efficiency and robustness. Second, it devises a weak-to-strong spatial-temporal consistency regularization, which seamlessly integrates the simpler TF-AUG and the more complex STF-AUG. Leveraging the consistency regularization, it steers the learning process from simple to complex augmentations. Both of them work together to achieve a better trade-off between clean accuracy and robustness. Extensive experiments on UCF-101 and HMDB-51 with both CNN and Transformer-based models demonstrate that VFAT-WS achieves great improvements in adversarial robustness and corruption robustness, while accelerating training by nearly 490%.



## **36. Information Leakage of Sentence Embeddings via Generative Embedding Inversion Attacks**

cs.IR

This is a preprint of our paper accepted at SIGIR 2025

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.16609v1) [paper-pdf](http://arxiv.org/pdf/2504.16609v1)

**Authors**: Antonios Tragoudaras, Theofanis Aslanidis, Emmanouil Georgios Lionis, Marina Orozco González, Panagiotis Eustratiadis

**Abstract**: Text data are often encoded as dense vectors, known as embeddings, which capture semantic, syntactic, contextual, and domain-specific information. These embeddings, widely adopted in various applications, inherently contain rich information that may be susceptible to leakage under certain attacks. The GEIA framework highlights vulnerabilities in sentence embeddings, demonstrating that they can reveal the original sentences they represent. In this study, we reproduce GEIA's findings across various neural sentence embedding models. Additionally, we contribute new analysis to examine whether these models leak sensitive information from their training datasets. We propose a simple yet effective method without any modification to the attacker's architecture proposed in GEIA. The key idea is to examine differences between log-likelihood for masked and original variants of data that sentence embedding models have been pre-trained on, calculated on the embedding space of the attacker. Our findings indicate that following our approach, an adversary party can recover meaningful sensitive information related to the pre-training knowledge of the popular models used for creating sentence embeddings, seriously undermining their security. Our code is available on: https://github.com/taslanidis/GEIA



## **37. Seeking Flat Minima over Diverse Surrogates for Improved Adversarial Transferability: A Theoretical Framework and Algorithmic Instantiation**

cs.CR

26 pages, 6 figures

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.16474v1) [paper-pdf](http://arxiv.org/pdf/2504.16474v1)

**Authors**: Meixi Zheng, Kehan Wu, Yanbo Fan, Rui Huang, Baoyuan Wu

**Abstract**: The transfer-based black-box adversarial attack setting poses the challenge of crafting an adversarial example (AE) on known surrogate models that remain effective against unseen target models. Due to the practical importance of this task, numerous methods have been proposed to address this challenge. However, most previous methods are heuristically designed and intuitively justified, lacking a theoretical foundation. To bridge this gap, we derive a novel transferability bound that offers provable guarantees for adversarial transferability. Our theoretical analysis has the advantages of \textit{(i)} deepening our understanding of previous methods by building a general attack framework and \textit{(ii)} providing guidance for designing an effective attack algorithm. Our theoretical results demonstrate that optimizing AEs toward flat minima over the surrogate model set, while controlling the surrogate-target model shift measured by the adversarial model discrepancy, yields a comprehensive guarantee for AE transferability. The results further lead to a general transfer-based attack framework, within which we observe that previous methods consider only partial factors contributing to the transferability. Algorithmically, inspired by our theoretical results, we first elaborately construct the surrogate model set in which models exhibit diverse adversarial vulnerabilities with respect to AEs to narrow an instantiated adversarial model discrepancy. Then, a \textit{model-Diversity-compatible Reverse Adversarial Perturbation} (DRAP) is generated to effectively promote the flatness of AEs over diverse surrogate models to improve transferability. Extensive experiments on NIPS2017 and CIFAR-10 datasets against various target models demonstrate the effectiveness of our proposed attack.



## **38. Automated Trustworthiness Oracle Generation for Machine Learning Text Classifiers**

cs.SE

24 pages, 5 tables, 9 figures, Camera-ready version accepted to FSE  2025

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2410.22663v3) [paper-pdf](http://arxiv.org/pdf/2410.22663v3)

**Authors**: Lam Nguyen Tung, Steven Cho, Xiaoning Du, Neelofar Neelofar, Valerio Terragni, Stefano Ruberto, Aldeida Aleti

**Abstract**: Machine learning (ML) for text classification has been widely used in various domains. These applications can significantly impact ethics, economics, and human behavior, raising serious concerns about trusting ML decisions. Studies indicate that conventional metrics are insufficient to build human trust in ML models. These models often learn spurious correlations and predict based on them. In the real world, their performance can deteriorate significantly. To avoid this, a common practice is to test whether predictions are reasonable based on valid patterns in the data. Along with this, a challenge known as the trustworthiness oracle problem has been introduced. Due to the lack of automated trustworthiness oracles, the assessment requires manual validation of the decision process disclosed by explanation methods. However, this is time-consuming, error-prone, and unscalable.   We propose TOKI, the first automated trustworthiness oracle generation method for text classifiers. TOKI automatically checks whether the words contributing the most to a prediction are semantically related to the predicted class. Specifically, we leverage ML explanations to extract the decision-contributing words and measure their semantic relatedness with the class based on word embeddings. We also introduce a novel adversarial attack method that targets trustworthiness vulnerabilities identified by TOKI. To evaluate their alignment with human judgement, experiments are conducted. We compare TOKI with a naive baseline based solely on model confidence and TOKI-guided adversarial attack method with A2T, a SOTA adversarial attack method. Results show that relying on prediction uncertainty cannot effectively distinguish between trustworthy and untrustworthy predictions, TOKI achieves 142% higher accuracy than the naive baseline, and TOKI-guided attack method is more effective with fewer perturbations than A2T.



## **39. Large Language Model Sentinel: LLM Agent for Adversarial Purification**

cs.CL

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2405.20770v4) [paper-pdf](http://arxiv.org/pdf/2405.20770v4)

**Authors**: Guang Lin, Toshihisa Tanaka, Qibin Zhao

**Abstract**: Over the past two years, the use of large language models (LLMs) has advanced rapidly. While these LLMs offer considerable convenience, they also raise security concerns, as LLMs are vulnerable to adversarial attacks by some well-designed textual perturbations. In this paper, we introduce a novel defense technique named Large LAnguage MOdel Sentinel (LLAMOS), which is designed to enhance the adversarial robustness of LLMs by purifying the adversarial textual examples before feeding them into the target LLM. Our method comprises two main components: a) Agent instruction, which can simulate a new agent for adversarial defense, altering minimal characters to maintain the original meaning of the sentence while defending against attacks; b) Defense guidance, which provides strategies for modifying clean or adversarial examples to ensure effective defense and accurate outputs from the target LLMs. Remarkably, the defense agent demonstrates robust defensive capabilities even without learning from adversarial examples. Additionally, we conduct an intriguing adversarial experiment where we develop two agents, one for defense and one for attack, and engage them in mutual confrontation. During the adversarial interactions, neither agent completely beat the other. Extensive experiments on both open-source and closed-source LLMs demonstrate that our method effectively defends against adversarial attacks, thereby enhancing adversarial robustness.



## **40. Property-Preserving Hashing for $\ell_1$-Distance Predicates: Applications to Countering Adversarial Input Attacks**

cs.CR

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.16355v1) [paper-pdf](http://arxiv.org/pdf/2504.16355v1)

**Authors**: Hassan Asghar, Chenhan Zhang, Dali Kaafar

**Abstract**: Perceptual hashing is used to detect whether an input image is similar to a reference image with a variety of security applications. Recently, they have been shown to succumb to adversarial input attacks which make small imperceptible changes to the input image yet the hashing algorithm does not detect its similarity to the original image. Property-preserving hashing (PPH) is a recent construct in cryptography, which preserves some property (predicate) of its inputs in the hash domain. Researchers have so far shown constructions of PPH for Hamming distance predicates, which, for instance, outputs 1 if two inputs are within Hamming distance $t$. A key feature of PPH is its strong correctness guarantee, i.e., the probability that the predicate will not be correctly evaluated in the hash domain is negligible. Motivated by the use case of detecting similar images under adversarial setting, we propose the first PPH construction for an $\ell_1$-distance predicate. Roughly, this predicate checks if the two one-sided $\ell_1$-distances between two images are within a threshold $t$. Since many adversarial attacks use $\ell_2$-distance (related to $\ell_1$-distance) as the objective function to perturb the input image, by appropriately choosing the threshold $t$, we can force the attacker to add considerable noise to evade detection, and hence significantly deteriorate the image quality. Our proposed scheme is highly efficient, and runs in time $O(t^2)$. For grayscale images of size $28 \times 28$, we can evaluate the predicate in $0.0784$ seconds when pixel values are perturbed by up to $1 \%$. For larger RGB images of size $224 \times 224$, by dividing the image into 1,000 blocks, we achieve times of $0.0128$ seconds per block for $1 \%$ change, and up to $0.2641$ seconds per block for $14\%$ change.



## **41. Blockchain Meets Adaptive Honeypots: A Trust-Aware Approach to Next-Gen IoT Security**

cs.CR

This paper has been submitted to the IEEE Transactions on Network  Science and Engineering (TNSE) for possible publication

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.16226v1) [paper-pdf](http://arxiv.org/pdf/2504.16226v1)

**Authors**: Yazan Otoum, Arghavan Asad, Amiya Nayak

**Abstract**: Edge computing-based Next-Generation Wireless Networks (NGWN)-IoT offer enhanced bandwidth capacity for large-scale service provisioning but remain vulnerable to evolving cyber threats. Existing intrusion detection and prevention methods provide limited security as adversaries continually adapt their attack strategies. We propose a dynamic attack detection and prevention approach to address this challenge. First, blockchain-based authentication uses the Deoxys Authentication Algorithm (DAA) to verify IoT device legitimacy before data transmission. Next, a bi-stage intrusion detection system is introduced: the first stage uses signature-based detection via an Improved Random Forest (IRF) algorithm. In contrast, the second stage applies feature-based anomaly detection using a Diffusion Convolution Recurrent Neural Network (DCRNN). To ensure Quality of Service (QoS) and maintain Service Level Agreements (SLA), trust-aware service migration is performed using Heap-Based Optimization (HBO). Additionally, on-demand virtual High-Interaction honeypots deceive attackers and extract attack patterns, which are securely stored using the Bimodal Lattice Signature Scheme (BLISS) to enhance signature-based Intrusion Detection Systems (IDS). The proposed framework is implemented in the NS3 simulation environment and evaluated against existing methods across multiple performance metrics, including accuracy, attack detection rate, false negative rate, precision, recall, ROC curve, memory usage, CPU usage, and execution time. Experimental results demonstrate that the framework significantly outperforms existing approaches, reinforcing the security of NGWN-enabled IoT ecosystems



## **42. Adversarial Observations in Weather Forecasting**

cs.CR

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.15942v1) [paper-pdf](http://arxiv.org/pdf/2504.15942v1)

**Authors**: Erik Imgrund, Thorsten Eisenhofer, Konrad Rieck

**Abstract**: AI-based systems, such as Google's GenCast, have recently redefined the state of the art in weather forecasting, offering more accurate and timely predictions of both everyday weather and extreme events. While these systems are on the verge of replacing traditional meteorological methods, they also introduce new vulnerabilities into the forecasting process. In this paper, we investigate this threat and present a novel attack on autoregressive diffusion models, such as those used in GenCast, capable of manipulating weather forecasts and fabricating extreme events, including hurricanes, heat waves, and intense rainfall. The attack introduces subtle perturbations into weather observations that are statistically indistinguishable from natural noise and change less than 0.1% of the measurements - comparable to tampering with data from a single meteorological satellite. As modern forecasting integrates data from nearly a hundred satellites and many other sources operated by different countries, our findings highlight a critical security risk with the potential to cause large-scale disruptions and undermine public trust in weather prediction.



## **43. Human-Imperceptible Physical Adversarial Attack for NIR Face Recognition Models**

cs.CV

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.15823v1) [paper-pdf](http://arxiv.org/pdf/2504.15823v1)

**Authors**: Songyan Xie, Jinghang Wen, Encheng Su, Qiucheng Yu

**Abstract**: Near-infrared (NIR) face recognition systems, which can operate effectively in low-light conditions or in the presence of makeup, exhibit vulnerabilities when subjected to physical adversarial attacks. To further demonstrate the potential risks in real-world applications, we design a novel, stealthy, and practical adversarial patch to attack NIR face recognition systems in a black-box setting. We achieved this by utilizing human-imperceptible infrared-absorbing ink to generate multiple patches with digitally optimized shapes and positions for infrared images. To address the optimization mismatch between digital and real-world NIR imaging, we develop a light reflection model for human skin to minimize pixel-level discrepancies by simulating NIR light reflection.   Compared to state-of-the-art (SOTA) physical attacks on NIR face recognition systems, the experimental results show that our method improves the attack success rate in both digital and physical domains, particularly maintaining effectiveness across various face postures. Notably, the proposed approach outperforms SOTA methods, achieving an average attack success rate of 82.46% in the physical domain across different models, compared to 64.18% for existing methods. The artifact is available at https://anonymous.4open.science/r/Human-imperceptible-adversarial-patch-0703/.



## **44. Graph Neural Networks for Next-Generation-IoT: Recent Advances and Open Challenges**

cs.IT

28 pages, 15 figures, and 6 tables. Submitted for publication

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2412.20634v2) [paper-pdf](http://arxiv.org/pdf/2412.20634v2)

**Authors**: Nguyen Xuan Tung, Le Tung Giang, Bui Duc Son, Seon Geun Jeong, Trinh Van Chien, Won Joo Hwang, Lajos Hanzo

**Abstract**: Graph Neural Networks (GNNs) have emerged as a critical tool for optimizing and managing the complexities of the Internet of Things (IoT) in next-generation networks. This survey presents a comprehensive exploration of how GNNs may be harnessed in 6G IoT environments, focusing on key challenges and opportunities through a series of open questions. We commence with an exploration of GNN paradigms and the roles of node, edge, and graph-level tasks in solving wireless networking problems and highlight GNNs' ability to overcome the limitations of traditional optimization methods. This guidance enhances problem-solving efficiency across various next-generation (NG) IoT scenarios. Next, we provide a detailed discussion of the application of GNN in advanced NG enabling technologies, including massive MIMO, reconfigurable intelligent surfaces, satellites, THz, mobile edge computing (MEC), and ultra-reliable low latency communication (URLLC). We then delve into the challenges posed by adversarial attacks, offering insights into defense mechanisms to secure GNN-based NG-IoT networks. Next, we examine how GNNs can be integrated with future technologies like integrated sensing and communication (ISAC), satellite-air-ground-sea integrated networks (SAGSIN), and quantum computing. Our findings highlight the transformative potential of GNNs in improving efficiency, scalability, and security within NG-IoT systems, paving the way for future advances. Finally, we propose a set of design guidelines to facilitate the development of efficient, scalable, and secure GNN models tailored for NG IoT applications.



## **45. BaThe: Defense against the Jailbreak Attack in Multimodal Large Language Models by Treating Harmful Instruction as Backdoor Trigger**

cs.CR

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2408.09093v3) [paper-pdf](http://arxiv.org/pdf/2408.09093v3)

**Authors**: Yulin Chen, Haoran Li, Yirui Zhang, Zihao Zheng, Yangqiu Song, Bryan Hooi

**Abstract**: Multimodal Large Language Models (MLLMs) have showcased impressive performance in a variety of multimodal tasks. On the other hand, the integration of additional image modality may allow the malicious users to inject harmful content inside the images for jailbreaking. Unlike text-based LLMs, where adversaries need to select discrete tokens to conceal their malicious intent using specific algorithms, the continuous nature of image signals provides a direct opportunity for adversaries to inject harmful intentions. In this work, we propose $\textbf{BaThe}$ ($\textbf{Ba}$ckdoor $\textbf{T}$rigger S$\textbf{h}$i$\textbf{e}$ld), a simple yet effective jailbreak defense mechanism. Our work is motivated by recent research on jailbreak backdoor attack and virtual prompt backdoor attack in generative language models. Jailbreak backdoor attack uses harmful instructions combined with manually crafted strings as triggers to make the backdoored model generate prohibited responses. We assume that harmful instructions can function as triggers, and if we alternatively set rejection responses as the triggered response, the backdoored model then can defend against jailbreak attacks. We achieve this by utilizing virtual rejection prompt, similar to the virtual prompt backdoor attack. We embed the virtual rejection prompt into the soft text embeddings, which we call ``wedge''. Our comprehensive experiments demonstrate that BaThe effectively mitigates various types of jailbreak attacks and is adaptable to defend against unseen attacks, with minimal impact on MLLMs' performance.



## **46. Red Team Diffuser: Exposing Toxic Continuation Vulnerabilities in Vision-Language Models via Reinforcement Learning**

cs.CV

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2503.06223v2) [paper-pdf](http://arxiv.org/pdf/2503.06223v2)

**Authors**: Ruofan Wang, Xiang Zheng, Xiaosen Wang, Cong Wang, Xingjun Ma

**Abstract**: The growing deployment of large Vision-Language Models (VLMs) exposes critical safety gaps in their alignment mechanisms. While existing jailbreak studies primarily focus on VLMs' susceptibility to harmful instructions, we reveal a fundamental yet overlooked vulnerability: toxic text continuation, where VLMs produce highly toxic completions when prompted with harmful text prefixes paired with semantically adversarial images. To systematically study this threat, we propose Red Team Diffuser (RTD), the first red teaming diffusion model that coordinates adversarial image generation and toxic continuation through reinforcement learning. Our key innovations include dynamic cross-modal attack and stealth-aware optimization. For toxic text prefixes from an LLM safety benchmark, we conduct greedy search to identify optimal image prompts that maximally induce toxic completions. The discovered image prompts then drive RL-based diffusion model fine-tuning, producing semantically aligned adversarial images that boost toxicity rates. Stealth-aware optimization introduces joint adversarial rewards that balance toxicity maximization (via Detoxify classifier) and stealthiness (via BERTScore), circumventing traditional noise-based adversarial patterns. Experimental results demonstrate the effectiveness of RTD, increasing the toxicity rate of LLaVA outputs by 10.69% over text-only baselines on the original attack set and 8.91% on an unseen set, proving generalization capability. Moreover, RTD exhibits strong cross-model transferability, raising the toxicity rate by 5.1% on Gemini and 26.83% on LLaMA. Our findings expose two critical flaws in current VLM alignment: (1) failure to prevent toxic continuation from harmful prefixes, and (2) overlooking cross-modal attack vectors. These results necessitate a paradigm shift toward multimodal red teaming in safety evaluations.



## **47. TrojanDam: Detection-Free Backdoor Defense in Federated Learning through Proactive Model Robustification utilizing OOD Data**

cs.CR

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.15674v1) [paper-pdf](http://arxiv.org/pdf/2504.15674v1)

**Authors**: Yanbo Dai, Songze Li, Zihan Gan, Xueluan Gong

**Abstract**: Federated learning (FL) systems allow decentralized data-owning clients to jointly train a global model through uploading their locally trained updates to a centralized server. The property of decentralization enables adversaries to craft carefully designed backdoor updates to make the global model misclassify only when encountering adversary-chosen triggers. Existing defense mechanisms mainly rely on post-training detection after receiving updates. These methods either fail to identify updates which are deliberately fabricated statistically close to benign ones, or show inconsistent performance in different FL training stages. The effect of unfiltered backdoor updates will accumulate in the global model, and eventually become functional. Given the difficulty of ruling out every backdoor update, we propose a backdoor defense paradigm, which focuses on proactive robustification on the global model against potential backdoor attacks. We first reveal that the successful launching of backdoor attacks in FL stems from the lack of conflict between malicious and benign updates on redundant neurons of ML models. We proceed to prove the feasibility of activating redundant neurons utilizing out-of-distribution (OOD) samples in centralized settings, and migrating to FL settings to propose a novel backdoor defense mechanism, TrojanDam. The proposed mechanism has the FL server continuously inject fresh OOD mappings into the global model to activate redundant neurons, canceling the effect of backdoor updates during aggregation. We conduct systematic and extensive experiments to illustrate the superior performance of TrojanDam, over several SOTA backdoor defense methods across a wide range of FL settings.



## **48. Research on Cloud Platform Network Traffic Monitoring and Anomaly Detection System based on Large Language Models**

cs.NI

Proceedings of 2025 IEEE 7th International Conference on  Communications, Information System and Computer Engineering (CISCE 2025)

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.17807v1) [paper-pdf](http://arxiv.org/pdf/2504.17807v1)

**Authors**: Ze Yang, Yihong Jin, Juntian Liu, Xinhe Xu, Yihan Zhang, Shuyang Ji

**Abstract**: The rapidly evolving cloud platforms and the escalating complexity of network traffic demand proper network traffic monitoring and anomaly detection to ensure network security and performance. This paper introduces a large language model (LLM)-based network traffic monitoring and anomaly detection system. In addition to existing models such as autoencoders and decision trees, we harness the power of large language models for processing sequence data from network traffic, which allows us a better capture of underlying complex patterns, as well as slight fluctuations in the dataset. We show for a given detection task, the need for a hybrid model that incorporates the attention mechanism of the transformer architecture into a supervised learning framework in order to achieve better accuracy. A pre-trained large language model analyzes and predicts the probable network traffic, and an anomaly detection layer that considers temporality and context is added. Moreover, we present a novel transfer learning-based methodology to enhance the model's effectiveness to quickly adapt to unknown network structures and adversarial conditions without requiring extensive labeled datasets. Actual results show that the designed model outperforms traditional methods in detection accuracy and computational efficiency, effectively identify various network anomalies such as zero-day attacks and traffic congestion pattern, and significantly reduce the false positive rate.



## **49. Gungnir: Exploiting Stylistic Features in Images for Backdoor Attacks on Diffusion Models**

cs.CV

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2502.20650v2) [paper-pdf](http://arxiv.org/pdf/2502.20650v2)

**Authors**: Yu Pan, Bingrong Dai, Jiahao Chen, Lin Wang, Yi Du, Jiao Liu

**Abstract**: In recent years, Diffusion Models (DMs) have demonstrated significant advances in the field of image generation. However, according to current research, DMs are vulnerable to backdoor attacks, which allow attackers to control the model's output by inputting data containing covert triggers, such as a specific visual patch or phrase. Existing defense strategies are well equipped to thwart such attacks through backdoor detection and trigger inversion because previous attack methods are constrained by limited input spaces and low-dimensional triggers. For example, visual triggers are easily observed by defenders, text-based or attention-based triggers are more susceptible to neural network detection. To explore more possibilities of backdoor attack in DMs, we propose Gungnir, a novel method that enables attackers to activate the backdoor in DMs through style triggers within input images. Our approach proposes using stylistic features as triggers for the first time and implements backdoor attacks successfully in image-to-image tasks by introducing Reconstructing-Adversarial Noise (RAN) and Short-Term Timesteps-Retention (STTR). Our technique generates trigger-embedded images that are perceptually indistinguishable from clean images, thus bypassing both manual inspection and automated detection neural networks. Experiments demonstrate that Gungnir can easily bypass existing defense methods. Among existing DM defense frameworks, our approach achieves a 0 backdoor detection rate (BDR). Our codes are available at https://github.com/paoche11/Gungnir.



## **50. Evaluating the Robustness of Multimodal Agents Against Active Environmental Injection Attacks**

cs.CL

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2502.13053v2) [paper-pdf](http://arxiv.org/pdf/2502.13053v2)

**Authors**: Yurun Chen, Xavier Hu, Keting Yin, Juncheng Li, Shengyu Zhang

**Abstract**: As researchers continue to optimize AI agents for more effective task execution within operating systems, they often overlook a critical security concern: the ability of these agents to detect "impostors" within their environment. Through an analysis of the agents' operational context, we identify a significant threat-attackers can disguise malicious attacks as environmental elements, injecting active disturbances into the agents' execution processes to manipulate their decision-making. We define this novel threat as the Active Environment Injection Attack (AEIA). Focusing on the interaction mechanisms of the Android OS, we conduct a risk assessment of AEIA and identify two critical security vulnerabilities: (1) Adversarial content injection in multimodal interaction interfaces, where attackers embed adversarial instructions within environmental elements to mislead agent decision-making; and (2) Reasoning gap vulnerabilities in the agent's task execution process, which increase susceptibility to AEIA attacks during reasoning. To evaluate the impact of these vulnerabilities, we propose AEIA-MN, an attack scheme that exploits interaction vulnerabilities in mobile operating systems to assess the robustness of MLLM-based agents. Experimental results show that even advanced MLLMs are highly vulnerable to this attack, achieving a maximum attack success rate of 93% on the AndroidWorld benchmark by combining two vulnerabilities.



