# Latest Adversarial Attack Papers
**update at 2025-04-25 10:06:38**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Siren -- Advancing Cybersecurity through Deception and Adaptive Analysis**

cs.CR

14 pages, 5 figures, 13th Computing Conference 2025 - London, United  Kingdom

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2406.06225v2) [paper-pdf](http://arxiv.org/pdf/2406.06225v2)

**Authors**: Samhruth Ananthanarayanan, Girish Kulathumani, Ganesh Narayanan

**Abstract**: Siren represents a pioneering research effort aimed at fortifying cybersecurity through strategic integration of deception, machine learning, and proactive threat analysis. Drawing inspiration from mythical sirens, this project employs sophisticated methods to lure potential threats into controlled environments. The system features a dynamic machine learning model for realtime analysis and classification, ensuring continuous adaptability to emerging cyber threats. The architectural framework includes a link monitoring proxy, a purpose-built machine learning model for dynamic link analysis, and a honeypot enriched with simulated user interactions to intensify threat engagement. Data protection within the honeypot is fortified with probabilistic encryption. Additionally, the incorporation of simulated user activity extends the system's capacity to capture and learn from potential attackers even after user disengagement. Overall, Siren introduces a paradigm shift in cybersecurity, transforming traditional defense mechanisms into proactive systems that actively engage and learn from potential adversaries. The research strives to enhance user protection while yielding valuable insights for ongoing refinement in response to the evolving landscape of cybersecurity threats.



## **2. On the Generalization of Adversarially Trained Quantum Classifiers**

quant-ph

22 pages, 6 figures

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17690v1) [paper-pdf](http://arxiv.org/pdf/2504.17690v1)

**Authors**: Petros Georgiou, Aaron Mark Thomas, Sharu Theresa Jose, Osvaldo Simeone

**Abstract**: Quantum classifiers are vulnerable to adversarial attacks that manipulate their input classical or quantum data. A promising countermeasure is adversarial training, where quantum classifiers are trained by using an attack-aware, adversarial loss function. This work establishes novel bounds on the generalization error of adversarially trained quantum classifiers when tested in the presence of perturbation-constrained adversaries. The bounds quantify the excess generalization error incurred to ensure robustness to adversarial attacks as scaling with the training sample size $m$ as $1/\sqrt{m}$, while yielding insights into the impact of the quantum embedding. For quantum binary classifiers employing \textit{rotation embedding}, we find that, in the presence of adversarial attacks on classical inputs $\mathbf{x}$, the increase in sample complexity due to adversarial training over conventional training vanishes in the limit of high dimensional inputs $\mathbf{x}$. In contrast, when the adversary can directly attack the quantum state $\rho(\mathbf{x})$ encoding the input $\mathbf{x}$, the excess generalization error depends on the choice of embedding only through its Hilbert space dimension. The results are also extended to multi-class classifiers. We validate our theoretical findings with numerical experiments.



## **3. Evaluating the Vulnerability of ML-Based Ethereum Phishing Detectors to Single-Feature Adversarial Perturbations**

cs.CR

24 pages; an extension of a paper that appeared at WISA 2024

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17684v1) [paper-pdf](http://arxiv.org/pdf/2504.17684v1)

**Authors**: Ahod Alghuried, Ali Alkinoon, Abdulaziz Alghamdi, Soohyeon Choi, Manar Mohaisen, David Mohaisen

**Abstract**: This paper explores the vulnerability of machine learning models to simple single-feature adversarial attacks in the context of Ethereum fraudulent transaction detection. Through comprehensive experimentation, we investigate the impact of various adversarial attack strategies on model performance metrics. Our findings, highlighting how prone those techniques are to simple attacks, are alarming, and the inconsistency in the attacks' effect on different algorithms promises ways for attack mitigation. We examine the effectiveness of different mitigation strategies, including adversarial training and enhanced feature selection, in enhancing model robustness and show their effectiveness.



## **4. Regulatory Markets for AI Safety**

cs.CY

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2001.00078v2) [paper-pdf](http://arxiv.org/pdf/2001.00078v2)

**Authors**: Jack Clark, Gillian K. Hadfield

**Abstract**: We propose a new model for regulation to achieve AI safety: global regulatory markets. We first sketch the model in general terms and provide an overview of the costs and benefits of this approach. We then demonstrate how the model might work in practice: responding to the risk of adversarial attacks on AI models employed in commercial drones.



## **5. GRANITE : a Byzantine-Resilient Dynamic Gossip Learning Framework**

cs.LG

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17471v1) [paper-pdf](http://arxiv.org/pdf/2504.17471v1)

**Authors**: Yacine Belal, Mohamed Maouche, Sonia Ben Mokhtar, Anthony Simonet-Boulogne

**Abstract**: Gossip Learning (GL) is a decentralized learning paradigm where users iteratively exchange and aggregate models with a small set of neighboring peers. Recent GL approaches rely on dynamic communication graphs built and maintained using Random Peer Sampling (RPS) protocols. Thanks to graph dynamics, GL can achieve fast convergence even over extremely sparse topologies. However, the robustness of GL over dy- namic graphs to Byzantine (model poisoning) attacks remains unaddressed especially when Byzantine nodes attack the RPS protocol to scale up model poisoning. We address this issue by introducing GRANITE, a framework for robust learning over sparse, dynamic graphs in the presence of a fraction of Byzantine nodes. GRANITE relies on two key components (i) a History-aware Byzantine-resilient Peer Sampling protocol (HaPS), which tracks previously encountered identifiers to reduce adversarial influence over time, and (ii) an Adaptive Probabilistic Threshold (APT), which leverages an estimate of Byzantine presence to set aggregation thresholds with formal guarantees. Empirical results confirm that GRANITE maintains convergence with up to 30% Byzantine nodes, improves learning speed via adaptive filtering of poisoned models and obtains these results in up to 9 times sparser graphs than dictated by current theory.



## **6. Evaluating Time Series Models for Urban Wastewater Management: Predictive Performance, Model Complexity and Resilience**

cs.LG

6 pages, 6 figures, accepted at 10th International Conference on  Smart and Sustainable Technologies (SpliTech) 2025, GitHub:  https://github.com/calgo-lab/resilient-timeseries-evaluation

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17461v1) [paper-pdf](http://arxiv.org/pdf/2504.17461v1)

**Authors**: Vipin Singh, Tianheng Ling, Teodor Chiaburu, Felix Biessmann

**Abstract**: Climate change increases the frequency of extreme rainfall, placing a significant strain on urban infrastructures, especially Combined Sewer Systems (CSS). Overflows from overburdened CSS release untreated wastewater into surface waters, posing environmental and public health risks. Although traditional physics-based models are effective, they are costly to maintain and difficult to adapt to evolving system dynamics. Machine Learning (ML) approaches offer cost-efficient alternatives with greater adaptability. To systematically assess the potential of ML for modeling urban infrastructure systems, we propose a protocol for evaluating Neural Network architectures for CSS time series forecasting with respect to predictive performance, model complexity, and robustness to perturbations. In addition, we assess model performance on peak events and critical fluctuations, as these are the key regimes for urban wastewater management. To investigate the feasibility of lightweight models suitable for IoT deployment, we compare global models, which have access to all information, with local models, which rely solely on nearby sensor readings. Additionally, to explore the security risks posed by network outages or adversarial attacks on urban infrastructure, we introduce error models that assess the resilience of models. Our results demonstrate that while global models achieve higher predictive performance, local models provide sufficient resilience in decentralized scenarios, ensuring robust modeling of urban infrastructure. Furthermore, models with longer native forecast horizons exhibit greater robustness to data perturbations. These findings contribute to the development of interpretable and reliable ML solutions for sustainable urban wastewater management. The implementation is available in our GitHub repository.



## **7. Unveiling Hidden Vulnerabilities in Digital Human Generation via Adversarial Attacks**

cs.CV

14 pages, 7 figures

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17457v1) [paper-pdf](http://arxiv.org/pdf/2504.17457v1)

**Authors**: Zhiying Li, Yeying Jin, Fan Shen, Zhi Liu, Weibin Chen, Pengju Zhang, Xiaomei Zhang, Boyu Chen, Michael Shen, Kejian Wu, Zhaoxin Fan, Jin Dong

**Abstract**: Expressive human pose and shape estimation (EHPS) is crucial for digital human generation, especially in applications like live streaming. While existing research primarily focuses on reducing estimation errors, it largely neglects robustness and security aspects, leaving these systems vulnerable to adversarial attacks. To address this significant challenge, we propose the \textbf{Tangible Attack (TBA)}, a novel framework designed to generate adversarial examples capable of effectively compromising any digital human generation model. Our approach introduces a \textbf{Dual Heterogeneous Noise Generator (DHNG)}, which leverages Variational Autoencoders (VAE) and ControlNet to produce diverse, targeted noise tailored to the original image features. Additionally, we design a custom \textbf{adversarial loss function} to optimize the noise, ensuring both high controllability and potent disruption. By iteratively refining the adversarial sample through multi-gradient signals from both the noise and the state-of-the-art EHPS model, TBA substantially improves the effectiveness of adversarial attacks. Extensive experiments demonstrate TBA's superiority, achieving a remarkable 41.0\% increase in estimation error, with an average improvement of approximately 17.0\%. These findings expose significant security vulnerabilities in current EHPS models and highlight the need for stronger defenses in digital human generation systems.



## **8. Analysis and Mitigation of Data injection Attacks against Data-Driven Control**

eess.SY

Under review for publication

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17347v1) [paper-pdf](http://arxiv.org/pdf/2504.17347v1)

**Authors**: Sribalaji C. Anand

**Abstract**: This paper investigates the impact of false data injection attacks on data-driven control systems. Specifically, we consider an adversary injecting false data into the sensor channels during the learning phase. When the operator seeks to learn a stable state-feedback controller, we propose an attack strategy capable of misleading the operator into learning an unstable feedback gain. We also investigate the effects of constant-bias injection attacks on data-driven linear quadratic regulation (LQR). Finally, we explore potential mitigation strategies and support our findings with numerical examples.



## **9. JailbreakLens: Interpreting Jailbreak Mechanism in the Lens of Representation and Circuit**

cs.CR

17 pages, 11 figures

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2411.11114v2) [paper-pdf](http://arxiv.org/pdf/2411.11114v2)

**Authors**: Zeqing He, Zhibo Wang, Zhixuan Chu, Huiyu Xu, Wenhui Zhang, Qinglong Wang, Rui Zheng

**Abstract**: Despite the outstanding performance of Large language Models (LLMs) in diverse tasks, they are vulnerable to jailbreak attacks, wherein adversarial prompts are crafted to bypass their security mechanisms and elicit unexpected responses. Although jailbreak attacks are prevalent, the understanding of their underlying mechanisms remains limited. Recent studies have explained typical jailbreaking behavior (e.g., the degree to which the model refuses to respond) of LLMs by analyzing representation shifts in their latent space caused by jailbreak prompts or identifying key neurons that contribute to the success of jailbreak attacks. However, these studies neither explore diverse jailbreak patterns nor provide a fine-grained explanation from the failure of circuit to the changes of representational, leaving significant gaps in uncovering the jailbreak mechanism. In this paper, we propose JailbreakLens, an interpretation framework that analyzes jailbreak mechanisms from both representation (which reveals how jailbreaks alter the model's harmfulness perception) and circuit perspectives~(which uncovers the causes of these deceptions by identifying key circuits contributing to the vulnerability), tracking their evolution throughout the entire response generation process. We then conduct an in-depth evaluation of jailbreak behavior on five mainstream LLMs under seven jailbreak strategies. Our evaluation reveals that jailbreak prompts amplify components that reinforce affirmative responses while suppressing those that produce refusal. This manipulation shifts model representations toward safe clusters to deceive the LLM, leading it to provide detailed responses instead of refusals. Notably, we find a strong and consistent correlation between representation deception and activation shift of key circuits across diverse jailbreak methods and multiple LLMs.



## **10. Enhancing Variational Autoencoders with Smooth Robust Latent Encoding**

cs.LG

Under review

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17219v1) [paper-pdf](http://arxiv.org/pdf/2504.17219v1)

**Authors**: Hyomin Lee, Minseon Kim, Sangwon Jang, Jongheon Jeong, Sung Ju Hwang

**Abstract**: Variational Autoencoders (VAEs) have played a key role in scaling up diffusion-based generative models, as in Stable Diffusion, yet questions regarding their robustness remain largely underexplored. Although adversarial training has been an established technique for enhancing robustness in predictive models, it has been overlooked for generative models due to concerns about potential fidelity degradation by the nature of trade-offs between performance and robustness. In this work, we challenge this presumption, introducing Smooth Robust Latent VAE (SRL-VAE), a novel adversarial training framework that boosts both generation quality and robustness. In contrast to conventional adversarial training, which focuses on robustness only, our approach smooths the latent space via adversarial perturbations, promoting more generalizable representations while regularizing with originality representation to sustain original fidelity. Applied as a post-training step on pre-trained VAEs, SRL-VAE improves image robustness and fidelity with minimal computational overhead. Experiments show that SRL-VAE improves both generation quality, in image reconstruction and text-guided image editing, and robustness, against Nightshade attacks and image editing attacks. These results establish a new paradigm, showing that adversarial training, once thought to be detrimental to generative models, can instead enhance both fidelity and robustness.



## **11. CheatAgent: Attacking LLM-Empowered Recommender Systems via LLM Agent**

cs.CR

Accepted by KDD 2024;

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.13192v2) [paper-pdf](http://arxiv.org/pdf/2504.13192v2)

**Authors**: Liang-bo Ning, Shijie Wang, Wenqi Fan, Qing Li, Xin Xu, Hao Chen, Feiran Huang

**Abstract**: Recently, Large Language Model (LLM)-empowered recommender systems (RecSys) have brought significant advances in personalized user experience and have attracted considerable attention. Despite the impressive progress, the research question regarding the safety vulnerability of LLM-empowered RecSys still remains largely under-investigated. Given the security and privacy concerns, it is more practical to focus on attacking the black-box RecSys, where attackers can only observe the system's inputs and outputs. However, traditional attack approaches employing reinforcement learning (RL) agents are not effective for attacking LLM-empowered RecSys due to the limited capabilities in processing complex textual inputs, planning, and reasoning. On the other hand, LLMs provide unprecedented opportunities to serve as attack agents to attack RecSys because of their impressive capability in simulating human-like decision-making processes. Therefore, in this paper, we propose a novel attack framework called CheatAgent by harnessing the human-like capabilities of LLMs, where an LLM-based agent is developed to attack LLM-Empowered RecSys. Specifically, our method first identifies the insertion position for maximum impact with minimal input modification. After that, the LLM agent is designed to generate adversarial perturbations to insert at target positions. To further improve the quality of generated perturbations, we utilize the prompt tuning technique to improve attacking strategies via feedback from the victim RecSys iteratively. Extensive experiments across three real-world datasets demonstrate the effectiveness of our proposed attacking method.



## **12. Range and Topology Mutation Based Wireless Agility**

cs.ET

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17164v1) [paper-pdf](http://arxiv.org/pdf/2504.17164v1)

**Authors**: Qi Duan, Ehab Al-Shae, Jiang Xie

**Abstract**: In this paper, we present formal foundations for two wireless agility techniques: (1) Random Range Mutation (RNM) that allows for periodic changes of AP coverage range randomly, and (2) Ran- dom Topology Mutation (RTM) that allows for random motion and placement of APs in the wireless infrastructure. The goal of these techniques is to proactively defend against targeted attacks (e.g., DoS and eavesdropping) by forcing the wireless clients to change their AP association randomly. We apply Satisfiability Modulo The- ories (SMT) and Answer Set Programming (ASP) based constraint solving methods that allow for optimizing wireless AP mutation while maintaining service requirements including coverage, secu- rity and energy properties under incomplete information about the adversary strategies. Our evaluation validates the feasibility, scalability, and effectiveness of the formal methods based technical approaches.



## **13. On Minimizing Adversarial Counterfactual Error in Adversarial RL**

cs.LG

Presented at ICLR 2025

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2406.04724v4) [paper-pdf](http://arxiv.org/pdf/2406.04724v4)

**Authors**: Roman Belaire, Arunesh Sinha, Pradeep Varakantham

**Abstract**: Deep Reinforcement Learning (DRL) policies are highly susceptible to adversarial noise in observations, which poses significant risks in safety-critical scenarios. The challenge inherent to adversarial perturbations is that by altering the information observed by the agent, the state becomes only partially observable. Existing approaches address this by either enforcing consistent actions across nearby states or maximizing the worst-case value within adversarially perturbed observations. However, the former suffers from performance degradation when attacks succeed, while the latter tends to be overly conservative, leading to suboptimal performance in benign settings. We hypothesize that these limitations stem from their failing to account for partial observability directly. To this end, we introduce a novel objective called Adversarial Counterfactual Error (ACoE), defined on the beliefs about the true state and balancing value optimization with robustness. To make ACoE scalable in model-free settings, we propose the theoretically-grounded surrogate objective Cumulative-ACoE (C-ACoE). Our empirical evaluations on standard benchmarks (MuJoCo, Atari, and Highway) demonstrate that our method significantly outperforms current state-of-the-art approaches for addressing adversarial RL challenges, offering a promising direction for improving robustness in DRL under adversarial conditions. Our code is available at https://github.com/romanbelaire/acoe-robust-rl.



## **14. BadVideo: Stealthy Backdoor Attack against Text-to-Video Generation**

cs.CV

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.16907v1) [paper-pdf](http://arxiv.org/pdf/2504.16907v1)

**Authors**: Ruotong Wang, Mingli Zhu, Jiarong Ou, Rui Chen, Xin Tao, Pengfei Wan, Baoyuan Wu

**Abstract**: Text-to-video (T2V) generative models have rapidly advanced and found widespread applications across fields like entertainment, education, and marketing. However, the adversarial vulnerabilities of these models remain rarely explored. We observe that in T2V generation tasks, the generated videos often contain substantial redundant information not explicitly specified in the text prompts, such as environmental elements, secondary objects, and additional details, providing opportunities for malicious attackers to embed hidden harmful content. Exploiting this inherent redundancy, we introduce BadVideo, the first backdoor attack framework tailored for T2V generation. Our attack focuses on designing target adversarial outputs through two key strategies: (1) Spatio-Temporal Composition, which combines different spatiotemporal features to encode malicious information; (2) Dynamic Element Transformation, which introduces transformations in redundant elements over time to convey malicious information. Based on these strategies, the attacker's malicious target seamlessly integrates with the user's textual instructions, providing high stealthiness. Moreover, by exploiting the temporal dimension of videos, our attack successfully evades traditional content moderation systems that primarily analyze spatial information within individual frames. Extensive experiments demonstrate that BadVideo achieves high attack success rates while preserving original semantics and maintaining excellent performance on clean inputs. Overall, our work reveals the adversarial vulnerability of T2V models, calling attention to potential risks and misuse. Our project page is at https://wrt2000.github.io/BadVideo2025/.



## **15. aiXamine: Simplified LLM Safety and Security**

cs.CR

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.14985v2) [paper-pdf](http://arxiv.org/pdf/2504.14985v2)

**Authors**: Fatih Deniz, Dorde Popovic, Yazan Boshmaf, Euisuh Jeong, Minhaj Ahmad, Sanjay Chawla, Issa Khalil

**Abstract**: Evaluating Large Language Models (LLMs) for safety and security remains a complex task, often requiring users to navigate a fragmented landscape of ad hoc benchmarks, datasets, metrics, and reporting formats. To address this challenge, we present aiXamine, a comprehensive black-box evaluation platform for LLM safety and security. aiXamine integrates over 40 tests (i.e., benchmarks) organized into eight key services targeting specific dimensions of safety and security: adversarial robustness, code security, fairness and bias, hallucination, model and data privacy, out-of-distribution (OOD) robustness, over-refusal, and safety alignment. The platform aggregates the evaluation results into a single detailed report per model, providing a detailed breakdown of model performance, test examples, and rich visualizations. We used aiXamine to assess over 50 publicly available and proprietary LLMs, conducting over 2K examinations. Our findings reveal notable vulnerabilities in leading models, including susceptibility to adversarial attacks in OpenAI's GPT-4o, biased outputs in xAI's Grok-3, and privacy weaknesses in Google's Gemini 2.0. Additionally, we observe that open-source models can match or exceed proprietary models in specific services such as safety alignment, fairness and bias, and OOD robustness. Finally, we identify trade-offs between distillation strategies, model size, training methods, and architectural choices.



## **16. A robust and composable device-independent protocol for oblivious transfer using (fully) untrusted quantum devices in the bounded storage model**

quant-ph

Major improvement in the main result (security against non-IID  devices)

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2404.11283v2) [paper-pdf](http://arxiv.org/pdf/2404.11283v2)

**Authors**: Rishabh Batra, Sayantan Chakraborty, Rahul Jain, Upendra Kapshikar

**Abstract**: We present a robust and composable device-independent (DI) quantum protocol between two parties for oblivious transfer (OT) using Magic Square devices in the bounded storage model in which the (honest and cheating) devices and parties have no long-term quantum memory. After a fixed constant (real-world) time interval, referred to as DELAY, the quantum states decohere completely. The adversary (cheating party), with full control over the devices, is allowed joint (non-IID) quantum operations on the devices, and there are no time and space complexity bounds placed on its powers. The running time of the honest parties is polylog({\lambda}) (where {\lambda} is the security parameter). Our protocol has negligible (in {\lambda}) correctness and security errors and can be implemented in the NISQ (Noisy Intermediate Scale Quantum) era. By robustness, we mean that our protocol is correct even when devices are slightly off (by a small constant) from their ideal specification. This is an important property since small manufacturing errors in the real-world devices are inevitable. Our protocol is sequentially composable and, hence, can be used as a building block to construct larger protocols (including DI bit-commitment and DI secure multi-party computation) while still preserving correctness and security guarantees.   None of the known DI protocols for OT in the literature are robust and secure against joint quantum attacks. This was a major open question in device-independent two-party distrustful cryptography, which we resolve.   We prove a parallel repetition theorem for a certain class of entangled games with a hybrid (quantum-classical) strategy to show the security of our protocol. The hybrid strategy helps to incorporate DELAY in our protocol. This parallel repetition theorem is a main technical contribution of our work.



## **17. Exploring Adversarial Transferability between Kolmogorov-arnold Networks**

cs.CV

After the submission of the paper, we realized that the study still  has room for expansion. In order to make the research findings more profound  and comprehensive, we have decided to withdraw the paper so that we can  conduct further research and expansion

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2503.06276v2) [paper-pdf](http://arxiv.org/pdf/2503.06276v2)

**Authors**: Songping Wang, Xinquan Yue, Yueming Lyu, Caifeng Shan

**Abstract**: Kolmogorov-Arnold Networks (KANs) have emerged as a transformative model paradigm, significantly impacting various fields. However, their adversarial robustness remains less underexplored, especially across different KAN architectures. To explore this critical safety issue, we conduct an analysis and find that due to overfitting to the specific basis functions of KANs, they possess poor adversarial transferability among different KANs. To tackle this challenge, we propose AdvKAN, the first transfer attack method for KANs. AdvKAN integrates two key components: 1) a Breakthrough-Defense Surrogate Model (BDSM), which employs a breakthrough-defense training strategy to mitigate overfitting to the specific structures of KANs. 2) a Global-Local Interaction (GLI) technique, which promotes sufficient interaction between adversarial gradients of hierarchical levels, further smoothing out loss surfaces of KANs. Both of them work together to enhance the strength of transfer attack among different KANs. Extensive experimental results on various KANs and datasets demonstrate the effectiveness of AdvKAN, which possesses notably superior attack capabilities and deeply reveals the vulnerabilities of KANs. Code will be released upon acceptance.



## **18. Fast Adversarial Training with Weak-to-Strong Spatial-Temporal Consistency in the Frequency Domain on Videos**

cs.CV

After the submission of the paper, we realized that the study still  has room for expansion. In order to make the research findings more profound  and comprehensive, we have decided to withdraw the paper so that we can  conduct further research and expansion

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.14921v2) [paper-pdf](http://arxiv.org/pdf/2504.14921v2)

**Authors**: Songping Wang, Hanqing Liu, Yueming Lyu, Xiantao Hu, Ziwen He, Wei Wang, Caifeng Shan, Liang Wang

**Abstract**: Adversarial Training (AT) has been shown to significantly enhance adversarial robustness via a min-max optimization approach. However, its effectiveness in video recognition tasks is hampered by two main challenges. First, fast adversarial training for video models remains largely unexplored, which severely impedes its practical applications. Specifically, most video adversarial training methods are computationally costly, with long training times and high expenses. Second, existing methods struggle with the trade-off between clean accuracy and adversarial robustness. To address these challenges, we introduce Video Fast Adversarial Training with Weak-to-Strong consistency (VFAT-WS), the first fast adversarial training method for video data. Specifically, VFAT-WS incorporates the following key designs: First, it integrates a straightforward yet effective temporal frequency augmentation (TF-AUG), and its spatial-temporal enhanced form STF-AUG, along with a single-step PGD attack to boost training efficiency and robustness. Second, it devises a weak-to-strong spatial-temporal consistency regularization, which seamlessly integrates the simpler TF-AUG and the more complex STF-AUG. Leveraging the consistency regularization, it steers the learning process from simple to complex augmentations. Both of them work together to achieve a better trade-off between clean accuracy and robustness. Extensive experiments on UCF-101 and HMDB-51 with both CNN and Transformer-based models demonstrate that VFAT-WS achieves great improvements in adversarial robustness and corruption robustness, while accelerating training by nearly 490%.



## **19. Information Leakage of Sentence Embeddings via Generative Embedding Inversion Attacks**

cs.IR

This is a preprint of our paper accepted at SIGIR 2025

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.16609v1) [paper-pdf](http://arxiv.org/pdf/2504.16609v1)

**Authors**: Antonios Tragoudaras, Theofanis Aslanidis, Emmanouil Georgios Lionis, Marina Orozco González, Panagiotis Eustratiadis

**Abstract**: Text data are often encoded as dense vectors, known as embeddings, which capture semantic, syntactic, contextual, and domain-specific information. These embeddings, widely adopted in various applications, inherently contain rich information that may be susceptible to leakage under certain attacks. The GEIA framework highlights vulnerabilities in sentence embeddings, demonstrating that they can reveal the original sentences they represent. In this study, we reproduce GEIA's findings across various neural sentence embedding models. Additionally, we contribute new analysis to examine whether these models leak sensitive information from their training datasets. We propose a simple yet effective method without any modification to the attacker's architecture proposed in GEIA. The key idea is to examine differences between log-likelihood for masked and original variants of data that sentence embedding models have been pre-trained on, calculated on the embedding space of the attacker. Our findings indicate that following our approach, an adversary party can recover meaningful sensitive information related to the pre-training knowledge of the popular models used for creating sentence embeddings, seriously undermining their security. Our code is available on: https://github.com/taslanidis/GEIA



## **20. Seeking Flat Minima over Diverse Surrogates for Improved Adversarial Transferability: A Theoretical Framework and Algorithmic Instantiation**

cs.CR

26 pages, 6 figures

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.16474v1) [paper-pdf](http://arxiv.org/pdf/2504.16474v1)

**Authors**: Meixi Zheng, Kehan Wu, Yanbo Fan, Rui Huang, Baoyuan Wu

**Abstract**: The transfer-based black-box adversarial attack setting poses the challenge of crafting an adversarial example (AE) on known surrogate models that remain effective against unseen target models. Due to the practical importance of this task, numerous methods have been proposed to address this challenge. However, most previous methods are heuristically designed and intuitively justified, lacking a theoretical foundation. To bridge this gap, we derive a novel transferability bound that offers provable guarantees for adversarial transferability. Our theoretical analysis has the advantages of \textit{(i)} deepening our understanding of previous methods by building a general attack framework and \textit{(ii)} providing guidance for designing an effective attack algorithm. Our theoretical results demonstrate that optimizing AEs toward flat minima over the surrogate model set, while controlling the surrogate-target model shift measured by the adversarial model discrepancy, yields a comprehensive guarantee for AE transferability. The results further lead to a general transfer-based attack framework, within which we observe that previous methods consider only partial factors contributing to the transferability. Algorithmically, inspired by our theoretical results, we first elaborately construct the surrogate model set in which models exhibit diverse adversarial vulnerabilities with respect to AEs to narrow an instantiated adversarial model discrepancy. Then, a \textit{model-Diversity-compatible Reverse Adversarial Perturbation} (DRAP) is generated to effectively promote the flatness of AEs over diverse surrogate models to improve transferability. Extensive experiments on NIPS2017 and CIFAR-10 datasets against various target models demonstrate the effectiveness of our proposed attack.



## **21. Automated Trustworthiness Oracle Generation for Machine Learning Text Classifiers**

cs.SE

24 pages, 5 tables, 9 figures, Camera-ready version accepted to FSE  2025

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2410.22663v3) [paper-pdf](http://arxiv.org/pdf/2410.22663v3)

**Authors**: Lam Nguyen Tung, Steven Cho, Xiaoning Du, Neelofar Neelofar, Valerio Terragni, Stefano Ruberto, Aldeida Aleti

**Abstract**: Machine learning (ML) for text classification has been widely used in various domains. These applications can significantly impact ethics, economics, and human behavior, raising serious concerns about trusting ML decisions. Studies indicate that conventional metrics are insufficient to build human trust in ML models. These models often learn spurious correlations and predict based on them. In the real world, their performance can deteriorate significantly. To avoid this, a common practice is to test whether predictions are reasonable based on valid patterns in the data. Along with this, a challenge known as the trustworthiness oracle problem has been introduced. Due to the lack of automated trustworthiness oracles, the assessment requires manual validation of the decision process disclosed by explanation methods. However, this is time-consuming, error-prone, and unscalable.   We propose TOKI, the first automated trustworthiness oracle generation method for text classifiers. TOKI automatically checks whether the words contributing the most to a prediction are semantically related to the predicted class. Specifically, we leverage ML explanations to extract the decision-contributing words and measure their semantic relatedness with the class based on word embeddings. We also introduce a novel adversarial attack method that targets trustworthiness vulnerabilities identified by TOKI. To evaluate their alignment with human judgement, experiments are conducted. We compare TOKI with a naive baseline based solely on model confidence and TOKI-guided adversarial attack method with A2T, a SOTA adversarial attack method. Results show that relying on prediction uncertainty cannot effectively distinguish between trustworthy and untrustworthy predictions, TOKI achieves 142% higher accuracy than the naive baseline, and TOKI-guided attack method is more effective with fewer perturbations than A2T.



## **22. Large Language Model Sentinel: LLM Agent for Adversarial Purification**

cs.CL

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2405.20770v4) [paper-pdf](http://arxiv.org/pdf/2405.20770v4)

**Authors**: Guang Lin, Toshihisa Tanaka, Qibin Zhao

**Abstract**: Over the past two years, the use of large language models (LLMs) has advanced rapidly. While these LLMs offer considerable convenience, they also raise security concerns, as LLMs are vulnerable to adversarial attacks by some well-designed textual perturbations. In this paper, we introduce a novel defense technique named Large LAnguage MOdel Sentinel (LLAMOS), which is designed to enhance the adversarial robustness of LLMs by purifying the adversarial textual examples before feeding them into the target LLM. Our method comprises two main components: a) Agent instruction, which can simulate a new agent for adversarial defense, altering minimal characters to maintain the original meaning of the sentence while defending against attacks; b) Defense guidance, which provides strategies for modifying clean or adversarial examples to ensure effective defense and accurate outputs from the target LLMs. Remarkably, the defense agent demonstrates robust defensive capabilities even without learning from adversarial examples. Additionally, we conduct an intriguing adversarial experiment where we develop two agents, one for defense and one for attack, and engage them in mutual confrontation. During the adversarial interactions, neither agent completely beat the other. Extensive experiments on both open-source and closed-source LLMs demonstrate that our method effectively defends against adversarial attacks, thereby enhancing adversarial robustness.



## **23. Property-Preserving Hashing for $\ell_1$-Distance Predicates: Applications to Countering Adversarial Input Attacks**

cs.CR

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.16355v1) [paper-pdf](http://arxiv.org/pdf/2504.16355v1)

**Authors**: Hassan Asghar, Chenhan Zhang, Dali Kaafar

**Abstract**: Perceptual hashing is used to detect whether an input image is similar to a reference image with a variety of security applications. Recently, they have been shown to succumb to adversarial input attacks which make small imperceptible changes to the input image yet the hashing algorithm does not detect its similarity to the original image. Property-preserving hashing (PPH) is a recent construct in cryptography, which preserves some property (predicate) of its inputs in the hash domain. Researchers have so far shown constructions of PPH for Hamming distance predicates, which, for instance, outputs 1 if two inputs are within Hamming distance $t$. A key feature of PPH is its strong correctness guarantee, i.e., the probability that the predicate will not be correctly evaluated in the hash domain is negligible. Motivated by the use case of detecting similar images under adversarial setting, we propose the first PPH construction for an $\ell_1$-distance predicate. Roughly, this predicate checks if the two one-sided $\ell_1$-distances between two images are within a threshold $t$. Since many adversarial attacks use $\ell_2$-distance (related to $\ell_1$-distance) as the objective function to perturb the input image, by appropriately choosing the threshold $t$, we can force the attacker to add considerable noise to evade detection, and hence significantly deteriorate the image quality. Our proposed scheme is highly efficient, and runs in time $O(t^2)$. For grayscale images of size $28 \times 28$, we can evaluate the predicate in $0.0784$ seconds when pixel values are perturbed by up to $1 \%$. For larger RGB images of size $224 \times 224$, by dividing the image into 1,000 blocks, we achieve times of $0.0128$ seconds per block for $1 \%$ change, and up to $0.2641$ seconds per block for $14\%$ change.



## **24. Blockchain Meets Adaptive Honeypots: A Trust-Aware Approach to Next-Gen IoT Security**

cs.CR

This paper has been submitted to the IEEE Transactions on Network  Science and Engineering (TNSE) for possible publication

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.16226v1) [paper-pdf](http://arxiv.org/pdf/2504.16226v1)

**Authors**: Yazan Otoum, Arghavan Asad, Amiya Nayak

**Abstract**: Edge computing-based Next-Generation Wireless Networks (NGWN)-IoT offer enhanced bandwidth capacity for large-scale service provisioning but remain vulnerable to evolving cyber threats. Existing intrusion detection and prevention methods provide limited security as adversaries continually adapt their attack strategies. We propose a dynamic attack detection and prevention approach to address this challenge. First, blockchain-based authentication uses the Deoxys Authentication Algorithm (DAA) to verify IoT device legitimacy before data transmission. Next, a bi-stage intrusion detection system is introduced: the first stage uses signature-based detection via an Improved Random Forest (IRF) algorithm. In contrast, the second stage applies feature-based anomaly detection using a Diffusion Convolution Recurrent Neural Network (DCRNN). To ensure Quality of Service (QoS) and maintain Service Level Agreements (SLA), trust-aware service migration is performed using Heap-Based Optimization (HBO). Additionally, on-demand virtual High-Interaction honeypots deceive attackers and extract attack patterns, which are securely stored using the Bimodal Lattice Signature Scheme (BLISS) to enhance signature-based Intrusion Detection Systems (IDS). The proposed framework is implemented in the NS3 simulation environment and evaluated against existing methods across multiple performance metrics, including accuracy, attack detection rate, false negative rate, precision, recall, ROC curve, memory usage, CPU usage, and execution time. Experimental results demonstrate that the framework significantly outperforms existing approaches, reinforcing the security of NGWN-enabled IoT ecosystems



## **25. Adversarial Observations in Weather Forecasting**

cs.CR

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.15942v1) [paper-pdf](http://arxiv.org/pdf/2504.15942v1)

**Authors**: Erik Imgrund, Thorsten Eisenhofer, Konrad Rieck

**Abstract**: AI-based systems, such as Google's GenCast, have recently redefined the state of the art in weather forecasting, offering more accurate and timely predictions of both everyday weather and extreme events. While these systems are on the verge of replacing traditional meteorological methods, they also introduce new vulnerabilities into the forecasting process. In this paper, we investigate this threat and present a novel attack on autoregressive diffusion models, such as those used in GenCast, capable of manipulating weather forecasts and fabricating extreme events, including hurricanes, heat waves, and intense rainfall. The attack introduces subtle perturbations into weather observations that are statistically indistinguishable from natural noise and change less than 0.1% of the measurements - comparable to tampering with data from a single meteorological satellite. As modern forecasting integrates data from nearly a hundred satellites and many other sources operated by different countries, our findings highlight a critical security risk with the potential to cause large-scale disruptions and undermine public trust in weather prediction.



## **26. Human-Imperceptible Physical Adversarial Attack for NIR Face Recognition Models**

cs.CV

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.15823v1) [paper-pdf](http://arxiv.org/pdf/2504.15823v1)

**Authors**: Songyan Xie, Jinghang Wen, Encheng Su, Qiucheng Yu

**Abstract**: Near-infrared (NIR) face recognition systems, which can operate effectively in low-light conditions or in the presence of makeup, exhibit vulnerabilities when subjected to physical adversarial attacks. To further demonstrate the potential risks in real-world applications, we design a novel, stealthy, and practical adversarial patch to attack NIR face recognition systems in a black-box setting. We achieved this by utilizing human-imperceptible infrared-absorbing ink to generate multiple patches with digitally optimized shapes and positions for infrared images. To address the optimization mismatch between digital and real-world NIR imaging, we develop a light reflection model for human skin to minimize pixel-level discrepancies by simulating NIR light reflection.   Compared to state-of-the-art (SOTA) physical attacks on NIR face recognition systems, the experimental results show that our method improves the attack success rate in both digital and physical domains, particularly maintaining effectiveness across various face postures. Notably, the proposed approach outperforms SOTA methods, achieving an average attack success rate of 82.46% in the physical domain across different models, compared to 64.18% for existing methods. The artifact is available at https://anonymous.4open.science/r/Human-imperceptible-adversarial-patch-0703/.



## **27. Graph Neural Networks for Next-Generation-IoT: Recent Advances and Open Challenges**

cs.IT

28 pages, 15 figures, and 6 tables. Submitted for publication

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2412.20634v2) [paper-pdf](http://arxiv.org/pdf/2412.20634v2)

**Authors**: Nguyen Xuan Tung, Le Tung Giang, Bui Duc Son, Seon Geun Jeong, Trinh Van Chien, Won Joo Hwang, Lajos Hanzo

**Abstract**: Graph Neural Networks (GNNs) have emerged as a critical tool for optimizing and managing the complexities of the Internet of Things (IoT) in next-generation networks. This survey presents a comprehensive exploration of how GNNs may be harnessed in 6G IoT environments, focusing on key challenges and opportunities through a series of open questions. We commence with an exploration of GNN paradigms and the roles of node, edge, and graph-level tasks in solving wireless networking problems and highlight GNNs' ability to overcome the limitations of traditional optimization methods. This guidance enhances problem-solving efficiency across various next-generation (NG) IoT scenarios. Next, we provide a detailed discussion of the application of GNN in advanced NG enabling technologies, including massive MIMO, reconfigurable intelligent surfaces, satellites, THz, mobile edge computing (MEC), and ultra-reliable low latency communication (URLLC). We then delve into the challenges posed by adversarial attacks, offering insights into defense mechanisms to secure GNN-based NG-IoT networks. Next, we examine how GNNs can be integrated with future technologies like integrated sensing and communication (ISAC), satellite-air-ground-sea integrated networks (SAGSIN), and quantum computing. Our findings highlight the transformative potential of GNNs in improving efficiency, scalability, and security within NG-IoT systems, paving the way for future advances. Finally, we propose a set of design guidelines to facilitate the development of efficient, scalable, and secure GNN models tailored for NG IoT applications.



## **28. BaThe: Defense against the Jailbreak Attack in Multimodal Large Language Models by Treating Harmful Instruction as Backdoor Trigger**

cs.CR

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2408.09093v3) [paper-pdf](http://arxiv.org/pdf/2408.09093v3)

**Authors**: Yulin Chen, Haoran Li, Yirui Zhang, Zihao Zheng, Yangqiu Song, Bryan Hooi

**Abstract**: Multimodal Large Language Models (MLLMs) have showcased impressive performance in a variety of multimodal tasks. On the other hand, the integration of additional image modality may allow the malicious users to inject harmful content inside the images for jailbreaking. Unlike text-based LLMs, where adversaries need to select discrete tokens to conceal their malicious intent using specific algorithms, the continuous nature of image signals provides a direct opportunity for adversaries to inject harmful intentions. In this work, we propose $\textbf{BaThe}$ ($\textbf{Ba}$ckdoor $\textbf{T}$rigger S$\textbf{h}$i$\textbf{e}$ld), a simple yet effective jailbreak defense mechanism. Our work is motivated by recent research on jailbreak backdoor attack and virtual prompt backdoor attack in generative language models. Jailbreak backdoor attack uses harmful instructions combined with manually crafted strings as triggers to make the backdoored model generate prohibited responses. We assume that harmful instructions can function as triggers, and if we alternatively set rejection responses as the triggered response, the backdoored model then can defend against jailbreak attacks. We achieve this by utilizing virtual rejection prompt, similar to the virtual prompt backdoor attack. We embed the virtual rejection prompt into the soft text embeddings, which we call ``wedge''. Our comprehensive experiments demonstrate that BaThe effectively mitigates various types of jailbreak attacks and is adaptable to defend against unseen attacks, with minimal impact on MLLMs' performance.



## **29. Red Team Diffuser: Exposing Toxic Continuation Vulnerabilities in Vision-Language Models via Reinforcement Learning**

cs.CV

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2503.06223v2) [paper-pdf](http://arxiv.org/pdf/2503.06223v2)

**Authors**: Ruofan Wang, Xiang Zheng, Xiaosen Wang, Cong Wang, Xingjun Ma

**Abstract**: The growing deployment of large Vision-Language Models (VLMs) exposes critical safety gaps in their alignment mechanisms. While existing jailbreak studies primarily focus on VLMs' susceptibility to harmful instructions, we reveal a fundamental yet overlooked vulnerability: toxic text continuation, where VLMs produce highly toxic completions when prompted with harmful text prefixes paired with semantically adversarial images. To systematically study this threat, we propose Red Team Diffuser (RTD), the first red teaming diffusion model that coordinates adversarial image generation and toxic continuation through reinforcement learning. Our key innovations include dynamic cross-modal attack and stealth-aware optimization. For toxic text prefixes from an LLM safety benchmark, we conduct greedy search to identify optimal image prompts that maximally induce toxic completions. The discovered image prompts then drive RL-based diffusion model fine-tuning, producing semantically aligned adversarial images that boost toxicity rates. Stealth-aware optimization introduces joint adversarial rewards that balance toxicity maximization (via Detoxify classifier) and stealthiness (via BERTScore), circumventing traditional noise-based adversarial patterns. Experimental results demonstrate the effectiveness of RTD, increasing the toxicity rate of LLaVA outputs by 10.69% over text-only baselines on the original attack set and 8.91% on an unseen set, proving generalization capability. Moreover, RTD exhibits strong cross-model transferability, raising the toxicity rate by 5.1% on Gemini and 26.83% on LLaMA. Our findings expose two critical flaws in current VLM alignment: (1) failure to prevent toxic continuation from harmful prefixes, and (2) overlooking cross-modal attack vectors. These results necessitate a paradigm shift toward multimodal red teaming in safety evaluations.



## **30. TrojanDam: Detection-Free Backdoor Defense in Federated Learning through Proactive Model Robustification utilizing OOD Data**

cs.CR

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.15674v1) [paper-pdf](http://arxiv.org/pdf/2504.15674v1)

**Authors**: Yanbo Dai, Songze Li, Zihan Gan, Xueluan Gong

**Abstract**: Federated learning (FL) systems allow decentralized data-owning clients to jointly train a global model through uploading their locally trained updates to a centralized server. The property of decentralization enables adversaries to craft carefully designed backdoor updates to make the global model misclassify only when encountering adversary-chosen triggers. Existing defense mechanisms mainly rely on post-training detection after receiving updates. These methods either fail to identify updates which are deliberately fabricated statistically close to benign ones, or show inconsistent performance in different FL training stages. The effect of unfiltered backdoor updates will accumulate in the global model, and eventually become functional. Given the difficulty of ruling out every backdoor update, we propose a backdoor defense paradigm, which focuses on proactive robustification on the global model against potential backdoor attacks. We first reveal that the successful launching of backdoor attacks in FL stems from the lack of conflict between malicious and benign updates on redundant neurons of ML models. We proceed to prove the feasibility of activating redundant neurons utilizing out-of-distribution (OOD) samples in centralized settings, and migrating to FL settings to propose a novel backdoor defense mechanism, TrojanDam. The proposed mechanism has the FL server continuously inject fresh OOD mappings into the global model to activate redundant neurons, canceling the effect of backdoor updates during aggregation. We conduct systematic and extensive experiments to illustrate the superior performance of TrojanDam, over several SOTA backdoor defense methods across a wide range of FL settings.



## **31. Manipulating Multimodal Agents via Cross-Modal Prompt Injection**

cs.CV

17 pages, 5 figures

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.14348v2) [paper-pdf](http://arxiv.org/pdf/2504.14348v2)

**Authors**: Le Wang, Zonghao Ying, Tianyuan Zhang, Siyuan Liang, Shengshan Hu, Mingchuan Zhang, Aishan Liu, Xianglong Liu

**Abstract**: The emergence of multimodal large language models has redefined the agent paradigm by integrating language and vision modalities with external data sources, enabling agents to better interpret human instructions and execute increasingly complex tasks. However, in this work, we identify a critical yet previously overlooked security vulnerability in multimodal agents: cross-modal prompt injection attacks. To exploit this vulnerability, we propose CrossInject, a novel attack framework in which attackers embed adversarial perturbations across multiple modalities to align with target malicious content, allowing external instructions to hijack the agent's decision-making process and execute unauthorized tasks. Our approach consists of two key components. First, we introduce Visual Latent Alignment, where we optimize adversarial features to the malicious instructions in the visual embedding space based on a text-to-image generative model, ensuring that adversarial images subtly encode cues for malicious task execution. Subsequently, we present Textual Guidance Enhancement, where a large language model is leveraged to infer the black-box defensive system prompt through adversarial meta prompting and generate an malicious textual command that steers the agent's output toward better compliance with attackers' requests. Extensive experiments demonstrate that our method outperforms existing injection attacks, achieving at least a +26.4% increase in attack success rates across diverse tasks. Furthermore, we validate our attack's effectiveness in real-world multimodal autonomous agents, highlighting its potential implications for safety-critical applications.



## **32. Gungnir: Exploiting Stylistic Features in Images for Backdoor Attacks on Diffusion Models**

cs.CV

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2502.20650v2) [paper-pdf](http://arxiv.org/pdf/2502.20650v2)

**Authors**: Yu Pan, Bingrong Dai, Jiahao Chen, Lin Wang, Yi Du, Jiao Liu

**Abstract**: In recent years, Diffusion Models (DMs) have demonstrated significant advances in the field of image generation. However, according to current research, DMs are vulnerable to backdoor attacks, which allow attackers to control the model's output by inputting data containing covert triggers, such as a specific visual patch or phrase. Existing defense strategies are well equipped to thwart such attacks through backdoor detection and trigger inversion because previous attack methods are constrained by limited input spaces and low-dimensional triggers. For example, visual triggers are easily observed by defenders, text-based or attention-based triggers are more susceptible to neural network detection. To explore more possibilities of backdoor attack in DMs, we propose Gungnir, a novel method that enables attackers to activate the backdoor in DMs through style triggers within input images. Our approach proposes using stylistic features as triggers for the first time and implements backdoor attacks successfully in image-to-image tasks by introducing Reconstructing-Adversarial Noise (RAN) and Short-Term Timesteps-Retention (STTR). Our technique generates trigger-embedded images that are perceptually indistinguishable from clean images, thus bypassing both manual inspection and automated detection neural networks. Experiments demonstrate that Gungnir can easily bypass existing defense methods. Among existing DM defense frameworks, our approach achieves a 0 backdoor detection rate (BDR). Our codes are available at https://github.com/paoche11/Gungnir.



## **33. Evaluating the Robustness of Multimodal Agents Against Active Environmental Injection Attacks**

cs.CL

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2502.13053v2) [paper-pdf](http://arxiv.org/pdf/2502.13053v2)

**Authors**: Yurun Chen, Xavier Hu, Keting Yin, Juncheng Li, Shengyu Zhang

**Abstract**: As researchers continue to optimize AI agents for more effective task execution within operating systems, they often overlook a critical security concern: the ability of these agents to detect "impostors" within their environment. Through an analysis of the agents' operational context, we identify a significant threat-attackers can disguise malicious attacks as environmental elements, injecting active disturbances into the agents' execution processes to manipulate their decision-making. We define this novel threat as the Active Environment Injection Attack (AEIA). Focusing on the interaction mechanisms of the Android OS, we conduct a risk assessment of AEIA and identify two critical security vulnerabilities: (1) Adversarial content injection in multimodal interaction interfaces, where attackers embed adversarial instructions within environmental elements to mislead agent decision-making; and (2) Reasoning gap vulnerabilities in the agent's task execution process, which increase susceptibility to AEIA attacks during reasoning. To evaluate the impact of these vulnerabilities, we propose AEIA-MN, an attack scheme that exploits interaction vulnerabilities in mobile operating systems to assess the robustness of MLLM-based agents. Experimental results show that even advanced MLLMs are highly vulnerable to this attack, achieving a maximum attack success rate of 93% on the AndroidWorld benchmark by combining two vulnerabilities.



## **34. Unifying Image Counterfactuals and Feature Attributions with Latent-Space Adversarial Attacks**

cs.LG

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15479v1) [paper-pdf](http://arxiv.org/pdf/2504.15479v1)

**Authors**: Jeremy Goldwasser, Giles Hooker

**Abstract**: Counterfactuals are a popular framework for interpreting machine learning predictions. These what if explanations are notoriously challenging to create for computer vision models: standard gradient-based methods are prone to produce adversarial examples, in which imperceptible modifications to image pixels provoke large changes in predictions. We introduce a new, easy-to-implement framework for counterfactual images that can flexibly adapt to contemporary advances in generative modeling. Our method, Counterfactual Attacks, resembles an adversarial attack on the representation of the image along a low-dimensional manifold. In addition, given an auxiliary dataset of image descriptors, we show how to accompany counterfactuals with feature attribution that quantify the changes between the original and counterfactual images. These importance scores can be aggregated into global counterfactual explanations that highlight the overall features driving model predictions. While this unification is possible for any counterfactual method, it has particular computational efficiency for ours. We demonstrate the efficacy of our approach with the MNIST and CelebA datasets.



## **35. An Undetectable Watermark for Generative Image Models**

cs.CR

ICLR 2025

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2410.07369v4) [paper-pdf](http://arxiv.org/pdf/2410.07369v4)

**Authors**: Sam Gunn, Xuandong Zhao, Dawn Song

**Abstract**: We present the first undetectable watermarking scheme for generative image models. Undetectability ensures that no efficient adversary can distinguish between watermarked and un-watermarked images, even after making many adaptive queries. In particular, an undetectable watermark does not degrade image quality under any efficiently computable metric. Our scheme works by selecting the initial latents of a diffusion model using a pseudorandom error-correcting code (Christ and Gunn, 2024), a strategy which guarantees undetectability and robustness. We experimentally demonstrate that our watermarks are quality-preserving and robust using Stable Diffusion 2.1. Our experiments verify that, in contrast to every prior scheme we tested, our watermark does not degrade image quality. Our experiments also demonstrate robustness: existing watermark removal attacks fail to remove our watermark from images without significantly degrading the quality of the images. Finally, we find that we can robustly encode 512 bits in our watermark, and up to 2500 bits when the images are not subjected to watermark removal attacks. Our code is available at https://github.com/XuandongZhao/PRC-Watermark.



## **36. A Framework for Evaluating Emerging Cyberattack Capabilities of AI**

cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2503.11917v3) [paper-pdf](http://arxiv.org/pdf/2503.11917v3)

**Authors**: Mikel Rodriguez, Raluca Ada Popa, Four Flynn, Lihao Liang, Allan Dafoe, Anna Wang

**Abstract**: As frontier AI models become more capable, evaluating their potential to enable cyberattacks is crucial for ensuring the safe development of Artificial General Intelligence (AGI). Current cyber evaluation efforts are often ad-hoc, lacking systematic analysis of attack phases and guidance on targeted defenses. This work introduces a novel evaluation framework that addresses these limitations by: (1) examining the end-to-end attack chain, (2) identifying gaps in AI threat evaluation, and (3) helping defenders prioritize targeted mitigations and conduct AI-enabled adversary emulation for red teaming. Our approach adapts existing cyberattack chain frameworks for AI systems. We analyzed over 12,000 real-world instances of AI involvement in cyber incidents, catalogued by Google's Threat Intelligence Group, to curate seven representative attack chain archetypes. Through a bottleneck analysis on these archetypes, we pinpointed phases most susceptible to AI-driven disruption. We then identified and utilized externally developed cybersecurity model evaluations focused on these critical phases. We report on AI's potential to amplify offensive capabilities across specific attack stages, and offer recommendations for prioritizing defenses. We believe this represents the most comprehensive AI cyber risk evaluation framework published to date.



## **37. MST3 Encryption improvement with three-parameter group of Hermitian function field**

cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15391v1) [paper-pdf](http://arxiv.org/pdf/2504.15391v1)

**Authors**: Gennady Khalimov, Yevgen Kotukh

**Abstract**: This scholarly work presents an advanced cryptographic framework utilizing automorphism groups as the foundational structure for encryption scheme implementation. The proposed methodology employs a three-parameter group construction, distinguished by its application of logarithmic signatures positioned outside the group's center, a significant departure from conventional approaches. A key innovation in this implementation is utilizing the Hermitian function field as the underlying mathematical framework. This particular function field provides enhanced structural properties that strengthen the cryptographic protocol when integrated with the three-parameter group architecture. The encryption mechanism features phased key de-encapsulation from ciphertext, representing a substantial advantage over alternative implementations. This sequential extraction process introduces additional computational complexity for potential adversaries while maintaining efficient legitimate decryption. A notable characteristic of this cryptosystem is the direct correlation between the underlying group's mathematical strength and both the attack complexity and message size parameters. This relationship enables precise security-efficiency calibration based on specific implementation requirements and threat models. The application of automorphism groups with logarithmic signatures positioned outside the center represents a significant advancement in non-traditional cryptographic designs, particularly relevant in the context of post-quantum cryptographic resilience.



## **38. MR. Guard: Multilingual Reasoning Guardrail using Curriculum Learning**

cs.CL

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15241v1) [paper-pdf](http://arxiv.org/pdf/2504.15241v1)

**Authors**: Yahan Yang, Soham Dan, Shuo Li, Dan Roth, Insup Lee

**Abstract**: Large Language Models (LLMs) are susceptible to adversarial attacks such as jailbreaking, which can elicit harmful or unsafe behaviors. This vulnerability is exacerbated in multilingual setting, where multilingual safety-aligned data are often limited. Thus, developing a guardrail capable of detecting and filtering unsafe content across diverse languages is critical for deploying LLMs in real-world applications. In this work, we propose an approach to build a multilingual guardrail with reasoning. Our method consists of: (1) synthetic multilingual data generation incorporating culturally and linguistically nuanced variants, (2) supervised fine-tuning, and (3) a curriculum-guided Group Relative Policy Optimization (GRPO) framework that further improves performance. Experimental results demonstrate that our multilingual guardrail consistently outperforms recent baselines across both in-domain and out-of-domain languages. The multilingual reasoning capability of our guardrail enables it to generate multilingual explanations, which are particularly useful for understanding language-specific risks and ambiguities in multilingual content moderation.



## **39. Progressive Pruning: Analyzing the Impact of Intersection Attacks**

cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2410.08700v2) [paper-pdf](http://arxiv.org/pdf/2410.08700v2)

**Authors**: Christoph Döpmann, Maximilian Weisenseel, Florian Tschorsch

**Abstract**: Stream-based communication dominates today's Internet, posing unique challenges for anonymous communication networks (ACNs). Traditionally designed for independent messages, ACNs struggle to account for the inherent vulnerabilities of streams, such as susceptibility to intersection attacks. In this work, we address this gap and introduce progressive pruning, a novel methodology for quantifying the susceptibility to intersection attacks. Progressive pruning quantifies and monitors anonymity sets over time, providing an assessment of an adversary's success in correlating senders and receivers. We leverage this methodology to analyze synthetic scenarios and large-scale simulations of the Tor network using our newly developed TorFS simulator. Our findings reveal that anonymity is significantly influenced by stream length, user population, and stream distribution across the network. These insights highlight critical design challenges for future ACNs seeking to safeguard stream-based communication against traffic analysis attacks.



## **40. HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States**

cs.CL

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2502.14744v3) [paper-pdf](http://arxiv.org/pdf/2502.14744v3)

**Authors**: Yilei Jiang, Xinyan Gao, Tianshuo Peng, Yingshui Tan, Xiaoyong Zhu, Bo Zheng, Xiangyu Yue

**Abstract**: The integration of additional modalities increases the susceptibility of large vision-language models (LVLMs) to safety risks, such as jailbreak attacks, compared to their language-only counterparts. While existing research primarily focuses on post-hoc alignment techniques, the underlying safety mechanisms within LVLMs remain largely unexplored. In this work , we investigate whether LVLMs inherently encode safety-relevant signals within their internal activations during inference. Our findings reveal that LVLMs exhibit distinct activation patterns when processing unsafe prompts, which can be leveraged to detect and mitigate adversarial inputs without requiring extensive fine-tuning. Building on this insight, we introduce HiddenDetect, a novel tuning-free framework that harnesses internal model activations to enhance safety. Experimental results show that {HiddenDetect} surpasses state-of-the-art methods in detecting jailbreak attacks against LVLMs. By utilizing intrinsic safety-aware patterns, our method provides an efficient and scalable solution for strengthening LVLM robustness against multimodal threats. Our code will be released publicly at https://github.com/leigest519/HiddenDetect.



## **41. Scalable Discrete Event Simulation Tool for Large-Scale Cyber-Physical Energy Systems: Advancing System Efficiency and Scalability**

eess.SY

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15198v1) [paper-pdf](http://arxiv.org/pdf/2504.15198v1)

**Authors**: Khandaker Akramul Haque, Shining Sun, Xiang Huo, Ana E. Goulart, Katherine R. Davis

**Abstract**: Modern power systems face growing risks from cyber-physical attacks, necessitating enhanced resilience due to their societal function as critical infrastructures. The challenge is that defense of large-scale systems-of-systems requires scalability in their threat and risk assessment environment for cyber physical analysis including cyber-informed transmission planning, decision-making, and intrusion response. Hence, we present a scalable discrete event simulation tool for analysis of energy systems, called DESTinE. The tool is tailored for largescale cyber-physical systems, with a focus on power systems. It supports faster-than-real-time traffic generation and models packet flow and congestion under both normal and adversarial conditions. Using three well-established power system synthetic cases with 500, 2000, and 10,000 buses, we overlay a constructed cyber network employing star and radial topologies. Experiments are conducted to identify critical nodes within a communication network in response to a disturbance. The findings are incorporated into a constrained optimization problem to assess the impact of the disturbance on a specific node and its cascading effects on the overall network. Based on the solution of the optimization problem, a new hybrid network topology is also derived, combining the strengths of star and radial structures to improve network resilience. Furthermore, DESTinE is integrated with a virtual server and a hardware-in-the-loop (HIL) system using Raspberry Pi 5.



## **42. RainbowPlus: Enhancing Adversarial Prompt Generation via Evolutionary Quality-Diversity Search**

cs.CL

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.15047v1) [paper-pdf](http://arxiv.org/pdf/2504.15047v1)

**Authors**: Quy-Anh Dang, Chris Ngo, Truong-Son Hy

**Abstract**: Large Language Models (LLMs) exhibit remarkable capabilities but are susceptible to adversarial prompts that exploit vulnerabilities to produce unsafe or biased outputs. Existing red-teaming methods often face scalability challenges, resource-intensive requirements, or limited diversity in attack strategies. We propose RainbowPlus, a novel red-teaming framework rooted in evolutionary computation, enhancing adversarial prompt generation through an adaptive quality-diversity (QD) search that extends classical evolutionary algorithms like MAP-Elites with innovations tailored for language models. By employing a multi-element archive to store diverse high-quality prompts and a comprehensive fitness function to evaluate multiple prompts concurrently, RainbowPlus overcomes the constraints of single-prompt archives and pairwise comparisons in prior QD methods like Rainbow Teaming. Experiments comparing RainbowPlus to QD methods across six benchmark datasets and four open-source LLMs demonstrate superior attack success rate (ASR) and diversity (Diverse-Score $\approx 0.84$), generating up to 100 times more unique prompts (e.g., 10,418 vs. 100 for Ministral-8B-Instruct-2410). Against nine state-of-the-art methods on the HarmBench dataset with twelve LLMs (ten open-source, two closed-source), RainbowPlus achieves an average ASR of 81.1%, surpassing AutoDAN-Turbo by 3.9%, and is 9 times faster (1.45 vs. 13.50 hours). Our open-source implementation fosters further advancements in LLM safety, offering a scalable tool for vulnerability assessment. Code and resources are publicly available at https://github.com/knoveleng/rainbowplus, supporting reproducibility and future research in LLM red-teaming.



## **43. An Information-theoretic Security Analysis of Honeyword**

cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2311.10960v2) [paper-pdf](http://arxiv.org/pdf/2311.10960v2)

**Authors**: Pengcheng Su, Haibo Cheng, Wenting Li, Ping Wang

**Abstract**: Honeyword is a representative "honey" technique that employs decoy objects to mislead adversaries and protect the real ones. To assess the security of a Honeyword system, two metrics--flatness and success-number--have been proposed and evaluated using various simulated attackers. Existing evaluations typically apply statistical learning methods to distinguish real passwords from decoys on real-world datasets. However, such evaluations may overestimate the system's security, as more effective distinguishing attacks could potentially exist.   In this paper, we aim to analyze the security of Honeyword systems under the strongest theoretical attack, rather than relying on specific, expert-crafted attacks evaluated in prior experimental studies. We first derive mathematical expressions for the flatness and success-number under the strongest attack. We conduct analyses and computations for several typical scenarios, and determine the security of honeyword generation methods using a uniform distribution and the List model as examples.   We further evaluate the security of existing honeyword generation methods based on password probability models (PPMs), which depends on the sample size used for training. We investigate, for the first time, the sample complexity of several representative PPMs, introducing two novel polynomial-time approximation schemes for computing the total variation between PCFG models and between higher-order Markov models. Our experimental results show that for small-scale password distributions, sample sizes on the order of millions--often tens of millions--are required to reduce the total variation below 0.1. A surprising result is that we establish an equivalence between flatness and total variation, thus bridging the theoretical study of Honeyword systems with classical information theory. Finally, we discuss the practical implications of our findings.



## **44. Transferable Adversarial Attacks on SAM and Its Downstream Models**

cs.LG

update fig 1

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2410.20197v3) [paper-pdf](http://arxiv.org/pdf/2410.20197v3)

**Authors**: Song Xia, Wenhan Yang, Yi Yu, Xun Lin, Henghui Ding, Ling-Yu Duan, Xudong Jiang

**Abstract**: The utilization of large foundational models has a dilemma: while fine-tuning downstream tasks from them holds promise for making use of the well-generalized knowledge in practical applications, their open accessibility also poses threats of adverse usage. This paper, for the first time, explores the feasibility of adversarial attacking various downstream models fine-tuned from the segment anything model (SAM), by solely utilizing the information from the open-sourced SAM. In contrast to prevailing transfer-based adversarial attacks, we demonstrate the existence of adversarial dangers even without accessing the downstream task and dataset to train a similar surrogate model. To enhance the effectiveness of the adversarial attack towards models fine-tuned on unknown datasets, we propose a universal meta-initialization (UMI) algorithm to extract the intrinsic vulnerability inherent in the foundation model, which is then utilized as the prior knowledge to guide the generation of adversarial perturbations. Moreover, by formulating the gradient difference in the attacking process between the open-sourced SAM and its fine-tuned downstream models, we theoretically demonstrate that a deviation occurs in the adversarial update direction by directly maximizing the distance of encoded feature embeddings in the open-sourced SAM. Consequently, we propose a gradient robust loss that simulates the associated uncertainty with gradient-based noise augmentation to enhance the robustness of generated adversarial examples (AEs) towards this deviation, thus improving the transferability. Extensive experiments demonstrate the effectiveness of the proposed universal meta-initialized and gradient robust adversarial attack (UMI-GRAT) toward SAMs and their downstream models. Code is available at https://github.com/xiasong0501/GRAT.



## **45. PA-Boot: A Formally Verified Authentication Protocol for Multiprocessor Secure Boot**

cs.CR

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2209.07936v3) [paper-pdf](http://arxiv.org/pdf/2209.07936v3)

**Authors**: Zhuoruo Zhang, Rui Chang, Mingshuai Chen, Wenbo Shen, Chenyang Yu, He Huang, Qinming Dai, Yongwang Zhao

**Abstract**: Hardware supply-chain attacks are raising significant security threats to the boot process of multiprocessor systems. This paper identifies a new, prevalent hardware supply-chain attack surface that can bypass multiprocessor secure boot due to the absence of processor-authentication mechanisms. To defend against such attacks, we present PA-Boot, the first formally verified processor-authentication protocol for secure boot in multiprocessor systems. PA-Boot is proved functionally correct and is guaranteed to detect multiple adversarial behaviors, e.g., processor replacements, man-in-the-middle attacks, and tampering with certificates. The fine-grained formalization of PA-Boot and its fully mechanized security proofs are carried out in the Isabelle/HOL theorem prover with 306 lemmas/theorems and ~7,100 LoC. Experiments on a proof-of-concept implementation indicate that PA-Boot can effectively identify boot-process attacks with a considerably minor overhead and thereby improve the security of multiprocessor systems.



## **46. Verifying Robust Unlearning: Probing Residual Knowledge in Unlearned Models**

cs.LG

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2504.14798v1) [paper-pdf](http://arxiv.org/pdf/2504.14798v1)

**Authors**: Hao Xuan, Xingyu Li

**Abstract**: Machine Unlearning (MUL) is crucial for privacy protection and content regulation, yet recent studies reveal that traces of forgotten information persist in unlearned models, enabling adversaries to resurface removed knowledge. Existing verification methods only confirm whether unlearning was executed, failing to detect such residual information leaks. To address this, we introduce the concept of Robust Unlearning, ensuring models are indistinguishable from retraining and resistant to adversarial recovery. To empirically evaluate whether unlearning techniques meet this security standard, we propose the Unlearning Mapping Attack (UMA), a post-unlearning verification framework that actively probes models for forgotten traces using adversarial queries. Extensive experiments on discriminative and generative tasks show that existing unlearning techniques remain vulnerable, even when passing existing verification metrics. By establishing UMA as a practical verification tool, this study sets a new standard for assessing and enhancing machine unlearning security.



## **47. Modality Unified Attack for Omni-Modality Person Re-Identification**

cs.CV

9 pages,3 figures

**SubmitDate**: 2025-04-21    [abs](http://arxiv.org/abs/2501.12761v2) [paper-pdf](http://arxiv.org/pdf/2501.12761v2)

**Authors**: Yuan Bian, Min Liu, Yunqi Yi, Xueping Wang, Yunfeng Ma, Yaonan Wang

**Abstract**: Deep learning based person re-identification (re-id) models have been widely employed in surveillance systems. Recent studies have demonstrated that black-box single-modality and cross-modality re-id models are vulnerable to adversarial examples (AEs), leaving the robustness of multi-modality re-id models unexplored. Due to the lack of knowledge about the specific type of model deployed in the target black-box surveillance system, we aim to generate modality unified AEs for omni-modality (single-, cross- and multi-modality) re-id models. Specifically, we propose a novel Modality Unified Attack method to train modality-specific adversarial generators to generate AEs that effectively attack different omni-modality models. A multi-modality model is adopted as the surrogate model, wherein the features of each modality are perturbed by metric disruption loss before fusion. To collapse the common features of omni-modality models, Cross Modality Simulated Disruption approach is introduced to mimic the cross-modality feature embeddings by intentionally feeding images to non-corresponding modality-specific subnetworks of the surrogate model. Moreover, Multi Modality Collaborative Disruption strategy is devised to facilitate the attacker to comprehensively corrupt the informative content of person images by leveraging a multi modality feature collaborative metric disruption loss. Extensive experiments show that our MUA method can effectively attack the omni-modality re-id models, achieving 55.9%, 24.4%, 49.0% and 62.7% mean mAP Drop Rate, respectively.



## **48. LookAhead: Preventing DeFi Attacks via Unveiling Adversarial Contracts**

cs.CR

23 pages, 7 figures; Accepted to FSE 2025, to be published in The  Proceedings of the ACM on Software Engineering (PACMSE)

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2401.07261v6) [paper-pdf](http://arxiv.org/pdf/2401.07261v6)

**Authors**: Shoupeng Ren, Lipeng He, Tianyu Tu, Di Wu, Jian Liu, Kui Ren, Chun Chen

**Abstract**: The exploitation of smart contract vulnerabilities in Decentralized Finance (DeFi) has resulted in financial losses exceeding 3 billion US dollars. Existing defense mechanisms primarily focus on detecting and reacting to adversarial transactions executed by attackers that target victim contracts. However, with the emergence of private transaction pools where transactions are sent directly to miners without first appearing in public mempools, current detection tools face significant challenges in identifying attack activities effectively. Based on the fact that most attack logic rely on deploying intermediate smart contracts as supporting components to the exploitation of victim contracts, novel detection methods have been proposed that focus on identifying these adversarial contracts instead of adversarial transactions. However, previous state-of-the-art approaches in this direction have failed to produce results satisfactory enough for real-world deployment. In this paper, we propose LookAhead, a new framework for detecting DeFi attacks via unveiling adversarial contracts. LookAhead leverages common attack patterns, code semantics and intrinsic characteristics found in adversarial smart contracts to train Machine Learning (ML)-based classifiers that can effectively distinguish adversarial contracts from benign ones and make timely predictions of different types of potential attacks. Experiments on our labeled datasets show that LookAhead achieves an F1-score as high as 0.8966, which represents an improvement of over 44.4% compared to the previous state-of-the-art solution, with a False Positive Rate (FPR) at only 0.16%.



## **49. Human-AI Collaboration in Cloud Security: Cognitive Hierarchy-Driven Deep Reinforcement Learning**

cs.CR

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2502.16054v2) [paper-pdf](http://arxiv.org/pdf/2502.16054v2)

**Authors**: Zahra Aref, Sheng Wei, Narayan B. Mandayam

**Abstract**: Given the complexity of multi-tenant cloud environments and the growing need for real-time threat mitigation, Security Operations Centers (SOCs) must adopt AI-driven adaptive defense mechanisms to counter Advanced Persistent Threats (APTs). However, SOC analysts face challenges in handling adaptive adversarial tactics, requiring intelligent decision-support frameworks. We propose a Cognitive Hierarchy Theory-driven Deep Q-Network (CHT-DQN) framework that models interactive decision-making between SOC analysts and AI-driven APT bots. The SOC analyst (defender) operates at cognitive level-1, anticipating attacker strategies, while the APT bot (attacker) follows a level-0 policy. By incorporating CHT into DQN, our framework enhances adaptive SOC defense using Attack Graph (AG)-based reinforcement learning. Simulation experiments across varying AG complexities show that CHT-DQN consistently achieves higher data protection and lower action discrepancies compared to standard DQN. A theoretical lower bound further confirms its superiority as AG complexity increases. A human-in-the-loop (HITL) evaluation on Amazon Mechanical Turk (MTurk) reveals that SOC analysts using CHT-DQN-derived transition probabilities align more closely with adaptive attackers, leading to better defense outcomes. Moreover, human behavior aligns with Prospect Theory (PT) and Cumulative Prospect Theory (CPT): participants are less likely to reselect failed actions and more likely to persist with successful ones. This asymmetry reflects amplified loss sensitivity and biased probability weighting -- underestimating gains after failure and overestimating continued success. Our findings highlight the potential of integrating cognitive models into deep reinforcement learning to improve real-time SOC decision-making for cloud security.



## **50. Large Language Models as Robust Data Generators in Software Analytics: Are We There Yet?**

cs.SE

**SubmitDate**: 2025-04-20    [abs](http://arxiv.org/abs/2411.10565v2) [paper-pdf](http://arxiv.org/pdf/2411.10565v2)

**Authors**: Md. Abdul Awal, Mrigank Rochan, Chanchal K. Roy

**Abstract**: Large Language Model (LLM)-generated data is increasingly used in software analytics, but it is unclear how this data compares to human-written data, particularly when models are exposed to adversarial scenarios. Adversarial attacks can compromise the reliability and security of software systems, so understanding how LLM-generated data performs under these conditions, compared to human-written data, which serves as the benchmark for model performance, can provide valuable insights into whether LLM-generated data offers similar robustness and effectiveness. To address this gap, we systematically evaluate and compare the quality of human-written and LLM-generated data for fine-tuning robust pre-trained models (PTMs) in the context of adversarial attacks. We evaluate the robustness of six widely used PTMs, fine-tuned on human-written and LLM-generated data, before and after adversarial attacks. This evaluation employs nine state-of-the-art (SOTA) adversarial attack techniques across three popular software analytics tasks: clone detection, code summarization, and sentiment analysis in code review discussions. Additionally, we analyze the quality of the generated adversarial examples using eleven similarity metrics. Our findings reveal that while PTMs fine-tuned on LLM-generated data perform competitively with those fine-tuned on human-written data, they exhibit less robustness against adversarial attacks in software analytics tasks. Our study underscores the need for further exploration into enhancing the quality of LLM-generated training data to develop models that are both high-performing and capable of withstanding adversarial attacks in software analytics.



