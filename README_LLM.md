# Latest Large Language Model Attack Papers
**update at 2025-08-04 09:17:14**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. LeakSealer: A Semisupervised Defense for LLMs Against Prompt Injection and Leakage Attacks**

cs.CR

22 pages, preprint

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2508.00602v1) [paper-pdf](http://arxiv.org/pdf/2508.00602v1)

**Authors**: Francesco Panebianco, Stefano Bonfanti, Francesco Trovò, Michele Carminati

**Abstract**: The generalization capabilities of Large Language Models (LLMs) have led to their widespread deployment across various applications. However, this increased adoption has introduced several security threats, notably in the forms of jailbreaking and data leakage attacks. Additionally, Retrieval Augmented Generation (RAG), while enhancing context-awareness in LLM responses, has inadvertently introduced vulnerabilities that can result in the leakage of sensitive information. Our contributions are twofold. First, we introduce a methodology to analyze historical interaction data from an LLM system, enabling the generation of usage maps categorized by topics (including adversarial interactions). This approach further provides forensic insights for tracking the evolution of jailbreaking attack patterns. Second, we propose LeakSealer, a model-agnostic framework that combines static analysis for forensic insights with dynamic defenses in a Human-In-The-Loop (HITL) pipeline. This technique identifies topic groups and detects anomalous patterns, allowing for proactive defense mechanisms. We empirically evaluate LeakSealer under two scenarios: (1) jailbreak attempts, employing a public benchmark dataset, and (2) PII leakage, supported by a curated dataset of labeled LLM interactions. In the static setting, LeakSealer achieves the highest precision and recall on the ToxicChat dataset when identifying prompt injection. In the dynamic setting, PII leakage detection achieves an AUPRC of $0.97$, significantly outperforming baselines such as Llama Guard.



## **2. From LLMs to MLLMs to Agents: A Survey of Emerging Paradigms in Jailbreak Attacks and Defenses within LLM Ecosystem**

cs.CR

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2506.15170v3) [paper-pdf](http://arxiv.org/pdf/2506.15170v3)

**Authors**: Yanxu Mao, Tiehan Cui, Peipei Liu, Datao You, Hongsong Zhu

**Abstract**: Large language models (LLMs) are rapidly evolving from single-modal systems to multimodal LLMs and intelligent agents, significantly expanding their capabilities while introducing increasingly severe security risks. This paper presents a systematic survey of the growing complexity of jailbreak attacks and corresponding defense mechanisms within the expanding LLM ecosystem. We first trace the developmental trajectory from LLMs to MLLMs and Agents, highlighting the core security challenges emerging at each stage. Next, we categorize mainstream jailbreak techniques from both the attack impact and visibility perspectives, and provide a comprehensive analysis of representative attack methods, related datasets, and evaluation metrics. On the defense side, we organize existing strategies based on response timing and technical approach, offering a structured understanding of their applicability and implementation. Furthermore, we identify key limitations in existing surveys, such as insufficient attention to agent-specific security issues, the absence of a clear taxonomy for hybrid jailbreak methods, a lack of detailed analysis of experimental setups, and outdated coverage of recent advancements. To address these limitations, we provide an updated synthesis of recent work and outline future research directions in areas such as dataset construction, evaluation framework optimization, and strategy generalization. Our study seeks to enhance the understanding of jailbreak mechanisms and facilitate the advancement of more resilient and adaptive defense strategies in the context of ever more capable LLMs.



## **3. CyGATE: Game-Theoretic Cyber Attack-Defense Engine for Patch Strategy Optimization**

cs.CR

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2508.00478v1) [paper-pdf](http://arxiv.org/pdf/2508.00478v1)

**Authors**: Yuning Jiang, Nay Oo, Qiaoran Meng, Lu Lin, Dusit Niyato, Zehui Xiong, Hoon Wei Lim, Biplab Sikdar

**Abstract**: Modern cyber attacks unfold through multiple stages, requiring defenders to dynamically prioritize mitigations under uncertainty. While game-theoretic models capture attacker-defender interactions, existing approaches often rely on static assumptions and lack integration with real-time threat intelligence, limiting their adaptability. This paper presents CyGATE, a game-theoretic framework modeling attacker-defender interactions, using large language models (LLMs) with retrieval-augmented generation (RAG) to enhance tactic selection and patch prioritization. Applied to a two-agent scenario, CyGATE frames cyber conflicts as a partially observable stochastic game (POSG) across Cyber Kill Chain stages. Both agents use belief states to navigate uncertainty, with the attacker adapting tactics and the defender re-prioritizing patches based on evolving risks and observed adversary behavior. The framework's flexible architecture enables extension to multi-agent scenarios involving coordinated attackers, collaborative defenders, or complex enterprise environments with multiple stakeholders. Evaluated in a dynamic patch scheduling scenario, CyGATE effectively prioritizes high-risk vulnerabilities, enhancing adaptability through dynamic threat integration, strategic foresight by anticipating attacker moves under uncertainty, and efficiency by optimizing resource use.



## **4. Large AI Model-Enabled Secure Communications in Low-Altitude Wireless Networks: Concepts, Perspectives and Case Study**

cs.NI

This paper has been submitted to IEEE Communications Magazine for  consideration

**SubmitDate**: 2025-08-01    [abs](http://arxiv.org/abs/2508.00256v1) [paper-pdf](http://arxiv.org/pdf/2508.00256v1)

**Authors**: Chuang Zhang, Geng Sun, Jiacheng Wang, Yijing Lin, Weijie Yuan, Sinem Coleri, Dusit Niyato, Tony Q. S. Quek

**Abstract**: Low-altitude wireless networks (LAWNs) have the potential to revolutionize communications by supporting a range of applications, including urban parcel delivery, aerial inspections and air taxis. However, compared with traditional wireless networks, LAWNs face unique security challenges due to low-altitude operations, frequent mobility and reliance on unlicensed spectrum, making it more vulnerable to some malicious attacks. In this paper, we investigate some large artificial intelligence model (LAM)-enabled solutions for secure communications in LAWNs. Specifically, we first explore the amplified security risks and important limitations of traditional AI methods in LAWNs. Then, we introduce the basic concepts of LAMs and delve into the role of LAMs in addressing these challenges. To demonstrate the practical benefits of LAMs for secure communications in LAWNs, we propose a novel LAM-based optimization framework that leverages large language models (LLMs) to generate enhanced state features on top of handcrafted representations, and to design intrinsic rewards accordingly, thereby improving reinforcement learning performance for secure communication tasks. Through a typical case study, simulation results validate the effectiveness of the proposed framework. Finally, we outline future directions for integrating LAMs into secure LAWN applications.



## **5. Watch the Weights: Unsupervised monitoring and control of fine-tuned LLMs**

cs.LG

**SubmitDate**: 2025-07-31    [abs](http://arxiv.org/abs/2508.00161v1) [paper-pdf](http://arxiv.org/pdf/2508.00161v1)

**Authors**: Ziqian Zhong, Aditi Raghunathan

**Abstract**: The releases of powerful open-weight large language models (LLMs) are often not accompanied by access to their full training data. Existing interpretability methods, particularly those based on activations, often require or assume distributionally similar data. This is a significant limitation when detecting and defending against novel potential threats like backdoors, which are by definition out-of-distribution.   In this work, we introduce a new method for understanding, monitoring and controlling fine-tuned LLMs that interprets weights, rather than activations, thereby side stepping the need for data that is distributionally similar to the unknown training data. We demonstrate that the top singular vectors of the weight difference between a fine-tuned model and its base model correspond to newly acquired behaviors. By monitoring the cosine similarity of activations along these directions, we can detect salient behaviors introduced during fine-tuning with high precision.   For backdoored models that bypasses safety mechanisms when a secret trigger is present, our method stops up to 100% of attacks with a false positive rate below 1.2%. For models that have undergone unlearning, we detect inference on erased topics with accuracy up to 95.42% and can even steer the model to recover "unlearned" information. Besides monitoring, our method also shows potential for pre-deployment model auditing: by analyzing commercial instruction-tuned models (OLMo, Llama, Qwen), we are able to uncover model-specific fine-tuning focus including marketing strategies and Midjourney prompt generation.   Our implementation can be found at https://github.com/fjzzq2002/WeightWatch.



## **6. Graph Representation-based Model Poisoning on Federated Large Language Models**

cs.CR

7 pages, 5 figures (Submitted to IEEE Communication Magazine)

**SubmitDate**: 2025-07-31    [abs](http://arxiv.org/abs/2507.01694v2) [paper-pdf](http://arxiv.org/pdf/2507.01694v2)

**Authors**: Hanlin Cai, Haofan Dong, Houtianfu Wang, Kai Li, Ozgur B. Akan

**Abstract**: Federated large language models (FedLLMs) enable powerful generative capabilities within wireless networks while preserving data privacy. Nonetheless, FedLLMs remain vulnerable to model poisoning attacks. This article first reviews recent advancements in model poisoning techniques and existing defense mechanisms for FedLLMs, underscoring critical limitations, especially when dealing with non-IID textual data distributions. Current defense strategies predominantly employ distance or similarity-based outlier detection mechanisms, relying on the assumption that malicious updates markedly differ from benign statistical patterns. However, this assumption becomes inadequate against adaptive adversaries targeting billion-parameter LLMs. The article further investigates graph representation-based model poisoning (GRMP), an emerging attack paradigm that exploits higher-order correlations among benign client gradients to craft malicious updates indistinguishable from legitimate ones. GRMP can effectively circumvent advanced defense systems, causing substantial degradation in model accuracy and overall performance. Moreover, the article outlines a forward-looking research roadmap that emphasizes the necessity of graph-aware secure aggregation methods, specialized vulnerability metrics tailored for FedLLMs, and evaluation frameworks to enhance the robustness of federated language model deployments.



## **7. Probabilistic Modeling of Jailbreak on Multimodal LLMs: From Quantification to Application**

cs.CR

**SubmitDate**: 2025-07-31    [abs](http://arxiv.org/abs/2503.06989v2) [paper-pdf](http://arxiv.org/pdf/2503.06989v2)

**Authors**: Wenzhuo Xu, Zhipeng Wei, Xiongtao Sun, Zonghao Ying, Deyue Zhang, Dongdong Yang, Xiangzheng Zhang, Quanchen Zou

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have demonstrated their superior ability in understanding multimodal content. However, they remain vulnerable to jailbreak attacks, which exploit weaknesses in their safety alignment to generate harmful responses. Previous studies categorize jailbreaks as successful or failed based on whether responses contain malicious content. However, given the stochastic nature of MLLM responses, this binary classification of an input's ability to jailbreak MLLMs is inappropriate. Derived from this viewpoint, we introduce jailbreak probability to quantify the jailbreak potential of an input, which represents the likelihood that MLLMs generated a malicious response when prompted with this input. We approximate this probability through multiple queries to MLLMs. After modeling the relationship between input hidden states and their corresponding jailbreak probability using Jailbreak Probability Prediction Network (JPPN), we use continuous jailbreak probability for optimization. Specifically, we propose Jailbreak-Probability-based Attack (JPA) that optimizes adversarial perturbations on input image to maximize jailbreak probability, and further enhance it as Multimodal JPA (MJPA) by including monotonic text rephrasing. To counteract attacks, we also propose Jailbreak-Probability-based Finetuning (JPF), which minimizes jailbreak probability through MLLM parameter updates. Extensive experiments show that (1) (M)JPA yields significant improvements when attacking a wide range of models under both white and black box settings. (2) JPF vastly reduces jailbreaks by at most over 60\%. Both of the above results demonstrate the significance of introducing jailbreak probability to make nuanced distinctions among input jailbreak abilities.



## **8. CEE: An Inference-Time Jailbreak Defense for Embodied Intelligence via Subspace Concept Rotation**

cs.CR

**SubmitDate**: 2025-07-31    [abs](http://arxiv.org/abs/2504.13201v2) [paper-pdf](http://arxiv.org/pdf/2504.13201v2)

**Authors**: Jirui Yang, Zheyu Lin, Zhihui Lu, Yinggui Wang, Lei Wang, Tao Wei, Xin Du, Shuhan Yang

**Abstract**: Large Language Models (LLMs) are increasingly becoming the cognitive core of Embodied Intelligence (EI) systems, such as robots and autonomous vehicles. However, this integration also exposes them to serious jailbreak risks, where malicious instructions can be transformed into dangerous physical actions. Existing defense mechanisms suffer from notable drawbacks--including high training costs, significant inference delays, and complex hyperparameter tuning--which limit their practical applicability. To address these challenges, we propose a novel and efficient inference-time defense framework: Concept Enhancement Engineering (CEE). CEE enhances the model's inherent safety mechanisms by directly manipulating its internal representations, requiring neither additional training nor external modules, thereby improving defense efficiency. Furthermore, CEE introduces a rotation-based control mechanism that enables stable and linearly tunable behavioral control of the model. This design eliminates the need for tedious manual tuning and avoids the output degradation issues commonly observed in other representation engineering methods. Extensive experiments across multiple EI safety benchmarks and diverse attack scenarios demonstrate that CEE significantly improves the defense success rates of various multimodal LLMs. It effectively mitigates safety risks while preserving high-quality generation and inference efficiency, offering a promising solution for deploying safer embodied intelligence systems.



## **9. Fine-Grained Privacy Extraction from Retrieval-Augmented Generation Systems via Knowledge Asymmetry Exploitation**

cs.CR

**SubmitDate**: 2025-07-31    [abs](http://arxiv.org/abs/2507.23229v1) [paper-pdf](http://arxiv.org/pdf/2507.23229v1)

**Authors**: Yufei Chen, Yao Wang, Haibin Zhang, Tao Gu

**Abstract**: Retrieval-augmented generation (RAG) systems enhance large language models (LLMs) by integrating external knowledge bases, but this advancement introduces significant privacy risks. Existing privacy attacks on RAG systems can trigger data leakage but often fail to accurately isolate knowledge-base-derived sentences within mixed responses. They also lack robustness when applied across multiple domains. This paper addresses these challenges by presenting a novel black-box attack framework that exploits knowledge asymmetry between RAG and standard LLMs to achieve fine-grained privacy extraction across heterogeneous knowledge landscapes. We propose a chain-of-thought reasoning strategy that creates adaptive prompts to steer RAG systems away from sensitive content. Specifically, we first decompose adversarial queries to maximize information disparity and then apply a semantic relationship scoring to resolve lexical and syntactic ambiguities. We finally train a neural network on these feature scores to precisely identify sentences containing private information. Unlike prior work, our framework generalizes to unseen domains through iterative refinement without pre-defined knowledge. Experimental results show that we achieve over 91% privacy extraction rate in single-domain and 83% in multi-domain scenarios, reducing sensitive sentence exposure by over 65% in case studies. This work bridges the gap between attack and defense in RAG systems, enabling precise extraction of private information while providing a foundation for adaptive mitigation.



## **10. Adversarial-Guided Diffusion for Multimodal LLM Attacks**

cs.CV

**SubmitDate**: 2025-07-31    [abs](http://arxiv.org/abs/2507.23202v1) [paper-pdf](http://arxiv.org/pdf/2507.23202v1)

**Authors**: Chengwei Xia, Fan Ma, Ruijie Quan, Kun Zhan, Yi Yang

**Abstract**: This paper addresses the challenge of generating adversarial image using a diffusion model to deceive multimodal large language models (MLLMs) into generating the targeted responses, while avoiding significant distortion of the clean image. To address the above challenges, we propose an adversarial-guided diffusion (AGD) approach for adversarial attack MLLMs. We introduce adversarial-guided noise to ensure attack efficacy. A key observation in our design is that, unlike most traditional adversarial attacks which embed high-frequency perturbations directly into the clean image, AGD injects target semantics into the noise component of the reverse diffusion. Since the added noise in a diffusion model spans the entire frequency spectrum, the adversarial signal embedded within it also inherits this full-spectrum property. Importantly, during reverse diffusion, the adversarial image is formed as a linear combination of the clean image and the noise. Thus, when applying defenses such as a simple low-pass filtering, which act independently on each component, the adversarial image within the noise component is less likely to be suppressed, as it is not confined to the high-frequency band. This makes AGD inherently robust to variety defenses. Extensive experiments demonstrate that our AGD outperforms state-of-the-art methods in attack performance as well as in model robustness to some defenses.



## **11. Don't Lag, RAG: Training-Free Adversarial Detection Using RAG**

cs.AI

Accepted at VecDB @ ICML 2025

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2504.04858v3) [paper-pdf](http://arxiv.org/pdf/2504.04858v3)

**Authors**: Roie Kazoom, Raz Lapid, Moshe Sipper, Ofer Hadar

**Abstract**: Adversarial patch attacks pose a major threat to vision systems by embedding localized perturbations that mislead deep models. Traditional defense methods often require retraining or fine-tuning, making them impractical for real-world deployment. We propose a training-free Visual Retrieval-Augmented Generation (VRAG) framework that integrates Vision-Language Models (VLMs) for adversarial patch detection. By retrieving visually similar patches and images that resemble stored attacks in a continuously expanding database, VRAG performs generative reasoning to identify diverse attack types, all without additional training or fine-tuning. We extensively evaluate open-source large-scale VLMs, including Qwen-VL-Plus, Qwen2.5-VL-72B, and UI-TARS-72B-DPO, alongside Gemini-2.0, a closed-source model. Notably, the open-source UI-TARS-72B-DPO model achieves up to 95 percent classification accuracy, setting a new state-of-the-art for open-source adversarial patch detection. Gemini-2.0 attains the highest overall accuracy, 98 percent, but remains closed-source. Experimental results demonstrate VRAG's effectiveness in identifying a variety of adversarial patches with minimal human annotation, paving the way for robust, practical defenses against evolving adversarial patch attacks.



## **12. Efficient Differentially Private Fine-Tuning of LLMs via Reinforcement Learning**

cs.LG

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22565v1) [paper-pdf](http://arxiv.org/pdf/2507.22565v1)

**Authors**: Afshin Khadangi, Amir Sartipi, Igor Tchappi, Ramin Bahmani, Gilbert Fridgen

**Abstract**: The tension between data privacy and model utility has become the defining bottleneck for the practical deployment of large language models (LLMs) trained on sensitive corpora including healthcare. Differentially private stochastic gradient descent (DP-SGD) guarantees formal privacy, yet it does so at a pronounced cost: gradients are forcibly clipped and perturbed with noise, degrading sample efficiency and final accuracy. Numerous variants have been proposed to soften this trade-off, but they all share a handicap: their control knobs are hard-coded, global, and oblivious to the evolving optimization landscape. Consequently, practitioners are forced either to over-spend privacy budget in pursuit of utility, or to accept mediocre models in order to stay within privacy constraints. We present RLDP, the first framework to cast DP optimization itself as a closed-loop control problem amenable to modern deep reinforcement learning (RL). RLDP continuously senses rich statistics of the learning dynamics and acts by selecting fine-grained per parameter gradient-clipping thresholds as well as the magnitude of injected Gaussian noise. A soft actor-critic (SAC) hyper-policy is trained online during language model fine-tuning; it learns, from scratch, how to allocate the privacy budget where it matters and when it matters. Across more than 1,600 ablation experiments on GPT2-small, Llama-1B, Llama-3B, and Mistral-7B, RLDP delivers perplexity reductions of 1.3-30.5% (mean 5.4%) and an average 5.6% downstream utility gain. RLDP reaches each baseline's final utility after only 13-43% of the gradient-update budget (mean speed-up 71%), all while honoring the same ($\epsilon$, $\delta$)-DP contract and exhibiting equal or lower susceptibility to membership-inference and canary-extraction attacks.



## **13. Exploiting Synergistic Cognitive Biases to Bypass Safety in LLMs**

cs.CL

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22564v1) [paper-pdf](http://arxiv.org/pdf/2507.22564v1)

**Authors**: Xikang Yang, Biyu Zhou, Xuehai Tang, Jizhong Han, Songlin Hu

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities across a wide range of tasks, yet their safety mechanisms remain susceptible to adversarial attacks that exploit cognitive biases -- systematic deviations from rational judgment. Unlike prior jailbreaking approaches focused on prompt engineering or algorithmic manipulation, this work highlights the overlooked power of multi-bias interactions in undermining LLM safeguards. We propose CognitiveAttack, a novel red-teaming framework that systematically leverages both individual and combined cognitive biases. By integrating supervised fine-tuning and reinforcement learning, CognitiveAttack generates prompts that embed optimized bias combinations, effectively bypassing safety protocols while maintaining high attack success rates. Experimental results reveal significant vulnerabilities across 30 diverse LLMs, particularly in open-source models. CognitiveAttack achieves a substantially higher attack success rate compared to the SOTA black-box method PAP (60.1% vs. 31.6%), exposing critical limitations in current defense mechanisms. These findings highlight multi-bias interactions as a powerful yet underexplored attack vector. This work introduces a novel interdisciplinary perspective by bridging cognitive science and LLM safety, paving the way for more robust and human-aligned AI systems.



## **14. ETrace:Event-Driven Vulnerability Detection in Smart Contracts via LLM-Based Trace Analysis**

cs.CR

4 pages, 1 figure. To appear in Proceedings of the 16th Asia-Pacific  Symposium on Internetware (Internetware 2025), ACM ICPS. DOI:  https://doi.org/10.1145/3755881.3755934

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2506.15790v3) [paper-pdf](http://arxiv.org/pdf/2506.15790v3)

**Authors**: Chenyang Peng, Haijun Wang, Yin Wu, Hao Wu, Ming Fan, Yitao Zhao, Ting Liu

**Abstract**: With the advance application of blockchain technology in various fields, ensuring the security and stability of smart contracts has emerged as a critical challenge. Current security analysis methodologies in vulnerability detection can be categorized into static analysis and dynamic analysis methods.However, these existing traditional vulnerability detection methods predominantly rely on analyzing original contract code, not all smart contracts provide accessible code.We present ETrace, a novel event-driven vulnerability detection framework for smart contracts, which uniquely identifies potential vulnerabilities through LLM-powered trace analysis without requiring source code access. By extracting fine-grained event sequences from transaction logs, the framework leverages Large Language Models (LLMs) as adaptive semantic interpreters to reconstruct event analysis through chain-of-thought reasoning. ETrace implements pattern-matching to establish causal links between transaction behavior patterns and known attack behaviors. Furthermore, we validate the effectiveness of ETrace through preliminary experimental results.



## **15. Invisible Injections: Exploiting Vision-Language Models Through Steganographic Prompt Embedding**

cs.CR

14 Pages

**SubmitDate**: 2025-07-30    [abs](http://arxiv.org/abs/2507.22304v1) [paper-pdf](http://arxiv.org/pdf/2507.22304v1)

**Authors**: Chetan Pathade

**Abstract**: Vision-language models (VLMs) have revolutionized multimodal AI applications but introduce novel security vulnerabilities that remain largely unexplored. We present the first comprehensive study of steganographic prompt injection attacks against VLMs, where malicious instructions are invisibly embedded within images using advanced steganographic techniques. Our approach demonstrates that current VLM architectures can inadvertently extract and execute hidden prompts during normal image processing, leading to covert behavioral manipulation. We develop a multi-domain embedding framework combining spatial, frequency, and neural steganographic methods, achieving an overall attack success rate of 24.3% (plus or minus 3.2%, 95% CI) across leading VLMs including GPT-4V, Claude, and LLaVA, with neural steganography methods reaching up to 31.8%, while maintaining reasonable visual imperceptibility (PSNR greater than 38 dB, SSIM greater than 0.94). Through systematic evaluation on 12 diverse datasets and 8 state-of-the-art models, we reveal moderate but meaningful vulnerabilities in current VLM architectures and propose effective countermeasures. Our findings have significant implications for VLM deployment in security-critical applications and highlight the need for proportionate multimodal AI security frameworks.



## **16. Large Language Model-Based Framework for Explainable Cyberattack Detection in Automatic Generation Control Systems**

cs.CR

Accepted Publication

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.22239v1) [paper-pdf](http://arxiv.org/pdf/2507.22239v1)

**Authors**: Muhammad Sharshar, Ahmad Mohammad Saber, Davor Svetinovic, Amr M. Youssef, Deepa Kundur, Ehab F. El-Saadany

**Abstract**: The increasing digitization of smart grids has improved operational efficiency but also introduced new cybersecurity vulnerabilities, such as False Data Injection Attacks (FDIAs) targeting Automatic Generation Control (AGC) systems. While machine learning (ML) and deep learning (DL) models have shown promise in detecting such attacks, their opaque decision-making limits operator trust and real-world applicability. This paper proposes a hybrid framework that integrates lightweight ML-based attack detection with natural language explanations generated by Large Language Models (LLMs). Classifiers such as LightGBM achieve up to 95.13% attack detection accuracy with only 0.004 s inference latency. Upon detecting a cyberattack, the system invokes LLMs, including GPT-3.5 Turbo, GPT-4 Turbo, and GPT-4o mini, to generate human-readable explanation of the event. Evaluated on 100 test samples, GPT-4o mini with 20-shot prompting achieved 93% accuracy in identifying the attack target, a mean absolute error of 0.075 pu in estimating attack magnitude, and 2.19 seconds mean absolute error (MAE) in estimating attack onset. These results demonstrate that the proposed framework effectively balances real-time detection with interpretable, high-fidelity explanations, addressing a critical need for actionable AI in smart grid cybersecurity.



## **17. Can adversarial attacks by large language models be attributed?**

cs.AI

22 pages, 5 figures, 2 tables

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2411.08003v3) [paper-pdf](http://arxiv.org/pdf/2411.08003v3)

**Authors**: Manuel Cebrian, Andres Abeliuk, Jan Arne Telle

**Abstract**: Attributing outputs from Large Language Models (LLMs) in adversarial settings-such as cyberattacks and disinformation campaigns-presents significant challenges that are likely to grow in importance. We approach this attribution problem from both a theoretical and an empirical perspective, drawing on formal language theory (identification in the limit) and data-driven analysis of the expanding LLM ecosystem. By modeling an LLM's set of possible outputs as a formal language, we analyze whether finite samples of text can uniquely pinpoint the originating model. Our results show that, under mild assumptions of overlapping capabilities among models, certain classes of LLMs are fundamentally non-identifiable from their outputs alone. We delineate four regimes of theoretical identifiability: (1) an infinite class of deterministic (discrete) LLM languages is not identifiable (Gold's classical result from 1967); (2) an infinite class of probabilistic LLMs is also not identifiable (by extension of the deterministic case); (3) a finite class of deterministic LLMs is identifiable (consistent with Angluin's tell-tale criterion); and (4) even a finite class of probabilistic LLMs can be non-identifiable (we provide a new counterexample establishing this negative result). Complementing these theoretical insights, we quantify the explosion in the number of plausible model origins (hypothesis space) for a given output in recent years. Even under conservative assumptions-each open-source model fine-tuned on at most one new dataset-the count of distinct candidate models doubles approximately every 0.5 years, and allowing multi-dataset fine-tuning combinations yields doubling times as short as 0.28 years. This combinatorial growth, alongside the extraordinary computational cost of brute-force likelihood attribution across all models and potential users, renders exhaustive attribution infeasible in practice.



## **18. Strategic Deflection: Defending LLMs from Logit Manipulation**

cs.CR

20 pages

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.22160v1) [paper-pdf](http://arxiv.org/pdf/2507.22160v1)

**Authors**: Yassine Rachidy, Jihad Rbaiti, Youssef Hmamouche, Faissal Sehbaoui, Amal El Fallah Seghrouchni

**Abstract**: With the growing adoption of Large Language Models (LLMs) in critical areas, ensuring their security against jailbreaking attacks is paramount. While traditional defenses primarily rely on refusing malicious prompts, recent logit-level attacks have demonstrated the ability to bypass these safeguards by directly manipulating the token-selection process during generation. We introduce Strategic Deflection (SDeflection), a defense that redefines the LLM's response to such advanced attacks. Instead of outright refusal, the model produces an answer that is semantically adjacent to the user's request yet strips away the harmful intent, thereby neutralizing the attacker's harmful intent. Our experiments demonstrate that SDeflection significantly lowers Attack Success Rate (ASR) while maintaining model performance on benign queries. This work presents a critical shift in defensive strategies, moving from simple refusal to strategic content redirection to neutralize advanced threats.



## **19. Prompt Optimization and Evaluation for LLM Automated Red Teaming**

cs.CR

9 pages, 5 Figures, and 1 Appendix item

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.22133v1) [paper-pdf](http://arxiv.org/pdf/2507.22133v1)

**Authors**: Michael Freenor, Lauren Alvarez, Milton Leal, Lily Smith, Joel Garrett, Yelyzaveta Husieva, Madeline Woodruff, Ryan Miller, Erich Kummerfeld, Rafael Medeiros, Sander Schulhoff

**Abstract**: Applications that use Large Language Models (LLMs) are becoming widespread, making the identification of system vulnerabilities increasingly important. Automated Red Teaming accelerates this effort by using an LLM to generate and execute attacks against target systems. Attack generators are evaluated using the Attack Success Rate (ASR) the sample mean calculated over the judgment of success for each attack. In this paper, we introduce a method for optimizing attack generator prompts that applies ASR to individual attacks. By repeating each attack multiple times against a randomly seeded target, we measure an attack's discoverability the expectation of the individual attack success. This approach reveals exploitable patterns that inform prompt optimization, ultimately enabling more robust evaluation and refinement of generators.



## **20. Secure Tug-of-War (SecTOW): Iterative Defense-Attack Training with Reinforcement Learning for Multimodal Model Security**

cs.CR

10 pages, 4 figures

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.22037v1) [paper-pdf](http://arxiv.org/pdf/2507.22037v1)

**Authors**: Muzhi Dai, Shixuan Liu, Zhiyuan Zhao, Junyu Gao, Hao Sun, Xuelong Li

**Abstract**: The rapid advancement of multimodal large language models (MLLMs) has led to breakthroughs in various applications, yet their security remains a critical challenge. One pressing issue involves unsafe image-query pairs--jailbreak inputs specifically designed to bypass security constraints and elicit unintended responses from MLLMs. Compared to general multimodal data, such unsafe inputs are relatively sparse, which limits the diversity and richness of training samples available for developing robust defense models. Meanwhile, existing guardrail-type methods rely on external modules to enforce security constraints but fail to address intrinsic vulnerabilities within MLLMs. Traditional supervised fine-tuning (SFT), on the other hand, often over-refuses harmless inputs, compromising general performance. Given these challenges, we propose Secure Tug-of-War (SecTOW), an innovative iterative defense-attack training method to enhance the security of MLLMs. SecTOW consists of two modules: a defender and an auxiliary attacker, both trained iteratively using reinforcement learning (GRPO). During the iterative process, the attacker identifies security vulnerabilities in the defense model and expands jailbreak data. The expanded data are then used to train the defender, enabling it to address identified security vulnerabilities. We also design reward mechanisms used for GRPO to simplify the use of response labels, reducing dependence on complex generative labels and enabling the efficient use of synthetic data. Additionally, a quality monitoring mechanism is used to mitigate the defender's over-refusal of harmless inputs and ensure the diversity of the jailbreak data generated by the attacker. Experimental results on safety-specific and general benchmarks demonstrate that SecTOW significantly improves security while preserving general performance.



## **21. Anyone Can Jailbreak: Prompt-Based Attacks on LLMs and T2Is**

cs.CV

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21820v1) [paper-pdf](http://arxiv.org/pdf/2507.21820v1)

**Authors**: Ahmed B Mustafa, Zihan Ye, Yang Lu, Michael P Pound, Shreyank N Gowda

**Abstract**: Despite significant advancements in alignment and content moderation, large language models (LLMs) and text-to-image (T2I) systems remain vulnerable to prompt-based attacks known as jailbreaks. Unlike traditional adversarial examples requiring expert knowledge, many of today's jailbreaks are low-effort, high-impact crafted by everyday users with nothing more than cleverly worded prompts. This paper presents a systems-style investigation into how non-experts reliably circumvent safety mechanisms through techniques such as multi-turn narrative escalation, lexical camouflage, implication chaining, fictional impersonation, and subtle semantic edits. We propose a unified taxonomy of prompt-level jailbreak strategies spanning both text-output and T2I models, grounded in empirical case studies across popular APIs. Our analysis reveals that every stage of the moderation pipeline, from input filtering to output validation, can be bypassed with accessible strategies. We conclude by highlighting the urgent need for context-aware defenses that reflect the ease with which these jailbreaks can be reproduced in real-world settings.



## **22. Latent Adversarial Training Improves Robustness to Persistent Harmful Behaviors in LLMs**

cs.LG

Code at https://github.com/aengusl/latent-adversarial-training.  Models at https://huggingface.co/LLM-LAT

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2407.15549v3) [paper-pdf](http://arxiv.org/pdf/2407.15549v3)

**Authors**: Abhay Sheshadri, Aidan Ewart, Phillip Guo, Aengus Lynch, Cindy Wu, Vivek Hebbar, Henry Sleight, Asa Cooper Stickland, Ethan Perez, Dylan Hadfield-Menell, Stephen Casper

**Abstract**: Large language models (LLMs) can often be made to behave in undesirable ways that they are explicitly fine-tuned not to. For example, the LLM red-teaming literature has produced a wide variety of 'jailbreaking' techniques to elicit harmful text from models that were fine-tuned to be harmless. Recent work on red-teaming, model editing, and interpretability suggests that this challenge stems from how (adversarial) fine-tuning largely serves to suppress rather than remove undesirable capabilities from LLMs. Prior work has introduced latent adversarial training (LAT) as a way to improve robustness to broad classes of failures. These prior works have considered untargeted latent space attacks where the adversary perturbs latent activations to maximize loss on examples of desirable behavior. Untargeted LAT can provide a generic type of robustness but does not leverage information about specific failure modes. Here, we experiment with targeted LAT where the adversary seeks to minimize loss on a specific competing task. We find that it can augment a wide variety of state-of-the-art methods. First, we use targeted LAT to improve robustness to jailbreaks, outperforming a strong R2D2 baseline with orders of magnitude less compute. Second, we use it to more effectively remove backdoors with no knowledge of the trigger. Finally, we use it to more effectively unlearn knowledge for specific undesirable tasks in a way that is also more robust to re-learning. Overall, our results suggest that targeted LAT can be an effective tool for defending against harmful behaviors from LLMs.



## **23. PRISM: Programmatic Reasoning with Image Sequence Manipulation for LVLM Jailbreaking**

cs.CR

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21540v1) [paper-pdf](http://arxiv.org/pdf/2507.21540v1)

**Authors**: Quanchen Zou, Zonghao Ying, Moyang Chen, Wenzhuo Xu, Yisong Xiao, Yakai Li, Deyue Zhang, Dongdong Yang, Zhao Liu, Xiangzheng Zhang

**Abstract**: The increasing sophistication of large vision-language models (LVLMs) has been accompanied by advances in safety alignment mechanisms designed to prevent harmful content generation. However, these defenses remain vulnerable to sophisticated adversarial attacks. Existing jailbreak methods typically rely on direct and semantically explicit prompts, overlooking subtle vulnerabilities in how LVLMs compose information over multiple reasoning steps. In this paper, we propose a novel and effective jailbreak framework inspired by Return-Oriented Programming (ROP) techniques from software security. Our approach decomposes a harmful instruction into a sequence of individually benign visual gadgets. A carefully engineered textual prompt directs the sequence of inputs, prompting the model to integrate the benign visual gadgets through its reasoning process to produce a coherent and harmful output. This makes the malicious intent emergent and difficult to detect from any single component. We validate our method through extensive experiments on established benchmarks including SafeBench and MM-SafetyBench, targeting popular LVLMs. Results show that our approach consistently and substantially outperforms existing baselines on state-of-the-art models, achieving near-perfect attack success rates (over 0.90 on SafeBench) and improving ASR by up to 0.39. Our findings reveal a critical and underexplored vulnerability that exploits the compositional reasoning abilities of LVLMs, highlighting the urgent need for defenses that secure the entire reasoning process.



## **24. Can We End the Cat-and-Mouse Game? Simulating Self-Evolving Phishing Attacks with LLMs and Genetic Algorithms**

cs.CR

**SubmitDate**: 2025-07-29    [abs](http://arxiv.org/abs/2507.21538v1) [paper-pdf](http://arxiv.org/pdf/2507.21538v1)

**Authors**: Seiji Sato, Tetsushi Ohki, Masakatsu Nishigaki

**Abstract**: Anticipating emerging attack methodologies is crucial for proactive cybersecurity. Recent advances in Large Language Models (LLMs) have enabled the automated generation of phishing messages and accelerated research into potential attack techniques. However, predicting future threats remains challenging due to reliance on existing training data. To address this limitation, we propose a novel framework that integrates LLM-based phishing attack simulations with a genetic algorithm in a psychological context, enabling phishing strategies to evolve dynamically through adversarial interactions with simulated victims. Through simulations using Llama 3.1, we demonstrate that (1) self-evolving phishing strategies employ increasingly sophisticated psychological manipulation techniques, surpassing naive LLM-generated attacks, (2) variations in a victim's prior knowledge significantly influence the evolution of attack strategies, and (3) adversarial interactions between evolving attacks and adaptive defenses create a cat-and-mouse dynamic, revealing an inherent asymmetry in cybersecurity -- attackers continuously refine their methods, whereas defenders struggle to comprehensively counter all evolving threats. Our approach provides a scalable, cost-effective method for analyzing the evolution of phishing strategies and defenses, offering insights into future social engineering threats and underscoring the necessity of proactive cybersecurity measures.



## **25. Memorization in Fine-Tuned Large Language Models**

cs.CL

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.21009v1) [paper-pdf](http://arxiv.org/pdf/2507.21009v1)

**Authors**: Danil Savine, Muni Sreenivas Pydi, Jamal Atif, Olivier Cappé

**Abstract**: This study investigates the mechanisms and factors influencing memorization in fine-tuned large language models (LLMs), with a focus on the medical domain due to its privacy-sensitive nature. We examine how different aspects of the fine-tuning process affect a model's propensity to memorize training data, using the PHEE dataset of pharmacovigilance events.   Our research employs two main approaches: a membership inference attack to detect memorized data, and a generation task with prompted prefixes to assess verbatim reproduction. We analyze the impact of adapting different weight matrices in the transformer architecture, the relationship between perplexity and memorization, and the effect of increasing the rank in low-rank adaptation (LoRA) fine-tuning.   Key findings include: (1) Value and Output matrices contribute more significantly to memorization compared to Query and Key matrices; (2) Lower perplexity in the fine-tuned model correlates with increased memorization; (3) Higher LoRA ranks lead to increased memorization, but with diminishing returns at higher ranks.   These results provide insights into the trade-offs between model performance and privacy risks in fine-tuned LLMs. Our findings have implications for developing more effective and responsible strategies for adapting large language models while managing data privacy concerns.



## **26. A Large Language Model-Supported Threat Modeling Framework for Transportation Cyber-Physical Systems**

cs.CR

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2506.00831v2) [paper-pdf](http://arxiv.org/pdf/2506.00831v2)

**Authors**: M Sabbir Salek, Mashrur Chowdhury, Muhaimin Bin Munir, Yuchen Cai, Mohammad Imtiaz Hasan, Jean-Michel Tine, Latifur Khan, Mizanur Rahman

**Abstract**: Existing threat modeling frameworks related to transportation cyber-physical systems (CPS) are often narrow in scope, labor-intensive, and require substantial cybersecurity expertise. To this end, we introduce the Transportation Cybersecurity and Resiliency Threat Modeling Framework (TraCR-TMF), a large language model (LLM)-based threat modeling framework for transportation CPS that requires limited cybersecurity expert intervention. TraCR-TMF identifies threats, potential attack techniques, and relevant countermeasures for transportation CPS. Three LLM-based approaches support these identifications: (i) a retrieval-augmented generation approach requiring no cybersecurity expert intervention, (ii) an in-context learning approach with low expert intervention, and (iii) a supervised fine-tuning approach with moderate expert intervention. TraCR-TMF offers LLM-based attack path identification for critical assets based on vulnerabilities across transportation CPS entities. Additionally, it incorporates the Common Vulnerability Scoring System (CVSS) scores of known exploited vulnerabilities to prioritize threat mitigations. The framework was evaluated through two cases. First, the framework identified relevant attack techniques for various transportation CPS applications, 73% of which were validated by cybersecurity experts as correct. Second, the framework was used to identify attack paths for a target asset in a real-world cyberattack incident. TraCR-TMF successfully predicted exploitations, like lateral movement of adversaries, data exfiltration, and data encryption for ransomware, as reported in the incident. These findings show the efficacy of TraCR-TMF in transportation CPS threat modeling, while reducing the need for extensive involvement of cybersecurity experts. To facilitate real-world adoptions, all our codes are shared via an open-source repository.



## **27. Enhancing Jailbreak Attacks on LLMs via Persona Prompts**

cs.CR

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.22171v1) [paper-pdf](http://arxiv.org/pdf/2507.22171v1)

**Authors**: Zheng Zhang, Peilin Zhao, Deheng Ye, Hao Wang

**Abstract**: Jailbreak attacks aim to exploit large language models (LLMs) by inducing them to generate harmful content, thereby revealing their vulnerabilities. Understanding and addressing these attacks is crucial for advancing the field of LLM safety. Previous jailbreak approaches have mainly focused on direct manipulations of harmful intent, with limited attention to the impact of persona prompts. In this study, we systematically explore the efficacy of persona prompts in compromising LLM defenses. We propose a genetic algorithm-based method that automatically crafts persona prompts to bypass LLM's safety mechanisms. Our experiments reveal that: (1) our evolved persona prompts reduce refusal rates by 50-70% across multiple LLMs, and (2) these prompts demonstrate synergistic effects when combined with existing attack methods, increasing success rates by 10-20%. Our code and data are available at https://github.com/CjangCjengh/Generic_Persona.



## **28. Uncovering Gradient Inversion Risks in Practical Language Model Training**

cs.LG

15 Pages, 5 figures, 10 tables. Accepted by ACM CCS 2024

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.21198v1) [paper-pdf](http://arxiv.org/pdf/2507.21198v1)

**Authors**: Xinguo Feng, Zhongkui Ma, Zihan Wang, Eu Joe Chegne, Mengyao Ma, Alsharif Abuadbba, Guangdong Bai

**Abstract**: The gradient inversion attack has been demonstrated as a significant privacy threat to federated learning (FL), particularly in continuous domains such as vision models. In contrast, it is often considered less effective or highly dependent on impractical training settings when applied to language models, due to the challenges posed by the discrete nature of tokens in text data. As a result, its potential privacy threats remain largely underestimated, despite FL being an emerging training method for language models. In this work, we propose a domain-specific gradient inversion attack named Grab (gradient inversion with hybrid optimization). Grab features two alternating optimization processes to address the challenges caused by practical training settings, including a simultaneous optimization on dropout masks between layers for improved token recovery and a discrete optimization for effective token sequencing. Grab can recover a significant portion (up to 92.9% recovery rate) of the private training data, outperforming the attack strategy of utilizing discrete optimization with an auxiliary model by notable improvements of up to 28.9% recovery rate in benchmark settings and 48.5% recovery rate in practical settings. Grab provides a valuable step forward in understanding this privacy threat in the emerging FL training mode of language models.



## **29. Accidental Vulnerability: Factors in Fine-Tuning that Shift Model Safeguards**

cs.CL

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2505.16789v2) [paper-pdf](http://arxiv.org/pdf/2505.16789v2)

**Authors**: Punya Syon Pandey, Samuel Simko, Kellin Pelrine, Zhijing Jin

**Abstract**: As large language models (LLMs) gain popularity, their vulnerability to adversarial attacks emerges as a primary concern. While fine-tuning models on domain-specific datasets is often employed to improve model performance, it can inadvertently introduce vulnerabilities within the underlying model. In this work, we investigate Accidental Vulnerability, unexpected vulnerabilities arising from characteristics of fine-tuning data. We begin by identifying potential correlation factors such as linguistic features, semantic similarity, and toxicity across multiple experimental datasets. We then evaluate the adversarial robustness of these fine-tuned models, analyzing persona shifts and interpretability traits to understand how dataset factors contribute to attack success rates. Lastly, we explore causal relationships that offer new insights into adversarial defense strategies, highlighting the crucial role of dataset design in preserving model alignment. Our code is available at https://github.com/psyonp/accidental_vulnerability.



## **30. Security Challenges in AI Agent Deployment: Insights from a Large Scale Public Competition**

cs.AI

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2507.20526v1) [paper-pdf](http://arxiv.org/pdf/2507.20526v1)

**Authors**: Andy Zou, Maxwell Lin, Eliot Jones, Micha Nowak, Mateusz Dziemian, Nick Winter, Alexander Grattan, Valent Nathanael, Ayla Croft, Xander Davies, Jai Patel, Robert Kirk, Nate Burnikell, Yarin Gal, Dan Hendrycks, J. Zico Kolter, Matt Fredrikson

**Abstract**: Recent advances have enabled LLM-powered AI agents to autonomously execute complex tasks by combining language model reasoning with tools, memory, and web access. But can these systems be trusted to follow deployment policies in realistic environments, especially under attack? To investigate, we ran the largest public red-teaming competition to date, targeting 22 frontier AI agents across 44 realistic deployment scenarios. Participants submitted 1.8 million prompt-injection attacks, with over 60,000 successfully eliciting policy violations such as unauthorized data access, illicit financial actions, and regulatory noncompliance. We use these results to build the Agent Red Teaming (ART) benchmark - a curated set of high-impact attacks - and evaluate it across 19 state-of-the-art models. Nearly all agents exhibit policy violations for most behaviors within 10-100 queries, with high attack transferability across models and tasks. Importantly, we find limited correlation between agent robustness and model size, capability, or inference-time compute, suggesting that additional defenses are needed against adversarial misuse. Our findings highlight critical and persistent vulnerabilities in today's AI agents. By releasing the ART benchmark and accompanying evaluation framework, we aim to support more rigorous security assessment and drive progress toward safer agent deployment.



## **31. More is Less: The Pitfalls of Multi-Model Synthetic Preference Data in DPO Safety Alignment**

cs.AI

This version includes updated results and expanded discussion

**SubmitDate**: 2025-07-28    [abs](http://arxiv.org/abs/2504.02193v3) [paper-pdf](http://arxiv.org/pdf/2504.02193v3)

**Authors**: Yifan Wang, Runjin Chen, Bolian Li, David Cho, Yihe Deng, Ruqi Zhang, Tianlong Chen, Zhangyang Wang, Ananth Grama, Junyuan Hong

**Abstract**: Aligning large language models (LLMs) with human values is an increasingly critical step in post-training. Direct Preference Optimization (DPO) has emerged as a simple, yet effective alternative to reinforcement learning from human feedback (RLHF). Synthetic preference data with its low cost and high quality enable effective alignment through single- or multi-model generated preference data. Our study reveals a striking, safety-specific phenomenon associated with DPO alignment: Although multi-model generated data enhances performance on general tasks (ARC, Hellaswag, MMLU, TruthfulQA, Winogrande) by providing diverse responses, it also tends to facilitate reward hacking during training. This can lead to a high attack success rate (ASR) when models encounter jailbreaking prompts. The issue is particularly pronounced when employing stronger models like GPT-4o or larger models in the same family to generate chosen responses paired with target model self-generated rejected responses, resulting in dramatically poorer safety outcomes. Furthermore, with respect to safety, using solely self-generated responses (single-model generation) for both chosen and rejected pairs significantly outperforms configurations that incorporate responses from stronger models, whether used directly as chosen data or as part of a multi-model response pool. We demonstrate that multi-model preference data exhibits high linear separability between chosen and rejected responses, which allows models to exploit superficial cues rather than internalizing robust safety constraints. Our experiments, conducted on models from the Llama, Mistral, and Qwen families, consistently validate these findings.



## **32. Interpretable Anomaly-Based DDoS Detection in AI-RAN with XAI and LLMs**

cs.CR

**SubmitDate**: 2025-07-27    [abs](http://arxiv.org/abs/2507.21193v1) [paper-pdf](http://arxiv.org/pdf/2507.21193v1)

**Authors**: Sotiris Chatzimiltis, Mohammad Shojafar, Mahdi Boloursaz Mashhadi, Rahim Tafazolli

**Abstract**: Next generation Radio Access Networks (RANs) introduce programmability, intelligence, and near real-time control through intelligent controllers, enabling enhanced security within the RAN and across broader 5G/6G infrastructures. This paper presents a comprehensive survey highlighting opportunities, challenges, and research gaps for Large Language Models (LLMs)-assisted explainable (XAI) intrusion detection (IDS) for secure future RAN environments. Motivated by this, we propose an LLM interpretable anomaly-based detection system for distributed denial-of-service (DDoS) attacks using multivariate time series key performance measures (KPMs), extracted from E2 nodes, within the Near Real-Time RAN Intelligent Controller (Near-RT RIC). An LSTM-based model is trained to identify malicious User Equipment (UE) behavior based on these KPMs. To enhance transparency, we apply post-hoc local explainability methods such as LIME and SHAP to interpret individual predictions. Furthermore, LLMs are employed to convert technical explanations into natural-language insights accessible to non-expert users. Experimental results on real 5G network KPMs demonstrate that our framework achieves high detection accuracy (F1-score > 0.96) while delivering actionable and interpretable outputs.



## **33. Manipulating Multimodal Agents via Cross-Modal Prompt Injection**

cs.CV

16 pages, 5 figures

**SubmitDate**: 2025-07-27    [abs](http://arxiv.org/abs/2504.14348v4) [paper-pdf](http://arxiv.org/pdf/2504.14348v4)

**Authors**: Le Wang, Zonghao Ying, Tianyuan Zhang, Siyuan Liang, Shengshan Hu, Mingchuan Zhang, Aishan Liu, Xianglong Liu

**Abstract**: The emergence of multimodal large language models has redefined the agent paradigm by integrating language and vision modalities with external data sources, enabling agents to better interpret human instructions and execute increasingly complex tasks. However, in this paper, we identify a critical yet previously overlooked security vulnerability in multimodal agents: cross-modal prompt injection attacks. To exploit this vulnerability, we propose CrossInject, a novel attack framework in which attackers embed adversarial perturbations across multiple modalities to align with target malicious content, allowing external instructions to hijack the agent's decision-making process and execute unauthorized tasks. Our approach incorporates two key coordinated components. First, we introduce Visual Latent Alignment, where we optimize adversarial features to the malicious instructions in the visual embedding space based on a text-to-image generative model, ensuring that adversarial images subtly encode cues for malicious task execution. Subsequently, we present Textual Guidance Enhancement, where a large language model is leveraged to construct the black-box defensive system prompt through adversarial meta prompting and generate an malicious textual command that steers the agent's output toward better compliance with attackers' requests. Extensive experiments demonstrate that our method outperforms state-of-the-art attacks, achieving at least a +30.1% increase in attack success rates across diverse tasks. Furthermore, we validate our attack's effectiveness in real-world multimodal autonomous agents, highlighting its potential implications for safety-critical applications.



## **34. SDD: Self-Degraded Defense against Malicious Fine-tuning**

cs.CR

Accepted by ACL2025

**SubmitDate**: 2025-07-27    [abs](http://arxiv.org/abs/2507.21182v1) [paper-pdf](http://arxiv.org/pdf/2507.21182v1)

**Authors**: Zixuan Chen, Weikai Lu, Xin Lin, Ziqian Zeng

**Abstract**: Open-source Large Language Models (LLMs) often employ safety alignment methods to resist harmful instructions. However, recent research shows that maliciously fine-tuning these LLMs on harmful data can easily bypass these safeguards. To counter this, we theoretically uncover why malicious fine-tuning succeeds and identify potential defense strategies. Building on the theoretical analysis, we introduce the Self-Degraded Defense (SDD) framework. SDD encourages LLMs to produce high-quality but irrelevant responses to harmful prompts. When attackers attempt malicious fine-tuning, the general capability of the LLM aligned by SDD will significantly decrease, rendering it incapable of following harmful instructions. Our experimental results confirm SDD's effectiveness against such attacks.



## **35. Risks & Benefits of LLMs & GenAI for Platform Integrity, Healthcare Diagnostics, Financial Trust and Compliance, Cybersecurity, Privacy & AI Safety: A Comprehensive Survey, Roadmap & Implementation Blueprint**

cs.CR

**SubmitDate**: 2025-07-26    [abs](http://arxiv.org/abs/2506.12088v2) [paper-pdf](http://arxiv.org/pdf/2506.12088v2)

**Authors**: Kiarash Ahi

**Abstract**: Large Language Models (LLMs) and generative AI (GenAI) systems, such as ChatGPT, Claude, Gemini, LLaMA, and Copilot (by OpenAI, Anthropic, Google, Meta, and Microsoft, respectively), are reshaping digital platforms and app ecosystems while introducing critical challenges in cybersecurity, privacy, and platform integrity. Our analysis reveals alarming trends: LLM-assisted malware is projected to rise from 2% (2021) to 50% (2025); AI-generated Google reviews grew nearly tenfold (1.2% in 2021 to 12.21% in 2023, expected to reach 30% by 2025); AI scam reports surged 456%; misinformation sites increased over 1500%; and deepfake attacks are projected to rise over 900% in 2025. In finance, LLM-driven threats like synthetic identity fraud and AI-generated scams are accelerating. Platforms such as JPMorgan Chase, Stripe, and Plaid deploy LLMs for fraud detection, regulation parsing, and KYC/AML automation, reducing fraud loss by up to 21% and accelerating onboarding by 40-60%. LLM-facilitated code development has driven mobile app submissions from 1.8 million (2020) to 3.0 million (2024), projected to reach 3.6 million (2025). To address AI threats, platforms like Google Play, Apple App Store, GitHub Copilot, TikTok, Facebook, and Amazon deploy LLM-based defenses, highlighting their dual nature as both threat sources and mitigation tools. In clinical diagnostics, LLMs raise concerns about accuracy, bias, and safety, necessitating strong governance. Drawing on 445 references, this paper surveys LLM/GenAI and proposes a strategic roadmap and operational blueprint integrating policy auditing (such as CCPA and GDPR compliance), fraud detection, and demonstrates an advanced LLM-DA stack with modular components, multi-LLM routing, agentic memory, and governance layers. We provide actionable insights, best practices, and real-world case studies for scalable trust and responsible innovation.



## **36. LoX: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning**

cs.LG

**SubmitDate**: 2025-07-25    [abs](http://arxiv.org/abs/2506.15606v3) [paper-pdf](http://arxiv.org/pdf/2506.15606v3)

**Authors**: Gabriel J. Perin, Runjin Chen, Xuxi Chen, Nina S. T. Hirata, Zhangyang Wang, Junyuan Hong

**Abstract**: Large Language Models (LLMs) have become indispensable in real-world applications. However, their widespread adoption raises significant safety concerns, particularly in responding to socially harmful questions. Despite substantial efforts to improve model safety through alignment, aligned models can still have their safety protections undermined by subsequent fine-tuning - even when the additional training data appears benign. In this paper, we empirically demonstrate that this vulnerability stems from the sensitivity of safety-critical low-rank subspaces in LLM parameters to fine-tuning. Building on this insight, we propose a novel training-free method, termed Low-Rank Extrapolation (LoX), to enhance safety robustness by extrapolating the safety subspace of an aligned LLM. Our experimental results confirm the effectiveness of LoX, demonstrating significant improvements in robustness against both benign and malicious fine-tuning attacks while preserving the model's adaptability to new tasks. For instance, LoX leads to 11% to 54% absolute reductions in attack success rates (ASR) facing benign or malicious fine-tuning attacks. By investigating the ASR landscape of parameters, we attribute the success of LoX to that the extrapolation moves LLM parameters to a flatter zone, thereby less sensitive to perturbations. The code is available at github.com/VITA-Group/LoX.



## **37. MOCHA: Are Code Language Models Robust Against Multi-Turn Malicious Coding Prompts?**

cs.CL

Winner Defender Team at Amazon Nova AI Challenge 2025

**SubmitDate**: 2025-07-25    [abs](http://arxiv.org/abs/2507.19598v1) [paper-pdf](http://arxiv.org/pdf/2507.19598v1)

**Authors**: Muntasir Wahed, Xiaona Zhou, Kiet A. Nguyen, Tianjiao Yu, Nirav Diwan, Gang Wang, Dilek Hakkani-Tür, Ismini Lourentzou

**Abstract**: Recent advancements in Large Language Models (LLMs) have significantly enhanced their code generation capabilities. However, their robustness against adversarial misuse, particularly through multi-turn malicious coding prompts, remains underexplored. In this work, we introduce code decomposition attacks, where a malicious coding task is broken down into a series of seemingly benign subtasks across multiple conversational turns to evade safety filters. To facilitate systematic evaluation, we introduce \benchmarkname{}, a large-scale benchmark designed to evaluate the robustness of code LLMs against both single-turn and multi-turn malicious prompts. Empirical results across open- and closed-source models reveal persistent vulnerabilities, especially under multi-turn scenarios. Fine-tuning on MOCHA improves rejection rates while preserving coding ability, and importantly, enhances robustness on external adversarial datasets with up to 32.4% increase in rejection rates without any additional supervision.



## **38. Jailbreaking Large Language Diffusion Models: Revealing Hidden Safety Flaws in Diffusion-Based Text Generation**

cs.CL

**SubmitDate**: 2025-07-25    [abs](http://arxiv.org/abs/2507.19227v1) [paper-pdf](http://arxiv.org/pdf/2507.19227v1)

**Authors**: Yuanhe Zhang, Fangzhou Xie, Zhenhong Zhou, Zherui Li, Hao Chen, Kun Wang, Yufei Guo

**Abstract**: Large Language Diffusion Models (LLDMs) exhibit comparable performance to LLMs while offering distinct advantages in inference speed and mathematical reasoning tasks.The precise and rapid generation capabilities of LLDMs amplify concerns of harmful generations, while existing jailbreak methodologies designed for Large Language Models (LLMs) prove limited effectiveness against LLDMs and fail to expose safety vulnerabilities.Successful defense cannot definitively resolve harmful generation concerns, as it remains unclear whether LLDMs possess safety robustness or existing attacks are incompatible with diffusion-based architectures.To address this, we first reveal the vulnerability of LLDMs to jailbreak and demonstrate that attack failure in LLDMs stems from fundamental architectural differences.We present a PArallel Decoding jailbreak (PAD) for diffusion-based language models. PAD introduces Multi-Point Attention Attack, which guides parallel generative processes toward harmful outputs that inspired by affirmative response patterns in LLMs. Experimental evaluations across four LLDMs demonstrate that PAD achieves jailbreak attack success rates by 97%, revealing significant safety vulnerabilities. Furthermore, compared to autoregressive LLMs of the same size, LLDMs increase the harmful generation speed by 2x, significantly highlighting risks of uncontrolled misuse.Through comprehensive analysis, we provide an investigation into LLDM architecture, offering critical insights for the secure deployment of diffusion-based language models.



## **39. Blind Spot Navigation: Evolutionary Discovery of Sensitive Semantic Concepts for LVLMs**

cs.CV

The paper needs major revisions, so it is being withdrawn

**SubmitDate**: 2025-07-25    [abs](http://arxiv.org/abs/2505.15265v2) [paper-pdf](http://arxiv.org/pdf/2505.15265v2)

**Authors**: Zihao Pan, Yu Tong, Weibin Wu, Jingyi Wang, Lifeng Chen, Zhe Zhao, Jiajia Wei, Yitong Qiao, Zibin Zheng

**Abstract**: Adversarial attacks aim to generate malicious inputs that mislead deep models, but beyond causing model failure, they cannot provide certain interpretable information such as ``\textit{What content in inputs make models more likely to fail?}'' However, this information is crucial for researchers to specifically improve model robustness. Recent research suggests that models may be particularly sensitive to certain semantics in visual inputs (such as ``wet,'' ``foggy''), making them prone to errors. Inspired by this, in this paper we conducted the first exploration on large vision-language models (LVLMs) and found that LVLMs indeed are susceptible to hallucinations and various errors when facing specific semantic concepts in images. To efficiently search for these sensitive concepts, we integrated large language models (LLMs) and text-to-image (T2I) models to propose a novel semantic evolution framework. Randomly initialized semantic concepts undergo LLM-based crossover and mutation operations to form image descriptions, which are then converted by T2I models into visual inputs for LVLMs. The task-specific performance of LVLMs on each input is quantified as fitness scores for the involved semantics and serves as reward signals to further guide LLMs in exploring concepts that induce LVLMs. Extensive experiments on seven mainstream LVLMs and two multimodal tasks demonstrate the effectiveness of our method. Additionally, we provide interesting findings about the sensitive semantics of LVLMs, aiming to inspire further in-depth research.



## **40. KGV: Integrating Large Language Models with Knowledge Graphs for Cyber Threat Intelligence Credibility Assessment**

cs.CR

**SubmitDate**: 2025-07-25    [abs](http://arxiv.org/abs/2408.08088v2) [paper-pdf](http://arxiv.org/pdf/2408.08088v2)

**Authors**: Zongzong Wu, Fengxiao Tang, Ming Zhao, Yufeng Li

**Abstract**: Cyber threat intelligence (CTI) is a crucial tool to prevent sophisticated, organized, and weaponized cyber attacks. However, few studies have focused on the credibility assessment of CTI, and this work still requires manual analysis by cybersecurity experts. In this paper, we propose Knowledge Graph-based Verifier (KGV), the first framework integrating large language models (LLMs) with simple structured knowledge graphs (KGs) for automated CTI credibility assessment. Unlike entity-centric KGs, KGV constructs paragraph-level semantic graphs where nodes represent text segments connected through similarity analysis, which effectively enhances the semantic understanding ability of the model, reduces KG density and greatly improves response speed. Experimental results demonstrate that our KGV outperforms state-of-the-art fact reasoning methods on the CTI-200 dataset, achieving a 5.7\% improvement in F1. Additionally, it shows strong scalability on factual QA and fake news detection datasets. Compared to entity-based knowledge graphs (KGs) for equivalent-length texts, our structurally simple KG reduces node quantities by nearly two-thirds while boosting precision by 1.7\% and cutting response time by 46.7\%. In addition, we have created and publicly released the first CTI credibility assessment dataset, CTI-200. Distinct from CTI identification datasets, CTI-200 refines CTI summaries and key sentences to focus specifically on credibility assessment.



## **41. Kill two birds with one stone: generalized and robust AI-generated text detection via dynamic perturbations**

cs.CL

Accepted by NAACL 2025 main conference

**SubmitDate**: 2025-07-25    [abs](http://arxiv.org/abs/2504.21019v2) [paper-pdf](http://arxiv.org/pdf/2504.21019v2)

**Authors**: Yinghan Zhou, Juan Wen, Wanli Peng, Yiming Xue, Ziwei Zhang, Zhengxian Wu

**Abstract**: The growing popularity of large language models has raised concerns regarding the potential to misuse AI-generated text (AIGT). It becomes increasingly critical to establish an excellent AIGT detection method with high generalization and robustness. However, existing methods either focus on model generalization or concentrate on robustness. The unified mechanism, to simultaneously address the challenges of generalization and robustness, is less explored. In this paper, we argue that robustness can be view as a specific form of domain shift, and empirically reveal an intrinsic mechanism for model generalization of AIGT detection task. Then, we proposed a novel AIGT detection method (DP-Net) via dynamic perturbations introduced by a reinforcement learning with elaborated reward and action. Experimentally, extensive results show that the proposed DP-Net significantly outperforms some state-of-the-art AIGT detection methods for generalization capacity in three cross-domain scenarios. Meanwhile, the DP-Net achieves best robustness under two text adversarial attacks. The code is publicly available at https://github.com/CAU-ISS-Lab/AIGT-Detection-Evade-Detection/tree/main/DP-Net.



## **42. Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities**

cs.CR

Accepted to TMLR

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2502.05209v4) [paper-pdf](http://arxiv.org/pdf/2502.05209v4)

**Authors**: Zora Che, Stephen Casper, Robert Kirk, Anirudh Satheesh, Stewart Slocum, Lev E McKinney, Rohit Gandikota, Aidan Ewart, Domenic Rosati, Zichu Wu, Zikui Cai, Bilal Chughtai, Yarin Gal, Furong Huang, Dylan Hadfield-Menell

**Abstract**: Evaluations of large language model (LLM) risks and capabilities are increasingly being incorporated into AI risk management and governance frameworks. Currently, most risk evaluations are conducted by designing inputs that elicit harmful behaviors from the system. However, this approach suffers from two limitations. First, input-output evaluations cannot fully evaluate realistic risks from open-weight models. Second, the behaviors identified during any particular input-output evaluation can only lower-bound the model's worst-possible-case input-output behavior. As a complementary method for eliciting harmful behaviors, we propose evaluating LLMs with model tampering attacks which allow for modifications to latent activations or weights. We pit state-of-the-art techniques for removing harmful LLM capabilities against a suite of 5 input-space and 6 model tampering attacks. In addition to benchmarking these methods against each other, we show that (1) model resilience to capability elicitation attacks lies on a low-dimensional robustness subspace; (2) the success rate of model tampering attacks can empirically predict and offer conservative estimates for the success of held-out input-space attacks; and (3) state-of-the-art unlearning methods can easily be undone within 16 steps of fine-tuning. Together, these results highlight the difficulty of suppressing harmful LLM capabilities and show that model tampering attacks enable substantially more rigorous evaluations than input-space attacks alone.



## **43. Noise Contrastive Estimation-based Matching Framework for Low-Resource Security Attack Pattern Recognition**

cs.LG

accepted at EACL 2024, in ARR October 2023

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2401.10337v4) [paper-pdf](http://arxiv.org/pdf/2401.10337v4)

**Authors**: Tu Nguyen, Nedim Šrndić, Alexander Neth

**Abstract**: Tactics, Techniques and Procedures (TTPs) represent sophisticated attack patterns in the cybersecurity domain, described encyclopedically in textual knowledge bases. Identifying TTPs in cybersecurity writing, often called TTP mapping, is an important and challenging task. Conventional learning approaches often target the problem in the classical multi-class or multilabel classification setting. This setting hinders the learning ability of the model due to a large number of classes (i.e., TTPs), the inevitable skewness of the label distribution and the complex hierarchical structure of the label space. We formulate the problem in a different learning paradigm, where the assignment of a text to a TTP label is decided by the direct semantic similarity between the two, thus reducing the complexity of competing solely over the large labeling space. To that end, we propose a neural matching architecture with an effective sampling-based learn-to-compare mechanism, facilitating the learning process of the matching model despite constrained resources.



## **44. A comprehensive study of LLM-based argument classification: from LLAMA through GPT-4o to Deepseek-R1**

cs.CL

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.08621v2) [paper-pdf](http://arxiv.org/pdf/2507.08621v2)

**Authors**: Marcin Pietroń, Rafał Olszowski, Jakub Gomułka, Filip Gampel, Andrzej Tomski

**Abstract**: Argument mining (AM) is an interdisciplinary research field that integrates insights from logic, philosophy, linguistics, rhetoric, law, psychology, and computer science. It involves the automatic identification and extraction of argumentative components, such as premises and claims, and the detection of relationships between them, such as support, attack, or neutrality. Recently, the field has advanced significantly, especially with the advent of large language models (LLMs), which have enhanced the efficiency of analyzing and extracting argument semantics compared to traditional methods and other deep learning models. There are many benchmarks for testing and verifying the quality of LLM, but there is still a lack of research and results on the operation of these models in publicly available argument classification databases. This paper presents a study of a selection of LLM's, using diverse datasets such as Args.me and UKP. The models tested include versions of GPT, Llama, and DeepSeek, along with reasoning-enhanced variants incorporating the Chain-of-Thoughts algorithm. The results indicate that ChatGPT-4o outperforms the others in the argument classification benchmarks. In case of models incorporated with reasoning capabilities, the Deepseek-R1 shows its superiority. However, despite their superiority, GPT-4o and Deepseek-R1 still make errors. The most common errors are discussed for all models. To our knowledge, the presented work is the first broader analysis of the mentioned datasets using LLM and prompt algorithms. The work also shows some weaknesses of known prompt algorithms in argument analysis, while indicating directions for their improvement. The added value of the work is the in-depth analysis of the available argument datasets and the demonstration of their shortcomings.



## **45. BadReasoner: Planting Tunable Overthinking Backdoors into Large Reasoning Models for Fun or Profit**

cs.CL

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18305v1) [paper-pdf](http://arxiv.org/pdf/2507.18305v1)

**Authors**: Biao Yi, Zekun Fei, Jianing Geng, Tong Li, Lihai Nie, Zheli Liu, Yiming Li

**Abstract**: Large reasoning models (LRMs) have emerged as a significant advancement in artificial intelligence, representing a specialized class of large language models (LLMs) designed to tackle complex reasoning tasks. The defining characteristic of LRMs lies in their extensive chain-of-thought (CoT) reasoning capabilities. In this paper, we identify a previously unexplored attack vector against LRMs, which we term "overthinking backdoors". We advance this concept by proposing a novel tunable backdoor, which moves beyond simple on/off attacks to one where an attacker can precisely control the extent of the model's reasoning verbosity. Our attack is implemented through a novel data poisoning methodology. It pairs a tunable trigger-where the number of repetitions signals the desired intensity-with a correspondingly verbose CoT response. These responses are programmatically generated by instructing a teacher LLM to inject a controlled number of redundant refinement steps into a correct reasoning process. The approach preserves output correctness, which ensures stealth and establishes the attack as a pure resource-consumption vector. Extensive empirical results on various LRMs demonstrate that our method can reliably trigger a controllable, multi-fold increase in the length of the reasoning process, without degrading the final answer's correctness. Our source code is available at https://github.com/FZaKK/BadReasoner.



## **46. Auto-SGCR: Automated Generation of Smart Grid Cyber Range Using IEC 61850 Standard Models**

cs.CR

12 pages

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18249v1) [paper-pdf](http://arxiv.org/pdf/2507.18249v1)

**Authors**: Muhammad M. Roomi, S. M. Suhail Hussain, Ee-Chien Chang, David M. Nicol, Daisuke Mashima

**Abstract**: Digitalization of power grids have made them increasingly susceptible to cyber-attacks in the past decade. Iterative cybersecurity testing is indispensable to counter emerging attack vectors and to ensure dependability of critical infrastructure. Furthermore, these can be used to evaluate cybersecurity configuration, effectiveness of the cybersecurity measures against various attack vectors, as well as to train smart grid cybersecurity experts defending the system. Enabling extensive experiments narrows the gap between academic research and production environment. A high-fidelity cyber range is vital as it is often infeasible to conduct such experiments and training using production environment. However, the design and implementation of cyber range requires extensive domain knowledge of physical and cyber aspect of the infrastructure. Furthermore, costs incurred for setup and maintenance of cyber range are significant. Moreover, most existing smart grid cyber ranges are designed as a one-off, proprietary system, and are limited in terms of configurability, accessibility, portability, and reproducibility. To address these challenges, an automated Smart grid Cyber Range generation framework is presented in this paper. Initially a human-/machine-friendly, XML-based modeling language called Smart Grid Modeling Language was defined, which incorporates IEC 61850 System Configuration Language files. Subsequently, a toolchain to parse SG-ML model files and automatically instantiate a functional smart grid cyber range was developed. The developed SG-ML models can be easily shared and/or modified to reproduce or customize for any cyber range. The application of Auto-SGCR is demonstrated through case studies with large-scale substation models. The toolchain along with example SG-ML models have been open-sourced.



## **47. Safeguarding RAG Pipelines with GMTP: A Gradient-based Masked Token Probability Method for Poisoned Document Detection**

cs.CL

18 pages, accepted to ACL Findings 2025

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18202v1) [paper-pdf](http://arxiv.org/pdf/2507.18202v1)

**Authors**: San Kim, Jonghwi Kim, Yejin Jeon, Gary Geunbae Lee

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by providing external knowledge for accurate and up-to-date responses. However, this reliance on external sources exposes a security risk, attackers can inject poisoned documents into the knowledge base to steer the generation process toward harmful or misleading outputs. In this paper, we propose Gradient-based Masked Token Probability (GMTP), a novel defense method to detect and filter out adversarially crafted documents. Specifically, GMTP identifies high-impact tokens by examining gradients of the retriever's similarity function. These key tokens are then masked, and their probabilities are checked via a Masked Language Model (MLM). Since injected tokens typically exhibit markedly low masked-token probabilities, this enables GMTP to easily detect malicious documents and achieve high-precision filtering. Experiments demonstrate that GMTP is able to eliminate over 90% of poisoned content while retaining relevant documents, thus maintaining robust retrieval and generation performance across diverse datasets and adversarial settings.



## **48. Policy Disruption in Reinforcement Learning:Adversarial Attack with Large Language Models and Critical State Identification**

cs.LG

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18113v1) [paper-pdf](http://arxiv.org/pdf/2507.18113v1)

**Authors**: Junyong Jiang, Buwei Tian, Chenxing Xu, Songze Li, Lu Dong

**Abstract**: Reinforcement learning (RL) has achieved remarkable success in fields like robotics and autonomous driving, but adversarial attacks designed to mislead RL systems remain challenging. Existing approaches often rely on modifying the environment or policy, limiting their practicality. This paper proposes an adversarial attack method in which existing agents in the environment guide the target policy to output suboptimal actions without altering the environment. We propose a reward iteration optimization framework that leverages large language models (LLMs) to generate adversarial rewards explicitly tailored to the vulnerabilities of the target agent, thereby enhancing the effectiveness of inducing the target agent toward suboptimal decision-making. Additionally, a critical state identification algorithm is designed to pinpoint the target agent's most vulnerable states, where suboptimal behavior from the victim leads to significant degradation in overall performance. Experimental results in diverse environments demonstrate the superiority of our method over existing approaches.



## **49. RECALLED: An Unbounded Resource Consumption Attack on Large Vision-Language Models**

cs.CR

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18053v1) [paper-pdf](http://arxiv.org/pdf/2507.18053v1)

**Authors**: Haoran Gao, Yuanhe Zhang, Zhenhong Zhou, Lei Jiang, Fanyu Meng, Yujia Xiao, Kun Wang, Yang Liu, Junlan Feng

**Abstract**: Resource Consumption Attacks (RCAs) have emerged as a significant threat to the deployment of Large Language Models (LLMs). With the integration of vision modalities, additional attack vectors exacerbate the risk of RCAs in large vision-language models (LVLMs). However, existing red-teaming studies have largely overlooked visual inputs as a potential attack surface, resulting in insufficient mitigation strategies against RCAs in LVLMs. To address this gap, we propose RECALLED (\textbf{RE}source \textbf{C}onsumption \textbf{A}ttack on \textbf{L}arge Vision-\textbf{L}anguag\textbf{E} Mo\textbf{D}els), the first approach for exploiting visual modalities to trigger unbounded RCAs red-teaming. First, we present \textit{Vision Guided Optimization}, a fine-grained pixel-level optimization, to obtain \textit{Output Recall} adversarial perturbations, which can induce repeating output. Then, we inject the perturbations into visual inputs, triggering unbounded generations to achieve the goal of RCAs. Additionally, we introduce \textit{Multi-Objective Parallel Losses} to generate universal attack templates and resolve optimization conflicts when intending to implement parallel attacks. Empirical results demonstrate that RECALLED increases service response latency by over 26 $\uparrow$, resulting in an additional 20\% increase in GPU utilization and memory consumption. Our study exposes security vulnerabilities in LVLMs and establishes a red-teaming framework that can facilitate future defense development against RCAs.



## **50. ViGText: Deepfake Image Detection with Vision-Language Model Explanations and Graph Neural Networks**

cs.CV

**SubmitDate**: 2025-07-24    [abs](http://arxiv.org/abs/2507.18031v1) [paper-pdf](http://arxiv.org/pdf/2507.18031v1)

**Authors**: Ahmad ALBarqawi, Mahmoud Nazzal, Issa Khalil, Abdallah Khreishah, NhatHai Phan

**Abstract**: The rapid rise of deepfake technology, which produces realistic but fraudulent digital content, threatens the authenticity of media. Traditional deepfake detection approaches often struggle with sophisticated, customized deepfakes, especially in terms of generalization and robustness against malicious attacks. This paper introduces ViGText, a novel approach that integrates images with Vision Large Language Model (VLLM) Text explanations within a Graph-based framework to improve deepfake detection. The novelty of ViGText lies in its integration of detailed explanations with visual data, as it provides a more context-aware analysis than captions, which often lack specificity and fail to reveal subtle inconsistencies. ViGText systematically divides images into patches, constructs image and text graphs, and integrates them for analysis using Graph Neural Networks (GNNs) to identify deepfakes. Through the use of multi-level feature extraction across spatial and frequency domains, ViGText captures details that enhance its robustness and accuracy to detect sophisticated deepfakes. Extensive experiments demonstrate that ViGText significantly enhances generalization and achieves a notable performance boost when it detects user-customized deepfakes. Specifically, average F1 scores rise from 72.45% to 98.32% under generalization evaluation, and reflects the model's superior ability to generalize to unseen, fine-tuned variations of stable diffusion models. As for robustness, ViGText achieves an increase of 11.1% in recall compared to other deepfake detection approaches. When facing targeted attacks that exploit its graph-based architecture, ViGText limits classification performance degradation to less than 4%. ViGText uses detailed visual and textual analysis to set a new standard for detecting deepfakes, helping ensure media authenticity and information integrity.



