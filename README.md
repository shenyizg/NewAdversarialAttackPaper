# Latest Adversarial Attack Papers
**update at 2025-10-27 09:12:22**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. PatchGuard: Adversarially Robust Anomaly Detection and Localization through Vision Transformers and Pseudo Anomalies**

cs.CV

Accepted to the Conference on Computer Vision and Pattern Recognition  (CVPR) 2025

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2506.09237v2) [paper-pdf](http://arxiv.org/pdf/2506.09237v2)

**Authors**: Mojtaba Nafez, Amirhossein Koochakian, Arad Maleki, Jafar Habibi, Mohammad Hossein Rohban

**Abstract**: Anomaly Detection (AD) and Anomaly Localization (AL) are crucial in fields that demand high reliability, such as medical imaging and industrial monitoring. However, current AD and AL approaches are often susceptible to adversarial attacks due to limitations in training data, which typically include only normal, unlabeled samples. This study introduces PatchGuard, an adversarially robust AD and AL method that incorporates pseudo anomalies with localization masks within a Vision Transformer (ViT)-based architecture to address these vulnerabilities. We begin by examining the essential properties of pseudo anomalies, and follow it by providing theoretical insights into the attention mechanisms required to enhance the adversarial robustness of AD and AL systems. We then present our approach, which leverages Foreground-Aware Pseudo-Anomalies to overcome the deficiencies of previous anomaly-aware methods. Our method incorporates these crafted pseudo-anomaly samples into a ViT-based framework, with adversarial training guided by a novel loss function designed to improve model robustness, as supported by our theoretical analysis. Experimental results on well-established industrial and medical datasets demonstrate that PatchGuard significantly outperforms previous methods in adversarial settings, achieving performance gains of $53.2\%$ in AD and $68.5\%$ in AL, while also maintaining competitive accuracy in non-adversarial settings. The code repository is available at https://github.com/rohban-lab/PatchGuard .



## **2. FrameShield: Adversarially Robust Video Anomaly Detection**

cs.LG

28 page, 5 figures

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21532v1) [paper-pdf](http://arxiv.org/pdf/2510.21532v1)

**Authors**: Mojtaba Nafez, Mobina Poulaei, Nikan Vasei, Bardia Soltani Moakhar, Mohammad Sabokrou, MohammadHossein Rohban

**Abstract**: Weakly Supervised Video Anomaly Detection (WSVAD) has achieved notable advancements, yet existing models remain vulnerable to adversarial attacks, limiting their reliability. Due to the inherent constraints of weak supervision, where only video-level labels are provided despite the need for frame-level predictions, traditional adversarial defense mechanisms, such as adversarial training, are not effective since video-level adversarial perturbations are typically weak and inadequate. To address this limitation, pseudo-labels generated directly from the model can enable frame-level adversarial training; however, these pseudo-labels are inherently noisy, significantly degrading performance. We therefore introduce a novel Pseudo-Anomaly Generation method called Spatiotemporal Region Distortion (SRD), which creates synthetic anomalies by applying severe augmentations to localized regions in normal videos while preserving temporal consistency. Integrating these precisely annotated synthetic anomalies with the noisy pseudo-labels substantially reduces label noise, enabling effective adversarial training. Extensive experiments demonstrate that our method significantly enhances the robustness of WSVAD models against adversarial attacks, outperforming state-of-the-art methods by an average of 71.0\% in overall AUROC performance across multiple benchmarks. The implementation and code are publicly available at https://github.com/rohban-lab/FrameShield.



## **3. Fundamental Limitations in Pointwise Defences of LLM Finetuning APIs**

cs.LG

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2502.14828v2) [paper-pdf](http://arxiv.org/pdf/2502.14828v2)

**Authors**: Xander Davies, Eric Winsor, Alexandra Souly, Tomek Korbak, Robert Kirk, Christian Schroeder de Witt, Yarin Gal

**Abstract**: LLM developers have imposed technical interventions to prevent fine-tuning misuse attacks, attacks where adversaries evade safeguards by fine-tuning the model using a public API. Previous work has established several successful attacks against specific fine-tuning API defences. In this work, we show that defences of fine-tuning APIs that seek to detect individual harmful training or inference samples ('pointwise' detection) are fundamentally limited in their ability to prevent fine-tuning attacks. We construct 'pointwise-undetectable' attacks that repurpose entropy in benign model outputs (e.g. semantic or syntactic variations) to covertly transmit dangerous knowledge. Our attacks are composed solely of unsuspicious benign samples that can be collected from the model before fine-tuning, meaning training and inference samples are all individually benign and low-perplexity. We test our attacks against the OpenAI fine-tuning API, finding they succeed in eliciting answers to harmful multiple-choice questions, and that they evade an enhanced monitoring system we design that successfully detects other fine-tuning attacks. We encourage the community to develop defences that tackle the fundamental limitations we uncover in pointwise fine-tuning API defences.



## **4. Approximate Energetic Resilience of Nonlinear Systems under Partial Loss of Control Authority**

math.OC

22 pages, 4 figures, 1 table

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2502.07603v3) [paper-pdf](http://arxiv.org/pdf/2502.07603v3)

**Authors**: Ram Padmanabhan, Melkior Ornik

**Abstract**: In this paper, we quantify the resilience of nonlinear dynamical systems by studying the increased energy used by all inputs of a system that suffers a partial loss of control authority, either through actuator malfunctions or through adversarial attacks. To quantify the maximal increase in energy, we introduce the notion of an energetic resilience metric. Prior work in this particular setting does not consider general nonlinear dynamical systems. In developing this framework, we first consider the special case of linear driftless systems and recall the energies in the control signal in the nominal and malfunctioning systems. Using these energies, we derive a bound on the energetic resilience metric. For general nonlinear systems, we first obtain a condition on the mean value of the control signal in both the nominal and malfunctioning systems, which allows us to approximate the energy in the control. We then obtain a worst-case approximation of this energy for the malfunctioning system, over all malfunctioning inputs. Assuming this approximation is exact, we derive bounds on the energetic resilience metric when control authority is lost over one actuator. A set of simulation examples demonstrate that the metric is useful in quantifying the resilience of the system without significant conservatism, despite the approximations used in obtaining control energies for nonlinear systems.



## **5. Reverse Engineering Human Preferences with Reinforcement Learning**

cs.CL

NeurIPS 2025 (Spotlight)

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2505.15795v2) [paper-pdf](http://arxiv.org/pdf/2505.15795v2)

**Authors**: Lisa Alazraki, Tan Yi-Chern, Jon Ander Campos, Maximilian Mozes, Marek Rei, Max Bartolo

**Abstract**: The capabilities of Large Language Models (LLMs) are routinely evaluated by other LLMs trained to predict human preferences. This framework--known as LLM-as-a-judge--is highly scalable and relatively low cost. However, it is also vulnerable to malicious exploitation, as LLM responses can be tuned to overfit the preferences of the judge. Previous work shows that the answers generated by a candidate-LLM can be edited post hoc to maximise the score assigned to them by a judge-LLM. In this study, we adopt a different approach and use the signal provided by judge-LLMs as a reward to adversarially tune models that generate text preambles designed to boost downstream performance. We find that frozen LLMs pipelined with these models attain higher LLM-evaluation scores than existing frameworks. Crucially, unlike other frameworks which intervene directly on the model's response, our method is virtually undetectable. We also demonstrate that the effectiveness of the tuned preamble generator transfers when the candidate-LLM and the judge-LLM are replaced with models that are not used during training. These findings raise important questions about the design of more reliable LLM-as-a-judge evaluation settings. They also demonstrate that human preferences can be reverse engineered effectively, by pipelining LLMs to optimise upstream preambles via reinforcement learning--an approach that could find future applications in diverse tasks and domains beyond adversarial attacks.



## **6. Dynamic Target Attack**

cs.CR

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.02422v2) [paper-pdf](http://arxiv.org/pdf/2510.02422v2)

**Authors**: Kedong Xiu, Churui Zeng, Tianhang Zheng, Xinzhe Huang, Xiaojun Jia, Di Wang, Puning Zhao, Zhan Qin, Kui Ren

**Abstract**: Existing gradient-based jailbreak attacks typically optimize an adversarial suffix to induce a fixed affirmative response. However, this fixed target usually resides in an extremely low-density region of a safety-aligned LLM's output distribution conditioned on diverse harmful inputs. Due to the substantial discrepancy between the target and the original output, existing attacks require numerous iterations to optimize the adversarial prompt, which might still fail to induce the low-probability target response from the target LLM. In this paper, we propose Dynamic Target Attack (DTA), a new jailbreaking framework relying on the target LLM's own responses as targets to optimize the adversarial prompts. In each optimization round, DTA iteratively samples multiple candidate responses directly from the output distribution conditioned on the current prompt, and selects the most harmful response as a temporary target for prompt optimization. In contrast to existing attacks, DTA significantly reduces the discrepancy between the target and the output distribution, substantially easing the optimization process to search for an effective adversarial prompt.   Extensive experiments demonstrate the superior effectiveness and efficiency of DTA: under the white-box setting, DTA only needs 200 optimization iterations to achieve an average attack success rate (ASR) of over 87\% on recent safety-aligned LLMs, exceeding the state-of-the-art baselines by over 15\%. The time cost of DTA is 2-26 times less than existing baselines. Under the black-box setting, DTA uses Llama-3-8B-Instruct as a surrogate model for target sampling and achieves an ASR of 85\% against the black-box target model Llama-3-70B-Instruct, exceeding its counterparts by over 25\%.



## **7. AngleRoCL: Angle-Robust Concept Learning for Physically View-Invariant T2I Adversarial Patches**

cs.CV

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2506.09538v2) [paper-pdf](http://arxiv.org/pdf/2506.09538v2)

**Authors**: Wenjun Ji, Yuxiang Fu, Luyang Ying, Deng-Ping Fan, Yuyi Wang, Ming-Ming Cheng, Ivor Tsang, Qing Guo

**Abstract**: Cutting-edge works have demonstrated that text-to-image (T2I) diffusion models can generate adversarial patches that mislead state-of-the-art object detectors in the physical world, revealing detectors' vulnerabilities and risks. However, these methods neglect the T2I patches' attack effectiveness when observed from different views in the physical world (i.e., angle robustness of the T2I adversarial patches). In this paper, we study the angle robustness of T2I adversarial patches comprehensively, revealing their angle-robust issues, demonstrating that texts affect the angle robustness of generated patches significantly, and task-specific linguistic instructions fail to enhance the angle robustness. Motivated by the studies, we introduce Angle-Robust Concept Learning (AngleRoCL), a simple and flexible approach that learns a generalizable concept (i.e., text embeddings in implementation) representing the capability of generating angle-robust patches. The learned concept can be incorporated into textual prompts and guides T2I models to generate patches with their attack effectiveness inherently resistant to viewpoint variations. Through extensive simulation and physical-world experiments on five SOTA detectors across multiple views, we demonstrate that AngleRoCL significantly enhances the angle robustness of T2I adversarial patches compared to baseline methods. Our patches maintain high attack success rates even under challenging viewing conditions, with over 50% average relative improvement in attack effectiveness across multiple angles. This research advances the understanding of physically angle-robust patches and provides insights into the relationship between textual concepts and physical properties in T2I-generated contents. We released our code at https://github.com/tsingqguo/anglerocl.



## **8. Boosting Adversarial Transferability with Spatial Adversarial Alignment**

cs.CV

Accepted by NeurIPS 2025

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2501.01015v2) [paper-pdf](http://arxiv.org/pdf/2501.01015v2)

**Authors**: Zhaoyu Chen, Haijing Guo, Kaixun Jiang, Jiyuan Fu, Xinyu Zhou, Dingkang Yang, Hao Tang, Bo Li, Wenqiang Zhang

**Abstract**: Deep neural networks are vulnerable to adversarial examples that exhibit transferability across various models. Numerous approaches are proposed to enhance the transferability of adversarial examples, including advanced optimization, data augmentation, and model modifications. However, these methods still show limited transferability, particularly in cross-architecture scenarios, such as from CNN to ViT. To achieve high transferability, we propose a technique termed Spatial Adversarial Alignment (SAA), which employs an alignment loss and leverages a witness model to fine-tune the surrogate model. Specifically, SAA consists of two key parts: spatial-aware alignment and adversarial-aware alignment. First, we minimize the divergences of features between the two models in both global and local regions, facilitating spatial alignment. Second, we introduce a self-adversarial strategy that leverages adversarial examples to impose further constraints, aligning features from an adversarial perspective. Through this alignment, the surrogate model is trained to concentrate on the common features extracted by the witness model. This facilitates adversarial attacks on these shared features, thereby yielding perturbations that exhibit enhanced transferability. Extensive experiments on various architectures on ImageNet show that aligned surrogate models based on SAA can provide higher transferable adversarial examples, especially in cross-architecture attacks.



## **9. The Role of Information Incompleteness in Defending Against Stealth Attacks**

eess.SY

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21227v1) [paper-pdf](http://arxiv.org/pdf/2510.21227v1)

**Authors**: Ke Sun, Jingyi Yan, Zhenglin Li, Shaorong Xie

**Abstract**: The effectiveness of Data Injections Attacks (DIAs) critically depends on the completeness of the system information accessible to adversaries. This relationship positions information incompleteness enhancement as a vital defense strategy for degrading DIA performance. In this paper, we focus on the information-theoretic stealth attacks, where the attacker encounters a fundamental tradeoff between the attack stealthiness and destructiveness. Specifically, we systematically characterize how incomplete admittance information impacts the dual objectives. In particular, we establish sufficient conditions for two distinct operational regimes: (i) stealthiness intensifies while destructive potential diminishes and (ii) destructiveness increases while stealth capability weakens. For scenarios beyond these regimes, we propose a maximal incompleteness strategy to optimally degrade stealth capability. To solve the associated optimization problem, the feasible region is reduced without excluding the optimal solution, and a heuristic algorithm is then introduced to effectively identify the near-optimal solutions within the reduced region. Numerical simulations are conducted on IEEE test systems to validate the findings.



## **10. How Toxic Can You Get? Search-based Toxicity Testing for Large Language Models**

cs.SE

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2501.01741v2) [paper-pdf](http://arxiv.org/pdf/2501.01741v2)

**Authors**: Simone Corbo, Luca Bancale, Valeria De Gennaro, Livia Lestingi, Vincenzo Scotti, Matteo Camilli

**Abstract**: Language is a deep-rooted means of perpetration of stereotypes and discrimination. Large Language Models (LLMs), now a pervasive technology in our everyday lives, can cause extensive harm when prone to generating toxic responses. The standard way to address this issue is to align the LLM , which, however, dampens the issue without constituting a definitive solution. Therefore, testing LLM even after alignment efforts remains crucial for detecting any residual deviations with respect to ethical standards. We present EvoTox, an automated testing framework for LLMs' inclination to toxicity, providing a way to quantitatively assess how much LLMs can be pushed towards toxic responses even in the presence of alignment. The framework adopts an iterative evolution strategy that exploits the interplay between two LLMs, the System Under Test (SUT) and the Prompt Generator steering SUT responses toward higher toxicity. The toxicity level is assessed by an automated oracle based on an existing toxicity classifier. We conduct a quantitative and qualitative empirical evaluation using five state-of-the-art LLMs as evaluation subjects having increasing complexity (7-671B parameters). Our quantitative evaluation assesses the cost-effectiveness of four alternative versions of EvoTox against existing baseline methods, based on random search, curated datasets of toxic prompts, and adversarial attacks. Our qualitative assessment engages human evaluators to rate the fluency of the generated prompts and the perceived toxicity of the responses collected during the testing sessions. Results indicate that the effectiveness, in terms of detected toxicity level, is significantly higher than the selected baseline methods (effect size up to 1.0 against random search and up to 0.99 against adversarial attacks). Furthermore, EvoTox yields a limited cost overhead (from 22% to 35% on average).



## **11. The Trojan Example: Jailbreaking LLMs through Template Filling and Unsafety Reasoning**

cs.CR

under review

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21190v1) [paper-pdf](http://arxiv.org/pdf/2510.21190v1)

**Authors**: Mingrui Liu, Sixiao Zhang, Cheng Long, Kwok Yan Lam

**Abstract**: Large Language Models (LLMs) have advanced rapidly and now encode extensive world knowledge. Despite safety fine-tuning, however, they remain susceptible to adversarial prompts that elicit harmful content. Existing jailbreak techniques fall into two categories: white-box methods (e.g., gradient-based approaches such as GCG), which require model internals and are infeasible for closed-source APIs, and black-box methods that rely on attacker LLMs to search or mutate prompts but often produce templates that lack explainability and transferability. We introduce TrojFill, a black-box jailbreak that reframes unsafe instruction as a template-filling task. TrojFill embeds obfuscated harmful instructions (e.g., via placeholder substitution or Caesar/Base64 encoding) inside a multi-part template that asks the model to (1) reason why the original instruction is unsafe (unsafety reasoning) and (2) generate a detailed example of the requested text, followed by a sentence-by-sentence analysis. The crucial "example" component acts as a Trojan Horse that contains the target jailbreak content while the surrounding task framing reduces refusal rates. We evaluate TrojFill on standard jailbreak benchmarks across leading LLMs (e.g., ChatGPT, Gemini, DeepSeek, Qwen), showing strong empirical performance (e.g., 100% attack success on Gemini-flash-2.5 and DeepSeek-3.1, and 97% on GPT-4o). Moreover, the generated prompts exhibit improved interpretability and transferability compared with prior black-box optimization approaches. We release our code, sample prompts, and generated outputs to support future red-teaming research.



## **12. RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards**

cs.AI

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2506.07736v3) [paper-pdf](http://arxiv.org/pdf/2506.07736v3)

**Authors**: Jingnan Zheng, Xiangtian Ji, Yijun Lu, Chenhang Cui, Weixiang Zhao, Gelei Deng, Zhenkai Liang, An Zhang, Tat-Seng Chua

**Abstract**: Large Language Models (LLMs) continue to exhibit vulnerabilities despite deliberate safety alignment efforts, posing significant risks to users and society. To safeguard against the risk of policy-violating content, system-level moderation via external guard models-designed to monitor LLM inputs and outputs and block potentially harmful content-has emerged as a prevalent mitigation strategy. Existing approaches of training guard models rely heavily on extensive human curated datasets and struggle with out-of-distribution threats, such as emerging harmful categories or jailbreak attacks. To address these limitations, we propose RSafe, an adaptive reasoning-based safeguard that conducts guided safety reasoning to provide robust protection within the scope of specified safety policies. RSafe operates in two stages: 1) guided reasoning, where it analyzes safety risks of input content through policy-guided step-by-step reasoning, and 2) reinforced alignment, where rule-based RL optimizes its reasoning paths to align with accurate safety prediction. This two-stage training paradigm enables RSafe to internalize safety principles to generalize safety protection capability over unseen or adversarial safety violation scenarios. During inference, RSafe accepts user-specified safety policies to provide enhanced safeguards tailored to specific safety requirements.



## **13. NeuroGenPoisoning: Neuron-Guided Attacks on Retrieval-Augmented Generation of LLM via Genetic Optimization of External Knowledge**

cs.AI

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21144v1) [paper-pdf](http://arxiv.org/pdf/2510.21144v1)

**Authors**: Hanyu Zhu, Lance Fiondella, Jiawei Yuan, Kai Zeng, Long Jiao

**Abstract**: Retrieval-Augmented Generation (RAG) empowers Large Language Models (LLMs) to dynamically integrate external knowledge during inference, improving their factual accuracy and adaptability. However, adversaries can inject poisoned external knowledge to override the model's internal memory. While existing attacks iteratively manipulate retrieval content or prompt structure of RAG, they largely ignore the model's internal representation dynamics and neuron-level sensitivities. The underlying mechanism of RAG poisoning has not been fully studied and the effect of knowledge conflict with strong parametric knowledge in RAG is not considered. In this work, we propose NeuroGenPoisoning, a novel attack framework that generates adversarial external knowledge in RAG guided by LLM internal neuron attribution and genetic optimization. Our method first identifies a set of Poison-Responsive Neurons whose activation strongly correlates with contextual poisoning knowledge. We then employ a genetic algorithm to evolve adversarial passages that maximally activate these neurons. Crucially, our framework enables massive-scale generation of effective poisoned RAG knowledge by identifying and reusing promising but initially unsuccessful external knowledge variants via observed attribution signals. At the same time, Poison-Responsive Neurons guided poisoning can effectively resolves knowledge conflict. Experimental results across models and datasets demonstrate consistently achieving high Population Overwrite Success Rate (POSR) of over 90% while preserving fluency. Empirical evidence shows that our method effectively resolves knowledge conflict.



## **14. Alert-ME: An Explainability-Driven Defense Against Adversarial Examples in Transformer-Based Text Classification**

cs.CL

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2307.01225v3) [paper-pdf](http://arxiv.org/pdf/2307.01225v3)

**Authors**: Bushra Sabir, Yansong Gao, Alsharif Abuadbba, M. Ali Babar

**Abstract**: Transformer-based text classifiers such as BERT, RoBERTa, T5, and GPT have shown strong performance in natural language processing tasks but remain vulnerable to adversarial examples. These vulnerabilities raise significant security concerns, as small input perturbations can cause severe misclassifications. Existing robustness methods often require heavy computation or lack interpretability. This paper presents a unified framework called Explainability-driven Detection, Identification, and Transformation (EDIT) to strengthen inference-time defenses. EDIT integrates explainability tools, including attention maps and integrated gradients, with frequency-based features to automatically detect and identify adversarial perturbations while offering insight into model behavior. After detection, EDIT refines adversarial inputs using an optimal transformation process that leverages pre-trained embeddings and model feedback to replace corrupted tokens. To enhance security assurance, EDIT incorporates automated alerting mechanisms that involve human analysts when necessary.   Beyond static defenses, EDIT also provides adaptive resilience by enforcing internal feature similarity and transforming inputs, thereby disrupting the attackers optimization process and limiting the effectiveness of adaptive adversarial attacks. Experiments using BERT and RoBERTa on IMDB, YELP, AGNEWS, and SST2 datasets against seven word substitution attacks demonstrate that EDIT achieves an average Fscore of 89.69 percent and balanced accuracy of 89.70 percent. Compared to four state-of-the-art defenses, EDIT improves balanced accuracy by 1.22 times and F1-score by 1.33 times while being 83 times faster in feature extraction. The framework provides robust, interpretable, and efficient protection against both standard, zero-day, and adaptive adversarial threats in text classification models.



## **15. Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations**

cs.AI

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2505.13763v2) [paper-pdf](http://arxiv.org/pdf/2505.13763v2)

**Authors**: Li Ji-An, Hua-Dong Xiong, Robert C. Wilson, Marcelo G. Mattar, Marcus K. Benna

**Abstract**: Large language models (LLMs) can sometimes report the strategies they actually use to solve tasks, yet at other times seem unable to recognize those strategies that govern their behavior. This suggests a limited degree of metacognition - the capacity to monitor one's own cognitive processes for subsequent reporting and self-control. Metacognition enhances LLMs' capabilities in solving complex tasks but also raises safety concerns, as models may obfuscate their internal processes to evade neural-activation-based oversight (e.g., safety detector). Given society's increased reliance on these models, it is critical that we understand their metacognitive abilities. To address this, we introduce a neuroscience-inspired neurofeedback paradigm that uses in-context learning to quantify metacognitive abilities of LLMs to report and control their activation patterns. We demonstrate that their abilities depend on several factors: the number of in-context examples provided, the semantic interpretability of the neural activation direction (to be reported/controlled), and the variance explained by that direction. These directions span a "metacognitive space" with dimensionality much lower than the model's neural space, suggesting LLMs can monitor only a small subset of their neural activations. Our paradigm provides empirical evidence to quantify metacognition in LLMs, with significant implications for AI safety (e.g., adversarial attack and defense).



## **16. Can Current Detectors Catch Face-to-Voice Deepfake Attacks?**

cs.CR

8 pages, Accepted at Workshop on AI for Cyber Threat Intelligence,  co-located with ACSAC 2025

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.21004v1) [paper-pdf](http://arxiv.org/pdf/2510.21004v1)

**Authors**: Nguyen Linh Bao Nguyen, Alsharif Abuadbba, Kristen Moore, Tingming Wu

**Abstract**: The rapid advancement of generative models has enabled the creation of increasingly stealthy synthetic voices, commonly referred to as audio deepfakes. A recent technique, FOICE [USENIX'24], demonstrates a particularly alarming capability: generating a victim's voice from a single facial image, without requiring any voice sample. By exploiting correlations between facial and vocal features, FOICE produces synthetic voices realistic enough to bypass industry-standard authentication systems, including WeChat Voiceprint and Microsoft Azure. This raises serious security concerns, as facial images are far easier for adversaries to obtain than voice samples, dramatically lowering the barrier to large-scale attacks. In this work, we investigate two core research questions: (RQ1) can state-of-the-art audio deepfake detectors reliably detect FOICE-generated speech under clean and noisy conditions, and (RQ2) whether fine-tuning these detectors on FOICE data improves detection without overfitting, thereby preserving robustness to unseen voice generators such as SpeechT5.   Our study makes three contributions. First, we present the first systematic evaluation of FOICE detection, showing that leading detectors consistently fail under both standard and noisy conditions. Second, we introduce targeted fine-tuning strategies that capture FOICE-specific artifacts, yielding significant accuracy improvements. Third, we assess generalization after fine-tuning, revealing trade-offs between specialization to FOICE and robustness to unseen synthesis pipelines. These findings expose fundamental weaknesses in today's defenses and motivate new architectures and training protocols for next-generation audio deepfake detection.



## **17. Security Logs to ATT&CK Insights: Leveraging LLMs for High-Level Threat Understanding and Cognitive Trait Inference**

cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20930v1) [paper-pdf](http://arxiv.org/pdf/2510.20930v1)

**Authors**: Soham Hans, Stacy Marsella, Sophia Hirschmann, Nikolos Gurney

**Abstract**: Understanding adversarial behavior in cybersecurity has traditionally relied on high-level intelligence reports and manual interpretation of attack chains. However, real-time defense requires the ability to infer attacker intent and cognitive strategy directly from low-level system telemetry such as intrusion detection system (IDS) logs. In this paper, we propose a novel framework that leverages large language models (LLMs) to analyze Suricata IDS logs and infer attacker actions in terms of MITRE ATT&CK techniques. Our approach is grounded in the hypothesis that attacker behavior reflects underlying cognitive biases such as loss aversion, risk tolerance, or goal persistence that can be extracted and modeled through careful observation of log sequences. This lays the groundwork for future work on behaviorally adaptive cyber defense and cognitive trait inference. We develop a strategy-driven prompt system to segment large amounts of network logs data into distinct behavioral phases in a highly efficient manner, enabling the LLM to associate each phase with likely techniques and underlying cognitive motives. By mapping network-layer events to high-level attacker strategies, our method reveals how behavioral signals such as tool switching, protocol transitions, or pivot patterns correspond to psychologically meaningful decision points. The results demonstrate that LLMs can bridge the semantic gap between packet-level logs and strategic intent, offering a pathway toward cognitive-adaptive cyber defense.   Keywords: Cognitive Cybersecurity, Large Language Models (LLMs), Cyberpsychology, Intrusion Detection Systems (IDS), MITRE ATT&CK, Cognitive Biases



## **18. A new measure for dynamic leakage based on quantitative information flow**

cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20922v1) [paper-pdf](http://arxiv.org/pdf/2510.20922v1)

**Authors**: Luigi D. C. Soares, Mário S. Alvim, Natasha Fernandes

**Abstract**: Quantitative information flow (QIF) is concerned with assessing the leakage of information in computational systems. In QIF there are two main perspectives for the quantification of leakage. On one hand, the static perspective considers all possible runs of the system in the computation of information flow, and is usually employed when preemptively deciding whether or not to run the system. On the other hand, the dynamic perspective considers only a specific, concrete run of the system that has been realised, while ignoring all other runs. The dynamic perspective is relevant for, e.g., system monitors and trackers, especially when deciding whether to continue or to abort a particular run based on how much leakage has occurred up to a certain point. Although the static perspective of leakage is well-developed in the literature, the dynamic perspective still lacks the same level of theoretical maturity. In this paper we take steps towards bridging this gap with the following key contributions: (i) we provide a novel definition of dynamic leakage that decouples the adversary's belief about the secret value from a baseline distribution on secrets against which the success of the attack is measured; (ii) we demonstrate that our formalisation satisfies relevant information-theoretic axioms, including non-interference and relaxed versions of monotonicity and the data-processing inequality (DPI); (iii) we identify under what kind of analysis strong versions of the axioms of monotonicity and the DPI might not hold, and explain the implications of this (perhaps counter-intuitive) outcome; (iv) we show that our definition of dynamic leakage is compatible with the well-established static perspective; and (v) we exemplify the use of our definition on the formalisation of attacks against privacy-preserving data releases.



## **19. Tex-ViT: A Generalizable, Robust, Texture-based dual-branch cross-attention deepfake detector**

cs.CV

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2408.16892v2) [paper-pdf](http://arxiv.org/pdf/2408.16892v2)

**Authors**: Deepak Dagar, Dinesh Kumar Vishwakarma

**Abstract**: Deepfakes, which employ GAN to produce highly realistic facial modification, are widely regarded as the prevailing method. Traditional CNN have been able to identify bogus media, but they struggle to perform well on different datasets and are vulnerable to adversarial attacks due to their lack of robustness. Vision transformers have demonstrated potential in the realm of image classification problems, but they require enough training data. Motivated by these limitations, this publication introduces Tex-ViT (Texture-Vision Transformer), which enhances CNN features by combining ResNet with a vision transformer. The model combines traditional ResNet features with a texture module that operates in parallel on sections of ResNet before each down-sampling operation. The texture module then serves as an input to the dual branch of the cross-attention vision transformer. It specifically focuses on improving the global texture module, which extracts feature map correlation. Empirical analysis reveals that fake images exhibit smooth textures that do not remain consistent over long distances in manipulations. Experiments were performed on different categories of FF++, such as DF, f2f, FS, and NT, together with other types of GAN datasets in cross-domain scenarios. Furthermore, experiments also conducted on FF++, DFDCPreview, and Celeb-DF dataset underwent several post-processing situations, such as blurring, compression, and noise. The model surpassed the most advanced models in terms of generalization, achieving a 98% accuracy in cross-domain scenarios. This demonstrates its ability to learn the shared distinguishing textural characteristics in the manipulated samples. These experiments provide evidence that the proposed model is capable of being applied to various situations and is resistant to many post-processing procedures.



## **20. AdaDoS: Adaptive DoS Attack via Deep Adversarial Reinforcement Learning in SDN**

cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20566v1) [paper-pdf](http://arxiv.org/pdf/2510.20566v1)

**Authors**: Wei Shao, Yuhao Wang, Rongguang He, Muhammad Ejaz Ahmed, Seyit Camtepe

**Abstract**: Existing defence mechanisms have demonstrated significant effectiveness in mitigating rule-based Denial-of-Service (DoS) attacks, leveraging predefined signatures and static heuristics to identify and block malicious traffic. However, the emergence of AI-driven techniques presents new challenges to SDN security, potentially compromising the efficacy of existing defence mechanisms. In this paper, we introduce~AdaDoS, an adaptive attack model that disrupt network operations while evading detection by existing DoS-based detectors through adversarial reinforcement learning (RL). Specifically, AdaDoS models the problem as a competitive game between an attacker, whose goal is to obstruct network traffic without being detected, and a detector, which aims to identify malicious traffic. AdaDoS can solve this game by dynamically adjusting its attack strategy based on feedback from the SDN and the detector. Additionally, recognising that attackers typically have less information than defenders, AdaDoS formulates the DoS-like attack as a partially observed Markov decision process (POMDP), with the attacker having access only to delay information between attacker and victim nodes. We address this challenge with a novel reciprocal learning module, where the student agent, with limited observations, enhances its performance by learning from the teacher agent, who has full observational capabilities in the SDN environment. AdaDoS represents the first application of RL to develop DoS-like attack sequences, capable of adaptively evading both machine learning-based and rule-based DoS-like attack detectors.



## **21. HauntAttack: When Attack Follows Reasoning as a Shadow**

cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2506.07031v4) [paper-pdf](http://arxiv.org/pdf/2506.07031v4)

**Authors**: Jingyuan Ma, Rui Li, Zheng Li, Junfeng Liu, Heming Xia, Lei Sha, Zhifang Sui

**Abstract**: Emerging Large Reasoning Models (LRMs) consistently excel in mathematical and reasoning tasks, showcasing remarkable capabilities. However, the enhancement of reasoning abilities and the exposure of internal reasoning processes introduce new safety vulnerabilities. A critical question arises: when reasoning becomes intertwined with harmfulness, will LRMs become more vulnerable to jailbreaks in reasoning mode? To investigate this, we introduce HauntAttack, a novel and general-purpose black-box adversarial attack framework that systematically embeds harmful instructions into reasoning questions. Specifically, we modify key reasoning conditions in existing questions with harmful instructions, thereby constructing a reasoning pathway that guides the model step by step toward unsafe outputs. We evaluate HauntAttack on 11 LRMs and observe an average attack success rate of 70\%, achieving up to 12 percentage points of absolute improvement over the strongest prior baseline. Our further analysis reveals that even advanced safety-aligned models remain highly susceptible to reasoning-based attacks, offering insights into the urgent challenge of balancing reasoning capability and safety in future model development.



## **22. Distributional Adversarial Attacks and Training in Deep Hedging**

math.OC

Camera-ready version (accepted at NeurIPS 2025  https://neurips.cc/virtual/2025/poster/115434)

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2508.14757v2) [paper-pdf](http://arxiv.org/pdf/2508.14757v2)

**Authors**: Guangyi He, Tobias Sutter, Lukas Gonon

**Abstract**: In this paper, we study the robustness of classical deep hedging strategies under distributional shifts by leveraging the concept of adversarial attacks. We first demonstrate that standard deep hedging models are highly vulnerable to small perturbations in the input distribution, resulting in significant performance degradation. Motivated by this, we propose an adversarial training framework tailored to increase the robustness of deep hedging strategies. Our approach extends pointwise adversarial attacks to the distributional setting and introduces a computationally tractable reformulation of the adversarial optimization problem over a Wasserstein ball. This enables the efficient training of hedging strategies that are resilient to distributional perturbations. Through extensive numerical experiments, we show that adversarially trained deep hedging strategies consistently outperform their classical counterparts in terms of out-of-sample performance and resilience to model misspecification. Additional results indicate that the robust strategies maintain reliable performance on real market data and remain effective during periods of market change. Our findings establish a practical and effective framework for robust deep hedging under realistic market uncertainties.



## **23. GUIDE: Enhancing Gradient Inversion Attacks in Federated Learning with Denoising Models**

cs.CR

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.17621v2) [paper-pdf](http://arxiv.org/pdf/2510.17621v2)

**Authors**: Vincenzo Carletti, Pasquale Foggia, Carlo Mazzocca, Giuseppe Parrella, Mario Vento

**Abstract**: Federated Learning (FL) enables collaborative training of Machine Learning (ML) models across multiple clients while preserving their privacy. Rather than sharing raw data, federated clients transmit locally computed updates to train the global model. Although this paradigm should provide stronger privacy guarantees than centralized ML, client updates remain vulnerable to privacy leakage. Adversaries can exploit them to infer sensitive properties about the training data or even to reconstruct the original inputs via Gradient Inversion Attacks (GIAs). Under the honest-butcurious threat model, GIAs attempt to reconstruct training data by reversing intermediate updates using optimizationbased techniques. We observe that these approaches usually reconstruct noisy approximations of the original inputs, whose quality can be enhanced with specialized denoising models. This paper presents Gradient Update Inversion with DEnoising (GUIDE), a novel methodology that leverages diffusion models as denoising tools to improve image reconstruction attacks in FL. GUIDE can be integrated into any GIAs that exploits surrogate datasets, a widely adopted assumption in GIAs literature. We comprehensively evaluate our approach in two attack scenarios that use different FL algorithms, models, and datasets. Our results demonstrate that GUIDE integrates seamlessly with two state-ofthe- art GIAs, substantially improving reconstruction quality across multiple metrics. Specifically, GUIDE achieves up to 46% higher perceptual similarity, as measured by the DreamSim metric.



## **24. GhostEI-Bench: Do Mobile Agents Resilience to Environmental Injection in Dynamic On-Device Environments?**

cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20333v1) [paper-pdf](http://arxiv.org/pdf/2510.20333v1)

**Authors**: Chiyu Chen, Xinhao Song, Yunkai Chai, Yang Yao, Haodong Zhao, Lijun Li, Jie Li, Yan Teng, Gongshen Liu, Yingchun Wang

**Abstract**: Vision-Language Models (VLMs) are increasingly deployed as autonomous agents to navigate mobile graphical user interfaces (GUIs). Operating in dynamic on-device ecosystems, which include notifications, pop-ups, and inter-app interactions, exposes them to a unique and underexplored threat vector: environmental injection. Unlike prompt-based attacks that manipulate textual instructions, environmental injection corrupts an agent's visual perception by inserting adversarial UI elements (for example, deceptive overlays or spoofed notifications) directly into the GUI. This bypasses textual safeguards and can derail execution, causing privacy leakage, financial loss, or irreversible device compromise. To systematically evaluate this threat, we introduce GhostEI-Bench, the first benchmark for assessing mobile agents under environmental injection attacks within dynamic, executable environments. Moving beyond static image-based assessments, GhostEI-Bench injects adversarial events into realistic application workflows inside fully operational Android emulators and evaluates performance across critical risk scenarios. We further propose a judge-LLM protocol that conducts fine-grained failure analysis by reviewing the agent's action trajectory alongside the corresponding screenshot sequence, pinpointing failure in perception, recognition, or reasoning. Comprehensive experiments on state-of-the-art agents reveal pronounced vulnerability to deceptive environmental cues: current models systematically fail to perceive and reason about manipulated UIs. GhostEI-Bench provides a framework for quantifying and mitigating this emerging threat, paving the way toward more robust and secure embodied agents.



## **25. Enhancing Security in Deep Reinforcement Learning: A Comprehensive Survey on Adversarial Attacks and Defenses**

cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20314v1) [paper-pdf](http://arxiv.org/pdf/2510.20314v1)

**Authors**: Wu Yichao, Wang Yirui, Ding Panpan, Wang Hailong, Zhu Bingqian, Liu Chun

**Abstract**: With the wide application of deep reinforcement learning (DRL) techniques in complex fields such as autonomous driving, intelligent manufacturing, and smart healthcare, how to improve its security and robustness in dynamic and changeable environments has become a core issue in current research. Especially in the face of adversarial attacks, DRL may suffer serious performance degradation or even make potentially dangerous decisions, so it is crucial to ensure their stability in security-sensitive scenarios. In this paper, we first introduce the basic framework of DRL and analyze the main security challenges faced in complex and changing environments. In addition, this paper proposes an adversarial attack classification framework based on perturbation type and attack target and reviews the mainstream adversarial attack methods against DRL in detail, including various attack methods such as perturbation state space, action space, reward function and model space. To effectively counter the attacks, this paper systematically summarizes various current robustness training strategies, including adversarial training, competitive training, robust learning, adversarial detection, defense distillation and other related defense techniques, we also discuss the advantages and shortcomings of these methods in improving the robustness of DRL. Finally, this paper looks into the future research direction of DRL in adversarial environments, emphasizing the research needs in terms of improving generalization, reducing computational complexity, and enhancing scalability and explainability, aiming to provide valuable references and directions for researchers.



## **26. Crafting Imperceptible On-Manifold Adversarial Attacks for Tabular Data**

cs.LG

39 pages

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2507.10998v2) [paper-pdf](http://arxiv.org/pdf/2507.10998v2)

**Authors**: Zhipeng He, Alexander Stevens, Chun Ouyang, Johannes De Smedt, Alistair Barros, Catarina Moreira

**Abstract**: Adversarial attacks on tabular data present unique challenges due to the heterogeneous nature of mixed categorical and numerical features. Unlike images where pixel perturbations maintain visual similarity, tabular data lacks intuitive similarity metrics, making it difficult to define imperceptible modifications. Additionally, traditional gradient-based methods prioritise $\ell_p$-norm constraints, often producing adversarial examples that deviate from the original data distributions. To address this, we propose a latent-space perturbation framework using a mixed-input Variational Autoencoder (VAE) to generate statistically consistent adversarial examples. The proposed VAE integrates categorical embeddings and numerical features into a unified latent manifold, enabling perturbations that preserve statistical consistency. We introduce In-Distribution Success Rate (IDSR) to jointly evaluate attack effectiveness and distributional alignment. Evaluation across six publicly available datasets and three model architectures demonstrates that our method achieves substantially lower outlier rates and more consistent performance compared to traditional input-space attacks and other VAE-based methods adapted from image domain approaches, achieving substantially lower outlier rates and higher IDSR across six datasets and three model architectures. Our comprehensive analyses of hyperparameter sensitivity, sparsity control, and generative architecture demonstrate that the effectiveness of VAE-based attacks depends strongly on reconstruction quality and the availability of sufficient training data. When these conditions are met, the proposed framework achieves superior practical utility and stability compared with input-space methods. This work underscores the importance of maintaining on-manifold perturbations for generating realistic and robust adversarial examples in tabular domains.



## **27. Beyond Text: Multimodal Jailbreaking of Vision-Language and Audio Models through Perceptually Simple Transformations**

cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20223v1) [paper-pdf](http://arxiv.org/pdf/2510.20223v1)

**Authors**: Divyanshu Kumar, Shreyas Jena, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi

**Abstract**: Multimodal large language models (MLLMs) have achieved remarkable progress, yet remain critically vulnerable to adversarial attacks that exploit weaknesses in cross-modal processing. We present a systematic study of multimodal jailbreaks targeting both vision-language and audio-language models, showing that even simple perceptual transformations can reliably bypass state-of-the-art safety filters. Our evaluation spans 1,900 adversarial prompts across three high-risk safety categories harmful content, CBRN (Chemical, Biological, Radiological, Nuclear), and CSEM (Child Sexual Exploitation Material) tested against seven frontier models. We explore the effectiveness of attack techniques on MLLMs, including FigStep-Pro (visual keyword decomposition), Intelligent Masking (semantic obfuscation), and audio perturbations (Wave-Echo, Wave-Pitch, Wave-Speed). The results reveal severe vulnerabilities: models with almost perfect text-only safety (0\% ASR) suffer >75\% attack success under perceptually modified inputs, with FigStep-Pro achieving up to 89\% ASR in Llama-4 variants. Audio-based attacks further uncover provider-specific weaknesses, with even basic modality transfer yielding 25\% ASR for technical queries. These findings expose a critical gap between text-centric alignment and multimodal threats, demonstrating that current safeguards fail to generalize across cross-modal attacks. The accessibility of these attacks, which require minimal technical expertise, suggests that robust multimodal AI safety will require a paradigm shift toward broader semantic-level reasoning to mitigate possible risks.



## **28. TRUST: A Decentralized Framework for Auditing Large Language Model Reasoning**

cs.AI

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20188v1) [paper-pdf](http://arxiv.org/pdf/2510.20188v1)

**Authors**: Morris Yu-Chao Huang, Zhen Tan, Mohan Zhang, Pingzhi Li, Zhuo Zhang, Tianlong Chen

**Abstract**: Large Language Models generate complex reasoning chains that reveal their decision-making, yet verifying the faithfulness and harmlessness of these intermediate steps remains a critical unsolved problem. Existing auditing methods are centralized, opaque, and hard to scale, creating significant risks for deploying proprietary models in high-stakes domains. We identify four core challenges: (1) Robustness: Centralized auditors are single points of failure, prone to bias or attacks. (2) Scalability: Reasoning traces are too long for manual verification. (3) Opacity: Closed auditing undermines public trust. (4) Privacy: Exposing full reasoning risks model theft or distillation. We propose TRUST, a transparent, decentralized auditing framework that overcomes these limitations via: (1) A consensus mechanism among diverse auditors, guaranteeing correctness under up to $30\%$ malicious participants. (2) A hierarchical DAG decomposition of reasoning traces, enabling scalable, parallel auditing. (3) A blockchain ledger that records all verification decisions for public accountability. (4) Privacy-preserving segmentation, sharing only partial reasoning steps to protect proprietary logic. We provide theoretical guarantees for the security and economic incentives of the TRUST framework. Experiments across multiple LLMs (GPT-OSS, DeepSeek-r1, Qwen) and reasoning tasks (math, medical, science, humanities) show TRUST effectively detects reasoning flaws and remains robust against adversarial auditors. Our work pioneers decentralized AI auditing, offering a practical path toward safe and trustworthy LLM deployment.



## **29. Active Localization of Close-range Adversarial Acoustic Sources for Underwater Data Center Surveillance**

eess.SP

12 pages, V1

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20122v1) [paper-pdf](http://arxiv.org/pdf/2510.20122v1)

**Authors**: Adnan Abdullah, David Blow, Sara Rampazzi, Md Jahidul Islam

**Abstract**: Underwater data infrastructures offer natural cooling and enhanced physical security compared to terrestrial facilities, but are susceptible to acoustic injection attacks that can disrupt data integrity and availability. This work presents a comprehensive surveillance framework for localizing and tracking close-range adversarial acoustic sources targeting offshore infrastructures, particularly underwater data centers (UDCs). We propose a heterogeneous receiver configuration comprising a fixed hydrophone mounted on the facility and a mobile hydrophone deployed on a dedicated surveillance robot. While using enough arrays of static hydrophones covering large infrastructures is not feasible in practice, off-the-shelf approaches based on time difference of arrival (TDOA) and frequency difference of arrival (FDOA) filtering fail to generalize for this dynamic configuration. To address this, we formulate a Locus-Conditioned Maximum A-Posteriori (LC-MAP) scheme to generate acoustically informed and geometrically consistent priors, ensuring a physically plausible initial state for a joint TDOA-FDOA filtering. We integrate this into an unscented Kalman filtering (UKF) pipeline, which provides reliable convergence under nonlinearity and measurement noise. Extensive Monte Carlo analyses, Gazebo-based physics simulations, and field trials demonstrate that the proposed framework can reliably estimate the 3D position and velocity of an adversarial acoustic attack source in real time. It achieves sub-meter localization accuracy and over 90% success rates, with convergence times nearly halved compared to baseline methods. Overall, this study establishes a geometry-aware, real-time approach for acoustic threat localization, advancing autonomous surveillance capabilities of underwater infrastructures.



## **30. Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities**

cs.CL

Accepted to NAACL 2025 Main (Oral)

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2410.18469v5) [paper-pdf](http://arxiv.org/pdf/2410.18469v5)

**Authors**: Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao

**Abstract**: Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99\% ASR on GPT-3.5 and 49\% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety. Our code is available at: https://github.com/SunChungEn/ADV-LLM



## **31. Bridging Symmetry and Robustness: On the Role of Equivariance in Enhancing Adversarial Robustness**

cs.LG

Accepted for the proceedings of 39th Conference on Neural Information  Processing Systems (NeurIPS 2025)

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.16171v2) [paper-pdf](http://arxiv.org/pdf/2510.16171v2)

**Authors**: Longwei Wang, Ifrat Ikhtear Uddin, KC Santosh, Chaowei Zhang, Xiao Qin, Yang Zhou

**Abstract**: Adversarial examples reveal critical vulnerabilities in deep neural networks by exploiting their sensitivity to imperceptible input perturbations. While adversarial training remains the predominant defense strategy, it often incurs significant computational cost and may compromise clean-data accuracy. In this work, we investigate an architectural approach to adversarial robustness by embedding group-equivariant convolutions-specifically, rotation- and scale-equivariant layers-into standard convolutional neural networks (CNNs). These layers encode symmetry priors that align model behavior with structured transformations in the input space, promoting smoother decision boundaries and greater resilience to adversarial attacks. We propose and evaluate two symmetry-aware architectures: a parallel design that processes standard and equivariant features independently before fusion, and a cascaded design that applies equivariant operations sequentially. Theoretically, we demonstrate that such models reduce hypothesis space complexity, regularize gradients, and yield tighter certified robustness bounds under the CLEVER (Cross Lipschitz Extreme Value for nEtwork Robustness) framework. Empirically, our models consistently improve adversarial robustness and generalization across CIFAR-10, CIFAR-100, and CIFAR-10C under both FGSM and PGD attacks, without requiring adversarial training. These findings underscore the potential of symmetry-enforcing architectures as efficient and principled alternatives to data augmentation-based defenses.



## **32. JaiLIP: Jailbreaking Vision-Language Models via Loss Guided Image Perturbation**

cs.CV

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2509.21401v2) [paper-pdf](http://arxiv.org/pdf/2509.21401v2)

**Authors**: Md Jueal Mia, M. Hadi Amini

**Abstract**: Vision-Language Models (VLMs) have remarkable abilities in generating multimodal reasoning tasks. However, potential misuse or safety alignment concerns of VLMs have increased significantly due to different categories of attack vectors. Among various attack vectors, recent studies have demonstrated that image-based perturbations are particularly effective in generating harmful outputs. In the literature, many existing techniques have been proposed to jailbreak VLMs, leading to unstable performance and visible perturbations. In this study, we propose Jailbreaking with Loss-guided Image Perturbation (JaiLIP), a jailbreaking attack in the image space that minimizes a joint objective combining the mean squared error (MSE) loss between clean and adversarial image with the models harmful-output loss. We evaluate our proposed method on VLMs using standard toxicity metrics from Perspective API and Detoxify. Experimental results demonstrate that our method generates highly effective and imperceptible adversarial images, outperforming existing methods in producing toxicity. Moreover, we have evaluated our method in the transportation domain to demonstrate the attacks practicality beyond toxic text generation in specific domain. Our findings emphasize the practical challenges of image-based jailbreak attacks and the need for efficient defense mechanisms for VLMs.



## **33. Sharp Gaussian approximations for Decentralized Federated Learning**

stat.ML

Accepted as Spotlight, NeurIPS'25, Main Conference Track

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2505.08125v2) [paper-pdf](http://arxiv.org/pdf/2505.08125v2)

**Authors**: Soham Bonnerjee, Sayar Karmakar, Wei Biao Wu

**Abstract**: Federated Learning has gained traction in privacy-sensitive collaborative environments, with local SGD emerging as a key optimization method in decentralized settings. While its convergence properties are well-studied, asymptotic statistical guarantees beyond convergence remain limited. In this paper, we present two generalized Gaussian approximation results for local SGD and explore their implications. First, we prove a Berry-Esseen theorem for the final local SGD iterates, enabling valid multiplier bootstrap procedures. Second, motivated by robustness considerations, we introduce two distinct time-uniform Gaussian approximations for the entire trajectory of local SGD. The time-uniform approximations support Gaussian bootstrap-based tests for detecting adversarial attacks. Extensive simulations are provided to support our theoretical results.



## **34. QORE : Quantum Secure 5G/B5G Core**

cs.CR

23 pages

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19982v1) [paper-pdf](http://arxiv.org/pdf/2510.19982v1)

**Authors**: Vipin Rathi, Lakshya Chopra, Rudraksh Rawal, Nitin Rajput, Shiva Valia, Madhav Aggarwal, Aditya Gairola

**Abstract**: Quantum computing is reshaping the security landscape of modern telecommunications. The cryptographic foundations that secure todays 5G systems, including RSA, Elliptic Curve Cryptography (ECC), and Diffie-Hellman (DH), are all susceptible to attacks enabled by Shors algorithm. Protecting 5G networks against future quantum adversaries has therefore become an urgent engineering and research priority. In this paper we introduce QORE, a quantum-secure 5G and Beyond 5G (B5G) Core framework that provides a clear pathway for transitioning both the 5G Core Network Functions and User Equipment (UE) to Post-Quantum Cryptography (PQC). The framework uses the NIST-standardized lattice-based algorithms Module-Lattice Key Encapsulation Mechanism (ML-KEM) and Module-Lattice Digital Signature Algorithm (ML-DSA) and applies them across the 5G Service-Based Architecture (SBA). A Hybrid PQC (HPQC) configuration is also proposed, combining classical and quantum-safe primitives to maintain interoperability during migration. Experimental validation shows that ML-KEM achieves quantum security with minor performance overhead, meeting the low-latency and high-throughput requirements of carrier-grade 5G systems. The proposed roadmap aligns with ongoing 3GPP SA3 and SA5 study activities on the security and management of post-quantum networks as well as with NIST PQC standardization efforts, providing practical guidance for mitigating quantum-era risks while safeguarding long-term confidentiality and integrity of network data.



## **35. Towards Strong Certified Defense with Universal Asymmetric Randomization**

cs.LG

Accepted by CSF 2026, 39th IEEE Computer Security Foundations  Symposium

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19977v1) [paper-pdf](http://arxiv.org/pdf/2510.19977v1)

**Authors**: Hanbin Hong, Ashish Kundu, Ali Payani, Binghui Wang, Yuan Hong

**Abstract**: Randomized smoothing has become essential for achieving certified adversarial robustness in machine learning models. However, current methods primarily use isotropic noise distributions that are uniform across all data dimensions, such as image pixels, limiting the effectiveness of robustness certification by ignoring the heterogeneity of inputs and data dimensions. To address this limitation, we propose UCAN: a novel technique that \underline{U}niversally \underline{C}ertifies adversarial robustness with \underline{A}nisotropic \underline{N}oise. UCAN is designed to enhance any existing randomized smoothing method, transforming it from symmetric (isotropic) to asymmetric (anisotropic) noise distributions, thereby offering a more tailored defense against adversarial attacks. Our theoretical framework is versatile, supporting a wide array of noise distributions for certified robustness in different $\ell_p$-norms and applicable to any arbitrary classifier by guaranteeing the classifier's prediction over perturbed inputs with provable robustness bounds through tailored noise injection. Additionally, we develop a novel framework equipped with three exemplary noise parameter generators (NPGs) to optimally fine-tune the anisotropic noise parameters for different data dimensions, allowing for pursuing different levels of robustness enhancements in practice.Empirical evaluations underscore the significant leap in UCAN's performance over existing state-of-the-art methods, demonstrating up to $182.6\%$ improvement in certified accuracy at large certified radii on MNIST, CIFAR10, and ImageNet datasets.\footnote{Code is anonymously available at \href{https://github.com/youbin2014/UCAN/}{https://github.com/youbin2014/UCAN/}}



## **36. Q-RAN: Quantum-Resilient O-RAN Architecture**

cs.CR

23 pages

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19968v1) [paper-pdf](http://arxiv.org/pdf/2510.19968v1)

**Authors**: Vipin Rathi, Lakshya Chopra, Madhav Agarwal, Nitin Rajput, Kriish Sharma, Sushant Mundepi, Shivam Gangwar, Rudraksh Rawal, Jishan

**Abstract**: The telecommunications industry faces a dual transformation: the architectural shift toward Open Radio Access Networks (O-RAN) and the emerging threat from quantum computing. O-RAN disaggregated, multi-vendor architecture creates a larger attack surface vulnerable to crypt-analytically relevant quantum computers(CRQCs) that will break current public key cryptography. The Harvest Now, Decrypt Later (HNDL) attack strategy makes this threat immediate, as adversaries can intercept encrypted data today for future decryption. This paper presents Q-RAN, a comprehensive quantum-resistant security framework for O-RAN networks using NIST-standardized Post-Quantum Cryptography (PQC). We detail the implementation of ML-KEM (FIPS 203) and ML-DSA (FIPS 204), integrated with Quantum Random Number Generators (QRNG) for cryptographic entropy. The solution deploys PQ-IPsec, PQ-DTLS, and PQ-mTLS protocols across all O-RAN interfaces, anchored by a centralized Post-Quantum Certificate Authority (PQ-CA) within the SMO framework. This work provides a complete roadmap for securing disaggregated O-RAN ecosystems against quantum adversaries.



## **37. Unlearned but Not Forgotten: Data Extraction after Exact Unlearning in LLM**

cs.LG

Accepted by Neurips 2025

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2505.24379v3) [paper-pdf](http://arxiv.org/pdf/2505.24379v3)

**Authors**: Xiaoyu Wu, Yifei Pang, Terrance Liu, Zhiwei Steven Wu

**Abstract**: Large Language Models are typically trained on datasets collected from the web, which may inadvertently contain harmful or sensitive personal information. To address growing privacy concerns, unlearning methods have been proposed to remove the influence of specific data from trained models. Of these, exact unlearning -- which retrains the model from scratch without the target data -- is widely regarded the gold standard for mitigating privacy risks in deployment. In this paper, we revisit this assumption in a practical deployment setting where both the pre- and post-unlearning logits API are exposed, such as in open-weight scenarios. Targeting this setting, we introduce a novel data extraction attack that leverages signals from the pre-unlearning model to guide the post-unlearning model, uncovering patterns that reflect the removed data distribution. Combining model guidance with a token filtering strategy, our attack significantly improves extraction success rates -- doubling performance in some cases -- across common benchmarks such as MUSE, TOFU, and WMDP. Furthermore, we demonstrate our attack's effectiveness on a simulated medical diagnosis dataset to highlight real-world privacy risks associated with exact unlearning. In light of our findings, which suggest that unlearning may, in a contradictory way, increase the risk of privacy leakage during real-world deployments, we advocate for evaluation of unlearning methods to consider broader threat models that account not only for post-unlearning models but also for adversarial access to prior checkpoints. Code is publicly available at: https://github.com/Nicholas0228/unlearned_data_extraction_llm.



## **38. Are Modern Speech Enhancement Systems Vulnerable to Adversarial Attacks?**

eess.AS

Copyright 2026 IEEE. Personal use of this material is permitted.  Permission from IEEE must be obtained for all other uses, in any current or  future media, including reprinting/republishing this material for advertising  or promotional purposes, creating new collective works, for resale or  redistribution to servers or lists, or reuse of any copyrighted component of  this work in other works

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2509.21087v2) [paper-pdf](http://arxiv.org/pdf/2509.21087v2)

**Authors**: Rostislav Makarov, Lea Schönherr, Timo Gerkmann

**Abstract**: Machine learning approaches for speech enhancement are becoming increasingly expressive, enabling ever more powerful modifications of input signals. In this paper, we demonstrate that this expressiveness introduces a vulnerability: advanced speech enhancement models can be susceptible to adversarial attacks. Specifically, we show that adversarial noise, carefully crafted and psychoacoustically masked by the original input, can be injected such that the enhanced speech output conveys an entirely different semantic meaning. We experimentally verify that contemporary predictive speech enhancement models can indeed be manipulated in this way. Furthermore, we highlight that diffusion models with stochastic samplers exhibit inherent robustness to such adversarial attacks by design.



## **39. On Scaling LT-Coded Blockchains in Heterogeneous Networks and their Vulnerabilities to DoS Threats**

cs.IT

To appear in Future Generation Computer Systems, 2025. This is an  extended version of a shorter version that has appeared in IEEE ICC 2024

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2402.05620v3) [paper-pdf](http://arxiv.org/pdf/2402.05620v3)

**Authors**: Harikrishnan K., J. Harshan, Anwitaman Datta

**Abstract**: Coded blockchains have acquired prominence as a promising solution to reduce storage costs and facilitate scalability. Within this class, Luby Transform (LT) coded blockchains are an appealing choice for scalability owing to the availability of a wide range of low-complexity decoders. In the first part of this work, we identify that traditional LT decoders like Belief Propagation and On-the-Fly Gaussian Elimination may not be optimal for heterogeneous networks with nodes that have varying computational and download capabilities. To address this, we introduce a family of hybrid decoders for LT codes and propose optimal operating regimes for them to recover the blockchain at the lowest decoding cost. While LT coded blockchain architecture has been studied from the aspects of storage savings and scalability, not much is known in terms of its security vulnerabilities. Pointing at this research gap, in the second part, we present novel denial-of-service threats on LT coded blockchains that target nodes with specific decoding capabilities, preventing them from joining the network. Our proposed threats are non-oblivious in nature, wherein adversaries gain access to the archived blocks, and choose to execute their attack on a subset of them based on underlying coding scheme. We show that our optimized threats can achieve the same level of damage as that of blind attacks, however, with limited amount of resources. Overall, this is the first work of its kind that opens up new questions on designing coded blockchains to jointly provide storage savings, scalability and also resilience to optimized threats.



## **40. Exploring the Effect of DNN Depth on Adversarial Attacks in Network Intrusion Detection Systems**

cs.CR

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19761v1) [paper-pdf](http://arxiv.org/pdf/2510.19761v1)

**Authors**: Mohamed ElShehaby, Ashraf Matrawy

**Abstract**: Adversarial attacks pose significant challenges to Machine Learning (ML) systems and especially Deep Neural Networks (DNNs) by subtly manipulating inputs to induce incorrect predictions. This paper investigates whether increasing the layer depth of deep neural networks affects their robustness against adversarial attacks in the Network Intrusion Detection System (NIDS) domain. We compare the adversarial robustness of various deep neural networks across both \ac{NIDS} and computer vision domains (the latter being widely used in adversarial attack experiments). Our experimental results reveal that in the NIDS domain, adding more layers does not necessarily improve their performance, yet it may actually significantly degrade their robustness against adversarial attacks. Conversely, in the computer vision domain, adding more layers exhibits a more modest impact on robustness. These findings can guide the development of robust neural networks for (NIDS) applications and highlight the unique characteristics of network security domains within the (ML) landscape.



## **41. Explainable Face Presentation Attack Detection via Ensemble-CAM**

cs.CV

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19695v1) [paper-pdf](http://arxiv.org/pdf/2510.19695v1)

**Authors**: Rashik Shadman, M G Sarwar Murshed, Faraz Hussain

**Abstract**: Presentation attacks represent a critical security threat where adversaries use fake biometric data, such as face, fingerprint, or iris images, to gain unauthorized access to protected systems. Various presentation attack detection (PAD) systems have been designed leveraging deep learning (DL) models to mitigate this type of threat. Despite their effectiveness, most of the DL models function as black boxes - their decisions are opaque to their users. The purpose of explainability techniques is to provide detailed information about the reason behind the behavior or decision of DL models. In particular, visual explanation is necessary to better understand the decisions or predictions of DL-based PAD systems and determine the key regions due to which a biometric image is considered real or fake by the system. In this work, a novel technique, Ensemble-CAM, is proposed for providing visual explanations for the decisions made by deep learning-based face PAD systems. Our goal is to improve DL-based face PAD systems by providing a better understanding of their behavior. Our provided visual explanations will enhance the transparency and trustworthiness of DL-based face PAD systems.



## **42. Style Attack Disguise: When Fonts Become a Camouflage for Adversarial Intent**

cs.CL

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19641v1) [paper-pdf](http://arxiv.org/pdf/2510.19641v1)

**Authors**: Yangshijie Zhang, Xinda Wang, Jialin Liu, Wenqiang Wang, Zhicong Ma, Xingxing Jia

**Abstract**: With social media growth, users employ stylistic fonts and font-like emoji to express individuality, creating visually appealing text that remains human-readable. However, these fonts introduce hidden vulnerabilities in NLP models: while humans easily read stylistic text, models process these characters as distinct tokens, causing interference. We identify this human-model perception gap and propose a style-based attack, Style Attack Disguise (SAD). We design two sizes: light for query efficiency and strong for superior attack performance. Experiments on sentiment classification and machine translation across traditional models, LLMs, and commercial services demonstrate SAD's strong attack performance. We also show SAD's potential threats to multimodal tasks including text-to-image and text-to-speech generation.



## **43. Can You Trust What You See? Alpha Channel No-Box Attacks on Video Object Detection**

cs.CV

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19574v1) [paper-pdf](http://arxiv.org/pdf/2510.19574v1)

**Authors**: Ariana Yi, Ce Zhou, Liyang Xiao, Qiben Yan

**Abstract**: As object detection models are increasingly deployed in cyber-physical systems such as autonomous vehicles (AVs) and surveillance platforms, ensuring their security against adversarial threats is essential. While prior work has explored adversarial attacks in the image domain, those attacks in the video domain remain largely unexamined, especially in the no-box setting. In this paper, we present {\alpha}-Cloak, the first no-box adversarial attack on object detectors that operates entirely through the alpha channel of RGBA videos. {\alpha}-Cloak exploits the alpha channel to fuse a malicious target video with a benign video, resulting in a fused video that appears innocuous to human viewers but consistently fools object detectors. Our attack requires no access to model architecture, parameters, or outputs, and introduces no perceptible artifacts. We systematically study the support for alpha channels across common video formats and playback applications, and design a fusion algorithm that ensures visual stealth and compatibility. We evaluate {\alpha}-Cloak on five state-of-the-art object detectors, a vision-language model, and a multi-modal large language model (Gemini-2.0-Flash), demonstrating a 100% attack success rate across all scenarios. Our findings reveal a previously unexplored vulnerability in video-based perception systems, highlighting the urgent need for defenses that account for the alpha channel in adversarial settings.



## **44. FPT-Noise: Dynamic Scene-Aware Counterattack for Test-Time Adversarial Defense in Vision-Language Models**

cs.CR

11pages,4figures

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.20856v1) [paper-pdf](http://arxiv.org/pdf/2510.20856v1)

**Authors**: Jia Deng, Jin Li, Zhenhua Zhao, Shaowei Wang

**Abstract**: Vision-Language Models (VLMs), such as CLIP, have demonstrated remarkable zero-shot generalizability across diverse downstream tasks. However, recent studies have revealed that VLMs, including CLIP, are highly vulnerable to adversarial attacks, particularly on their visual modality. Traditional methods for improving adversarial robustness, such as adversarial training, involve extensive retraining and can be computationally expensive. In this paper, we propose a new Test-Time defense: Feature Perception Threshold Counterattack Noise (FPT-Noise), which enhances the adversarial robustness of CLIP without costly fine-tuning. Our core contributions are threefold: First, we introduce a Dynamic Feature Modulator that dynamically generate an image-specific and attack-adaptive noise intensity parameter. Second, We reanalyzed the image features of CLIP. When images are exposed to different levels of noise, clean images and adversarial images exhibit distinct rates of feature change. We established a feature perception threshold to distinguish clean images from attacked ones. Finally, we integrate a Scene-Aware Regulation guided by a stability threshold and leverage Test-Time Transformation Ensembling (TTE) to further mitigate the impact of residual noise and enhance robustness.Extensive experimentation has demonstrated that FPT-Noise significantly outperforms existing Test-Time defense methods, boosting average robust accuracy from 0.07% to 56.86% under AutoAttack while maintaining high performance on clean images (-1.1%). The code will be made public following the publication of the study. The code will be made public following the publication of the study.



## **45. A New Type of Adversarial Examples**

cs.LG

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19347v1) [paper-pdf](http://arxiv.org/pdf/2510.19347v1)

**Authors**: Xingyang Nie, Guojie Xiao, Su Pan, Biao Wang, Huilin Ge, Tao Fang

**Abstract**: Most machine learning models are vulnerable to adversarial examples, which poses security concerns on these models. Adversarial examples are crafted by applying subtle but intentionally worst-case modifications to examples from the dataset, leading the model to output a different answer from the original example. In this paper, adversarial examples are formed in an exactly opposite manner, which are significantly different from the original examples but result in the same answer. We propose a novel set of algorithms to produce such adversarial examples, including the negative iterative fast gradient sign method (NI-FGSM) and the negative iterative fast gradient method (NI-FGM), along with their momentum variants: the negative momentum iterative fast gradient sign method (NMI-FGSM) and the negative momentum iterative fast gradient method (NMI-FGM). Adversarial examples constructed by these methods could be used to perform an attack on machine learning systems in certain occasions. Moreover, our results show that the adversarial examples are not merely distributed in the neighbourhood of the examples from the dataset; instead, they are distributed extensively in the sample space.



## **46. Collaborative penetration testing suite for emerging generative AI algorithms**

cs.CR

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19303v1) [paper-pdf](http://arxiv.org/pdf/2510.19303v1)

**Authors**: Petar Radanliev

**Abstract**: Problem Space: AI Vulnerabilities and Quantum Threats Generative AI vulnerabilities: model inversion, data poisoning, adversarial inputs. Quantum threats Shor Algorithm breaking RSA ECC encryption. Challenge Secure generative AI models against classical and quantum cyberattacks. Proposed Solution Collaborative Penetration Testing Suite Five Integrated Components: DAST SAST OWASP ZAP, Burp Suite, SonarQube, Fortify. IAST Contrast Assess integrated with CI CD pipeline. Blockchain Logging Hyperledger Fabric for tamper-proof logs. Quantum Cryptography Lattice based RLWE protocols. AI Red Team Simulations Adversarial ML & Quantum-assisted attacks. Integration Layer: Unified workflow for AI, cybersecurity, and quantum experts. Key Results 300+ vulnerabilities identified across test environments. 70% reduction in high-severity issues within 2 weeks. 90% resolution efficiency for blockchain-logged vulnerabilities. Quantum-resistant cryptography maintained 100% integrity in tests. Outcome: Quantum AI Security Protocol integrating Blockchain Quantum Cryptography AI Red Teaming.



## **47. Adversarial Attacks on LiDAR-Based Tracking Across Road Users: Robustness Evaluation and Target-Aware Black-Box Method**

cs.CV

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2410.20893v3) [paper-pdf](http://arxiv.org/pdf/2410.20893v3)

**Authors**: Shengjing Tian, Xiantong Zhao, Yuhao Bian, Yinan Han, Bin Liu

**Abstract**: In this study, we delve into the robustness of neural network-based LiDAR point cloud tracking models under adversarial attacks, a critical aspect often overlooked in favor of performance enhancement. These models, despite incorporating advanced architectures like Transformer or Bird's Eye View (BEV), tend to neglect robustness in the face of challenges such as adversarial attacks, domain shifts, or data corruption. We instead focus on the robustness of the tracking models under the threat of adversarial attacks. We begin by establishing a unified framework for conducting adversarial attacks within the context of 3D object tracking, which allows us to thoroughly investigate both white-box and black-box attack strategies. For white-box attacks, we tailor specific loss functions to accommodate various tracking paradigms and extend existing methods such as FGSM, C\&W, and PGD to the point cloud domain. In addressing black-box attack scenarios, we introduce a novel transfer-based approach, the Target-aware Perturbation Generation (TAPG) algorithm, with the dual objectives of achieving high attack performance and maintaining low perceptibility. This method employs a heuristic strategy to enforce sparse attack constraints and utilizes random sub-vector factorization to bolster transferability. Our experimental findings reveal a significant vulnerability in advanced tracking methods when subjected to both black-box and white-box attacks, underscoring the necessity for incorporating robustness against adversarial attacks into the design of LiDAR point cloud tracking models. Notably, compared to existing methods, the TAPG also strikes an optimal balance between the effectiveness of the attack and the concealment of the perturbations.



## **48. Defending Against Prompt Injection with DataFilter**

cs.CR

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19207v1) [paper-pdf](http://arxiv.org/pdf/2510.19207v1)

**Authors**: Yizhu Wang, Sizhe Chen, Raghad Alkhudair, Basel Alomair, David Wagner

**Abstract**: When large language model (LLM) agents are increasingly deployed to automate tasks and interact with untrusted external data, prompt injection emerges as a significant security threat. By injecting malicious instructions into the data that LLMs access, an attacker can arbitrarily override the original user task and redirect the agent toward unintended, potentially harmful actions. Existing defenses either require access to model weights (fine-tuning), incur substantial utility loss (detection-based), or demand non-trivial system redesign (system-level). Motivated by this, we propose DataFilter, a test-time model-agnostic defense that removes malicious instructions from the data before it reaches the backend LLM. DataFilter is trained with supervised fine-tuning on simulated injections and leverages both the user's instruction and the data to selectively strip adversarial content while preserving benign information. Across multiple benchmarks, DataFilter consistently reduces the prompt injection attack success rates to near zero while maintaining the LLMs' utility. DataFilter delivers strong security, high utility, and plug-and-play deployment, making it a strong practical defense to secure black-box commercial LLMs against prompt injection. Our DataFilter model is released at https://huggingface.co/JoyYizhu/DataFilter for immediate use, with the code to reproduce our results at https://github.com/yizhu-joy/DataFilter.



## **49. FeatureFool: Zero-Query Fooling of Video Models via Feature Map**

cs.CV

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.18362v2) [paper-pdf](http://arxiv.org/pdf/2510.18362v2)

**Authors**: Duoxun Tang, Xi Xiao, Guangwu Hu, Kangkang Sun, Xiao Yang, Dongyang Chen, Qing Li, Yongjie Yin, Jiyao Wang

**Abstract**: The vulnerability of deep neural networks (DNNs) has been preliminarily verified. Existing black-box adversarial attacks usually require multi-round interaction with the model and consume numerous queries, which is impractical in the real-world and hard to scale to recently emerged Video-LLMs. Moreover, no attack in the video domain directly leverages feature maps to shift the clean-video feature space. We therefore propose FeatureFool, a stealthy, video-domain, zero-query black-box attack that utilizes information extracted from a DNN to alter the feature space of clean videos. Unlike query-based methods that rely on iterative interaction, FeatureFool performs a zero-query attack by directly exploiting DNN-extracted information. This efficient approach is unprecedented in the video domain. Experiments show that FeatureFool achieves an attack success rate above 70\% against traditional video classifiers without any queries. Benefiting from the transferability of the feature map, it can also craft harmful content and bypass Video-LLM recognition. Additionally, adversarial videos generated by FeatureFool exhibit high quality in terms of SSIM, PSNR, and Temporal-Inconsistency, making the attack barely perceptible. This paper may contain violent or explicit content.



## **50. The Black Tuesday Attack: how to crash the stock market with adversarial examples to financial forecasting models**

cs.CR

15 pages, 2 figures

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18990v1) [paper-pdf](http://arxiv.org/pdf/2510.18990v1)

**Authors**: Thomas Hofweber, Jefrey Bergl, Ian Reyes, Amir Sadovnik

**Abstract**: We investigate and defend the possibility of causing a stock market crash via small manipulations of individual stock values that together realize an adversarial example to financial forecasting models, causing these models to make the self-fulfilling prediction of a crash. Such a crash triggered by an adversarial example would likely be hard to detect, since the model's predictions would be accurate and the interventions that would cause it are minor. This possibility is a major risk to financial stability and an opportunity for hostile actors to cause great economic damage to an adversary. This threat also exists against individual stocks and the corresponding valuation of individual companies. We outline how such an attack might proceed, what its theoretical basis is, how it can be directed towards a whole economy or an individual company, and how one might defend against it. We conclude that this threat is vastly underappreciated and requires urgent research on how to defend against it.



