# Latest Adversarial Attack Papers
**update at 2025-09-10 10:58:14**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_CN.md)

[Attacks and Defenses in Large language Models](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM.md)

## **1. Robust Adaptation of Large Multimodal Models for Retrieval Augmented Hateful Meme Detection**

cs.CL

EMNLP 2025 Main

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2502.13061v3) [paper-pdf](http://arxiv.org/pdf/2502.13061v3)

**Authors**: Jingbiao Mei, Jinghong Chen, Guangyu Yang, Weizhe Lin, Bill Byrne

**Abstract**: Hateful memes have become a significant concern on the Internet, necessitating robust automated detection systems. While Large Multimodal Models (LMMs) have shown promise in hateful meme detection, they face notable challenges like sub-optimal performance and limited out-of-domain generalization capabilities. Recent studies further reveal the limitations of both supervised fine-tuning (SFT) and in-context learning when applied to LMMs in this setting. To address these issues, we propose a robust adaptation framework for hateful meme detection that enhances in-domain accuracy and cross-domain generalization while preserving the general vision-language capabilities of LMMs. Analysis reveals that our approach achieves improved robustness under adversarial attacks compared to SFT models. Experiments on six meme classification datasets show that our approach achieves state-of-the-art performance, outperforming larger agentic systems. Moreover, our method generates higher-quality rationales for explaining hateful content compared to standard SFT, enhancing model interpretability. Code available at https://github.com/JingbiaoMei/RGCL



## **2. Empirical Security Analysis of Software-based Fault Isolation through Controlled Fault Injection**

cs.CR

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07757v1) [paper-pdf](http://arxiv.org/pdf/2509.07757v1)

**Authors**: Nils Bars, Lukas Bernhard, Moritz Schloegel, Thorsten Holz

**Abstract**: We use browsers daily to access all sorts of information. Because browsers routinely process scripts, media, and executable code from unknown sources, they form a critical security boundary between users and adversaries. A common attack vector is JavaScript, which exposes a large attack surface due to the sheer complexity of modern JavaScript engines. To mitigate these threats, modern engines increasingly adopt software-based fault isolation (SFI). A prominent example is Google's V8 heap sandbox, which represents the most widely deployed SFI mechanism, protecting billions of users across all Chromium-based browsers and countless applications built on Node.js and Electron. The heap sandbox splits the address space into two parts: one part containing trusted, security-sensitive metadata, and a sandboxed heap containing memory accessible to untrusted code. On a technical level, the sandbox enforces isolation by removing raw pointers and using translation tables to resolve references to trusted objects. Consequently, an attacker cannot corrupt trusted data even with full control of the sandboxed data, unless there is a bug in how code handles data from the sandboxed heap. Despite their widespread use, such SFI mechanisms have seen little security testing.   In this work, we propose a new testing technique that models the security boundary of modern SFI implementations. Following the SFI threat model, we assume a powerful attacker who fully controls the sandbox's memory. We implement this by instrumenting memory loads originating in the trusted domain and accessing untrusted, attacker-controlled sandbox memory. We then inject faults into the loaded data, aiming to trigger memory corruption in the trusted domain. In a comprehensive evaluation, we identify 19 security bugs in V8 that enable an attacker to bypass the sandbox.



## **3. Spectral Masking and Interpolation Attack (SMIA): A Black-box Adversarial Attack against Voice Authentication and Anti-Spoofing Systems**

cs.SD

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07677v1) [paper-pdf](http://arxiv.org/pdf/2509.07677v1)

**Authors**: Kamel Kamel, Hridoy Sankar Dutta, Keshav Sood, Sunil Aryal

**Abstract**: Voice Authentication Systems (VAS) use unique vocal characteristics for verification. They are increasingly integrated into high-security sectors such as banking and healthcare. Despite their improvements using deep learning, they face severe vulnerabilities from sophisticated threats like deepfakes and adversarial attacks. The emergence of realistic voice cloning complicates detection, as systems struggle to distinguish authentic from synthetic audio. While anti-spoofing countermeasures (CMs) exist to mitigate these risks, many rely on static detection models that can be bypassed by novel adversarial methods, leaving a critical security gap. To demonstrate this vulnerability, we propose the Spectral Masking and Interpolation Attack (SMIA), a novel method that strategically manipulates inaudible frequency regions of AI-generated audio. By altering the voice in imperceptible zones to the human ear, SMIA creates adversarial samples that sound authentic while deceiving CMs. We conducted a comprehensive evaluation of our attack against state-of-the-art (SOTA) models across multiple tasks, under simulated real-world conditions. SMIA achieved a strong attack success rate (ASR) of at least 82% against combined VAS/CM systems, at least 97.5% against standalone speaker verification systems, and 100% against countermeasures. These findings conclusively demonstrate that current security postures are insufficient against adaptive adversarial attacks. This work highlights the urgent need for a paradigm shift toward next-generation defenses that employ dynamic, context-aware frameworks capable of evolving with the threat landscape.



## **4. Transferable Direct Prompt Injection via Activation-Guided MCMC Sampling**

cs.AI

Accepted to EMNLP 2025

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07617v1) [paper-pdf](http://arxiv.org/pdf/2509.07617v1)

**Authors**: Minghui Li, Hao Zhang, Yechao Zhang, Wei Wan, Shengshan Hu, pei Xiaobing, Jing Wang

**Abstract**: Direct Prompt Injection (DPI) attacks pose a critical security threat to Large Language Models (LLMs) due to their low barrier of execution and high potential damage. To address the impracticality of existing white-box/gray-box methods and the poor transferability of black-box methods, we propose an activations-guided prompt injection attack framework. We first construct an Energy-based Model (EBM) using activations from a surrogate model to evaluate the quality of adversarial prompts. Guided by the trained EBM, we employ the token-level Markov Chain Monte Carlo (MCMC) sampling to adaptively optimize adversarial prompts, thereby enabling gradient-free black-box attacks. Experimental results demonstrate our superior cross-model transferability, achieving 49.6% attack success rate (ASR) across five mainstream LLMs and 34.6% improvement over human-crafted prompts, and maintaining 36.6% ASR on unseen task scenarios. Interpretability analysis reveals a correlation between activations and attack effectiveness, highlighting the critical role of semantic patterns in transferable vulnerability exploitation.



## **5. A Survey of Threats Against Voice Authentication and Anti-Spoofing Systems**

cs.CR

This paper is submitted to the Computer Science Review

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2508.16843v3) [paper-pdf](http://arxiv.org/pdf/2508.16843v3)

**Authors**: Kamel Kamel, Keshav Sood, Hridoy Sankar Dutta, Sunil Aryal

**Abstract**: Voice authentication has undergone significant changes from traditional systems that relied on handcrafted acoustic features to deep learning models that can extract robust speaker embeddings. This advancement has expanded its applications across finance, smart devices, law enforcement, and beyond. However, as adoption has grown, so have the threats. This survey presents a comprehensive review of the modern threat landscape targeting Voice Authentication Systems (VAS) and Anti-Spoofing Countermeasures (CMs), including data poisoning, adversarial, deepfake, and adversarial spoofing attacks. We chronologically trace the development of voice authentication and examine how vulnerabilities have evolved in tandem with technological advancements. For each category of attack, we summarize methodologies, highlight commonly used datasets, compare performance and limitations, and organize existing literature using widely accepted taxonomies. By highlighting emerging risks and open challenges, this survey aims to support the development of more secure and resilient voice authentication systems.



## **6. Generating Transferrable Adversarial Examples via Local Mixing and Logits Optimization for Remote Sensing Object Recognition**

cs.CV

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07495v1) [paper-pdf](http://arxiv.org/pdf/2509.07495v1)

**Authors**: Chun Liu, Hailong Wang, Bingqian Zhu, Panpan Ding, Zheng Zheng, Tao Xu, Zhigang Han, Jiayao Wang

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to adversarial attacks, posing significant security threats to their deployment in remote sensing applications. Research on adversarial attacks not only reveals model vulnerabilities but also provides critical insights for enhancing robustness. Although current mixing-based strategies have been proposed to increase the transferability of adversarial examples, they either perform global blending or directly exchange a region in the images, which may destroy global semantic features and mislead the optimization of adversarial examples. Furthermore, their reliance on cross-entropy loss for perturbation optimization leads to gradient diminishing during iterative updates, compromising adversarial example quality. To address these limitations, we focus on non-targeted attacks and propose a novel framework via local mixing and logits optimization. First, we present a local mixing strategy to generate diverse yet semantically consistent inputs. Different from MixUp, which globally blends two images, and MixCut, which stitches images together, our method merely blends local regions to preserve global semantic information. Second, we adapt the logit loss from targeted attacks to non-targeted scenarios, mitigating the gradient vanishing problem of cross-entropy loss. Third, a perturbation smoothing loss is applied to suppress high-frequency noise and enhance transferability. Extensive experiments on FGSCR-42 and MTARSI datasets demonstrate superior performance over 12 state-of-the-art methods across 6 surrogate models. Notably, with ResNet as the surrogate on MTARSI, our method achieves a 17.28% average improvement in black-box attack success rate.



## **7. Texture- and Shape-based Adversarial Attacks for Overhead Image Vehicle Detection**

cs.CV

This version corresponds to the paper accepted for presentation at  ICIP 2025

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2412.16358v2) [paper-pdf](http://arxiv.org/pdf/2412.16358v2)

**Authors**: Mikael Yeghiazaryan, Sai Abhishek Siddhartha Namburu, Emily Kim, Stanislav Panev, Celso de Melo, Fernando De la Torre, Jessica K. Hodgins

**Abstract**: Detecting vehicles in aerial images is difficult due to complex backgrounds, small object sizes, shadows, and occlusions. Although recent deep learning advancements have improved object detection, these models remain susceptible to adversarial attacks (AAs), challenging their reliability. Traditional AA strategies often ignore practical implementation constraints. Our work proposes realistic and practical constraints on texture (lowering resolution, limiting modified areas, and color ranges) and analyzes the impact of shape modifications on attack performance. We conducted extensive experiments with three object detector architectures, demonstrating the performance-practicality trade-off: more practical modifications tend to be less effective, and vice versa. We release both code and data to support reproducibility at https://github.com/humansensinglab/texture-shape-adversarial-attacks.



## **8. Prepared for the Worst: A Learning-Based Adversarial Attack for Resilience Analysis of the ICP Algorithm**

cs.RO

9 pages (6 content, 1 reference, 2 appendix). 7 figures, accepted to  2025 IEEE International Conference on Robotics and Automation (ICRA)

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2403.05666v3) [paper-pdf](http://arxiv.org/pdf/2403.05666v3)

**Authors**: Ziyu Zhang, Johann Laconte, Daniil Lisus, Timothy D. Barfoot

**Abstract**: This paper presents a novel method for assessing the resilience of the ICP algorithm via learning-based, worst-case attacks on lidar point clouds. For safety-critical applications such as autonomous navigation, ensuring the resilience of algorithms before deployments is crucial. The ICP algorithm is the standard for lidar-based localization, but its accuracy can be greatly affected by corrupted measurements from various sources, including occlusions, adverse weather, or mechanical sensor issues. Unfortunately, the complex and iterative nature of ICP makes assessing its resilience to corruption challenging. While there have been efforts to create challenging datasets and develop simulations to evaluate the resilience of ICP, our method focuses on finding the maximum possible ICP error that can arise from corrupted measurements at a location. We demonstrate that our perturbation-based adversarial attacks can be used pre-deployment to identify locations on a map where ICP is particularly vulnerable to corruptions in the measurements. With such information, autonomous robots can take safer paths when deployed, to mitigate against their measurements being corrupted. The proposed attack outperforms baselines more than 88% of the time across a wide range of scenarios.



## **9. When Fine-Tuning is Not Enough: Lessons from HSAD on Hybrid and Adversarial Audio Spoof Detection**

cs.SD

13 pages, 11 figures.This work has been submitted to the IEEE for  possible publication

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2509.07323v1) [paper-pdf](http://arxiv.org/pdf/2509.07323v1)

**Authors**: Bin Hu, Kunyang Huang, Daehan Kwak, Meng Xu, Kuan Huang

**Abstract**: The rapid advancement of AI has enabled highly realistic speech synthesis and voice cloning, posing serious risks to voice authentication, smart assistants, and telecom security. While most prior work frames spoof detection as a binary task, real-world attacks often involve hybrid utterances that mix genuine and synthetic speech, making detection substantially more challenging. To address this gap, we introduce the Hybrid Spoofed Audio Dataset (HSAD), a benchmark containing 1,248 clean and 41,044 degraded utterances across four classes: human, cloned, zero-shot AI-generated, and hybrid audio. Each sample is annotated with spoofing method, speaker identity, and degradation metadata to enable fine-grained analysis. We evaluate six transformer-based models, including spectrogram encoders (MIT-AST, MattyB95-AST) and self-supervised waveform models (Wav2Vec2, HuBERT). Results reveal critical lessons: pretrained models overgeneralize and collapse under hybrid conditions; spoof-specific fine-tuning improves separability but struggles with unseen compositions; and dataset-specific adaptation on HSAD yields large performance gains (AST greater than 97 percent and F1 score is approximately 99 percent), though residual errors persist for complex hybrids. These findings demonstrate that fine-tuning alone is not sufficient-robust hybrid-aware benchmarks like HSAD are essential to expose calibration failures, model biases, and factors affecting spoof detection in adversarial environments. HSAD thus provides both a dataset and an analytic framework for building resilient and trustworthy voice authentication systems.



## **10. GRADA: Graph-based Reranking against Adversarial Documents Attack**

cs.IR

**SubmitDate**: 2025-09-09    [abs](http://arxiv.org/abs/2505.07546v2) [paper-pdf](http://arxiv.org/pdf/2505.07546v2)

**Authors**: Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xuanli He, Pasquale Minervini, Youcheng Sun, Qiongkai Xu

**Abstract**: Retrieval Augmented Generation (RAG) frameworks improve the accuracy of large language models (LLMs) by integrating external knowledge from retrieved documents, thereby overcoming the limitations of models' static intrinsic knowledge. However, these systems are susceptible to adversarial attacks that manipulate the retrieval process by introducing documents that are adversarial yet semantically similar to the query. Notably, while these adversarial documents resemble the query, they exhibit weak similarity to benign documents in the retrieval set. Thus, we propose a simple yet effective Graph-based Reranking against Adversarial Document Attacks (GRADA) framework aiming at preserving retrieval quality while significantly reducing the success of adversaries. Our study evaluates the effectiveness of our approach through experiments conducted on five LLMs: GPT-3.5-Turbo, GPT-4o, Llama3.1-8b, Llama3.1-70b, and Qwen2.5-7b. We use three datasets to assess performance, with results from the Natural Questions dataset demonstrating up to an 80% reduction in attack success rates while maintaining minimal loss in accuracy.



## **11. Personalized Attacks of Social Engineering in Multi-turn Conversations: LLM Agents for Simulation and Detection**

cs.CR

Accepted as a paper at COLM 2025 Workshop on AI Agents: Capabilities  and Safety

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2503.15552v2) [paper-pdf](http://arxiv.org/pdf/2503.15552v2)

**Authors**: Tharindu Kumarage, Cameron Johnson, Jadie Adams, Lin Ai, Matthias Kirchner, Anthony Hoogs, Joshua Garland, Julia Hirschberg, Arslan Basharat, Huan Liu

**Abstract**: The rapid advancement of conversational agents, particularly chatbots powered by Large Language Models (LLMs), poses a significant risk of social engineering (SE) attacks on social media platforms. SE detection in multi-turn, chat-based interactions is considerably more complex than single-instance detection due to the dynamic nature of these conversations. A critical factor in mitigating this threat is understanding the SE attack mechanisms through which SE attacks operate, specifically how attackers exploit vulnerabilities and how victims' personality traits contribute to their susceptibility. In this work, we propose an LLM-agentic framework, SE-VSim, to simulate SE attack mechanisms by generating multi-turn conversations. We model victim agents with varying personality traits to assess how psychological profiles influence susceptibility to manipulation. Using a dataset of over 1000 simulated conversations, we examine attack scenarios in which adversaries, posing as recruiters, funding agencies, and journalists, attempt to extract sensitive information. Based on this analysis, we present a proof of concept, SE-OmniGuard, to offer personalized protection to users by leveraging prior knowledge of the victims personality, evaluating attack strategies, and monitoring information exchanges in conversations to identify potential SE attempts.



## **12. Adversarial Attacks on Audio Deepfake Detection: A Benchmark and Comparative Study**

cs.SD

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.07132v1) [paper-pdf](http://arxiv.org/pdf/2509.07132v1)

**Authors**: Kutub Uddin, Muhammad Umar Farooq, Awais Khan, Khalid Mahmood Malik

**Abstract**: The widespread use of generative AI has shown remarkable success in producing highly realistic deepfakes, posing a serious threat to various voice biometric applications, including speaker verification, voice biometrics, audio conferencing, and criminal investigations. To counteract this, several state-of-the-art (SoTA) audio deepfake detection (ADD) methods have been proposed to identify generative AI signatures to distinguish between real and deepfake audio. However, the effectiveness of these methods is severely undermined by anti-forensic (AF) attacks that conceal generative signatures. These AF attacks span a wide range of techniques, including statistical modifications (e.g., pitch shifting, filtering, noise addition, and quantization) and optimization-based attacks (e.g., FGSM, PGD, C \& W, and DeepFool). In this paper, we investigate the SoTA ADD methods and provide a comparative analysis to highlight their effectiveness in exposing deepfake signatures, as well as their vulnerabilities under adversarial conditions. We conducted an extensive evaluation of ADD methods on five deepfake benchmark datasets using two categories: raw and spectrogram-based approaches. This comparative analysis enables a deeper understanding of the strengths and limitations of SoTA ADD methods against diverse AF attacks. It does not only highlight vulnerabilities of ADD methods, but also informs the design of more robust and generalized detectors for real-world voice biometrics. It will further guide future research in developing adaptive defense strategies that can effectively counter evolving AF techniques.



## **13. Attacking LLMs and AI Agents: Advertisement Embedding Attacks Against Large Language Models**

cs.CR

6 pages, 2 figures

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2508.17674v2) [paper-pdf](http://arxiv.org/pdf/2508.17674v2)

**Authors**: Qiming Guo, Jinwen Tang, Xingran Huang

**Abstract**: We introduce Advertisement Embedding Attacks (AEA), a new class of LLM security threats that stealthily inject promotional or malicious content into model outputs and AI agents. AEA operate through two low-cost vectors: (1) hijacking third-party service-distribution platforms to prepend adversarial prompts, and (2) publishing back-doored open-source checkpoints fine-tuned with attacker data. Unlike conventional attacks that degrade accuracy, AEA subvert information integrity, causing models to return covert ads, propaganda, or hate speech while appearing normal. We detail the attack pipeline, map five stakeholder victim groups, and present an initial prompt-based self-inspection defense that mitigates these injections without additional model retraining. Our findings reveal an urgent, under-addressed gap in LLM security and call for coordinated detection, auditing, and policy responses from the AI-safety community.



## **14. From Noise to Narrative: Tracing the Origins of Hallucinations in Transformers**

cs.LG

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06938v1) [paper-pdf](http://arxiv.org/pdf/2509.06938v1)

**Authors**: Praneet Suresh, Jack Stanley, Sonia Joseph, Luca Scimeca, Danilo Bzdok

**Abstract**: As generative AI systems become competent and democratized in science, business, and government, deeper insight into their failure modes now poses an acute need. The occasional volatility in their behavior, such as the propensity of transformer models to hallucinate, impedes trust and adoption of emerging AI solutions in high-stakes areas. In the present work, we establish how and when hallucinations arise in pre-trained transformer models through concept representations captured by sparse autoencoders, under scenarios with experimentally controlled uncertainty in the input space. Our systematic experiments reveal that the number of semantic concepts used by the transformer model grows as the input information becomes increasingly unstructured. In the face of growing uncertainty in the input space, the transformer model becomes prone to activate coherent yet input-insensitive semantic features, leading to hallucinated output. At its extreme, for pure-noise inputs, we identify a wide variety of robustly triggered and meaningful concepts in the intermediate activations of pre-trained transformer models, whose functional integrity we confirm through targeted steering. We also show that hallucinations in the output of a transformer model can be reliably predicted from the concept patterns embedded in transformer layer activations. This collection of insights on transformer internal processing mechanics has immediate consequences for aligning AI models with human values, AI safety, opening the attack surface for potential adversarial attacks, and providing a basis for automatic quantification of a model's hallucination risk.



## **15. Evaluating the Impact of Adversarial Attacks on Traffic Sign Classification using the LISA Dataset**

cs.CV

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06835v1) [paper-pdf](http://arxiv.org/pdf/2509.06835v1)

**Authors**: Nabeyou Tadessa, Balaji Iyangar, Mashrur Chowdhury

**Abstract**: Adversarial attacks pose significant threats to machine learning models by introducing carefully crafted perturbations that cause misclassification. While prior work has primarily focused on MNIST and similar datasets, this paper investigates the vulnerability of traffic sign classifiers using the LISA Traffic Sign dataset. We train a convolutional neural network to classify 47 different traffic signs and evaluate its robustness against Fast Gradient Sign Method (FGSM) and Projected Gradient Descent (PGD) attacks. Our results show a sharp decline in classification accuracy as the perturbation magnitude increases, highlighting the models susceptibility to adversarial examples. This study lays the groundwork for future exploration into defense mechanisms tailored for real-world traffic sign recognition systems.



## **16. Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions**

cs.CL

Accepted at EMNLP 2025 (Main)

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2501.01872v4) [paper-pdf](http://arxiv.org/pdf/2501.01872v4)

**Authors**: Rachneet Sachdeva, Rima Hazra, Iryna Gurevych

**Abstract**: Large language models, despite extensive alignment with human values and ethical principles, remain vulnerable to sophisticated jailbreak attacks that exploit their reasoning abilities. Existing safety measures often detect overt malicious intent but fail to address subtle, reasoning-driven vulnerabilities. In this work, we introduce POATE (Polar Opposite query generation, Adversarial Template construction, and Elaboration), a novel jailbreak technique that harnesses contrastive reasoning to provoke unethical responses. POATE crafts semantically opposing intents and integrates them with adversarial templates, steering models toward harmful outputs with remarkable subtlety. We conduct extensive evaluation across six diverse language model families of varying parameter sizes to demonstrate the robustness of the attack, achieving significantly higher attack success rates (~44%) compared to existing methods. To counter this, we propose Intent-Aware CoT and Reverse Thinking CoT, which decompose queries to detect malicious intent and reason in reverse to evaluate and reject harmful responses. These methods enhance reasoning robustness and strengthen the model's defense against adversarial exploits.



## **17. On Hyperparameters and Backdoor-Resistance in Horizontal Federated Learning**

cs.CR

To appear in the Proceedings of the ACM Conference on Computer and  Communications Security (CCS) 2025

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.05192v2) [paper-pdf](http://arxiv.org/pdf/2509.05192v2)

**Authors**: Simon Lachnit, Ghassan Karame

**Abstract**: Horizontal Federated Learning (HFL) is particularly vulnerable to backdoor attacks as adversaries can easily manipulate both the training data and processes to execute sophisticated attacks. In this work, we study the impact of training hyperparameters on the effectiveness of backdoor attacks and defenses in HFL. More specifically, we show both analytically and by means of measurements that the choice of hyperparameters by benign clients does not only influence model accuracy but also significantly impacts backdoor attack success. This stands in sharp contrast with the multitude of contributions in the area of HFL security, which often rely on custom ad-hoc hyperparameter choices for benign clients$\unicode{x2013}$leading to more pronounced backdoor attack strength and diminished impact of defenses. Our results indicate that properly tuning benign clients' hyperparameters$\unicode{x2013}$such as learning rate, batch size, and number of local epochs$\unicode{x2013}$can significantly curb the effectiveness of backdoor attacks, regardless of the malicious clients' settings. We support this claim with an extensive robustness evaluation of state-of-the-art attack-defense combinations, showing that carefully chosen hyperparameters yield across-the-board improvements in robustness without sacrificing main task accuracy. For example, we show that the 50%-lifespan of the strong A3FL attack can be reduced by 98.6%, respectively$\unicode{x2013}$all without using any defense and while incurring only a 2.9 percentage points drop in clean task accuracy.



## **18. Mind Your Server: A Systematic Study of Parasitic Toolchain Attacks on the MCP Ecosystem**

cs.CR

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06572v1) [paper-pdf](http://arxiv.org/pdf/2509.06572v1)

**Authors**: Shuli Zhao, Qinsheng Hou, Zihan Zhan, Yanhao Wang, Yuchong Xie, Yu Guo, Libo Chen, Shenghong Li, Zhi Xue

**Abstract**: Large language models (LLMs) are increasingly integrated with external systems through the Model Context Protocol (MCP), which standardizes tool invocation and has rapidly become a backbone for LLM-powered applications. While this paradigm enhances functionality, it also introduces a fundamental security shift: LLMs transition from passive information processors to autonomous orchestrators of task-oriented toolchains, expanding the attack surface, elevating adversarial goals from manipulating single outputs to hijacking entire execution flows. In this paper, we reveal a new class of attacks, Parasitic Toolchain Attacks, instantiated as MCP Unintended Privacy Disclosure (MCP-UPD). These attacks require no direct victim interaction; instead, adversaries embed malicious instructions into external data sources that LLMs access during legitimate tasks. The malicious logic infiltrates the toolchain and unfolds in three phases: Parasitic Ingestion, Privacy Collection, and Privacy Disclosure, culminating in stealthy exfiltration of private data. Our root cause analysis reveals that MCP lacks both context-tool isolation and least-privilege enforcement, enabling adversarial instructions to propagate unchecked into sensitive tool invocations. To assess the severity, we design MCP-SEC and conduct the first large-scale security census of the MCP ecosystem, analyzing 12,230 tools across 1,360 servers. Our findings show that the MCP ecosystem is rife with exploitable gadgets and diverse attack methods, underscoring systemic risks in MCP platforms and the urgent need for defense mechanisms in LLM-integrated environments.



## **19. Robustness and accuracy of mean opinion scores with hard and soft outlier detection**

eess.IV

Accepted for 17th International Conference on Quality of Multimedia  Experience (QoMEX'25), September 2025, Madrid, Spain

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06554v1) [paper-pdf](http://arxiv.org/pdf/2509.06554v1)

**Authors**: Dietmar Saupe, Tim Bleile

**Abstract**: In subjective assessment of image and video quality, observers rate or compare selected stimuli. Before calculating the mean opinion scores (MOS) for these stimuli from the ratings, it is recommended to identify and deal with outliers that may have given unreliable ratings. Several methods are available for this purpose, some of which have been standardized. These methods are typically based on statistics and sometimes tested by introducing synthetic ratings from artificial outliers, such as random clickers. However, a reliable and comprehensive approach is lacking for comparative performance analysis of outlier detection methods. To fill this gap, this work proposes and applies an empirical worst-case analysis as a general solution. Our method involves evolutionary optimization of an adversarial black-box attack on outlier detection algorithms, where the adversary maximizes the distortion of scale values with respect to ground truth. We apply our analysis to several hard and soft outlier detection methods for absolute category ratings and show their differing performance in this stress test. In addition, we propose two new outlier detection methods with low complexity and excellent worst-case performance. Software for adversarial attacks and data analysis is available.



## **20. Byzantine-Robust Federated Learning Using Generative Adversarial Networks**

cs.CR

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2503.20884v2) [paper-pdf](http://arxiv.org/pdf/2503.20884v2)

**Authors**: Usama Zafar, André Teixeira, Salman Toor

**Abstract**: Federated learning (FL) enables collaborative model training across distributed clients without sharing raw data, but its robustness is threatened by Byzantine behaviors such as data and model poisoning. Existing defenses face fundamental limitations: robust aggregation rules incur error lower bounds that grow with client heterogeneity, while detection-based methods often rely on heuristics (e.g., a fixed number of malicious clients) or require trusted external datasets for validation. We present a defense framework that addresses these challenges by leveraging a conditional generative adversarial network (cGAN) at the server to synthesize representative data for validating client updates. This approach eliminates reliance on external datasets, adapts to diverse attack strategies, and integrates seamlessly into standard FL workflows. Extensive experiments on benchmark datasets demonstrate that our framework accurately distinguishes malicious from benign clients while maintaining overall model accuracy. Beyond Byzantine robustness, we also examine the representativeness of synthesized data, computational costs of cGAN training, and the transparency and scalability of our approach.



## **21. IGAff: Benchmarking Adversarial Iterative and Genetic Affine Algorithms on Deep Neural Networks**

cs.CV

10 pages, 7 figures, Accepted at ECAI 2025 (28th European Conference  on Artificial Intelligence)

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06459v1) [paper-pdf](http://arxiv.org/pdf/2509.06459v1)

**Authors**: Sebastian-Vasile Echim, Andrei-Alexandru Preda, Dumitru-Clementin Cercel, Florin Pop

**Abstract**: Deep neural networks currently dominate many fields of the artificial intelligence landscape, achieving state-of-the-art results on numerous tasks while remaining hard to understand and exhibiting surprising weaknesses. An active area of research focuses on adversarial attacks, which aim to generate inputs that uncover these weaknesses. However, this proves challenging, especially in the black-box scenario where model details are inaccessible. This paper explores in detail the impact of such adversarial algorithms on ResNet-18, DenseNet-121, Swin Transformer V2, and Vision Transformer network architectures. Leveraging the Tiny ImageNet, Caltech-256, and Food-101 datasets, we benchmark two novel black-box iterative adversarial algorithms based on affine transformations and genetic algorithms: 1) Affine Transformation Attack (ATA), an iterative algorithm maximizing our attack score function using random affine transformations, and 2) Affine Genetic Attack (AGA), a genetic algorithm that involves random noise and affine transformations. We evaluate the performance of the models in the algorithm parameter variation, data augmentation, and global and targeted attack configurations. We also compare our algorithms with two black-box adversarial algorithms, Pixle and Square Attack. Our experiments yield better results on the image classification task than similar methods in the literature, achieving an accuracy improvement of up to 8.82%. We provide noteworthy insights into successful adversarial defenses and attacks at both global and targeted levels, and demonstrate adversarial robustness through algorithm parameter variation.



## **22. Mask-GCG: Are All Tokens in Adversarial Suffixes Necessary for Jailbreak Attacks?**

cs.CL

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2509.06350v1) [paper-pdf](http://arxiv.org/pdf/2509.06350v1)

**Authors**: Junjie Mu, Zonghao Ying, Zhekui Fan, Zonglei Jing, Yaoyuan Zhang, Zhengmin Yu, Wenxin Zhang, Quanchen Zou, Xiangzheng Zhang

**Abstract**: Jailbreak attacks on Large Language Models (LLMs) have demonstrated various successful methods whereby attackers manipulate models into generating harmful responses that they are designed to avoid. Among these, Greedy Coordinate Gradient (GCG) has emerged as a general and effective approach that optimizes the tokens in a suffix to generate jailbreakable prompts. While several improved variants of GCG have been proposed, they all rely on fixed-length suffixes. However, the potential redundancy within these suffixes remains unexplored. In this work, we propose Mask-GCG, a plug-and-play method that employs learnable token masking to identify impactful tokens within the suffix. Our approach increases the update probability for tokens at high-impact positions while pruning those at low-impact positions. This pruning not only reduces redundancy but also decreases the size of the gradient space, thereby lowering computational overhead and shortening the time required to achieve successful attacks compared to GCG. We evaluate Mask-GCG by applying it to the original GCG and several improved variants. Experimental results show that most tokens in the suffix contribute significantly to attack success, and pruning a minority of low-impact tokens does not affect the loss values or compromise the attack success rate (ASR), thereby revealing token redundancy in LLM prompts. Our findings provide insights for developing efficient and interpretable LLMs from the perspective of jailbreak attacks.



## **23. Whisper Smarter, not Harder: Adversarial Attack on Partial Suppression**

cs.SD

14 pages, 7 figures

**SubmitDate**: 2025-09-08    [abs](http://arxiv.org/abs/2508.09994v2) [paper-pdf](http://arxiv.org/pdf/2508.09994v2)

**Authors**: Zheng Jie Wong, Bingquan Shen

**Abstract**: Currently, Automatic Speech Recognition (ASR) models are deployed in an extensive range of applications. However, recent studies have demonstrated the possibility of adversarial attack on these models which could potentially suppress or disrupt model output. We investigate and verify the robustness of these attacks and explore if it is possible to increase their imperceptibility. We additionally find that by relaxing the optimisation objective from complete suppression to partial suppression, we can further decrease the imperceptibility of the attack. We also explore possible defences against these attacks and show a low-pass filter defence could potentially serve as an effective defence.



## **24. Chronus: Understanding and Securing the Cutting-Edge Industry Solutions to DRAM Read Disturbance**

cs.CR

To appear in HPCA'25. arXiv admin note: text overlap with  arXiv:2406.19094. Appendix E added that describe the errata and new results

**SubmitDate**: 2025-09-07    [abs](http://arxiv.org/abs/2502.12650v2) [paper-pdf](http://arxiv.org/pdf/2502.12650v2)

**Authors**: Oğuzhan Canpolat, A. Giray Yağlıkçı, Geraldo F. Oliveira, Ataberk Olgun, Nisa Bostancı, İsmail Emir Yüksel, Haocong Luo, Oğuz Ergin, Onur Mutlu

**Abstract**: We 1) present the first rigorous security, performance, energy, and cost analyses of the state-of-the-art on-DRAM-die read disturbance mitigation method, Per Row Activation Counting (PRAC) and 2) propose Chronus, a new mechanism that addresses PRAC's two major weaknesses. Our analysis shows that PRAC's system performance overhead on benign applications is non-negligible for modern DRAM chips and prohibitively large for future DRAM chips that are more vulnerable to read disturbance. We identify two weaknesses of PRAC that cause these overheads. First, PRAC increases critical DRAM access latency parameters due to the additional time required to increment activation counters. Second, PRAC performs a constant number of preventive refreshes at a time, making it vulnerable to an adversarial access pattern, known as the wave attack, and consequently requiring it to be configured for significantly smaller activation thresholds. To address PRAC's two weaknesses, we propose a new on-DRAM-die RowHammer mitigation mechanism, Chronus. Chronus 1) updates row activation counters concurrently while serving accesses by separating counters from the data and 2) prevents the wave attack by dynamically controlling the number of preventive refreshes performed. Our performance analysis shows that Chronus's system performance overhead is near-zero for modern DRAM chips and very low for future DRAM chips. Chronus outperforms three variants of PRAC and three other state-of-the-art read disturbance solutions. We discuss Chronus's and PRAC's implications for future systems and foreshadow future research directions. To aid future research, we open-source our Chronus implementation at https://github.com/CMU-SAFARI/Chronus.



## **25. Cascading and Proxy Membership Inference Attacks**

cs.CR

Accepted by The Network and Distributed System Security (NDSS)  Symposium, 2026

**SubmitDate**: 2025-09-07    [abs](http://arxiv.org/abs/2507.21412v3) [paper-pdf](http://arxiv.org/pdf/2507.21412v3)

**Authors**: Yuntao Du, Jiacheng Li, Yuetian Chen, Kaiyuan Zhang, Zhizhen Yuan, Hanshen Xiao, Bruno Ribeiro, Ninghui Li

**Abstract**: A Membership Inference Attack (MIA) assesses how much a trained machine learning model reveals about its training data by determining whether specific query instances were included in the dataset. We classify existing MIAs into adaptive or non-adaptive, depending on whether the adversary is allowed to train shadow models on membership queries. In the adaptive setting, where the adversary can train shadow models after accessing query instances, we highlight the importance of exploiting membership dependencies between instances and propose an attack-agnostic framework called Cascading Membership Inference Attack (CMIA), which incorporates membership dependencies via conditional shadow training to boost membership inference performance.   In the non-adaptive setting, where the adversary is restricted to training shadow models before obtaining membership queries, we introduce Proxy Membership Inference Attack (PMIA). PMIA employs a proxy selection strategy that identifies samples with similar behaviors to the query instance and uses their behaviors in shadow models to perform a membership posterior odds test for membership inference. We provide theoretical analyses for both attacks, and extensive experimental results demonstrate that CMIA and PMIA substantially outperform existing MIAs in both settings, particularly in the low false-positive regime, which is crucial for evaluating privacy risks.



## **26. Quantum machine unlearning**

quant-ph

**SubmitDate**: 2025-09-07    [abs](http://arxiv.org/abs/2509.06086v1) [paper-pdf](http://arxiv.org/pdf/2509.06086v1)

**Authors**: Junjian Su, Runze He, Guanghui Li, Sujuan Qin, Zhimin He, Haozhen Situ, Fei Gao

**Abstract**: Quantum Machine Learning (QML) integrates quantum computation with classical Machine Learning (ML) and holds the potential to achieve the quantum advantage for specific tasks. In classical ML, Machine Unlearning (MU) is a crucial strategy for removing the influence of specified training data from a model, to meet regulatory requirements and mitigate privacy risks. However, both the risk of training-data membership leakage remains underexplored in QML. This motivates us to propose Quantum Machine Unlearning (QMU) to explore two core questions: do QML models require MU due to training-data membership leakage, and can MU mechanisms be efficiently implemented in the QML? To answer the two questions, we conducted experiments on the MNIST classification task, utilizing a class-wise unlearning paradigm in both noiseless simulations and quantum hardware. First, we quantify training-data privacy leakage using a Membership Inference Attack (MIA), observing average success rates of 90.2\% in noiseless simulations and 75.3\% on quantum hardware. These results indicate that QML models present training-data membership leakage with very high probability under adversarial access, motivating the need for MU. Second, we implement MU algorithms on the QML model, which reduces the average MIA success rate to 0\% in simulations and 3.7\% on quantum hardware while preserving accuracy on retained data. We conclude that implementing MU mechanisms in QML models renders them resistant to MIA. Overall, this paper reveals significant privacy vulnerabilities in QML models and provides effective corresponding defense strategies, providing a potential path toward privacy-preserving QML systems.



## **27. Asymmetry Vulnerability and Physical Attacks on Online Map Construction for Autonomous Driving**

cs.CR

CCS'25 (a shorter version of this paper will appear in the conference  proceeding)

**SubmitDate**: 2025-09-07    [abs](http://arxiv.org/abs/2509.06071v1) [paper-pdf](http://arxiv.org/pdf/2509.06071v1)

**Authors**: Yang Lou, Haibo Hu, Qun Song, Qian Xu, Yi Zhu, Rui Tan, Wei-Bin Lee, Jianping Wang

**Abstract**: High-definition maps provide precise environmental information essential for prediction and planning in autonomous driving systems. Due to the high cost of labeling and maintenance, recent research has turned to online HD map construction using onboard sensor data, offering wider coverage and more timely updates for autonomous vehicles. However, the robustness of online map construction under adversarial conditions remains underexplored. In this paper, we present a systematic vulnerability analysis of online map construction models, which reveals that these models exhibit an inherent bias toward predicting symmetric road structures. In asymmetric scenes like forks or merges, this bias often causes the model to mistakenly predict a straight boundary that mirrors the opposite side. We demonstrate that this vulnerability persists in the real-world and can be reliably triggered by obstruction or targeted interference. Leveraging this vulnerability, we propose a novel two-stage attack framework capable of manipulating online constructed maps. First, our method identifies vulnerable asymmetric scenes along the victim AV's potential route. Then, we optimize the location and pattern of camera-blinding attacks and adversarial patch attacks. Evaluations on a public AD dataset demonstrate that our attacks can degrade mapping accuracy by up to 9.9%, render up to 44% of targeted routes unreachable, and increase unsafe planned trajectory rates, colliding with real-world road boundaries, by up to 27%. These attacks are also validated on a real-world testbed vehicle. We further analyze root causes of the symmetry bias, attributing them to training data imbalance, model architecture, and map element representation. To the best of our knowledge, this study presents the first vulnerability assessment of online map construction models and introduces the first digital and physical attack against them.



## **28. ComplicitSplat: Downstream Models are Vulnerable to Blackbox Attacks by 3D Gaussian Splat Camouflages**

cs.CV

7 pages, 6 figures

**SubmitDate**: 2025-09-07    [abs](http://arxiv.org/abs/2508.11854v2) [paper-pdf](http://arxiv.org/pdf/2508.11854v2)

**Authors**: Matthew Hull, Haoyang Yang, Pratham Mehta, Mansi Phute, Aeree Cho, Haorang Wang, Matthew Lau, Wenke Lee, Wilian Lunardi, Martin Andreoni, Duen Horng Chau

**Abstract**: As 3D Gaussian Splatting (3DGS) gains rapid adoption in safety-critical tasks for efficient novel-view synthesis from static images, how might an adversary tamper images to cause harm? We introduce ComplicitSplat, the first attack that exploits standard 3DGS shading methods to create viewpoint-specific camouflage - colors and textures that change with viewing angle - to embed adversarial content in scene objects that are visible only from specific viewpoints and without requiring access to model architecture or weights. Our extensive experiments show that ComplicitSplat generalizes to successfully attack a variety of popular detector - both single-stage, multi-stage, and transformer-based models on both real-world capture of physical objects and synthetic scenes. To our knowledge, this is the first black-box attack on downstream object detectors using 3DGS, exposing a novel safety risk for applications like autonomous navigation and other mission-critical robotic systems.



## **29. Yours or Mine? Overwriting Attacks against Neural Audio Watermarking**

cs.CR

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2509.05835v1) [paper-pdf](http://arxiv.org/pdf/2509.05835v1)

**Authors**: Lingfeng Yao, Chenpei Huang, Shengyao Wang, Junpei Xue, Hanqing Guo, Jiang Liu, Phone Lin, Tomoaki Ohtsuki, Miao Pan

**Abstract**: As generative audio models are rapidly evolving, AI-generated audios increasingly raise concerns about copyright infringement and misinformation spread. Audio watermarking, as a proactive defense, can embed secret messages into audio for copyright protection and source verification. However, current neural audio watermarking methods focus primarily on the imperceptibility and robustness of watermarking, while ignoring its vulnerability to security attacks. In this paper, we develop a simple yet powerful attack: the overwriting attack that overwrites the legitimate audio watermark with a forged one and makes the original legitimate watermark undetectable. Based on the audio watermarking information that the adversary has, we propose three categories of overwriting attacks, i.e., white-box, gray-box, and black-box attacks. We also thoroughly evaluate the proposed attacks on state-of-the-art neural audio watermarking methods. Experimental results demonstrate that the proposed overwriting attacks can effectively compromise existing watermarking schemes across various settings and achieve a nearly 100% attack success rate. The practicality and effectiveness of the proposed overwriting attacks expose security flaws in existing neural audio watermarking systems, underscoring the need to enhance security in future audio watermarking designs.



## **30. Decoding Latent Attack Surfaces in LLMs: Prompt Injection via HTML in Web Summarization**

cs.CR

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2509.05831v1) [paper-pdf](http://arxiv.org/pdf/2509.05831v1)

**Authors**: Ishaan Verma

**Abstract**: Large Language Models (LLMs) are increasingly integrated into web-based systems for content summarization, yet their susceptibility to prompt injection attacks remains a pressing concern. In this study, we explore how non-visible HTML elements such as <meta>, aria-label, and alt attributes can be exploited to embed adversarial instructions without altering the visible content of a webpage. We introduce a novel dataset comprising 280 static web pages, evenly divided between clean and adversarial injected versions, crafted using diverse HTML-based strategies. These pages are processed through a browser automation pipeline to extract both raw HTML and rendered text, closely mimicking real-world LLM deployment scenarios. We evaluate two state-of-the-art open-source models, Llama 4 Scout (Meta) and Gemma 9B IT (Google), on their ability to summarize this content. Using both lexical (ROUGE-L) and semantic (SBERT cosine similarity) metrics, along with manual annotations, we assess the impact of these covert injections. Our findings reveal that over 29% of injected samples led to noticeable changes in the Llama 4 Scout summaries, while Gemma 9B IT showed a lower, yet non-trivial, success rate of 15%. These results highlight a critical and largely overlooked vulnerability in LLM driven web pipelines, where hidden adversarial content can subtly manipulate model outputs. Our work offers a reproducible framework and benchmark for evaluating HTML-based prompt injection and underscores the urgent need for robust mitigation strategies in LLM applications involving web content.



## **31. SEASONED: Semantic-Enhanced Self-Counterfactual Explainable Detection of Adversarial Exploiter Contracts**

cs.CR

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2509.05681v1) [paper-pdf](http://arxiv.org/pdf/2509.05681v1)

**Authors**: Xng Ai, Shudan Lin, Zecheng Li, Kai Zhou, Bixin Li, Bin Xiao

**Abstract**: Decentralized Finance (DeFi) attacks have resulted in significant losses, often orchestrated through Adversarial Exploiter Contracts (AECs) that exploit vulnerabilities in victim smart contracts. To proactively identify such threats, this paper targets the explainable detection of AECs.   Existing detection methods struggle to capture semantic dependencies and lack interpretability, limiting their effectiveness and leaving critical knowledge gaps in AEC analysis. To address these challenges, we introduce SEASONED, an effective, self-explanatory, and robust framework for AEC detection.   SEASONED extracts semantic information from contract bytecode to construct a semantic relation graph (SRG), and employs a self-counterfactual explainable detector (SCFED) to classify SRGs and generate explanations that highlight the core attack logic. SCFED further enhances robustness, generalizability, and data efficiency by extracting representative information from these explanations. Both theoretical analysis and experimental results demonstrate the effectiveness of SEASONED, which showcases outstanding detection performance, robustness, generalizability, and data efficiency learning ability. To support further research, we also release a new dataset of 359 AECs.



## **32. Privacy-Preserving Federated Learning via Homomorphic Adversarial Networks**

cs.CR

Accepted by KSEM 2025 (This is the extended version)

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2412.01650v3) [paper-pdf](http://arxiv.org/pdf/2412.01650v3)

**Authors**: Wenhan Dong, Chao Lin, Xinlei He, Shengmin Xu, Xinyi Huang

**Abstract**: Privacy-preserving federated learning (PPFL) aims to train a global model for multiple clients while maintaining their data privacy. However, current PPFL protocols exhibit one or more of the following insufficiencies: considerable degradation in accuracy, the requirement for sharing keys, and cooperation during the key generation or decryption processes. As a mitigation, we develop the first protocol that utilizes neural networks to implement PPFL, as well as incorporating an Aggregatable Hybrid Encryption scheme tailored to the needs of PPFL. We name these networks as Homomorphic Adversarial Networks (HANs) which demonstrate that neural networks are capable of performing tasks similar to multi-key homomorphic encryption (MK-HE) while solving the problems of key distribution and collaborative decryption. Our experiments show that HANs are robust against privacy attacks. Compared with non-private federated learning, experiments conducted on multiple datasets demonstrate that HANs exhibit a negligible accuracy loss (at most 1.35%). Compared to traditional MK-HE schemes, HANs increase encryption aggregation speed by 6,075 times while incurring a 29.2 times increase in communication overhead.



## **33. Evaluating the Robustness and Accuracy of Text Watermarking Under Real-World Cross-Lingual Manipulations**

cs.CL

Accepted by EMNLP 2025 Finding

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2502.16699v2) [paper-pdf](http://arxiv.org/pdf/2502.16699v2)

**Authors**: Mansour Al Ghanim, Jiaqi Xue, Rochana Prih Hastuti, Mengxin Zheng, Yan Solihin, Qian Lou

**Abstract**: We present a study to benchmark representative watermarking methods in cross-lingual settings. The current literature mainly focuses on the evaluation of watermarking methods for the English language. However, the literature for evaluating watermarking in cross-lingual settings is scarce. This results in overlooking important adversary scenarios in which a cross-lingual adversary could be in, leading to a gray area of practicality over cross-lingual watermarking. In this paper, we evaluate four watermarking methods in four different and vocabulary rich languages. Our experiments investigate the quality of text under different watermarking procedure and the detectability of watermarks with practical translation attack scenarios. Specifically, we investigate practical scenarios that an adversary with cross-lingual knowledge could take, and evaluate whether current watermarking methods are suitable for such scenarios. Finally, from our findings, we draw key insights about watermarking in cross-lingual settings.



## **34. Evo-MARL: Co-Evolutionary Multi-Agent Reinforcement Learning for Internalized Safety**

cs.AI

accepted by the Trustworthy FMs workshop in ICCV 2025

**SubmitDate**: 2025-09-06    [abs](http://arxiv.org/abs/2508.03864v2) [paper-pdf](http://arxiv.org/pdf/2508.03864v2)

**Authors**: Zhenyu Pan, Yiting Zhang, Yutong Zhang, Jianshu Zhang, Haozheng Luo, Yuwei Han, Dennis Wu, Hong-Yu Chen, Philip S. Yu, Manling Li, Han Liu

**Abstract**: Multi-agent systems (MAS) built on multimodal large language models exhibit strong collaboration and performance. However, their growing openness and interaction complexity pose serious risks, notably jailbreak and adversarial attacks. Existing defenses typically rely on external guard modules, such as dedicated safety agents, to handle unsafe behaviors. Unfortunately, this paradigm faces two challenges: (1) standalone agents offer limited protection, and (2) their independence leads to single-point failure-if compromised, system-wide safety collapses. Naively increasing the number of guard agents further raises cost and complexity. To address these challenges, we propose Evo-MARL, a novel multi-agent reinforcement learning (MARL) framework that enables all task agents to jointly acquire defensive capabilities. Rather than relying on external safety modules, Evo-MARL trains each agent to simultaneously perform its primary function and resist adversarial threats, ensuring robustness without increasing system overhead or single-node failure. Furthermore, Evo-MARL integrates evolutionary search with parameter-sharing reinforcement learning to co-evolve attackers and defenders. This adversarial training paradigm internalizes safety mechanisms and continually enhances MAS performance under co-evolving threats. Experiments show that Evo-MARL reduces attack success rates by up to 22% while boosting accuracy by up to 5% on reasoning tasks-demonstrating that safety and utility can be jointly improved.



## **35. Behind the Mask: Benchmarking Camouflaged Jailbreaks in Large Language Models**

cs.CR

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.05471v1) [paper-pdf](http://arxiv.org/pdf/2509.05471v1)

**Authors**: Youjia Zheng, Mohammad Zandsalimy, Shanu Sushmita

**Abstract**: Large Language Models (LLMs) are increasingly vulnerable to a sophisticated form of adversarial prompting known as camouflaged jailbreaking. This method embeds malicious intent within seemingly benign language to evade existing safety mechanisms. Unlike overt attacks, these subtle prompts exploit contextual ambiguity and the flexible nature of language, posing significant challenges to current defense systems. This paper investigates the construction and impact of camouflaged jailbreak prompts, emphasizing their deceptive characteristics and the limitations of traditional keyword-based detection methods. We introduce a novel benchmark dataset, Camouflaged Jailbreak Prompts, containing 500 curated examples (400 harmful and 100 benign prompts) designed to rigorously stress-test LLM safety protocols. In addition, we propose a multi-faceted evaluation framework that measures harmfulness across seven dimensions: Safety Awareness, Technical Feasibility, Implementation Safeguards, Harmful Potential, Educational Value, Content Quality, and Compliance Score. Our findings reveal a stark contrast in LLM behavior: while models demonstrate high safety and content quality with benign inputs, they exhibit a significant decline in performance and safety when confronted with camouflaged jailbreak attempts. This disparity underscores a pervasive vulnerability, highlighting the urgent need for more nuanced and adaptive security strategies to ensure the responsible and robust deployment of LLMs in real-world applications.



## **36. On Evaluating the Poisoning Robustness of Federated Learning under Local Differential Privacy**

cs.CR

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.05265v1) [paper-pdf](http://arxiv.org/pdf/2509.05265v1)

**Authors**: Zijian Wang, Wei Tong, Tingxuan Han, Haoyu Chen, Tianling Zhang, Yunlong Mao, Sheng Zhong

**Abstract**: Federated learning (FL) combined with local differential privacy (LDP) enables privacy-preserving model training across decentralized data sources. However, the decentralized data-management paradigm leaves LDPFL vulnerable to participants with malicious intent. The robustness of LDPFL protocols, particularly against model poisoning attacks (MPA), where adversaries inject malicious updates to disrupt global model convergence, remains insufficiently studied. In this paper, we propose a novel and extensible model poisoning attack framework tailored for LDPFL settings. Our approach is driven by the objective of maximizing the global training loss while adhering to local privacy constraints. To counter robust aggregation mechanisms such as Multi-Krum and trimmed mean, we develop adaptive attacks that embed carefully crafted constraints into a reverse training process, enabling evasion of these defenses. We evaluate our framework across three representative LDPFL protocols, three benchmark datasets, and two types of deep neural networks. Additionally, we investigate the influence of data heterogeneity and privacy budgets on attack effectiveness. Experimental results demonstrate that our adaptive attacks can significantly degrade the performance of the global model, revealing critical vulnerabilities and highlighting the need for more robust LDPFL defense strategies against MPA. Our code is available at https://github.com/ZiJW/LDPFL-Attack



## **37. Jamming Smarter, Not Harder: Exploiting O-RAN Y1 RAN Analytics for Efficient Interference**

cs.CR

8 pages, 7 figures

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.05161v1) [paper-pdf](http://arxiv.org/pdf/2509.05161v1)

**Authors**: Abiodun Ganiyu, Dara Ron, Syed Rafiul Hussain, Vijay K Shah

**Abstract**: The Y1 interface in O-RAN enables the sharing of RAN Analytics Information (RAI) between the near-RT RIC and authorized Y1 consumers, which may be internal applications within the operator's trusted domain or external systems accessing data through a secure exposure function. While this visibility enhances network optimization and enables advanced services, it also introduces a potential security risk -- a malicious or compromised Y1 consumer could misuse analytics to facilitate targeted interference. In this work, we demonstrate how an adversary can exploit the Y1 interface to launch selective jamming attacks by passively monitoring downlink metrics. We propose and evaluate two Y1-aided jamming strategies: a clustering-based jammer leveraging DBSCAN for traffic profiling and a threshold-based jammer. These are compared against two baselines strategies -- always-on jammer and random jammer -- on an over-the-air LTE/5G O-RAN testbed. Experimental results show that in unconstrained jamming budget scenarios, the threshold-based jammer can closely replicate the disruption caused by always-on jamming while reducing transmission time by 27\%. Under constrained jamming budgets, the clustering-based jammer proves most effective, causing up to an 18.1\% bitrate drop while remaining active only 25\% of the time. These findings reveal a critical trade-off between jamming stealthiness and efficiency, and illustrate how exposure of RAN analytics via the Y1 interface can enable highly targeted, low-overhead attacks, raising important security considerations for both civilian and mission-critical O-RAN deployments.



## **38. Robust Experts: the Effect of Adversarial Training on CNNs with Sparse Mixture-of-Experts Layers**

cs.CV

Accepted for publication at the STREAM workshop at ICCV 2025

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.05086v1) [paper-pdf](http://arxiv.org/pdf/2509.05086v1)

**Authors**: Svetlana Pavlitska, Haixi Fan, Konstantin Ditschuneit, J. Marius Zöllner

**Abstract**: Robustifying convolutional neural networks (CNNs) against adversarial attacks remains challenging and often requires resource-intensive countermeasures. We explore the use of sparse mixture-of-experts (MoE) layers to improve robustness by replacing selected residual blocks or convolutional layers, thereby increasing model capacity without additional inference cost. On ResNet architectures trained on CIFAR-100, we find that inserting a single MoE layer in the deeper stages leads to consistent improvements in robustness under PGD and AutoPGD attacks when combined with adversarial training. Furthermore, we discover that when switch loss is used for balancing, it causes routing to collapse onto a small set of overused experts, thereby concentrating adversarial training on these paths and inadvertently making them more robust. As a result, some individual experts outperform the gated MoE model in robustness, suggesting that robust subpaths emerge through specialization. Our code is available at https://github.com/KASTEL-MobilityLab/robust-sparse-moes.



## **39. Adversarial Augmentation and Active Sampling for Robust Cyber Anomaly Detection**

cs.CR

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.04999v1) [paper-pdf](http://arxiv.org/pdf/2509.04999v1)

**Authors**: Sidahmed Benabderrahmane, Talal Rahwan

**Abstract**: Advanced Persistent Threats (APTs) present a considerable challenge to cybersecurity due to their stealthy, long-duration nature. Traditional supervised learning methods typically require large amounts of labeled data, which is often scarce in real-world scenarios. This paper introduces a novel approach that combines AutoEncoders for anomaly detection with active learning to iteratively enhance APT detection. By selectively querying an oracle for labels on uncertain or ambiguous samples, our method reduces labeling costs while improving detection accuracy, enabling the model to effectively learn with minimal data and reduce reliance on extensive manual labeling. We present a comprehensive formulation of the Attention Adversarial Dual AutoEncoder-based anomaly detection framework and demonstrate how the active learning loop progressively enhances the model's performance. The framework is evaluated on real-world, imbalanced provenance trace data from the DARPA Transparent Computing program, where APT-like attacks account for just 0.004\% of the data. The datasets, which cover multiple operating systems including Android, Linux, BSD, and Windows, are tested in two attack scenarios. The results show substantial improvements in detection rates during active learning, outperforming existing methods.



## **40. Training a Perceptual Model for Evaluating Auditory Similarity in Music Adversarial Attack**

cs.SD

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.04985v1) [paper-pdf](http://arxiv.org/pdf/2509.04985v1)

**Authors**: Yuxuan Liu, Rui Sang, Peihong Zhang, Zhixin Li, Shengchen Li

**Abstract**: Music Information Retrieval (MIR) systems are highly vulnerable to adversarial attacks that are often imperceptible to humans, primarily due to a misalignment between model feature spaces and human auditory perception. Existing defenses and perceptual metrics frequently fail to adequately capture these auditory nuances, a limitation supported by our initial listening tests showing low correlation between common metrics and human judgments. To bridge this gap, we introduce Perceptually-Aligned MERT Transformer (PAMT), a novel framework for learning robust, perceptually-aligned music representations. Our core innovation lies in the psychoacoustically-conditioned sequential contrastive transformer, a lightweight projection head built atop a frozen MERT encoder. PAMT achieves a Spearman correlation coefficient of 0.65 with subjective scores, outperforming existing perceptual metrics. Our approach also achieves an average of 9.15\% improvement in robust accuracy on challenging MIR tasks, including Cover Song Identification and Music Genre Classification, under diverse perceptual adversarial attacks. This work pioneers architecturally-integrated psychoacoustic conditioning, yielding representations significantly more aligned with human perception and robust against music adversarial attacks.



## **41. MAIA: An Inpainting-Based Approach for Music Adversarial Attacks**

cs.SD

Accepted at ISMIR2025

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.04980v1) [paper-pdf](http://arxiv.org/pdf/2509.04980v1)

**Authors**: Yuxuan Liu, Peihong Zhang, Rui Sang, Zhixin Li, Shengchen Li

**Abstract**: Music adversarial attacks have garnered significant interest in the field of Music Information Retrieval (MIR). In this paper, we present Music Adversarial Inpainting Attack (MAIA), a novel adversarial attack framework that supports both white-box and black-box attack scenarios. MAIA begins with an importance analysis to identify critical audio segments, which are then targeted for modification. Utilizing generative inpainting models, these segments are reconstructed with guidance from the output of the attacked model, ensuring subtle and effective adversarial perturbations. We evaluate MAIA on multiple MIR tasks, demonstrating high attack success rates in both white-box and black-box settings while maintaining minimal perceptual distortion. Additionally, subjective listening tests confirm the high audio fidelity of the adversarial samples. Our findings highlight vulnerabilities in current MIR systems and emphasize the need for more robust and secure models.



## **42. RINSER: Accurate API Prediction Using Masked Language Models**

cs.CY

16 pages, 8 figures

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.04887v1) [paper-pdf](http://arxiv.org/pdf/2509.04887v1)

**Authors**: Muhammad Ejaz Ahmed, Christopher Cody, Muhammad Ikram, Sean Lamont, Alsharif Abuadbba, Seyit Camtepe, Surya Nepal, Muhammad Ali Kaafar

**Abstract**: Malware authors commonly use obfuscation to hide API identities in binary files, making analysis difficult and time-consuming for a human expert to understand the behavior and intent of the program. Automatic API prediction tools are necessary to efficiently analyze unknown binaries, facilitating rapid malware triage while reducing the workload on human analysts. In this paper, we present RINSER (AccuRate API predictioN using maSked languagE model leaRning), an automated framework for predicting Windows API (WinAPI) function names. RINSER introduces the novel concept of API codeprints, a set of API-relevant assembly instructions, and supports x86 PE binaries. RINSER relies on BERT's masked language model (LM) to predict API names at scale, achieving 85.77% accuracy for normal binaries and 82.88% accuracy for stripped binaries. We evaluate RINSER on a large dataset of 4.7M API codeprints from 11,098 malware binaries, covering 4,123 unique Windows APIs, making it the largest publicly available dataset of this type. RINSER successfully discovered 65 obfuscated Windows APIs related to C2 communication, spying, and evasion in our dataset, which the commercial disassembler IDA failed to identify. Furthermore, we compared RINSER against three state-of-the-art approaches, showing over 20% higher prediction accuracy. We also demonstrated RINSER's resilience to adversarial attacks, including instruction randomization and code displacement, with a performance drop of no more than 3%.



## **43. PersonaTeaming: Exploring How Introducing Personas Can Improve Automated AI Red-Teaming**

cs.AI

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2509.03728v2) [paper-pdf](http://arxiv.org/pdf/2509.03728v2)

**Authors**: Wesley Hanwen Deng, Sunnie S. Y. Kim, Akshita Jha, Ken Holstein, Motahhare Eslami, Lauren Wilcox, Leon A Gatys

**Abstract**: Recent developments in AI governance and safety research have called for red-teaming methods that can effectively surface potential risks posed by AI models. Many of these calls have emphasized how the identities and backgrounds of red-teamers can shape their red-teaming strategies, and thus the kinds of risks they are likely to uncover. While automated red-teaming approaches promise to complement human red-teaming by enabling larger-scale exploration of model behavior, current approaches do not consider the role of identity. As an initial step towards incorporating people's background and identities in automated red-teaming, we develop and evaluate a novel method, PersonaTeaming, that introduces personas in the adversarial prompt generation process to explore a wider spectrum of adversarial strategies. In particular, we first introduce a methodology for mutating prompts based on either "red-teaming expert" personas or "regular AI user" personas. We then develop a dynamic persona-generating algorithm that automatically generates various persona types adaptive to different seed prompts. In addition, we develop a set of new metrics to explicitly measure the "mutation distance" to complement existing diversity measurements of adversarial prompts. Our experiments show promising improvements (up to 144.1%) in the attack success rates of adversarial prompts through persona mutation, while maintaining prompt diversity, compared to RainbowPlus, a state-of-the-art automated red-teaming method. We discuss the strengths and limitations of different persona types and mutation methods, shedding light on future opportunities to explore complementarities between automated and human red-teaming approaches.



## **44. Adversarial Hubness in Multi-Modal Retrieval**

cs.CR

**SubmitDate**: 2025-09-05    [abs](http://arxiv.org/abs/2412.14113v3) [paper-pdf](http://arxiv.org/pdf/2412.14113v3)

**Authors**: Tingwei Zhang, Fnu Suya, Rishi Jha, Collin Zhang, Vitaly Shmatikov

**Abstract**: Hubness is a phenomenon in high-dimensional vector spaces where a point from the natural distribution is unusually close to many other points. This is a well-known problem in information retrieval that causes some items to accidentally (and incorrectly) appear relevant to many queries.   In this paper, we investigate how attackers can exploit hubness to turn any image or audio input in a multi-modal retrieval system into an adversarial hub. Adversarial hubs can be used to inject universal adversarial content (e.g., spam) that will be retrieved in response to thousands of different queries, and also for targeted attacks on queries related to specific, attacker-chosen concepts.   We present a method for creating adversarial hubs and evaluate the resulting hubs on benchmark multi-modal retrieval datasets and an image-to-image retrieval system implemented by Pinecone, a popular vector database. For example, in text-caption-to-image retrieval, a single adversarial hub, generated using 100 random queries, is retrieved as the top-1 most relevant image for more than 21,000 out of 25,000 test queries (by contrast, the most common natural hub is the top-1 response to only 102 queries), demonstrating the strong generalization capabilities of adversarial hubs. We also investigate whether techniques for mitigating natural hubness can also mitigate adversarial hubs, and show that they are not effective against hubs that target queries related to specific concepts.



## **45. Breaking to Build: A Threat Model of Prompt-Based Attacks for Securing LLMs**

cs.CL

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04615v1) [paper-pdf](http://arxiv.org/pdf/2509.04615v1)

**Authors**: Brennen Hill, Surendra Parla, Venkata Abhijeeth Balabhadruni, Atharv Prajod Padmalayam, Sujay Chandra Shekara Sharma

**Abstract**: The proliferation of Large Language Models (LLMs) has introduced critical security challenges, where adversarial actors can manipulate input prompts to cause significant harm and circumvent safety alignments. These prompt-based attacks exploit vulnerabilities in a model's design, training, and contextual understanding, leading to intellectual property theft, misinformation generation, and erosion of user trust. A systematic understanding of these attack vectors is the foundational step toward developing robust countermeasures. This paper presents a comprehensive literature survey of prompt-based attack methodologies, categorizing them to provide a clear threat model. By detailing the mechanisms and impacts of these exploits, this survey aims to inform the research community's efforts in building the next generation of secure LLMs that are inherently resistant to unauthorized distillation, fine-tuning, and editing.



## **46. Concept-ROT: Poisoning Concepts in Large Language Models with Model Editing**

cs.LG

Published at ICLR 2025

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2412.13341v2) [paper-pdf](http://arxiv.org/pdf/2412.13341v2)

**Authors**: Keltin Grimes, Marco Christiani, David Shriver, Marissa Connor

**Abstract**: Model editing methods modify specific behaviors of Large Language Models by altering a small, targeted set of network weights and require very little data and compute. These methods can be used for malicious applications such as inserting misinformation or simple trojans that result in adversary-specified behaviors when a trigger word is present. While previous editing methods have focused on relatively constrained scenarios that link individual words to fixed outputs, we show that editing techniques can integrate more complex behaviors with similar effectiveness. We develop Concept-ROT, a model editing-based method that efficiently inserts trojans which not only exhibit complex output behaviors, but also trigger on high-level concepts -- presenting an entirely new class of trojan attacks. Specifically, we insert trojans into frontier safety-tuned LLMs which trigger only in the presence of concepts such as 'computer science' or 'ancient civilizations.' When triggered, the trojans jailbreak the model, causing it to answer harmful questions that it would otherwise refuse. Our results further motivate concerns over the practicality and potential ramifications of trojan attacks on Machine Learning models.



## **47. DisPatch: Disarming Adversarial Patches in Object Detection with Diffusion Models**

cs.CV

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04597v1) [paper-pdf](http://arxiv.org/pdf/2509.04597v1)

**Authors**: Jin Ma, Mohammed Aldeen, Christopher Salas, Feng Luo, Mashrur Chowdhury, Mert Pesé, Long Cheng

**Abstract**: Object detection is fundamental to various real-world applications, such as security monitoring and surveillance video analysis. Despite their advancements, state-of-theart object detectors are still vulnerable to adversarial patch attacks, which can be easily applied to real-world objects to either conceal actual items or create non-existent ones, leading to severe consequences. Given the current diversity of adversarial patch attacks and potential unknown threats, an ideal defense method should be effective, generalizable, and robust against adaptive attacks. In this work, we introduce DISPATCH, the first diffusion-based defense framework for object detection. Unlike previous works that aim to "detect and remove" adversarial patches, DISPATCH adopts a "regenerate and rectify" strategy, leveraging generative models to disarm attack effects while preserving the integrity of the input image. Specifically, we utilize the in-distribution generative power of diffusion models to regenerate the entire image, aligning it with benign data. A rectification process is then employed to identify and replace adversarial regions with their regenerated benign counterparts. DISPATCH is attack-agnostic and requires no prior knowledge of the existing patches. Extensive experiments across multiple detectors and attacks demonstrate that DISPATCH consistently outperforms state-of-the-art defenses on both hiding attacks and creating attacks, achieving the best overall mAP.5 score of 89.3% on hiding attacks, and lowering the attack success rate to 24.8% on untargeted creating attacks. Moreover, it maintains strong robustness against adaptive attacks, making it a practical and reliable defense for object detection systems.



## **48. Manipulating Transformer-Based Models: Controllability, Steerability, and Robust Interventions**

cs.CL

13 pages

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.04549v1) [paper-pdf](http://arxiv.org/pdf/2509.04549v1)

**Authors**: Faruk Alpay, Taylan Alpay

**Abstract**: Transformer-based language models excel in NLP tasks, but fine-grained control remains challenging. This paper explores methods for manipulating transformer models through principled interventions at three levels: prompts, activations, and weights. We formalize controllable text generation as an optimization problem addressable via prompt engineering, parameter-efficient fine-tuning, model editing, and reinforcement learning. We introduce a unified framework encompassing prompt-level steering, activation interventions, and weight-space edits. We analyze robustness and safety implications, including adversarial attacks and alignment mitigations. Theoretically, we show minimal weight updates can achieve targeted behavior changes with limited side-effects. Empirically, we demonstrate >90% success in sentiment control and factual edits while preserving base performance, though generalization-specificity trade-offs exist. We discuss ethical dual-use risks and the need for rigorous evaluation. This work lays groundwork for designing controllable and robust language models.



## **49. LADSG: Label-Anonymized Distillation and Similar Gradient Substitution for Label Privacy in Vertical Federated Learning**

cs.CR

20 pages, 8 figures

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2506.06742v3) [paper-pdf](http://arxiv.org/pdf/2506.06742v3)

**Authors**: Zeyu Yan, Yanfei Yao, Xuanbing Wen, Shixiong Zhang, Juli Zhang, Kai Fan

**Abstract**: Vertical Federated Learning (VFL) has emerged as a promising paradigm for collaborative model training across distributed feature spaces, which enables privacy-preserving learning without sharing raw data. However, recent studies have confirmed the feasibility of label inference attacks by internal adversaries. By strategically exploiting gradient vectors and semantic embeddings, attackers-through passive, active, or direct attacks-can accurately reconstruct private labels, leading to catastrophic data leakage. Existing defenses, which typically address isolated leakage vectors or are designed for specific types of attacks, remain vulnerable to emerging hybrid attacks that exploit multiple pathways simultaneously. To bridge this gap, we propose Label-Anonymized Defense with Substitution Gradient (LADSG), a unified and lightweight defense framework for VFL. LADSG first anonymizes true labels via soft distillation to reduce semantic exposure, then generates semantically-aligned substitute gradients to disrupt gradient-based leakage, and finally filters anomalous updates through gradient norm detection. It is scalable and compatible with standard VFL pipelines. Extensive experiments on six real-world datasets show that LADSG reduces the success rates of all three types of label inference attacks by 30-60% with minimal computational overhead, demonstrating its practical effectiveness.



## **50. Enhancing Gradient Variance and Differential Privacy in Quantum Federated Learning**

quant-ph

**SubmitDate**: 2025-09-04    [abs](http://arxiv.org/abs/2509.05377v1) [paper-pdf](http://arxiv.org/pdf/2509.05377v1)

**Authors**: Duc-Thien Phan, Minh-Duong Nguyen, Quoc-Viet Pham, Huilong Pi

**Abstract**: Upon integrating Quantum Neural Network (QNN) as the local model, Quantum Federated Learning (QFL) has recently confronted notable challenges. Firstly, exploration is hindered over sharp minima, decreasing learning performance. Secondly, the steady gradient descent results in more stable and predictable model transmissions over wireless channels, making the model more susceptible to attacks from adversarial entities. Additionally, the local QFL model is vulnerable to noise produced by the quantum device's intermediate noise states, since it requires the use of quantum gates and circuits for training. This local noise becomes intertwined with learning parameters during training, impairing model precision and convergence rate. To address these issues, we propose a new QFL technique that incorporates differential privacy and introduces a dedicated noise estimation strategy to quantify and mitigate the impact of intermediate quantum noise. Furthermore, we design an adaptive noise generation scheme to alleviate privacy threats associated with the vanishing gradient variance phenomenon of QNN and enhance robustness against device noise. Experimental results demonstrate that our algorithm effectively balances convergence, reduces communication costs, and mitigates the adverse effects of intermediate quantum noise while maintaining strong privacy protection. Using real-world datasets, we achieved test accuracy of up to 98.47\% for the MNIST dataset and 83.85\% for the CIFAR-10 dataset while maintaining fast execution times.



