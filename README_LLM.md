# Latest Large Language Model Attack Papers
**update at 2025-10-24 10:12:07**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. RAGRank: Using PageRank to Counter Poisoning in CTI LLM Pipelines**

cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20768v1) [paper-pdf](http://arxiv.org/pdf/2510.20768v1)

**Authors**: Austin Jia, Avaneesh Ramesh, Zain Shamsi, Daniel Zhang, Alex Liu

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as the dominant architectural pattern to operationalize Large Language Model (LLM) usage in Cyber Threat Intelligence (CTI) systems. However, this design is susceptible to poisoning attacks, and previously proposed defenses can fail for CTI contexts as cyber threat information is often completely new for emerging attacks, and sophisticated threat actors can mimic legitimate formats, terminology, and stylistic conventions. To address this issue, we propose that the robustness of modern RAG defenses can be accelerated by applying source credibility algorithms on corpora, using PageRank as an example. In our experiments, we demonstrate quantitatively that our algorithm applies a lower authority score to malicious documents while promoting trusted content, using the standardized MS MARCO dataset. We also demonstrate proof-of-concept performance of our algorithm on CTI documents and feeds.



## **2. Breaking Bad Tokens: Detoxification of LLMs Using Sparse Autoencoders**

cs.CL

EMNLP 2025

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2505.14536v2) [paper-pdf](http://arxiv.org/pdf/2505.14536v2)

**Authors**: Agam Goyal, Vedant Rathi, William Yeh, Yian Wang, Yuen Chen, Hari Sundaram

**Abstract**: Large language models (LLMs) are now ubiquitous in user-facing applications, yet they still generate undesirable toxic outputs, including profanity, vulgarity, and derogatory remarks. Although numerous detoxification methods exist, most apply broad, surface-level fixes and can therefore easily be circumvented by jailbreak attacks. In this paper we leverage sparse autoencoders (SAEs) to identify toxicity-related directions in the residual stream of models and perform targeted activation steering using the corresponding decoder vectors. We introduce three tiers of steering aggressiveness and evaluate them on GPT-2 Small and Gemma-2-2B, revealing trade-offs between toxicity reduction and language fluency. At stronger steering strengths, these causal interventions surpass competitive baselines in reducing toxicity by up to 20%, though fluency can degrade noticeably on GPT-2 Small depending on the aggressiveness. Crucially, standard NLP benchmark scores upon steering remain stable, indicating that the model's knowledge and general abilities are preserved. We further show that feature-splitting in wider SAEs hampers safety interventions, underscoring the importance of disentangled feature learning. Our findings highlight both the promise and the current limitations of SAE-based causal interventions for LLM detoxification, further suggesting practical guidelines for safer language-model deployment.



## **3. HauntAttack: When Attack Follows Reasoning as a Shadow**

cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2506.07031v4) [paper-pdf](http://arxiv.org/pdf/2506.07031v4)

**Authors**: Jingyuan Ma, Rui Li, Zheng Li, Junfeng Liu, Heming Xia, Lei Sha, Zhifang Sui

**Abstract**: Emerging Large Reasoning Models (LRMs) consistently excel in mathematical and reasoning tasks, showcasing remarkable capabilities. However, the enhancement of reasoning abilities and the exposure of internal reasoning processes introduce new safety vulnerabilities. A critical question arises: when reasoning becomes intertwined with harmfulness, will LRMs become more vulnerable to jailbreaks in reasoning mode? To investigate this, we introduce HauntAttack, a novel and general-purpose black-box adversarial attack framework that systematically embeds harmful instructions into reasoning questions. Specifically, we modify key reasoning conditions in existing questions with harmful instructions, thereby constructing a reasoning pathway that guides the model step by step toward unsafe outputs. We evaluate HauntAttack on 11 LRMs and observe an average attack success rate of 70\%, achieving up to 12 percentage points of absolute improvement over the strongest prior baseline. Our further analysis reveals that even advanced safety-aligned models remain highly susceptible to reasoning-based attacks, offering insights into the urgent challenge of balancing reasoning capability and safety in future model development.



## **4. Beyond Text: Multimodal Jailbreaking of Vision-Language and Audio Models through Perceptually Simple Transformations**

cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20223v1) [paper-pdf](http://arxiv.org/pdf/2510.20223v1)

**Authors**: Divyanshu Kumar, Shreyas Jena, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi

**Abstract**: Multimodal large language models (MLLMs) have achieved remarkable progress, yet remain critically vulnerable to adversarial attacks that exploit weaknesses in cross-modal processing. We present a systematic study of multimodal jailbreaks targeting both vision-language and audio-language models, showing that even simple perceptual transformations can reliably bypass state-of-the-art safety filters. Our evaluation spans 1,900 adversarial prompts across three high-risk safety categories harmful content, CBRN (Chemical, Biological, Radiological, Nuclear), and CSEM (Child Sexual Exploitation Material) tested against seven frontier models. We explore the effectiveness of attack techniques on MLLMs, including FigStep-Pro (visual keyword decomposition), Intelligent Masking (semantic obfuscation), and audio perturbations (Wave-Echo, Wave-Pitch, Wave-Speed). The results reveal severe vulnerabilities: models with almost perfect text-only safety (0\% ASR) suffer >75\% attack success under perceptually modified inputs, with FigStep-Pro achieving up to 89\% ASR in Llama-4 variants. Audio-based attacks further uncover provider-specific weaknesses, with even basic modality transfer yielding 25\% ASR for technical queries. These findings expose a critical gap between text-centric alignment and multimodal threats, demonstrating that current safeguards fail to generalize across cross-modal attacks. The accessibility of these attacks, which require minimal technical expertise, suggests that robust multimodal AI safety will require a paradigm shift toward broader semantic-level reasoning to mitigate possible risks.



## **5. TRUST: A Decentralized Framework for Auditing Large Language Model Reasoning**

cs.AI

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20188v1) [paper-pdf](http://arxiv.org/pdf/2510.20188v1)

**Authors**: Morris Yu-Chao Huang, Zhen Tan, Mohan Zhang, Pingzhi Li, Zhuo Zhang, Tianlong Chen

**Abstract**: Large Language Models generate complex reasoning chains that reveal their decision-making, yet verifying the faithfulness and harmlessness of these intermediate steps remains a critical unsolved problem. Existing auditing methods are centralized, opaque, and hard to scale, creating significant risks for deploying proprietary models in high-stakes domains. We identify four core challenges: (1) Robustness: Centralized auditors are single points of failure, prone to bias or attacks. (2) Scalability: Reasoning traces are too long for manual verification. (3) Opacity: Closed auditing undermines public trust. (4) Privacy: Exposing full reasoning risks model theft or distillation. We propose TRUST, a transparent, decentralized auditing framework that overcomes these limitations via: (1) A consensus mechanism among diverse auditors, guaranteeing correctness under up to $30\%$ malicious participants. (2) A hierarchical DAG decomposition of reasoning traces, enabling scalable, parallel auditing. (3) A blockchain ledger that records all verification decisions for public accountability. (4) Privacy-preserving segmentation, sharing only partial reasoning steps to protect proprietary logic. We provide theoretical guarantees for the security and economic incentives of the TRUST framework. Experiments across multiple LLMs (GPT-OSS, DeepSeek-r1, Qwen) and reasoning tasks (math, medical, science, humanities) show TRUST effectively detects reasoning flaws and remains robust against adversarial auditors. Our work pioneers decentralized AI auditing, offering a practical path toward safe and trustworthy LLM deployment.



## **6. SAID: Empowering Large Language Models with Self-Activating Internal Defense**

cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20129v1) [paper-pdf](http://arxiv.org/pdf/2510.20129v1)

**Authors**: Yulong Chen, Yadong Liu, Jiawen Zhang, Mu Li, Chao Huang, Jie Wen

**Abstract**: Large Language Models (LLMs), despite advances in safety alignment, remain vulnerable to jailbreak attacks designed to circumvent protective mechanisms. Prevailing defense strategies rely on external interventions, such as input filtering or output modification, which often lack generalizability and compromise model utility while incurring significant computational overhead. In this work, we introduce a new, training-free defense paradigm, Self-Activating Internal Defense (SAID), which reframes the defense task from external correction to internal capability activation. SAID uniquely leverages the LLM's own reasoning abilities to proactively identify and neutralize malicious intent through a three-stage pipeline: model-native intent distillation to extract core semantics, optimal safety prefix probing to activate latent safety awareness, and a conservative aggregation strategy to ensure robust decision-making. Extensive experiments on five open-source LLMs against six advanced jailbreak attacks demonstrate that SAID substantially outperforms state-of-the-art defenses in reducing harmful outputs. Crucially, it achieves this while preserving model performance on benign tasks and incurring minimal computational overhead. Our work establishes that activating the intrinsic safety mechanisms of LLMs is a more robust and scalable path toward building safer and more reliable aligned AI systems.



## **7. Iterative Self-Tuning LLMs for Enhanced Jailbreaking Capabilities**

cs.CL

Accepted to NAACL 2025 Main (Oral)

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2410.18469v5) [paper-pdf](http://arxiv.org/pdf/2410.18469v5)

**Authors**: Chung-En Sun, Xiaodong Liu, Weiwei Yang, Tsui-Wei Weng, Hao Cheng, Aidan San, Michel Galley, Jianfeng Gao

**Abstract**: Recent research has shown that Large Language Models (LLMs) are vulnerable to automated jailbreak attacks, where adversarial suffixes crafted by algorithms appended to harmful queries bypass safety alignment and trigger unintended responses. Current methods for generating these suffixes are computationally expensive and have low Attack Success Rates (ASR), especially against well-aligned models like Llama2 and Llama3. To overcome these limitations, we introduce ADV-LLM, an iterative self-tuning process that crafts adversarial LLMs with enhanced jailbreak ability. Our framework significantly reduces the computational cost of generating adversarial suffixes while achieving nearly 100\% ASR on various open-source LLMs. Moreover, it exhibits strong attack transferability to closed-source models, achieving 99\% ASR on GPT-3.5 and 49\% ASR on GPT-4, despite being optimized solely on Llama3. Beyond improving jailbreak ability, ADV-LLM provides valuable insights for future safety alignment research through its ability to generate large datasets for studying LLM safety. Our code is available at: https://github.com/SunChungEn/ADV-LLM



## **8. Lexo: Eliminating Stealthy Supply-Chain Attacks via LLM-Assisted Program Regeneration**

cs.CR

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.14522v2) [paper-pdf](http://arxiv.org/pdf/2510.14522v2)

**Authors**: Evangelos Lamprou, Julian Dai, Grigoris Ntousakis, Martin C. Rinard, Nikos Vasilakis

**Abstract**: Software supply-chain attacks are an important and ongoing concern in the open source software ecosystem. These attacks maintain the standard functionality that a component implements, but additionally hide malicious functionality activated only when the component reaches its target environment. Lexo addresses such stealthy attacks by automatically learning and regenerating vulnerability-free versions of potentially malicious components. Lexo first generates a set of input-output pairs to model a component's full observable behavior, which it then uses to synthesize a new version of the original component. The new component implements the original functionality but avoids stealthy malicious behavior. Throughout this regeneration process, Lexo consults several distinct instances of Large Language Models (LLMs), uses correctness and coverage metrics to shepherd these instances, and guardrails their results. Our evaluation on 100+ real-world packages, including high profile stealthy supply-chain attacks, indicates that Lexo scales across multiple domains, regenerates code efficiently (<100s on average), maintains compatibility, and succeeds in eliminating malicious code in several real-world supply-chain-attacks, even in cases when a state-of-the-art LLM fails to eliminate malicious code when prompted to do so.



## **9. SecureInfer: Heterogeneous TEE-GPU Architecture for Privacy-Critical Tensors for Large Language Model Deployment**

cs.CR

Accepted at IEEE Intelligent Computing and Systems at the Edge  (ICEdge) 2025

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19979v1) [paper-pdf](http://arxiv.org/pdf/2510.19979v1)

**Authors**: Tushar Nayan, Ziqi Zhang, Ruimin Sun

**Abstract**: With the increasing deployment of Large Language Models (LLMs) on mobile and edge platforms, securing them against model extraction attacks has become a pressing concern. However, protecting model privacy without sacrificing the performance benefits of untrusted AI accelerators, such as GPUs, presents a challenging trade-off. In this paper, we initiate the study of high-performance execution on LLMs and present SecureInfer, a hybrid framework that leverages a heterogeneous Trusted Execution Environments (TEEs)-GPU architecture to isolate privacy-critical components while offloading compute-intensive operations to untrusted accelerators. Building upon an outsourcing scheme, SecureInfer adopts an information-theoretic and threat-informed partitioning strategy: security-sensitive components, including non-linear layers, projection of attention head, FNN transformations, and LoRA adapters, are executed inside an SGX enclave, while other linear operations (matrix multiplication) are performed on the GPU after encryption and are securely restored within the enclave. We implement a prototype of SecureInfer using the LLaMA-2 model and evaluate it across performance and security metrics. Our results show that SecureInfer offers strong security guarantees with reasonable performance, offering a practical solution for secure on-device model inference.



## **10. Unlearned but Not Forgotten: Data Extraction after Exact Unlearning in LLM**

cs.LG

Accepted by Neurips 2025

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2505.24379v3) [paper-pdf](http://arxiv.org/pdf/2505.24379v3)

**Authors**: Xiaoyu Wu, Yifei Pang, Terrance Liu, Zhiwei Steven Wu

**Abstract**: Large Language Models are typically trained on datasets collected from the web, which may inadvertently contain harmful or sensitive personal information. To address growing privacy concerns, unlearning methods have been proposed to remove the influence of specific data from trained models. Of these, exact unlearning -- which retrains the model from scratch without the target data -- is widely regarded the gold standard for mitigating privacy risks in deployment. In this paper, we revisit this assumption in a practical deployment setting where both the pre- and post-unlearning logits API are exposed, such as in open-weight scenarios. Targeting this setting, we introduce a novel data extraction attack that leverages signals from the pre-unlearning model to guide the post-unlearning model, uncovering patterns that reflect the removed data distribution. Combining model guidance with a token filtering strategy, our attack significantly improves extraction success rates -- doubling performance in some cases -- across common benchmarks such as MUSE, TOFU, and WMDP. Furthermore, we demonstrate our attack's effectiveness on a simulated medical diagnosis dataset to highlight real-world privacy risks associated with exact unlearning. In light of our findings, which suggest that unlearning may, in a contradictory way, increase the risk of privacy leakage during real-world deployments, we advocate for evaluation of unlearning methods to consider broader threat models that account not only for post-unlearning models but also for adversarial access to prior checkpoints. Code is publicly available at: https://github.com/Nicholas0228/unlearned_data_extraction_llm.



## **11. The Tail Tells All: Estimating Model-Level Membership Inference Vulnerability Without Reference Models**

cs.LG

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19773v1) [paper-pdf](http://arxiv.org/pdf/2510.19773v1)

**Authors**: Euodia Dodd, Nataša Krčo, Igor Shilov, Yves-Alexandre de Montjoye

**Abstract**: Membership inference attacks (MIAs) have emerged as the standard tool for evaluating the privacy risks of AI models. However, state-of-the-art attacks require training numerous, often computationally expensive, reference models, limiting their practicality. We present a novel approach for estimating model-level vulnerability, the TPR at low FPR, to membership inference attacks without requiring reference models. Empirical analysis shows loss distributions to be asymmetric and heavy-tailed and suggests that most points at risk from MIAs have moved from the tail (high-loss region) to the head (low-loss region) of the distribution after training. We leverage this insight to propose a method to estimate model-level vulnerability from the training and testing distribution alone: using the absence of outliers from the high-loss region as a predictor of the risk. We evaluate our method, the TNR of a simple loss attack, across a wide range of architectures and datasets and show it to accurately estimate model-level vulnerability to the SOTA MIA attack (LiRA). We also show our method to outperform both low-cost (few reference models) attacks such as RMIA and other measures of distribution difference. We finally evaluate the use of non-linear functions to evaluate risk and show the approach to be promising to evaluate the risk in large-language models.



## **12. Can You Trust What You See? Alpha Channel No-Box Attacks on Video Object Detection**

cs.CV

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19574v1) [paper-pdf](http://arxiv.org/pdf/2510.19574v1)

**Authors**: Ariana Yi, Ce Zhou, Liyang Xiao, Qiben Yan

**Abstract**: As object detection models are increasingly deployed in cyber-physical systems such as autonomous vehicles (AVs) and surveillance platforms, ensuring their security against adversarial threats is essential. While prior work has explored adversarial attacks in the image domain, those attacks in the video domain remain largely unexamined, especially in the no-box setting. In this paper, we present {\alpha}-Cloak, the first no-box adversarial attack on object detectors that operates entirely through the alpha channel of RGBA videos. {\alpha}-Cloak exploits the alpha channel to fuse a malicious target video with a benign video, resulting in a fused video that appears innocuous to human viewers but consistently fools object detectors. Our attack requires no access to model architecture, parameters, or outputs, and introduces no perceptible artifacts. We systematically study the support for alpha channels across common video formats and playback applications, and design a fusion algorithm that ensures visual stealth and compatibility. We evaluate {\alpha}-Cloak on five state-of-the-art object detectors, a vision-language model, and a multi-modal large language model (Gemini-2.0-Flash), demonstrating a 100% attack success rate across all scenarios. Our findings reveal a previously unexplored vulnerability in video-based perception systems, highlighting the urgent need for defenses that account for the alpha channel in adversarial settings.



## **13. Machine Text Detectors are Membership Inference Attacks**

cs.CL

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19492v1) [paper-pdf](http://arxiv.org/pdf/2510.19492v1)

**Authors**: Ryuto Koike, Liam Dugan, Masahiro Kaneko, Chris Callison-Burch, Naoaki Okazaki

**Abstract**: Although membership inference attacks (MIAs) and machine-generated text detection target different goals, identifying training samples and synthetic texts, their methods often exploit similar signals based on a language model's probability distribution. Despite this shared methodological foundation, the two tasks have been independently studied, which may lead to conclusions that overlook stronger methods and valuable insights developed in the other task. In this work, we theoretically and empirically investigate the transferability, i.e., how well a method originally developed for one task performs on the other, between MIAs and machine text detection. For our theoretical contribution, we prove that the metric that achieves the asymptotically highest performance on both tasks is the same. We unify a large proportion of the existing literature in the context of this optimal metric and hypothesize that the accuracy with which a given method approximates this metric is directly correlated with its transferability. Our large-scale empirical experiments, including 7 state-of-the-art MIA methods and 5 state-of-the-art machine text detectors across 13 domains and 10 generators, demonstrate very strong rank correlation (rho > 0.6) in cross-task performance. We notably find that Binoculars, originally designed for machine text detection, achieves state-of-the-art performance on MIA benchmarks as well, demonstrating the practical impact of the transferability. Our findings highlight the need for greater cross-task awareness and collaboration between the two research communities. To facilitate cross-task developments and fair evaluations, we introduce MINT, a unified evaluation suite for MIAs and machine-generated text detection, with implementation of 15 recent methods from both tasks.



## **14. Monitoring LLM-based Multi-Agent Systems Against Corruptions via Node Evaluation**

cs.CR

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19420v1) [paper-pdf](http://arxiv.org/pdf/2510.19420v1)

**Authors**: Chengcan Wu, Zhixin Zhang, Mingqian Xu, Zeming Wei, Meng Sun

**Abstract**: Large Language Model (LLM)-based Multi-Agent Systems (MAS) have become a popular paradigm of AI applications. However, trustworthiness issues in MAS remain a critical concern. Unlike challenges in single-agent systems, MAS involve more complex communication processes, making them susceptible to corruption attacks. To mitigate this issue, several defense mechanisms have been developed based on the graph representation of MAS, where agents represent nodes and communications form edges. Nevertheless, these methods predominantly focus on static graph defense, attempting to either detect attacks in a fixed graph structure or optimize a static topology with certain defensive capabilities. To address this limitation, we propose a dynamic defense paradigm for MAS graph structures, which continuously monitors communication within the MAS graph, then dynamically adjusts the graph topology, accurately disrupts malicious communications, and effectively defends against evolving and diverse dynamic attacks. Experimental results in increasingly complex and dynamic MAS environments demonstrate that our method significantly outperforms existing MAS defense mechanisms, contributing an effective guardrail for their trustworthy applications. Our code is available at https://github.com/ChengcanWu/Monitoring-LLM-Based-Multi-Agent-Systems.



## **15. Defending Against Prompt Injection with DataFilter**

cs.CR

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19207v1) [paper-pdf](http://arxiv.org/pdf/2510.19207v1)

**Authors**: Yizhu Wang, Sizhe Chen, Raghad Alkhudair, Basel Alomair, David Wagner

**Abstract**: When large language model (LLM) agents are increasingly deployed to automate tasks and interact with untrusted external data, prompt injection emerges as a significant security threat. By injecting malicious instructions into the data that LLMs access, an attacker can arbitrarily override the original user task and redirect the agent toward unintended, potentially harmful actions. Existing defenses either require access to model weights (fine-tuning), incur substantial utility loss (detection-based), or demand non-trivial system redesign (system-level). Motivated by this, we propose DataFilter, a test-time model-agnostic defense that removes malicious instructions from the data before it reaches the backend LLM. DataFilter is trained with supervised fine-tuning on simulated injections and leverages both the user's instruction and the data to selectively strip adversarial content while preserving benign information. Across multiple benchmarks, DataFilter consistently reduces the prompt injection attack success rates to near zero while maintaining the LLMs' utility. DataFilter delivers strong security, high utility, and plug-and-play deployment, making it a strong practical defense to secure black-box commercial LLMs against prompt injection. Our DataFilter model is released at https://huggingface.co/JoyYizhu/DataFilter for immediate use, with the code to reproduce our results at https://github.com/yizhu-joy/DataFilter.



## **16. OpenGuardrails: An Open-Source Context-Aware AI Guardrails Platform**

cs.CR

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.19169v1) [paper-pdf](http://arxiv.org/pdf/2510.19169v1)

**Authors**: Thomas Wang, Haowen Li

**Abstract**: As large language models (LLMs) become increasingly integrated into real-world applications, safeguarding them against unsafe, malicious, or privacy-violating content is critically important. We present OpenGuardrails, the first open-source project to provide both a context-aware safety and manipulation detection model and a deployable platform for comprehensive AI guardrails. OpenGuardrails protects against content-safety risks, model-manipulation attacks (e.g., prompt injection, jailbreaking, code-interpreter abuse, and the generation/execution of malicious code), and data leakage. Content-safety and model-manipulation detection are implemented by a unified large model, while data-leakage identification and redaction are performed by a separate lightweight NER pipeline (e.g., Presidio-style models or regex-based detectors). The system can be deployed as a security gateway or an API-based service, with enterprise-grade, fully private deployment options. OpenGuardrails achieves state-of-the-art (SOTA) performance on safety benchmarks, excelling in both prompt and response classification across English, Chinese, and multilingual tasks. All models are released under the Apache 2.0 license for public use.



## **17. PLAGUE: Plug-and-play framework for Lifelong Adaptive Generation of Multi-turn Exploits**

cs.CR

First two authors have equal author contributions

**SubmitDate**: 2025-10-22    [abs](http://arxiv.org/abs/2510.17947v2) [paper-pdf](http://arxiv.org/pdf/2510.17947v2)

**Authors**: Neeladri Bhuiya, Madhav Aggarwal, Diptanshu Purwar

**Abstract**: Large Language Models (LLMs) are improving at an exceptional rate. With the advent of agentic workflows, multi-turn dialogue has become the de facto mode of interaction with LLMs for completing long and complex tasks. While LLM capabilities continue to improve, they remain increasingly susceptible to jailbreaking, especially in multi-turn scenarios where harmful intent can be subtly injected across the conversation to produce nefarious outcomes. While single-turn attacks have been extensively explored, adaptability, efficiency and effectiveness continue to remain key challenges for their multi-turn counterparts. To address these gaps, we present PLAGUE, a novel plug-and-play framework for designing multi-turn attacks inspired by lifelong-learning agents. PLAGUE dissects the lifetime of a multi-turn attack into three carefully designed phases (Primer, Planner and Finisher) that enable a systematic and information-rich exploration of the multi-turn attack family. Evaluations show that red-teaming agents designed using PLAGUE achieve state-of-the-art jailbreaking results, improving attack success rates (ASR) by more than 30% across leading models in a lesser or comparable query budget. Particularly, PLAGUE enables an ASR (based on StrongReject) of 81.4% on OpenAI's o3 and 67.3% on Claude's Opus 4.1, two models that are considered highly resistant to jailbreaks in safety literature. Our work offers tools and insights to understand the importance of plan initialization, context optimization and lifelong learning in crafting multi-turn attacks for a comprehensive model vulnerability evaluation.



## **18. NEXUS: Network Exploration for eXploiting Unsafe Sequences in Multi-Turn LLM Jailbreaks**

cs.CR

This paper has been accepted in the main conference proceedings of  the 2025 Conference on Empirical Methods in Natural Language Processing  (EMNLP 2025). Javad Rafiei Asl and Sidhant Narula are co-first authors

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.03417v2) [paper-pdf](http://arxiv.org/pdf/2510.03417v2)

**Authors**: Javad Rafiei Asl, Sidhant Narula, Mohammad Ghasemigol, Eduardo Blanco, Daniel Takabi

**Abstract**: Large Language Models (LLMs) have revolutionized natural language processing but remain vulnerable to jailbreak attacks, especially multi-turn jailbreaks that distribute malicious intent across benign exchanges and bypass alignment mechanisms. Existing approaches often explore the adversarial space poorly, rely on hand-crafted heuristics, or lack systematic query refinement. We present NEXUS (Network Exploration for eXploiting Unsafe Sequences), a modular framework for constructing, refining, and executing optimized multi-turn attacks. NEXUS comprises: (1) ThoughtNet, which hierarchically expands a harmful intent into a structured semantic network of topics, entities, and query chains; (2) a feedback-driven Simulator that iteratively refines and prunes these chains through attacker-victim-judge LLM collaboration using harmfulness and semantic-similarity benchmarks; and (3) a Network Traverser that adaptively navigates the refined query space for real-time attacks. This pipeline uncovers stealthy, high-success adversarial paths across LLMs. On several closed-source and open-source LLMs, NEXUS increases attack success rate by 2.1% to 19.4% over prior methods. Code: https://github.com/inspire-lab/NEXUS



## **19. HarmNet: A Framework for Adaptive Multi-Turn Jailbreak Attacks on Large Language Models**

cs.CR

This paper has been accepted for presentation at the Conference on  Applied Machine Learning in Information Security (CAMLIS 2025)

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18728v1) [paper-pdf](http://arxiv.org/pdf/2510.18728v1)

**Authors**: Sidhant Narula, Javad Rafiei Asl, Mohammad Ghasemigol, Eduardo Blanco, Daniel Takabi

**Abstract**: Large Language Models (LLMs) remain vulnerable to multi-turn jailbreak attacks. We introduce HarmNet, a modular framework comprising ThoughtNet, a hierarchical semantic network; a feedback-driven Simulator for iterative query refinement; and a Network Traverser for real-time adaptive attack execution. HarmNet systematically explores and refines the adversarial space to uncover stealthy, high-success attack paths. Experiments across closed-source and open-source LLMs show that HarmNet outperforms state-of-the-art methods, achieving higher attack success rates. For example, on Mistral-7B, HarmNet achieves a 99.4% attack success rate, 13.9% higher than the best baseline. Index terms: jailbreak attacks; large language models; adversarial framework; query refinement.



## **20. Exploring Membership Inference Vulnerabilities in Clinical Large Language Models**

cs.CR

Accepted at the 1st IEEE Workshop on Healthcare and Medical Device  Security, Privacy, Resilience, and Trust (IEEE HMD-SPiRiT)

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18674v1) [paper-pdf](http://arxiv.org/pdf/2510.18674v1)

**Authors**: Alexander Nemecek, Zebin Yun, Zahra Rahmani, Yaniv Harel, Vipin Chaudhary, Mahmood Sharif, Erman Ayday

**Abstract**: As large language models (LLMs) become progressively more embedded in clinical decision-support, documentation, and patient-information systems, ensuring their privacy and trustworthiness has emerged as an imperative challenge for the healthcare sector. Fine-tuning LLMs on sensitive electronic health record (EHR) data improves domain alignment but also raises the risk of exposing patient information through model behaviors. In this work-in-progress, we present an exploratory empirical study on membership inference vulnerabilities in clinical LLMs, focusing on whether adversaries can infer if specific patient records were used during model training. Using a state-of-the-art clinical question-answering model, Llemr, we evaluate both canonical loss-based attacks and a domain-motivated paraphrasing-based perturbation strategy that more realistically reflects clinical adversarial conditions. Our preliminary findings reveal limited but measurable membership leakage, suggesting that current clinical LLMs provide partial resistance yet remain susceptible to subtle privacy risks that could undermine trust in clinical AI adoption. These results motivate continued development of context-aware, domain-specific privacy evaluations and defenses such as differential privacy fine-tuning and paraphrase-aware training, to strengthen the security and trustworthiness of healthcare AI systems.



## **21. SentinelNet: Safeguarding Multi-Agent Collaboration Through Credit-Based Dynamic Threat Detection**

cs.CR

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.16219v2) [paper-pdf](http://arxiv.org/pdf/2510.16219v2)

**Authors**: Yang Feng, Xudong Pan

**Abstract**: Malicious agents pose significant threats to the reliability and decision-making capabilities of Multi-Agent Systems (MAS) powered by Large Language Models (LLMs). Existing defenses often fall short due to reactive designs or centralized architectures which may introduce single points of failure. To address these challenges, we propose SentinelNet, the first decentralized framework for proactively detecting and mitigating malicious behaviors in multi-agent collaboration. SentinelNet equips each agent with a credit-based detector trained via contrastive learning on augmented adversarial debate trajectories, enabling autonomous evaluation of message credibility and dynamic neighbor ranking via bottom-k elimination to suppress malicious communications. To overcome the scarcity of attack data, it generates adversarial trajectories simulating diverse threats, ensuring robust training. Experiments on MAS benchmarks show SentinelNet achieves near-perfect detection of malicious agents, close to 100% within two debate rounds, and recovers 95% of system accuracy from compromised baselines. By exhibiting strong generalizability across domains and attack patterns, SentinelNet establishes a novel paradigm for safeguarding collaborative MAS.



## **22. The Attribution Story of WhisperGate: An Academic Perspective**

cs.CR

Virus Bulletin Conference 2025

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18484v1) [paper-pdf](http://arxiv.org/pdf/2510.18484v1)

**Authors**: Oleksandr Adamov, Anders Carlsson

**Abstract**: This paper explores the challenges of cyberattack attribution, specifically APTs, applying the case study approach for the WhisperGate cyber operation of January 2022 executed by the Russian military intelligence service (GRU) and targeting Ukrainian government entities. The study provides a detailed review of the threat actor identifiers and taxonomies used by leading cybersecurity vendors, focusing on the evolving attribution from Microsoft, ESET, and CrowdStrike researchers. Once the attribution to Ember Bear (GRU Unit 29155) is established through technical and intelligence reports, we use both traditional machine learning classifiers and a large language model (ChatGPT) to analyze the indicators of compromise (IoCs), tactics, and techniques to statistically and semantically attribute the WhisperGate attack. Our findings reveal overlapping indicators with the Sandworm group (GRU Unit 74455) but also strong evidence pointing to Ember Bear, especially when the LLM is fine-tuned or contextually augmented with additional intelligence. Thus, showing how AI/GenAI with proper fine-tuning are capable of solving the attribution challenge.



## **23. DeepTx: Real-Time Transaction Risk Analysis via Multi-Modal Features and LLM Reasoning**

cs.CR

Accepted to ASE'25

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18438v1) [paper-pdf](http://arxiv.org/pdf/2510.18438v1)

**Authors**: Yixuan Liu, Xinlei Li, Yi Li

**Abstract**: Phishing attacks in Web3 ecosystems are increasingly sophisticated, exploiting deceptive contract logic, malicious frontend scripts, and token approval patterns. We present DeepTx, a real-time transaction analysis system that detects such threats before user confirmation. DeepTx simulates pending transactions, extracts behavior, context, and UI features, and uses multiple large language models (LLMs) to reason about transaction intent. A consensus mechanism with self-reflection ensures robust and explainable decisions. Evaluated on our phishing dataset, DeepTx achieves high precision and recall (demo video: https://youtu.be/4OfK9KCEXUM).



## **24. SoK: Taxonomy and Evaluation of Prompt Security in Large Language Models**

cs.CR

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.15476v2) [paper-pdf](http://arxiv.org/pdf/2510.15476v2)

**Authors**: Hanbin Hong, Shuya Feng, Nima Naderloui, Shenao Yan, Jingyu Zhang, Biying Liu, Ali Arastehfard, Heqing Huang, Yuan Hong

**Abstract**: Large Language Models (LLMs) have rapidly become integral to real-world applications, powering services across diverse sectors. However, their widespread deployment has exposed critical security risks, particularly through jailbreak prompts that can bypass model alignment and induce harmful outputs. Despite intense research into both attack and defense techniques, the field remains fragmented: definitions, threat models, and evaluation criteria vary widely, impeding systematic progress and fair comparison. In this Systematization of Knowledge (SoK), we address these challenges by (1) proposing a holistic, multi-level taxonomy that organizes attacks, defenses, and vulnerabilities in LLM prompt security; (2) formalizing threat models and cost assumptions into machine-readable profiles for reproducible evaluation; (3) introducing an open-source evaluation toolkit for standardized, auditable comparison of attacks and defenses; (4) releasing JAILBREAKDB, the largest annotated dataset of jailbreak and benign prompts to date;\footnote{The dataset is released at \href{https://huggingface.co/datasets/youbin2014/JailbreakDB}{\textcolor{purple}{https://huggingface.co/datasets/youbin2014/JailbreakDB}}.} and (5) presenting a comprehensive evaluation platform and leaderboard of state-of-the-art methods \footnote{will be released soon.}. Our work unifies fragmented research, provides rigorous foundations for future studies, and supports the development of robust, trustworthy LLMs suitable for high-stakes deployment.



## **25. The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems**

cs.CR

This work was first submitted for review on Sept. 5, 2024, and the  initial version was uploaded to Arxiv on Sept. 30, 2024. The latest version  has accepted for publication by IEEE Transactions on Information Forensics  and Security (TIFS)

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2409.20002v5) [paper-pdf](http://arxiv.org/pdf/2409.20002v5)

**Authors**: Linke Song, Zixuan Pang, Wenhao Wang, Zihao Wang, XiaoFeng Wang, Hongbo Chen, Wei Song, Yier Jin, Dan Meng, Rui Hou

**Abstract**: The wide deployment of Large Language Models (LLMs) has given rise to strong demands for optimizing their inference performance. Today's techniques serving this purpose primarily focus on reducing latency and improving throughput through algorithmic and hardware enhancements, while largely overlooking their privacy side effects, particularly in a multi-user environment. In our research, for the first time, we discovered a set of new timing side channels in LLM systems, arising from shared caches and GPU memory allocations, which can be exploited to infer both confidential system prompts and those issued by other users. These vulnerabilities echo security challenges observed in traditional computing systems, highlighting an urgent need to address potential information leakage in LLM serving infrastructures. In this paper, we report novel attack strategies designed to exploit such timing side channels inherent in LLM deployments, specifically targeting the Key-Value (KV) cache and semantic cache widely used to enhance LLM inference performance. Our approach leverages timing measurements and classification models to detect cache hits, allowing an adversary to infer private prompts with high accuracy. We also propose a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches, showing the feasibility of stealing system prompts and those produced by peer users. Our experimental studies on black-box testing of popular online LLM services demonstrate that such privacy risks are completely realistic, with significant consequences. Our findings underscore the need for robust mitigation to protect LLM systems against such emerging threats.



## **26. Genesis: Evolving Attack Strategies for LLM Web Agent Red-Teaming**

cs.AI

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2510.18314v1) [paper-pdf](http://arxiv.org/pdf/2510.18314v1)

**Authors**: Zheng Zhang, Jiarui He, Yuchen Cai, Deheng Ye, Peilin Zhao, Ruili Feng, Hao Wang

**Abstract**: As large language model (LLM) agents increasingly automate complex web tasks, they boost productivity while simultaneously introducing new security risks. However, relevant studies on web agent attacks remain limited. Existing red-teaming approaches mainly rely on manually crafted attack strategies or static models trained offline. Such methods fail to capture the underlying behavioral patterns of web agents, making it difficult to generalize across diverse environments. In web agent attacks, success requires the continuous discovery and evolution of attack strategies. To this end, we propose Genesis, a novel agentic framework composed of three modules: Attacker, Scorer, and Strategist. The Attacker generates adversarial injections by integrating the genetic algorithm with a hybrid strategy representation. The Scorer evaluates the target web agent's responses to provide feedback. The Strategist dynamically uncovers effective strategies from interaction logs and compiles them into a continuously growing strategy library, which is then re-deployed to enhance the Attacker's effectiveness. Extensive experiments across various web tasks show that our framework discovers novel strategies and consistently outperforms existing attack baselines.



## **27. Secure and Efficient Access Control for Computer-Use Agents via Context Space**

cs.CR

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2509.22256v2) [paper-pdf](http://arxiv.org/pdf/2509.22256v2)

**Authors**: Haochen Gong, Chenxiao Li, Rui Chang, Wenbo Shen

**Abstract**: Large language model (LLM)-based computer-use agents represent a convergence of AI and OS capabilities, enabling natural language to control system- and application-level functions. However, due to LLMs' inherent uncertainty issues, granting agents control over computers poses significant security risks. When agent actions deviate from user intentions, they can cause irreversible consequences. Existing mitigation approaches, such as user confirmation and LLM-based dynamic action validation, still suffer from limitations in usability, security, and performance. To address these challenges, we propose CSAgent, a system-level, static policy-based access control framework for computer-use agents. To bridge the gap between static policy and dynamic context and user intent, CSAgent introduces intent- and context-aware policies, and provides an automated toolchain to assist developers in constructing and refining them. CSAgent enforces these policies through an optimized OS service, ensuring that agent actions can only be executed under specific user intents and contexts. CSAgent supports protecting agents that control computers through diverse interfaces, including API, CLI, and GUI. We implement and evaluate CSAgent, which successfully defends against more than 99.36% of attacks while introducing only 6.83% performance overhead.



## **28. DrunkAgent: Stealthy Memory Corruption in LLM-Powered Recommender Agents**

cs.CR

**SubmitDate**: 2025-10-21    [abs](http://arxiv.org/abs/2503.23804v3) [paper-pdf](http://arxiv.org/pdf/2503.23804v3)

**Authors**: Shiyi Yang, Zhibo Hu, Xinshu Li, Chen Wang, Tong Yu, Xiwei Xu, Liming Zhu, Lina Yao

**Abstract**: Large language model (LLM)-powered agents are increasingly used in recommender systems (RSs) to achieve personalized behavior modeling, where the memory mechanism plays a pivotal role in enabling the agents to autonomously explore, learn and self-evolve from real-world interactions. However, this very mechanism, serving as a contextual repository, inherently exposes an attack surface for potential adversarial manipulations. Despite its central role, the robustness of agentic RSs in the face of such threats remains largely underexplored. Previous works suffer from semantic mismatches or rely on static embeddings or pre-defined prompts, all of which are not designed for dynamic systems, especially for dynamic memory states of LLM agents. This challenge is exacerbated by the black-box nature of commercial recommenders.   To tackle the above problems, in this paper, we present the first systematic investigation of memory-based vulnerabilities in LLM-powered recommender agents, revealing their security limitations and guiding efforts to strengthen system resilience and trustworthiness. Specifically, we propose a novel black-box attack framework named DrunkAgent. DrunkAgent crafts semantically meaningful adversarial textual triggers for target item promotions and introduces a series of strategies to maximize the trigger effect by corrupting the memory updates during the interactions. The triggers and strategies are optimized on a surrogate model, enabling DrunkAgent transferable and stealthy. Extensive experiments on real-world datasets across diverse agentic RSs, including collaborative filtering, retrieval augmentation and sequential recommendations, demonstrate the generalizability, transferability and stealthiness of DrunkAgent.



## **29. Any-Depth Alignment: Unlocking Innate Safety Alignment of LLMs to Any-Depth**

cs.LG

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.18081v1) [paper-pdf](http://arxiv.org/pdf/2510.18081v1)

**Authors**: Jiawei Zhang, Andrew Estornell, David D. Baek, Bo Li, Xiaojun Xu

**Abstract**: Large Language Models (LLMs) exhibit strong but shallow alignment: they directly refuse harmful queries when a refusal is expected at the very start of an assistant turn, yet this protection collapses once a harmful continuation is underway (either through the adversarial attacks or via harmful assistant-prefill attacks). This raises a fundamental question: Can the innate shallow alignment in LLMs be unlocked to ensure safety at arbitrary generation depths? To achieve this goal, we propose Any-Depth Alignment (ADA), an effective inference-time defense with negligible overhead. ADA is built based on our observation that alignment is concentrated in the assistant header tokens through repeated use in shallow-refusal training, and these tokens possess the model's strong alignment priors. By reintroducing these tokens mid-stream, ADA induces the model to reassess harmfulness and recover refusals at any point in generation. Across diverse open-source model families (Llama, Gemma, Mistral, Qwen, DeepSeek, and gpt-oss), ADA achieves robust safety performance without requiring any changes to the base model's parameters. It secures a near-100% refusal rate against challenging adversarial prefill attacks ranging from dozens to thousands of tokens. Furthermore, ADA reduces the average success rate of prominent adversarial prompt attacks (such as GCG, AutoDAN, PAIR, and TAP) to below 3%. This is all accomplished while preserving utility on benign tasks with minimal over-refusal. ADA maintains this resilience even after the base model undergoes subsequent instruction tuning (benign or adversarial).



## **30. CourtGuard: A Local, Multiagent Prompt Injection Classifier**

cs.CR

11 pages, 7 figures

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.19844v1) [paper-pdf](http://arxiv.org/pdf/2510.19844v1)

**Authors**: Isaac Wu, Michael Maslowski

**Abstract**: As large language models (LLMs) become integrated into various sensitive applications, prompt injection, the use of prompting to induce harmful behaviors from LLMs, poses an ever increasing risk. Prompt injection attacks can cause LLMs to leak sensitive data, spread misinformation, and exhibit harmful behaviors. To defend against these attacks, we propose CourtGuard, a locally-runnable, multiagent prompt injection classifier. In it, prompts are evaluated in a court-like multiagent LLM system, where a "defense attorney" model argues the prompt is benign, a "prosecution attorney" model argues the prompt is a prompt injection, and a "judge" model gives the final classification. CourtGuard has a lower false positive rate than the Direct Detector, an LLM as-a-judge. However, CourtGuard is generally a worse prompt injection detector. Nevertheless, this lower false positive rate highlights the importance of considering both adversarial and benign scenarios for the classification of a prompt. Additionally, the relative performance of CourtGuard in comparison to other prompt injection classifiers advances the use of multiagent systems as a defense against prompt injection attacks. The implementations of CourtGuard and the Direct Detector with full prompts for Gemma-3-12b-it, Llama-3.3-8B, and Phi-4-mini-instruct are available at https://github.com/isaacwu2000/CourtGuard.



## **31. Echoes of Human Malice in Agents: Benchmarking LLMs for Multi-Turn Online Harassment Attacks**

cs.AI

13 pages, 4 figures

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.14207v2) [paper-pdf](http://arxiv.org/pdf/2510.14207v2)

**Authors**: Trilok Padhi, Pinxian Lu, Abdulkadir Erol, Tanmay Sutar, Gauri Sharma, Mina Sonmez, Munmun De Choudhury, Ugur Kursuncu

**Abstract**: Large Language Model (LLM) agents are powering a growing share of interactive web applications, yet remain vulnerable to misuse and harm. Prior jailbreak research has largely focused on single-turn prompts, whereas real harassment often unfolds over multi-turn interactions. In this work, we present the Online Harassment Agentic Benchmark consisting of: (i) a synthetic multi-turn harassment conversation dataset, (ii) a multi-agent (e.g., harasser, victim) simulation informed by repeated game theory, (iii) three jailbreak methods attacking agents across memory, planning, and fine-tuning, and (iv) a mixed-methods evaluation framework. We utilize two prominent LLMs, LLaMA-3.1-8B-Instruct (open-source) and Gemini-2.0-flash (closed-source). Our results show that jailbreak tuning makes harassment nearly guaranteed with an attack success rate of 95.78--96.89% vs. 57.25--64.19% without tuning in Llama, and 99.33% vs. 98.46% without tuning in Gemini, while sharply reducing refusal rate to 1-2% in both models. The most prevalent toxic behaviors are Insult with 84.9--87.8% vs. 44.2--50.8% without tuning, and Flaming with 81.2--85.1% vs. 31.5--38.8% without tuning, indicating weaker guardrails compared to sensitive categories such as sexual or racial harassment. Qualitative evaluation further reveals that attacked agents reproduce human-like aggression profiles, such as Machiavellian/psychopathic patterns under planning, and narcissistic tendencies with memory. Counterintuitively, closed-source and open-source models exhibit distinct escalation trajectories across turns, with closed-source models showing significant vulnerability. Overall, our findings show that multi-turn and theory-grounded attacks not only succeed at high rates but also mimic human-like harassment dynamics, motivating the development of robust safety guardrails to ultimately keep online platforms safe and responsible.



## **32. Is Multilingual LLM Watermarking Truly Multilingual? A Simple Back-Translation Solution**

cs.CL

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.18019v1) [paper-pdf](http://arxiv.org/pdf/2510.18019v1)

**Authors**: Asim Mohamed, Martin Gubri

**Abstract**: Multilingual watermarking aims to make large language model (LLM) outputs traceable across languages, yet current methods still fall short. Despite claims of cross-lingual robustness, they are evaluated only on high-resource languages. We show that existing multilingual watermarking methods are not truly multilingual: they fail to remain robust under translation attacks in medium- and low-resource languages. We trace this failure to semantic clustering, which fails when the tokenizer vocabulary contains too few full-word tokens for a given language. To address this, we introduce STEAM, a back-translation-based detection method that restores watermark strength lost through translation. STEAM is compatible with any watermarking method, robust across different tokenizers and languages, non-invasive, and easily extendable to new languages. With average gains of +0.19 AUC and +40%p TPR@1% on 17 languages, STEAM provides a simple and robust path toward fairer watermarking across diverse languages.



## **33. VERA-V: Variational Inference Framework for Jailbreaking Vision-Language Models**

cs.CR

18 pages, 7 Figures,

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.17759v1) [paper-pdf](http://arxiv.org/pdf/2510.17759v1)

**Authors**: Qilin Liao, Anamika Lochab, Ruqi Zhang

**Abstract**: Vision-Language Models (VLMs) extend large language models with visual reasoning, but their multimodal design also introduces new, underexplored vulnerabilities. Existing multimodal red-teaming methods largely rely on brittle templates, focus on single-attack settings, and expose only a narrow subset of vulnerabilities. To address these limitations, we introduce VERA-V, a variational inference framework that recasts multimodal jailbreak discovery as learning a joint posterior distribution over paired text-image prompts. This probabilistic view enables the generation of stealthy, coupled adversarial inputs that bypass model guardrails. We train a lightweight attacker to approximate the posterior, allowing efficient sampling of diverse jailbreaks and providing distributional insights into vulnerabilities. VERA-V further integrates three complementary strategies: (i) typography-based text prompts that embed harmful cues, (ii) diffusion-based image synthesis that introduces adversarial signals, and (iii) structured distractors to fragment VLM attention. Experiments on HarmBench and HADES benchmarks show that VERA-V consistently outperforms state-of-the-art baselines on both open-source and frontier VLMs, achieving up to 53.75% higher attack success rate (ASR) over the best baseline on GPT-4o.



## **34. CrossGuard: Safeguarding MLLMs against Joint-Modal Implicit Malicious Attacks**

cs.CR

14 pages, 8 figures, 2 tables

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.17687v1) [paper-pdf](http://arxiv.org/pdf/2510.17687v1)

**Authors**: Xu Zhang, Hao Li, Zhichao Lu

**Abstract**: Multimodal Large Language Models (MLLMs) achieve strong reasoning and perception capabilities but are increasingly vulnerable to jailbreak attacks. While existing work focuses on explicit attacks, where malicious content resides in a single modality, recent studies reveal implicit attacks, in which benign text and image inputs jointly express unsafe intent. Such joint-modal threats are difficult to detect and remain underexplored, largely due to the scarcity of high-quality implicit data. We propose ImpForge, an automated red-teaming pipeline that leverages reinforcement learning with tailored reward modules to generate diverse implicit samples across 14 domains. Building on this dataset, we further develop CrossGuard, an intent-aware safeguard providing robust and comprehensive defense against both explicit and implicit threats. Extensive experiments across safe and unsafe benchmarks, implicit and explicit attacks, and multiple out-of-domain settings demonstrate that CrossGuard significantly outperforms existing defenses, including advanced MLLMs and guardrails, achieving stronger security while maintaining high utility. This offers a balanced and practical solution for enhancing MLLM robustness against real-world multimodal threats.



## **35. Learning to Detect Unknown Jailbreak Attacks in Large Vision-Language Models**

cs.CR

16 pages; Previously this version appeared as arXiv:2510.15430 which  was submitted as a new work by accident

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2508.09201v2) [paper-pdf](http://arxiv.org/pdf/2508.09201v2)

**Authors**: Shuang Liang, Zhihao Xu, Jialing Tao, Hui Xue, Xiting Wang

**Abstract**: Despite extensive alignment efforts, Large Vision-Language Models (LVLMs) remain vulnerable to jailbreak attacks, posing serious safety risks. To address this, existing detection methods either learn attack-specific parameters, which hinders generalization to unseen attacks, or rely on heuristically sound principles, which limit accuracy and efficiency. To overcome these limitations, we propose Learning to Detect (LoD), a general framework that accurately detects unknown jailbreak attacks by shifting the focus from attack-specific learning to task-specific learning. This framework includes a Multi-modal Safety Concept Activation Vector module for safety-oriented representation learning and a Safety Pattern Auto-Encoder module for unsupervised attack classification. Extensive experiments show that our method achieves consistently higher detection AUROC on diverse unknown attacks while improving efficiency. The code is available at https://anonymous.4open.science/r/Learning-to-Detect-51CB.



## **36. Watch the Weights: Unsupervised monitoring and control of fine-tuned LLMs**

cs.LG

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2508.00161v2) [paper-pdf](http://arxiv.org/pdf/2508.00161v2)

**Authors**: Ziqian Zhong, Aditi Raghunathan

**Abstract**: The releases of powerful open-weight large language models (LLMs) are often not accompanied by access to their full training data. Existing interpretability methods, particularly those based on activations, often require or assume distributionally similar data. This is a significant limitation when detecting and defending against novel potential threats like backdoors, which are by definition out-of-distribution.   In this work, we introduce a new method for understanding, monitoring and controlling fine-tuned LLMs that interprets weights, rather than activations, thereby side stepping the need for data that is distributionally similar to the unknown training data. We demonstrate that the top singular vectors of the weight difference between a fine-tuned model and its base model correspond to newly acquired behaviors. By monitoring the cosine similarity of activations along these directions, we can detect salient behaviors introduced during fine-tuning with high precision.   For backdoored models that bypasses safety mechanisms when a secret trigger is present, our method stops up to 100% of attacks with a false positive rate below 1.2%. For models that have undergone unlearning, we detect inference on erased topics with accuracy up to 95.42% and can even steer the model to recover "unlearned" information. Besides monitoring, our method also shows potential for pre-deployment model auditing: by analyzing commercial instruction-tuned models (OLMo, Llama, Qwen), we are able to uncover model-specific fine-tuning focus including marketing strategies and Midjourney prompt generation.   Our implementation can be found at https://github.com/fjzzq2002/WeightWatch.



## **37. OCR-APT: Reconstructing APT Stories from Audit Logs using Subgraph Anomaly Detection and LLMs**

cs.CR

This is the authors' extended version of the paper accepted for  publication at the ACM SIGSAC Conference on Computer and Communications  Security (CCS 2025). The final published version is available at  https://doi.org/10.1145/3719027.3765219

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.15188v2) [paper-pdf](http://arxiv.org/pdf/2510.15188v2)

**Authors**: Ahmed Aly, Essam Mansour, Amr Youssef

**Abstract**: Advanced Persistent Threats (APTs) are stealthy cyberattacks that often evade detection in system-level audit logs. Provenance graphs model these logs as connected entities and events, revealing relationships that are missed by linear log representations. Existing systems apply anomaly detection to these graphs but often suffer from high false positive rates and coarse-grained alerts. Their reliance on node attributes like file paths or IPs leads to spurious correlations, reducing detection robustness and reliability. To fully understand an attack's progression and impact, security analysts need systems that can generate accurate, human-like narratives of the entire attack. To address these challenges, we introduce OCR-APT, a system for APT detection and reconstruction of human-like attack stories. OCR-APT uses Graph Neural Networks (GNNs) for subgraph anomaly detection, learning behavior patterns around nodes rather than fragile attributes such as file paths or IPs. This approach leads to a more robust anomaly detection. It then iterates over detected subgraphs using Large Language Models (LLMs) to reconstruct multi-stage attack stories. Each stage is validated before proceeding, reducing hallucinations and ensuring an interpretable final report. Our evaluations on the DARPA TC3, OpTC, and NODLINK datasets show that OCR-APT outperforms state-of-the-art systems in both detection accuracy and alert interpretability. Moreover, OCR-APT reconstructs human-like reports that comprehensively capture the attack story.



## **38. Learning to Detect Unknown Jailbreak Attacks in Large Vision-Language Models**

cs.CV

Withdrawn due to an accidental duplicate submission. This paper  (arXiv:2510.15430) was unintentionally submitted as a new entry instead of a  new version of our previous work (arXiv:2508.09201)

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.15430v2) [paper-pdf](http://arxiv.org/pdf/2510.15430v2)

**Authors**: Shuang Liang, Zhihao Xu, Jialing Tao, Hui Xue, Xiting Wang

**Abstract**: Despite extensive alignment efforts, Large Vision-Language Models (LVLMs) remain vulnerable to jailbreak attacks, posing serious safety risks. To address this, existing detection methods either learn attack-specific parameters, which hinders generalization to unseen attacks, or rely on heuristically sound principles, which limit accuracy and efficiency. To overcome these limitations, we propose Learning to Detect (LoD), a general framework that accurately detects unknown jailbreak attacks by shifting the focus from attack-specific learning to task-specific learning. This framework includes a Multi-modal Safety Concept Activation Vector module for safety-oriented representation learning and a Safety Pattern Auto-Encoder module for unsupervised attack classification. Extensive experiments show that our method achieves consistently higher detection AUROC on diverse unknown attacks while improving efficiency. The code is available at https://anonymous.4open.science/r/Learning-to-Detect-51CB.



## **39. Agentic Reinforcement Learning for Search is Unsafe**

cs.CL

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.17431v1) [paper-pdf](http://arxiv.org/pdf/2510.17431v1)

**Authors**: Yushi Yang, Shreyansh Padarha, Andrew Lee, Adam Mahdi

**Abstract**: Agentic reinforcement learning (RL) trains large language models to autonomously call tools during reasoning, with search as the most common application. These models excel at multi-step reasoning tasks, but their safety properties are not well understood. In this study, we show that RL-trained search models inherit refusal from instruction tuning and often deflect harmful requests by turning them into safe queries. However, this safety is fragile. Two simple attacks, one that forces the model to begin response with search (Search attack), another that encourages models to repeatedly search (Multi-search attack), trigger cascades of harmful searches and answers. Across two model families (Qwen, Llama) with both local and web search, these attacks lower refusal rates by up to 60.0%, answer safety by 82.5%, and search-query safety by 82.4%. The attacks succeed by triggering models to generate harmful, request-mirroring search queries before they can generate the inherited refusal tokens. This exposes a core weakness of current RL training: it rewards continued generation of effective queries without accounting for their harmfulness. As a result, RL search models have vulnerabilities that users can easily exploit, making it urgent to develop safety-aware agentic RL pipelines optimising for safe search.



## **40. Semantic Representation Attack against Aligned Large Language Models**

cs.CL

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2509.19360v2) [paper-pdf](http://arxiv.org/pdf/2509.19360v2)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau

**Abstract**: Large Language Models (LLMs) increasingly employ alignment techniques to prevent harmful outputs. Despite these safeguards, attackers can circumvent them by crafting prompts that induce LLMs to generate harmful content.   Current methods typically target exact affirmative responses, such as ``Sure, here is...'', suffering from limited convergence, unnatural prompts, and high computational costs.   We introduce Semantic Representation Attack, a novel paradigm that fundamentally reconceptualizes adversarial objectives against aligned LLMs.   Rather than targeting exact textual patterns, our approach exploits the semantic representation space comprising diverse responses with equivalent harmful meanings.   This innovation resolves the inherent trade-off between attack efficacy and prompt naturalness that plagues existing methods.   The Semantic Representation Heuristic Search algorithm is proposed to efficiently generate semantically coherent and concise adversarial prompts by maintaining interpretability during incremental expansion.   We establish rigorous theoretical guarantees for semantic convergence and demonstrate that our method achieves unprecedented attack success rates (89.41\% averaged across 18 LLMs, including 100\% on 11 models) while maintaining stealthiness and efficiency.   Comprehensive experimental results confirm the overall superiority of our Semantic Representation Attack.   The code will be publicly available.



## **41. Robustness in Text-Attributed Graph Learning: Insights, Trade-offs, and New Defenses**

cs.LG

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.17185v1) [paper-pdf](http://arxiv.org/pdf/2510.17185v1)

**Authors**: Runlin Lei, Lu Yi, Mingguo He, Pengyu Qiu, Zhewei Wei, Yongchao Liu, Chuntao Hong

**Abstract**: While Graph Neural Networks (GNNs) and Large Language Models (LLMs) are powerful approaches for learning on Text-Attributed Graphs (TAGs), a comprehensive understanding of their robustness remains elusive. Current evaluations are fragmented, failing to systematically investigate the distinct effects of textual and structural perturbations across diverse models and attack scenarios. To address these limitations, we introduce a unified and comprehensive framework to evaluate robustness in TAG learning. Our framework evaluates classical GNNs, robust GNNs (RGNNs), and GraphLLMs across ten datasets from four domains, under diverse text-based, structure-based, and hybrid perturbations in both poisoning and evasion scenarios. Our extensive analysis reveals multiple findings, among which three are particularly noteworthy: 1) models have inherent robustness trade-offs between text and structure, 2) the performance of GNNs and RGNNs depends heavily on the text encoder and attack type, and 3) GraphLLMs are particularly vulnerable to training data corruption. To overcome the identified trade-offs, we introduce SFT-auto, a novel framework that delivers superior and balanced robustness against both textual and structural attacks within a single model. Our work establishes a foundation for future research on TAG security and offers practical solutions for robust TAG learning in adversarial environments. Our code is available at: https://github.com/Leirunlin/TGRB.



## **42. ARMOR: Aligning Secure and Safe Large Language Models via Meticulous Reasoning**

cs.CR

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2507.11500v2) [paper-pdf](http://arxiv.org/pdf/2507.11500v2)

**Authors**: Zhengyue Zhao, Yingzi Ma, Somesh Jha, Marco Pavone, Patrick McDaniel, Chaowei Xiao

**Abstract**: Large Language Models have shown impressive generative capabilities across diverse tasks, but their safety remains a critical concern. Existing post-training alignment methods, such as SFT and RLHF, reduce harmful outputs yet leave LLMs vulnerable to jailbreak attacks, especially advanced optimization-based ones. Recent system-2 approaches enhance safety by adding inference-time reasoning, where models assess potential risks before producing responses. However, we find these methods fail against powerful out-of-distribution jailbreaks, such as AutoDAN-Turbo and Adversarial Reasoning, which conceal malicious goals behind seemingly benign prompts. We observe that all jailbreaks ultimately aim to embed a core malicious intent, suggesting that extracting this intent is key to defense. To this end, we propose ARMOR, which introduces a structured three-step reasoning pipeline: (1) analyze jailbreak strategies from an external, updatable strategy library, (2) extract the core intent, and (3) apply policy-based safety verification. We further develop ARMOR-Think, which decouples safety reasoning from general reasoning to improve both robustness and utility. Evaluations on advanced optimization-based jailbreaks and safety benchmarks show that ARMOR achieves state-of-the-art safety performance, with an average harmful rate of 0.002 and an attack success rate of 0.06 against advanced optimization-based jailbreaks, far below other reasoning-based models. Moreover, ARMOR demonstrates strong generalization to unseen jailbreak strategies, reducing their success rate to zero. These highlight ARMOR's effectiveness in defending against OOD jailbreak attacks, offering a practical path toward secure and reliable LLMs.



## **43. Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs**

cs.CV

Accepted by NeurIPS 2025 Dataset and Benchmark Track, Project page:  https://liuxuannan.github.io/Video-SafetyBench.github.io/

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2505.11842v2) [paper-pdf](http://arxiv.org/pdf/2505.11842v2)

**Authors**: Xuannan Liu, Zekun Li, Zheqi He, Peipei Li, Shuhan Xia, Xing Cui, Huaibo Huang, Xi Yang, Ran He

**Abstract**: The increasing deployment of Large Vision-Language Models (LVLMs) raises safety concerns under potential malicious inputs. However, existing multimodal safety evaluations primarily focus on model vulnerabilities exposed by static image inputs, ignoring the temporal dynamics of video that may induce distinct safety risks. To bridge this gap, we introduce Video-SafetyBench, the first comprehensive benchmark designed to evaluate the safety of LVLMs under video-text attacks. It comprises 2,264 video-text pairs spanning 48 fine-grained unsafe categories, each pairing a synthesized video with either a harmful query, which contains explicit malice, or a benign query, which appears harmless but triggers harmful behavior when interpreted alongside the video. To generate semantically accurate videos for safety evaluation, we design a controllable pipeline that decomposes video semantics into subject images (what is shown) and motion text (how it moves), which jointly guide the synthesis of query-relevant videos. To effectively evaluate uncertain or borderline harmful outputs, we propose RJScore, a novel LLM-based metric that incorporates the confidence of judge models and human-aligned decision threshold calibration. Extensive experiments show that benign-query video composition achieves average attack success rates of 67.2%, revealing consistent vulnerabilities to video-induced attacks. We believe Video-SafetyBench will catalyze future research into video-based safety evaluation and defense strategies.



## **44. Can Transformer Memory Be Corrupted? Investigating Cache-Side Vulnerabilities in Large Language Models**

cs.CR

**SubmitDate**: 2025-10-20    [abs](http://arxiv.org/abs/2510.17098v1) [paper-pdf](http://arxiv.org/pdf/2510.17098v1)

**Authors**: Elias Hossain, Swayamjit Saha, Somshubhra Roy, Ravi Prasad

**Abstract**: Even when prompts and parameters are secured, transformer language models remain vulnerable because their key-value (KV) cache during inference constitutes an overlooked attack surface. This paper introduces Malicious Token Injection (MTI), a modular framework that systematically perturbs cached key vectors at selected layers and timesteps through controlled magnitude and frequency, using additive Gaussian noise, zeroing, and orthogonal rotations. A theoretical analysis quantifies how these perturbations propagate through attention, linking logit deviations to the Frobenius norm of corruption and softmax Lipschitz dynamics. Empirical results show that MTI significantly alters next-token distributions and downstream task performance across GPT-2 and LLaMA-2/7B, as well as destabilizes retrieval-augmented and agentic reasoning pipelines. These findings identify cache integrity as a critical yet underexplored vulnerability in current LLM deployments, positioning cache corruption as a reproducible and theoretically grounded threat model for future robustness and security research.



## **45. System Prompt Poisoning: Persistent Attacks on Large Language Models Beyond User Injection**

cs.CR

**SubmitDate**: 2025-10-19    [abs](http://arxiv.org/abs/2505.06493v3) [paper-pdf](http://arxiv.org/pdf/2505.06493v3)

**Authors**: Zongze Li, Jiawei Guo, Haipeng Cai

**Abstract**: Large language models (LLMs) have gained widespread adoption across diverse applications due to their impressive generative capabilities. Their plug-and-play nature enables both developers and end users to interact with these models through simple prompts. However, as LLMs become more integrated into various systems in diverse domains, concerns around their security are growing. Existing studies mainly focus on threats arising from user prompts (e.g. prompt injection attack) and model output (e.g. model inversion attack), while the security of system prompts remains largely overlooked. This work bridges the critical gap. We introduce system prompt poisoning, a new attack vector against LLMs that, unlike traditional user prompt injection, poisons system prompts hence persistently impacts all subsequent user interactions and model responses. We systematically investigate four practical attack strategies in various poisoning scenarios. Through demonstration on both generative and reasoning LLMs, we show that system prompt poisoning is highly feasible without requiring jailbreak techniques, and effective across a wide range of tasks, including those in mathematics, coding, logical reasoning, and natural language processing. Importantly, our findings reveal that the attack remains effective even when user prompts employ advanced prompting techniques like chain-of-thought (CoT). We also show that such techniques, including CoT and retrieval-augmentation-generation (RAG), which are proven to be effective for improving LLM performance in a wide range of tasks, are significantly weakened in their effectiveness by system prompt poisoning.



## **46. Forgetting to Forget: Attention Sink as A Gateway for Backdooring LLM Unlearning**

cs.LG

**SubmitDate**: 2025-10-19    [abs](http://arxiv.org/abs/2510.17021v1) [paper-pdf](http://arxiv.org/pdf/2510.17021v1)

**Authors**: Bingqi Shang, Yiwei Chen, Yihua Zhang, Bingquan Shen, Sijia Liu

**Abstract**: Large language model (LLM) unlearning has become a critical mechanism for removing undesired data, knowledge, or behaviors from pre-trained models while retaining their general utility. Yet, with the rise of open-weight LLMs, we ask: can the unlearning process itself be backdoored, appearing successful under normal conditions yet reverting to pre-unlearned behavior when a hidden trigger is activated? Drawing inspiration from classical backdoor attacks that embed triggers into training data to enforce specific behaviors, we investigate backdoor unlearning, where models forget as intended in the clean setting but recover forgotten knowledge when the trigger appears. We show that designing such attacks presents unique challenges, hinging on where triggers are placed and how backdoor training is reinforced. We uncover a strong link between backdoor efficacy and the attention sink phenomenon, i.e., shallow input tokens consistently attract disproportionate attention in LLMs. Our analysis reveals that these attention sinks serve as gateways for backdoor unlearning: placing triggers at sink positions and aligning their attention values markedly enhances backdoor persistence. Extensive experiments validate these findings, showing that attention-sink-guided backdoor unlearning reliably restores forgotten knowledge in the presence of backdoor triggers, while behaving indistinguishably from a normally unlearned model when triggers are absent. Code is available at https://github.com/OPTML-Group/Unlearn-Backdoor.



## **47. Online Learning Defense against Iterative Jailbreak Attacks via Prompt Optimization**

cs.CL

**SubmitDate**: 2025-10-19    [abs](http://arxiv.org/abs/2510.17006v1) [paper-pdf](http://arxiv.org/pdf/2510.17006v1)

**Authors**: Masahiro Kaneko, Zeerak Talat, Timothy Baldwin

**Abstract**: Iterative jailbreak methods that repeatedly rewrite and input prompts into large language models (LLMs) to induce harmful outputs -- using the model's previous responses to guide each new iteration -- have been found to be a highly effective attack strategy. Despite being an effective attack strategy against LLMs and their safety mechanisms, existing defenses do not proactively disrupt this dynamic trial-and-error cycle. In this study, we propose a novel framework that dynamically updates its defense strategy through online learning in response to each new prompt from iterative jailbreak methods. Leveraging the distinctions between harmful jailbreak-generated prompts and typical harmless prompts, we introduce a reinforcement learning-based approach that optimizes prompts to ensure appropriate responses for harmless tasks while explicitly rejecting harmful prompts. Additionally, to curb overfitting to the narrow band of partial input rewrites explored during an attack, we introduce Past-Direction Gradient Damping (PDGD). Experiments conducted on three LLMs show that our approach significantly outperforms five existing defense methods against five iterative jailbreak methods. Moreover, our results indicate that our prompt optimization strategy simultaneously enhances response quality for harmless tasks.



## **48. Bits Leaked per Query: Information-Theoretic Bounds on Adversarial Attacks against LLMs**

cs.CR

NeurIPS 2025 (spotlight)

**SubmitDate**: 2025-10-19    [abs](http://arxiv.org/abs/2510.17000v1) [paper-pdf](http://arxiv.org/pdf/2510.17000v1)

**Authors**: Masahiro Kaneko, Timothy Baldwin

**Abstract**: Adversarial attacks by malicious users that threaten the safety of large language models (LLMs) can be viewed as attempts to infer a target property $T$ that is unknown when an instruction is issued, and becomes knowable only after the model's reply is observed. Examples of target properties $T$ include the binary flag that triggers an LLM's harmful response or rejection, and the degree to which information deleted by unlearning can be restored, both elicited via adversarial instructions. The LLM reveals an \emph{observable signal} $Z$ that potentially leaks hints for attacking through a response containing answer tokens, thinking process tokens, or logits. Yet the scale of information leaked remains anecdotal, leaving auditors without principled guidance and defenders blind to the transparency--risk trade-off. We fill this gap with an information-theoretic framework that computes how much information can be safely disclosed, and enables auditors to gauge how close their methods come to the fundamental limit. Treating the mutual information $I(Z;T)$ between the observation $Z$ and the target property $T$ as the leaked bits per query, we show that achieving error $\varepsilon$ requires at least $\log(1/\varepsilon)/I(Z;T)$ queries, scaling linearly with the inverse leak rate and only logarithmically with the desired accuracy. Thus, even a modest increase in disclosure collapses the attack cost from quadratic to logarithmic in terms of the desired accuracy. Experiments on seven LLMs across system-prompt leakage, jailbreak, and relearning attacks corroborate the theory: exposing answer tokens alone requires about a thousand queries; adding logits cuts this to about a hundred; and revealing the full thinking process trims it to a few dozen. Our results provide the first principled yardstick for balancing transparency and security when deploying LLMs.



## **49. BreakFun: Jailbreaking LLMs via Schema Exploitation**

cs.CR

**SubmitDate**: 2025-10-19    [abs](http://arxiv.org/abs/2510.17904v1) [paper-pdf](http://arxiv.org/pdf/2510.17904v1)

**Authors**: Amirkia Rafiei Oskooei, Mehmet S. Aktas

**Abstract**: The proficiency of Large Language Models (LLMs) in processing structured data and adhering to syntactic rules is a capability that drives their widespread adoption but also makes them paradoxically vulnerable. In this paper, we investigate this vulnerability through BreakFun, a jailbreak methodology that weaponizes an LLM's adherence to structured schemas. BreakFun employs a three-part prompt that combines an innocent framing and a Chain-of-Thought distraction with a core "Trojan Schema"--a carefully crafted data structure that compels the model to generate harmful content, exploiting the LLM's strong tendency to follow structures and schemas. We demonstrate this vulnerability is highly transferable, achieving an average success rate of 89% across 13 foundational and proprietary models on JailbreakBench, and reaching a 100% Attack Success Rate (ASR) on several prominent models. A rigorous ablation study confirms this Trojan Schema is the attack's primary causal factor. To counter this, we introduce the Adversarial Prompt Deconstruction guardrail, a defense that utilizes a secondary LLM to perform a "Literal Transcription"--extracting all human-readable text to isolate and reveal the user's true harmful intent. Our proof-of-concept guardrail demonstrates high efficacy against the attack, validating that targeting the deceptive schema is a viable mitigation strategy. Our work provides a look into how an LLM's core strengths can be turned into critical weaknesses, offering a fresh perspective for building more robustly aligned models.



## **50. Black-box Optimization of LLM Outputs by Asking for Directions**

cs.CR

**SubmitDate**: 2025-10-19    [abs](http://arxiv.org/abs/2510.16794v1) [paper-pdf](http://arxiv.org/pdf/2510.16794v1)

**Authors**: Jie Zhang, Meng Ding, Yang Liu, Jue Hong, Florian Tramèr

**Abstract**: We present a novel approach for attacking black-box large language models (LLMs) by exploiting their ability to express confidence in natural language. Existing black-box attacks require either access to continuous model outputs like logits or confidence scores (which are rarely available in practice), or rely on proxy signals from other models. Instead, we demonstrate how to prompt LLMs to express their internal confidence in a way that is sufficiently calibrated to enable effective adversarial optimization. We apply our general method to three attack scenarios: adversarial examples for vision-LLMs, jailbreaks and prompt injections. Our attacks successfully generate malicious inputs against systems that only expose textual outputs, thereby dramatically expanding the attack surface for deployed LLMs. We further find that better and larger models exhibit superior calibration when expressing confidence, creating a concerning security paradox where model capability improvements directly enhance vulnerability. Our code is available at this [link](https://github.com/zj-jayzhang/black_box_llm_optimization).



