# Latest Large Language Model Attack Papers
**update at 2025-09-01 10:11:45**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. WebInject: Prompt Injection Attack to Web Agents**

cs.LG

EMNLP 2025 main

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2505.11717v3) [paper-pdf](http://arxiv.org/pdf/2505.11717v3)

**Authors**: Xilong Wang, John Bloch, Zedian Shao, Yuepeng Hu, Shuyan Zhou, Neil Zhenqiang Gong

**Abstract**: Multi-modal large language model (MLLM)-based web agents interact with webpage environments by generating actions based on screenshots of the webpages. In this work, we propose WebInject, a prompt injection attack that manipulates the webpage environment to induce a web agent to perform an attacker-specified action. Our attack adds a perturbation to the raw pixel values of the rendered webpage. After these perturbed pixels are mapped into a screenshot, the perturbation induces the web agent to perform the attacker-specified action. We formulate the task of finding the perturbation as an optimization problem. A key challenge in solving this problem is that the mapping between raw pixel values and screenshot is non-differentiable, making it difficult to backpropagate gradients to the perturbation. To overcome this, we train a neural network to approximate the mapping and apply projected gradient descent to solve the reformulated optimization problem. Extensive evaluation on multiple datasets shows that WebInject is highly effective and significantly outperforms baselines.



## **2. Detecting Stealthy Data Poisoning Attacks in AI Code Generators**

cs.CR

Accepted to the 3rd IEEE International Workshop on Reliable and  Secure AI for Software Engineering (ReSAISE, 2025), co-located with ISSRE  2025

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.21636v1) [paper-pdf](http://arxiv.org/pdf/2508.21636v1)

**Authors**: Cristina Improta

**Abstract**: Deep learning (DL) models for natural language-to-code generation have become integral to modern software development pipelines. However, their heavy reliance on large amounts of data, often collected from unsanitized online sources, exposes them to data poisoning attacks, where adversaries inject malicious samples to subtly bias model behavior. Recent targeted attacks silently replace secure code with semantically equivalent but vulnerable implementations without relying on explicit triggers to launch the attack, making it especially hard for detection methods to distinguish clean from poisoned samples. We present a systematic study on the effectiveness of existing poisoning detection methods under this stealthy threat model. Specifically, we perform targeted poisoning on three DL models (CodeBERT, CodeT5+, AST-T5), and evaluate spectral signatures analysis, activation clustering, and static analysis as defenses. Our results show that all methods struggle to detect triggerless poisoning, with representation-based approaches failing to isolate poisoned samples and static analysis suffering false positives and false negatives, highlighting the need for more robust, trigger-independent defenses for AI-assisted code generation.



## **3. SoK: Large Language Model-Generated Textual Phishing Campaigns End-to-End Analysis of Generation, Characteristics, and Detection**

cs.CR

13 pages, 3 tables, 4 figures

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.21457v1) [paper-pdf](http://arxiv.org/pdf/2508.21457v1)

**Authors**: Fengchao Chen, Tingmin Wu, Van Nguyen, Carsten Rudolph

**Abstract**: Phishing is a pervasive form of social engineering in which attackers impersonate trusted entities to steal information or induce harmful actions. Text-based phishing dominates for its low cost, scalability, and concealability, advantages recently amplified by large language models (LLMs) that enable ``Phishing-as-a-Service'' attacks at scale within minutes. Despite the growing research into LLM-facilitated phishing attacks, consolidated systematic research on the phishing attack life cycle remains scarce. In this work, we present the first systematization of knowledge (SoK) on LLM-generated phishing, offering an end-to-end analysis that spans generation techniques, attack features, and mitigation strategies. We introduce Generation-Characterization-Defense (GenCharDef), which systematizes the ways in which LLM-generated phishing differs from traditional phishing across methodologies, security perspectives, data dependencies, and evaluation practices. This framework highlights unique challenges of LLM-driven phishing, providing a coherent foundation for understanding the evolving threat landscape and guiding the design of more resilient defenses.



## **4. Publish to Perish: Prompt Injection Attacks on LLM-Assisted Peer Review**

cs.CR

**SubmitDate**: 2025-08-29    [abs](http://arxiv.org/abs/2508.20863v2) [paper-pdf](http://arxiv.org/pdf/2508.20863v2)

**Authors**: Matteo Gioele Collu, Umberto Salviati, Roberto Confalonieri, Mauro Conti, Giovanni Apruzzese

**Abstract**: Large Language Models (LLMs) are increasingly being integrated into the scientific peer-review process, raising new questions about their reliability and resilience to manipulation. In this work, we investigate the potential for hidden prompt injection attacks, where authors embed adversarial text within a paper's PDF to influence the LLM-generated review. We begin by formalising three distinct threat models that envision attackers with different motivations -- not all of which implying malicious intent. For each threat model, we design adversarial prompts that remain invisible to human readers yet can steer an LLM's output toward the author's desired outcome. Using a user study with domain scholars, we derive four representative reviewing prompts used to elicit peer reviews from LLMs. We then evaluate the robustness of our adversarial prompts across (i) different reviewing prompts, (ii) different commercial LLM-based systems, and (iii) different peer-reviewed papers. Our results show that adversarial prompts can reliably mislead the LLM, sometimes in ways that adversely affect a "honest-but-lazy" reviewer. Finally, we propose and empirically assess methods to reduce detectability of adversarial prompts under automated content checks.



## **5. Lethe: Purifying Backdoored Large Language Models with Knowledge Dilution**

cs.CL

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.21004v1) [paper-pdf](http://arxiv.org/pdf/2508.21004v1)

**Authors**: Chen Chen, Yuchen Sun, Jiaxin Gao, Xueluan Gong, Qian Wang, Ziyao Wang, Yongsen Zheng, Kwok-Yan Lam

**Abstract**: Large language models (LLMs) have seen significant advancements, achieving superior performance in various Natural Language Processing (NLP) tasks. However, they remain vulnerable to backdoor attacks, where models behave normally for standard queries but generate harmful responses or unintended output when specific triggers are activated. Existing backdoor defenses either lack comprehensiveness, focusing on narrow trigger settings, detection-only mechanisms, and limited domains, or fail to withstand advanced scenarios like model-editing-based, multi-trigger, and triggerless attacks. In this paper, we present LETHE, a novel method to eliminate backdoor behaviors from LLMs through knowledge dilution using both internal and external mechanisms. Internally, LETHE leverages a lightweight dataset to train a clean model, which is then merged with the backdoored model to neutralize malicious behaviors by diluting the backdoor impact within the model's parametric memory. Externally, LETHE incorporates benign and semantically relevant evidence into the prompt to distract LLM's attention from backdoor features. Experimental results on classification and generation domains across 5 widely used LLMs demonstrate that LETHE outperforms 8 state-of-the-art defense baselines against 8 backdoor attacks. LETHE reduces the attack success rate of advanced backdoor attacks by up to 98% while maintaining model utility. Furthermore, LETHE has proven to be cost-efficient and robust against adaptive backdoor attacks.



## **6. PromptSleuth: Detecting Prompt Injection via Semantic Intent Invariance**

cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20890v1) [paper-pdf](http://arxiv.org/pdf/2508.20890v1)

**Authors**: Mengxiao Wang, Yuxuan Zhang, Guofei Gu

**Abstract**: Large Language Models (LLMs) are increasingly integrated into real-world applications, from virtual assistants to autonomous agents. However, their flexibility also introduces new attack vectors-particularly Prompt Injection (PI), where adversaries manipulate model behavior through crafted inputs. As attackers continuously evolve with paraphrased, obfuscated, and even multi-task injection strategies, existing benchmarks are no longer sufficient to capture the full spectrum of emerging threats.   To address this gap, we construct a new benchmark that systematically extends prior efforts. Our benchmark subsumes the two widely-used existing ones while introducing new manipulation techniques and multi-task scenarios, thereby providing a more comprehensive evaluation setting. We find that existing defenses, though effective on their original benchmarks, show clear weaknesses under our benchmark, underscoring the need for more robust solutions. Our key insight is that while attack forms may vary, the adversary's intent-injecting an unauthorized task-remains invariant. Building on this observation, we propose PromptSleuth, a semantic-oriented defense framework that detects prompt injection by reasoning over task-level intent rather than surface features. Evaluated across state-of-the-art benchmarks, PromptSleuth consistently outperforms existing defense while maintaining comparable runtime and cost efficiency. These results demonstrate that intent-based semantic reasoning offers a robust, efficient, and generalizable strategy for defending LLMs against evolving prompt injection threats.



## **7. Multi-Agent Penetration Testing AI for the Web**

cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20816v1) [paper-pdf](http://arxiv.org/pdf/2508.20816v1)

**Authors**: Isaac David, Arthur Gervais

**Abstract**: AI-powered development platforms are making software creation accessible to a broader audience, but this democratization has triggered a scalability crisis in security auditing. With studies showing that up to 40% of AI-generated code contains vulnerabilities, the pace of development now vastly outstrips the capacity for thorough security assessment.   We present MAPTA, a multi-agent system for autonomous web application security assessment that combines large language model orchestration with tool-grounded execution and end-to-end exploit validation. On the 104-challenge XBOW benchmark, MAPTA achieves 76.9% overall success with perfect performance on SSRF and misconfiguration vulnerabilities, 83% success on broken authorization, and strong results on injection attacks including server-side template injection (85%) and SQL injection (83%). Cross-site scripting (57%) and blind SQL injection (0%) remain challenging. Our comprehensive cost analysis across all challenges totals $21.38 with a median cost of $0.073 for successful attempts versus $0.357 for failures. Success correlates strongly with resource efficiency, enabling practical early-stopping thresholds at approximately 40 tool calls or $0.30 per challenge.   MAPTA's real-world findings are impactful given both the popularity of the respective scanned GitHub repositories (8K-70K stars) and MAPTA's low average operating cost of $3.67 per open-source assessment: MAPTA discovered critical vulnerabilities including RCEs, command injections, secret exposure, and arbitrary file write vulnerabilities. Findings are responsibly disclosed, 10 findings are under CVE review.



## **8. Addressing Tokenization Inconsistency in Steganography and Watermarking Based on Large Language Models**

cs.CL

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20718v1) [paper-pdf](http://arxiv.org/pdf/2508.20718v1)

**Authors**: Ruiyi Yan, Yugo Murawaki

**Abstract**: Large language models have significantly enhanced the capacities and efficiency of text generation. On the one hand, they have improved the quality of text-based steganography. On the other hand, they have also underscored the importance of watermarking as a safeguard against malicious misuse. In this study, we focus on tokenization inconsistency (TI) between Alice and Bob in steganography and watermarking, where TI can undermine robustness. Our investigation reveals that the problematic tokens responsible for TI exhibit two key characteristics: infrequency and temporariness. Based on these findings, we propose two tailored solutions for TI elimination: a stepwise verification method for steganography and a post-hoc rollback method for watermarking. Experiments show that (1) compared to traditional disambiguation methods in steganography, directly addressing TI leads to improvements in fluency, imperceptibility, and anti-steganalysis capacity; (2) for watermarking, addressing TI enhances detectability and robustness against attacks.



## **9. Token Buncher: Shielding LLMs from Harmful Reinforcement Learning Fine-Tuning**

cs.LG

Project Hompage: https://tokenbuncher.github.io/

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20697v1) [paper-pdf](http://arxiv.org/pdf/2508.20697v1)

**Authors**: Weitao Feng, Lixu Wang, Tianyi Wei, Jie Zhang, Chongyang Gao, Sinong Zhan, Peizhuo Lv, Wei Dong

**Abstract**: As large language models (LLMs) continue to grow in capability, so do the risks of harmful misuse through fine-tuning. While most prior studies assume that attackers rely on supervised fine-tuning (SFT) for such misuse, we systematically demonstrate that reinforcement learning (RL) enables adversaries to more effectively break safety alignment and facilitate advanced harmful task assistance, under matched computational budgets. To counter this emerging threat, we propose TokenBuncher, the first effective defense specifically targeting RL-based harmful fine-tuning. TokenBuncher suppresses the foundation on which RL relies: model response uncertainty. By constraining uncertainty, RL-based fine-tuning can no longer exploit distinct reward signals to drive the model toward harmful behaviors. We realize this defense through entropy-as-reward RL and a Token Noiser mechanism designed to prevent the escalation of expert-domain harmful capabilities. Extensive experiments across multiple models and RL algorithms show that TokenBuncher robustly mitigates harmful RL fine-tuning while preserving benign task utility and finetunability. Our results highlight that RL-based harmful fine-tuning poses a greater systemic risk than SFT, and that TokenBuncher provides an effective and general defense.



## **10. CyberSleuth: Autonomous Blue-Team LLM Agent for Web Attack Forensics**

cs.CR

Code:  https://github.com/SmartData-Polito/LLM_Agent_Cybersecurity_Forensic

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20643v1) [paper-pdf](http://arxiv.org/pdf/2508.20643v1)

**Authors**: Stefano Fumero, Kai Huang, Matteo Boffa, Danilo Giordano, Marco Mellia, Zied Ben Houidi, Dario Rossi

**Abstract**: Large Language Model (LLM) agents are powerful tools for automating complex tasks. In cybersecurity, researchers have primarily explored their use in red-team operations such as vulnerability discovery and penetration tests. Defensive uses for incident response and forensics have received comparatively less attention and remain at an early stage. This work presents a systematic study of LLM-agent design for the forensic investigation of realistic web application attacks. We propose CyberSleuth, an autonomous agent that processes packet-level traces and application logs to identify the targeted service, the exploited vulnerability (CVE), and attack success. We evaluate the consequences of core design decisions - spanning tool integration and agent architecture - and provide interpretable guidance for practitioners. We benchmark four agent architectures and six LLM backends on 20 incident scenarios of increasing complexity, identifying CyberSleuth as the best-performing design. In a separate set of 10 incidents from 2025, CyberSleuth correctly identifies the exact CVE in 80% of cases. At last, we conduct a human study with 22 experts, which rated the reports of CyberSleuth as complete, useful, and coherent. They also expressed a slight preference for DeepSeek R1, a good news for open source LLM. To foster progress in defensive LLM research, we release both our benchmark and the CyberSleuth platform as a foundation for fair, reproducible evaluation of forensic agents.



## **11. NetGPT: Generative Pretrained Transformer for Network Traffic**

cs.NI

Code is available at https://github.com/ict-net/NetGPT

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2304.09513v3) [paper-pdf](http://arxiv.org/pdf/2304.09513v3)

**Authors**: Xuying Meng, Chungang Lin, Yequan Wang, Yujun Zhang

**Abstract**: All data on the Internet are transferred by network traffic, thus accurately modeling network traffic can help improve network services quality and protect data privacy. Pretrained models for network traffic can utilize large-scale raw data to learn the essential characteristics of network traffic, and generate distinguishable results for input traffic without considering specific downstream tasks. Effective pretrained models can significantly optimize the training efficiency and effectiveness of downstream tasks, such as application classification, attack detection and traffic generation. Despite the great success of pretraining in natural language processing, there is no work in the network field. Considering the diverse demands and characteristics of network traffic and network tasks, it is non-trivial to build a pretrained model for network traffic and we face various challenges, especially the heterogeneous headers and payloads in the multi-pattern network traffic and the different dependencies for contexts of diverse downstream network tasks.   To tackle these challenges, in this paper, we make the first attempt to provide a generative pretrained model NetGPT for both traffic understanding and generation tasks. We propose the multi-pattern network traffic modeling to construct unified text inputs and support both traffic understanding and generation tasks. We further optimize the adaptation effect of the pretrained model to diversified tasks by shuffling header fields, segmenting packets in flows, and incorporating diverse task labels with prompts. With diverse traffic datasets from encrypted software, DNS, private industrial protocols and cryptocurrency mining, expensive experiments demonstrate the effectiveness of our NetGPT in a range of traffic understanding and generation tasks on traffic datasets, and outperform state-of-the-art baselines by a wide margin.



## **12. Probabilistic Modeling of Jailbreak on Multimodal LLMs: From Quantification to Application**

cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2503.06989v4) [paper-pdf](http://arxiv.org/pdf/2503.06989v4)

**Authors**: Wenzhuo Xu, Zhipeng Wei, Xiongtao Sun, Zonghao Ying, Deyue Zhang, Dongdong Yang, Xiangzheng Zhang, Quanchen Zou

**Abstract**: Recently, Multimodal Large Language Models (MLLMs) have demonstrated their superior ability in understanding multimodal content. However, they remain vulnerable to jailbreak attacks, which exploit weaknesses in their safety alignment to generate harmful responses. Previous studies categorize jailbreaks as successful or failed based on whether responses contain malicious content. However, given the stochastic nature of MLLM responses, this binary classification of an input's ability to jailbreak MLLMs is inappropriate. Derived from this viewpoint, we introduce jailbreak probability to quantify the jailbreak potential of an input, which represents the likelihood that MLLMs generated a malicious response when prompted with this input. We approximate this probability through multiple queries to MLLMs. After modeling the relationship between input hidden states and their corresponding jailbreak probability using Jailbreak Probability Prediction Network (JPPN), we use continuous jailbreak probability for optimization. Specifically, we propose Jailbreak-Probability-based Attack (JPA) that optimizes adversarial perturbations on input image to maximize jailbreak probability, and further enhance it as Multimodal JPA (MJPA) by including monotonic text rephrasing. To counteract attacks, we also propose Jailbreak-Probability-based Finetuning (JPF), which minimizes jailbreak probability through MLLM parameter updates. Extensive experiments show that (1) (M)JPA yields significant improvements when attacking a wide range of models under both white and black box settings. (2) JPF vastly reduces jailbreaks by at most over 60\%. Both of the above results demonstrate the significance of introducing jailbreak probability to make nuanced distinctions among input jailbreak abilities.



## **13. Ransomware 3.0: Self-Composing and LLM-Orchestrated**

cs.CR

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20444v1) [paper-pdf](http://arxiv.org/pdf/2508.20444v1)

**Authors**: Md Raz, Meet Udeshi, P. V. Sai Charan, Prashanth Krishnamurthy, Farshad Khorrami, Ramesh Karri

**Abstract**: Using automated reasoning, code synthesis, and contextual decision-making, we introduce a new threat that exploits large language models (LLMs) to autonomously plan, adapt, and execute the ransomware attack lifecycle. Ransomware 3.0 represents the first threat model and research prototype of LLM-orchestrated ransomware. Unlike conventional malware, the prototype only requires natural language prompts embedded in the binary; malicious code is synthesized dynamically by the LLM at runtime, yielding polymorphic variants that adapt to the execution environment. The system performs reconnaissance, payload generation, and personalized extortion, in a closed-loop attack campaign without human involvement. We evaluate this threat across personal, enterprise, and embedded environments using a phase-centric methodology that measures quantitative fidelity and qualitative coherence in each attack phase. We show that open source LLMs can generate functional ransomware components and sustain closed-loop execution across diverse environments. Finally, we present behavioral signals and multi-level telemetry of Ransomware 3.0 through a case study to motivate future development of better defenses and policy enforcements to address novel AI-enabled ransomware attacks.



## **14. Forewarned is Forearmed: Pre-Synthesizing Jailbreak-like Instructions to Enhance LLM Safety Guardrail to Potential Attacks**

cs.CL

EMNLP 2025 findings

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20038v2) [paper-pdf](http://arxiv.org/pdf/2508.20038v2)

**Authors**: Sheng Liu, Qiang Sheng, Danding Wang, Yang Li, Guang Yang, Juan Cao

**Abstract**: Despite advances in improving large language model (LLM) to refuse to answer malicious instructions, widely used LLMs remain vulnerable to jailbreak attacks where attackers generate instructions with distributions differing from safety alignment corpora. New attacks expose LLMs' inability to recognize unseen malicious instructions, highlighting a critical distributional mismatch between training data and real-world attacks that forces developers into reactive patching cycles. To tackle this challenge, we propose IMAGINE, a synthesis framework that leverages embedding space distribution analysis to generate jailbreak-like instructions. This approach effectively fills the distributional gap between authentic jailbreak patterns and safety alignment corpora. IMAGINE follows an iterative optimization process that dynamically evolves text generation distributions across iterations, thereby augmenting the coverage of safety alignment data distributions through synthesized data examples. Based on the safety-aligned corpus enhanced through IMAGINE, our framework demonstrates significant decreases in attack success rate on Qwen2.5, Llama3.1, and Llama3.2 without compromising their utility.



## **15. Poison Once, Refuse Forever: Weaponizing Alignment for Injecting Bias in LLMs**

cs.LG

**SubmitDate**: 2025-08-28    [abs](http://arxiv.org/abs/2508.20333v1) [paper-pdf](http://arxiv.org/pdf/2508.20333v1)

**Authors**: Md Abdullah Al Mamun, Ihsen Alouani, Nael Abu-Ghazaleh

**Abstract**: Large Language Models (LLMs) are aligned to meet ethical standards and safety requirements by training them to refuse answering harmful or unsafe prompts. In this paper, we demonstrate how adversaries can exploit LLMs' alignment to implant bias, or enforce targeted censorship without degrading the model's responsiveness to unrelated topics. Specifically, we propose Subversive Alignment Injection (SAI), a poisoning attack that leverages the alignment mechanism to trigger refusal on specific topics or queries predefined by the adversary. Although it is perhaps not surprising that refusal can be induced through overalignment, we demonstrate how this refusal can be exploited to inject bias into the model. Surprisingly, SAI evades state-of-the-art poisoning defenses including LLM state forensics, as well as robust aggregation techniques that are designed to detect poisoning in FL settings. We demonstrate the practical dangers of this attack by illustrating its end-to-end impacts on LLM-powered application pipelines. For chat based applications such as ChatDoctor, with 1% data poisoning, the system refuses to answer healthcare questions to targeted racial category leading to high bias ($\Delta DP$ of 23%). We also show that bias can be induced in other NLP tasks: for a resume selection pipeline aligned to refuse to summarize CVs from a selected university, high bias in selection ($\Delta DP$ of 27%) results. Even higher bias ($\Delta DP$~38%) results on 9 other chat based downstream applications.



## **16. CoCoTen: Detecting Adversarial Inputs to Large Language Models through Latent Space Features of Contextual Co-occurrence Tensors**

cs.CL

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.02997v3) [paper-pdf](http://arxiv.org/pdf/2508.02997v3)

**Authors**: Sri Durga Sai Sowmya Kadali, Evangelos E. Papalexakis

**Abstract**: The widespread use of Large Language Models (LLMs) in many applications marks a significant advance in research and practice. However, their complexity and hard-to-understand nature make them vulnerable to attacks, especially jailbreaks designed to produce harmful responses. To counter these threats, developing strong detection methods is essential for the safe and reliable use of LLMs. This paper studies this detection problem using the Contextual Co-occurrence Matrix, a structure recognized for its efficacy in data-scarce environments. We propose a novel method leveraging the latent space characteristics of Contextual Co-occurrence Matrices and Tensors for the effective identification of adversarial and jailbreak prompts. Our evaluations show that this approach achieves a notable F1 score of 0.83 using only 0.5% of labeled prompts, which is a 96.6% improvement over baselines. This result highlights the strength of our learned patterns, especially when labeled data is scarce. Our method is also significantly faster, speedup ranging from 2.3 to 128.4 times compared to the baseline models.



## **17. Disabling Self-Correction in Retrieval-Augmented Generation via Stealthy Retriever Poisoning**

cs.CR

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.20083v1) [paper-pdf](http://arxiv.org/pdf/2508.20083v1)

**Authors**: Yanbo Dai, Zhenlan Ji, Zongjie Li, Kuan Li, Shuai Wang

**Abstract**: Retrieval-Augmented Generation (RAG) has become a standard approach for improving the reliability of large language models (LLMs). Prior work demonstrates the vulnerability of RAG systems by misleading them into generating attacker-chosen outputs through poisoning the knowledge base. However, this paper uncovers that such attacks could be mitigated by the strong \textit{self-correction ability (SCA)} of modern LLMs, which can reject false context once properly configured. This SCA poses a significant challenge for attackers aiming to manipulate RAG systems.   In contrast to previous poisoning methods, which primarily target the knowledge base, we introduce \textsc{DisarmRAG}, a new poisoning paradigm that compromises the retriever itself to suppress the SCA and enforce attacker-chosen outputs. This compromisation enables the attacker to straightforwardly embed anti-SCA instructions into the context provided to the generator, thereby bypassing the SCA. To this end, we present a contrastive-learning-based model editing technique that performs localized and stealthy edits, ensuring the retriever returns a malicious instruction only for specific victim queries while preserving benign retrieval behavior. To further strengthen the attack, we design an iterative co-optimization framework that automatically discovers robust instructions capable of bypassing prompt-based defenses. We extensively evaluate DisarmRAG across six LLMs and three QA benchmarks. Our results show near-perfect retrieval of malicious instructions, which successfully suppress SCA and achieve attack success rates exceeding 90\% under diverse defensive prompts. Also, the edited retriever remains stealthy under several detection methods, highlighting the urgent need for retriever-centric defenses.



## **18. Scaling Decentralized Learning with FLock**

cs.LG

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2507.15349v2) [paper-pdf](http://arxiv.org/pdf/2507.15349v2)

**Authors**: Zehua Cheng, Rui Sun, Jiahao Sun, Yike Guo

**Abstract**: Fine-tuning the large language models (LLMs) are prevented by the deficiency of centralized control and the massive computing and communication overhead on the decentralized schemes. While the typical standard federated learning (FL) supports data privacy, the central server requirement creates a single point of attack and vulnerability to poisoning attacks. Generalizing the result in this direction to 70B-parameter models in the heterogeneous, trustless environments has turned out to be a huge, yet unbroken bottleneck. This paper introduces FLock, a decentralized framework for secure and efficient collaborative LLM fine-tuning. Integrating a blockchain-based trust layer with economic incentives, FLock replaces the central aggregator with a secure, auditable protocol for cooperation among untrusted parties. We present the first empirical validation of fine-tuning a 70B LLM in a secure, multi-domain, decentralized setting. Our experiments show the FLock framework defends against backdoor poisoning attacks that compromise standard FL optimizers and fosters synergistic knowledge transfer. The resulting models show a >68% reduction in adversarial attack success rates. The global model also demonstrates superior cross-domain generalization, outperforming models trained in isolation on their own specialized data.



## **19. IntentionReasoner: Facilitating Adaptive LLM Safeguards through Intent Reasoning and Selective Query Refinement**

cs.AI

17 pages, 9 figures

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.20151v1) [paper-pdf](http://arxiv.org/pdf/2508.20151v1)

**Authors**: Yuanzhe Shen, Zisu Huang, Zhengkang Guo, Yide Liu, Guanxu Chen, Ruicheng Yin, Xiaoqing Zheng, Xuanjing Huang

**Abstract**: The rapid advancement of large language models (LLMs) has driven their adoption across diverse domains, yet their ability to generate harmful content poses significant safety challenges. While extensive research has focused on mitigating harmful outputs, such efforts often come at the cost of excessively rejecting harmless prompts. Striking a balance among safety, over-refusal, and utility remains a critical challenge. In this work, we introduce IntentionReasoner, a novel safeguard mechanism that leverages a dedicated guard model to perform intent reasoning, multi-level safety classification, and query rewriting to neutralize potentially harmful intent in edge-case queries. Specifically, we first construct a comprehensive dataset comprising approximately 163,000 queries, each annotated with intent reasoning, safety labels, and rewritten versions. Supervised fine-tuning is then applied to equip the guard model with foundational capabilities in format adherence, intent analysis, and safe rewriting. Finally, we apply a tailored multi-reward optimization strategy that integrates rule-based heuristics and reward model signals within a reinforcement learning framework to further enhance performance. Extensive experiments show that IntentionReasoner excels in multiple safeguard benchmarks, generation quality evaluations, and jailbreak attack scenarios, significantly enhancing safety while effectively reducing over-refusal rates and improving the quality of responses.



## **20. Secure Multi-LLM Agentic AI and Agentification for Edge General Intelligence by Zero-Trust: A Survey**

cs.NI

35 pages

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.19870v1) [paper-pdf](http://arxiv.org/pdf/2508.19870v1)

**Authors**: Yinqiu Liu, Ruichen Zhang, Haoxiang Luo, Yijing Lin, Geng Sun, Dusit Niyato, Hongyang Du, Zehui Xiong, Yonggang Wen, Abbas Jamalipour, Dong In Kim, Ping Zhang

**Abstract**: Agentification serves as a critical enabler of Edge General Intelligence (EGI), transforming massive edge devices into cognitive agents through integrating Large Language Models (LLMs) and perception, reasoning, and acting modules. These agents collaborate across heterogeneous edge infrastructures, forming multi-LLM agentic AI systems that leverage collective intelligence and specialized capabilities to tackle complex, multi-step tasks. However, the collaborative nature of multi-LLM systems introduces critical security vulnerabilities, including insecure inter-LLM communications, expanded attack surfaces, and cross-domain data leakage that traditional perimeter-based security cannot adequately address. To this end, this survey introduces zero-trust security of multi-LLM in EGI, a paradigmatic shift following the ``never trust, always verify'' principle. We begin by systematically analyzing the security risks in multi-LLM systems within EGI contexts. Subsequently, we present the vision of a zero-trust multi-LLM framework in EGI. We then survey key technical progress to facilitate zero-trust multi-LLM systems in EGI. Particularly, we categorize zero-trust security mechanisms into model- and system-level approaches. The former and latter include strong identification, context-aware access control, etc., and proactive maintenance, blockchain-based management, etc., respectively. Finally, we identify critical research directions. This survey serves as the first systematic treatment of zero-trust applied to multi-LLM systems, providing both theoretical foundations and practical strategies.



## **21. Safety Alignment Should Be Made More Than Just A Few Attention Heads**

cs.CR

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2508.19697v1) [paper-pdf](http://arxiv.org/pdf/2508.19697v1)

**Authors**: Chao Huang, Zefeng Zhang, Juewei Yue, Quangang Li, Chuang Zhang, Tingwen Liu

**Abstract**: Current safety alignment for large language models(LLMs) continues to present vulnerabilities, given that adversarial prompting can effectively bypass their safety measures.Our investigation shows that these safety mechanisms predominantly depend on a limited subset of attention heads: removing or ablating these heads can severely compromise model safety. To identify and evaluate these safety-critical components, we introduce RDSHA, a targeted ablation method that leverages the model's refusal direction to pinpoint attention heads mostly responsible for safety behaviors. Further analysis shows that existing jailbreak attacks exploit this concentration by selectively bypassing or manipulating these critical attention heads. To address this issue, we propose AHD, a novel training strategy designed to promote the distributed encoding of safety-related behaviors across numerous attention heads. Experimental results demonstrate that AHD successfully distributes safety-related capabilities across more attention heads. Moreover, evaluations under several mainstream jailbreak attacks show that models trained with AHD exhibit considerably stronger safety robustness, while maintaining overall functional utility.



## **22. PromptKeeper: Safeguarding System Prompts for LLMs**

cs.CR

Accepted to the Findings of EMNLP 2025. 17 pages, 6 figures, 3 tables

**SubmitDate**: 2025-08-27    [abs](http://arxiv.org/abs/2412.13426v3) [paper-pdf](http://arxiv.org/pdf/2412.13426v3)

**Authors**: Zhifeng Jiang, Zhihua Jin, Guoliang He

**Abstract**: System prompts are widely used to guide the outputs of large language models (LLMs). These prompts often contain business logic and sensitive information, making their protection essential. However, adversarial and even regular user queries can exploit LLM vulnerabilities to expose these hidden prompts. To address this issue, we propose PromptKeeper, a defense mechanism designed to safeguard system prompts by tackling two core challenges: reliably detecting leakage and mitigating side-channel vulnerabilities when leakage occurs. By framing detection as a hypothesis-testing problem, PromptKeeper effectively identifies both explicit and subtle leakage. Upon leakage detected, it regenerates responses using a dummy prompt, ensuring that outputs remain indistinguishable from typical interactions when no leakage is present. PromptKeeper ensures robust protection against prompt extraction attacks via either adversarial or regular queries, while preserving conversational capability and runtime efficiency during benign user interactions.



## **23. MEraser: An Effective Fingerprint Erasure Approach for Large Language Models**

cs.CR

Accepted by ACL 2025, Main Conference, Long Paper

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2506.12551v2) [paper-pdf](http://arxiv.org/pdf/2506.12551v2)

**Authors**: Jingxuan Zhang, Zhenhua Xu, Rui Hu, Wenpeng Xing, Xuhong Zhang, Meng Han

**Abstract**: Large Language Models (LLMs) have become increasingly prevalent across various sectors, raising critical concerns about model ownership and intellectual property protection. Although backdoor-based fingerprinting has emerged as a promising solution for model authentication, effective attacks for removing these fingerprints remain largely unexplored. Therefore, we present Mismatched Eraser (MEraser), a novel method for effectively removing backdoor-based fingerprints from LLMs while maintaining model performance. Our approach leverages a two-phase fine-tuning strategy utilizing carefully constructed mismatched and clean datasets. Through extensive evaluation across multiple LLM architectures and fingerprinting methods, we demonstrate that MEraser achieves complete fingerprinting removal while maintaining model performance with minimal training data of fewer than 1,000 samples. Furthermore, we introduce a transferable erasure mechanism that enables effective fingerprinting removal across different models without repeated training. In conclusion, our approach provides a practical solution for fingerprinting removal in LLMs, reveals critical vulnerabilities in current fingerprinting techniques, and establishes comprehensive evaluation benchmarks for developing more resilient model protection methods in the future.



## **24. An Investigation on Group Query Hallucination Attacks**

cs.CR

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.19321v1) [paper-pdf](http://arxiv.org/pdf/2508.19321v1)

**Authors**: Kehao Miao, Xiaolong Jin

**Abstract**: With the widespread use of large language models (LLMs), understanding their potential failure modes during user interactions is essential. In practice, users often pose multiple questions in a single conversation with LLMs. Therefore, in this study, we propose Group Query Attack, a technique that simulates this scenario by presenting groups of queries to LLMs simultaneously. We investigate how the accumulated context from consecutive prompts influences the outputs of LLMs. Specifically, we observe that Group Query Attack significantly degrades the performance of models fine-tuned on specific tasks. Moreover, we demonstrate that Group Query Attack induces a risk of triggering potential backdoors of LLMs. Besides, Group Query Attack is also effective in tasks involving reasoning, such as mathematical reasoning and code generation for pre-trained and aligned models.



## **25. RePPL: Recalibrating Perplexity by Uncertainty in Semantic Propagation and Language Generation for Explainable QA Hallucination Detection**

cs.CL

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2505.15386v2) [paper-pdf](http://arxiv.org/pdf/2505.15386v2)

**Authors**: Yiming Huang, Junyan Zhang, Zihao Wang, Biquan Bie, Yunzhong Qiu, Yi R. Fung, Xinlei He

**Abstract**: Large Language Models (LLMs) have become powerful, but hallucinations remain a vital obstacle to their trustworthy use. While previous works improved the capability of hallucination detection by measuring uncertainty, they all lack the ability to explain the provenance behind why hallucinations occur, i.e., which part of the inputs tends to trigger hallucinations. Recent works on the prompt attack indicate that uncertainty exists in semantic propagation, where attention mechanisms gradually fuse local token information into high-level semantics across layers. Meanwhile, uncertainty also emerges in language generation, due to its probability-based selection of high-level semantics for sampled generations. Based on that, we propose RePPL to recalibrate uncertainty measurement by these two aspects, which dispatches explainable uncertainty scores to each token and aggregates in Perplexity-style Log-Average form as total score. Experiments show that our method achieves the best comprehensive detection performance across various QA datasets on advanced models (average AUC of 0.833), and our method is capable of producing token-level uncertainty scores as explanations for the hallucination. Leveraging these scores, we preliminarily find the chaotic pattern of hallucination and showcase its promising usage.



## **26. The Double-edged Sword of LLM-based Data Reconstruction: Understanding and Mitigating Contextual Vulnerability in Word-level Differential Privacy Text Sanitization**

cs.CR

15 pages, 4 figures, 8 tables. Accepted to WPES @ CCS 2025

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18976v1) [paper-pdf](http://arxiv.org/pdf/2508.18976v1)

**Authors**: Stephen Meisenbacher, Alexandra Klymenko, Andreea-Elena Bodea, Florian Matthes

**Abstract**: Differentially private text sanitization refers to the process of privatizing texts under the framework of Differential Privacy (DP), providing provable privacy guarantees while also empirically defending against adversaries seeking to harm privacy. Despite their simplicity, DP text sanitization methods operating at the word level exhibit a number of shortcomings, among them the tendency to leave contextual clues from the original texts due to randomization during sanitization $\unicode{x2013}$ this we refer to as $\textit{contextual vulnerability}$. Given the powerful contextual understanding and inference capabilities of Large Language Models (LLMs), we explore to what extent LLMs can be leveraged to exploit the contextual vulnerability of DP-sanitized texts. We expand on previous work not only in the use of advanced LLMs, but also in testing a broader range of sanitization mechanisms at various privacy levels. Our experiments uncover a double-edged sword effect of LLM-based data reconstruction attacks on privacy and utility: while LLMs can indeed infer original semantics and sometimes degrade empirical privacy protections, they can also be used for good, to improve the quality and privacy of DP-sanitized texts. Based on our findings, we propose recommendations for using LLM data reconstruction as a post-processing step, serving to increase privacy protection by thinking adversarially.



## **27. SDGO: Self-Discrimination-Guided Optimization for Consistent Safety in Large Language Models**

cs.CL

Accepted by EMNLP 2025 (Main Conference), 15 pages, 4 figures, 6  tables

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.15648v2) [paper-pdf](http://arxiv.org/pdf/2508.15648v2)

**Authors**: Peng Ding, Wen Sun, Dailin Li, Wei Zou, Jiaming Wang, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs) excel at various natural language processing tasks but remain vulnerable to jailbreaking attacks that induce harmful content generation. In this paper, we reveal a critical safety inconsistency: LLMs can more effectively identify harmful requests as discriminators than defend against them as generators. This insight inspires us to explore aligning the model's inherent discrimination and generation capabilities. To this end, we propose SDGO (Self-Discrimination-Guided Optimization), a reinforcement learning framework that leverages the model's own discrimination capabilities as a reward signal to enhance generation safety through iterative self-improvement. Our method does not require any additional annotated data or external models during the training phase. Extensive experiments demonstrate that SDGO significantly improves model safety compared to both prompt-based and training-based baselines while maintaining helpfulness on general benchmarks. By aligning LLMs' discrimination and generation capabilities, SDGO brings robust performance against out-of-distribution (OOD) jailbreaking attacks. This alignment achieves tighter coupling between these two capabilities, enabling the model's generation capability to be further enhanced with only a small amount of discriminative samples. Our code and datasets are available at https://github.com/NJUNLP/SDGO.



## **28. sudoLLM: On Multi-role Alignment of Language Models**

cs.CL

Accepted to EMNLP 2025 (findings)

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2505.14607v3) [paper-pdf](http://arxiv.org/pdf/2505.14607v3)

**Authors**: Soumadeep Saha, Akshay Chaturvedi, Joy Mahapatra, Utpal Garain

**Abstract**: User authorization-based access privileges are a key feature in many safety-critical systems, but have not been extensively studied in the large language model (LLM) realm. In this work, drawing inspiration from such access control systems, we introduce sudoLLM, a novel framework that results in multi-role aligned LLMs, i.e., LLMs that account for, and behave in accordance with, user access rights. sudoLLM injects subtle user-based biases into queries and trains an LLM to utilize this bias signal in order to produce sensitive information if and only if the user is authorized. We present empirical results demonstrating that this approach shows substantially improved alignment, generalization, resistance to prefix-based jailbreaking attacks, and ``fails-closed''. The persistent tension between the language modeling objective and safety alignment, which is often exploited to jailbreak LLMs, is somewhat resolved with the aid of the injected bias signal. Our framework is meant as an additional security layer, and complements existing guardrail mechanisms for enhanced end-to-end safety with LLMs.



## **29. FALCON: Autonomous Cyber Threat Intelligence Mining with LLMs for IDS Rule Generation**

cs.CR

11 pages, 5 figures, 4 tables

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18684v1) [paper-pdf](http://arxiv.org/pdf/2508.18684v1)

**Authors**: Shaswata Mitra, Azim Bazarov, Martin Duclos, Sudip Mittal, Aritran Piplai, Md Rayhanur Rahman, Edward Zieglar, Shahram Rahimi

**Abstract**: Signature-based Intrusion Detection Systems (IDS) detect malicious activities by matching network or host activity against predefined rules. These rules are derived from extensive Cyber Threat Intelligence (CTI), which includes attack signatures and behavioral patterns obtained through automated tools and manual threat analysis, such as sandboxing. The CTI is then transformed into actionable rules for the IDS engine, enabling real-time detection and prevention. However, the constant evolution of cyber threats necessitates frequent rule updates, which delay deployment time and weaken overall security readiness. Recent advancements in agentic systems powered by Large Language Models (LLMs) offer the potential for autonomous IDS rule generation with internal evaluation. We introduce FALCON, an autonomous agentic framework that generates deployable IDS rules from CTI data in real-time and evaluates them using built-in multi-phased validators. To demonstrate versatility, we target both network (Snort) and host-based (YARA) mediums and construct a comprehensive dataset of IDS rules with their corresponding CTIs. Our evaluations indicate FALCON excels in automatic rule generation, with an average of 95% accuracy validated by qualitative evaluation with 84% inter-rater agreement among multiple cybersecurity analysts across all metrics. These results underscore the feasibility and effectiveness of LLM-driven data mining for real-time cyber threat mitigation.



## **30. Can LLMs Handle WebShell Detection? Overcoming Detection Challenges with Behavioral Function-Aware Framework**

cs.CR

Published as a conference paper at COLM 2025

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2504.13811v3) [paper-pdf](http://arxiv.org/pdf/2504.13811v3)

**Authors**: Feijiang Han, Jiaming Zhang, Chuyi Deng, Jianheng Tang, Yunhuai Liu

**Abstract**: WebShell attacks, where malicious scripts are injected into web servers, pose a significant cybersecurity threat. Traditional ML and DL methods are often hampered by challenges such as the need for extensive training data, catastrophic forgetting, and poor generalization. Recently, Large Language Models have emerged as powerful alternatives for code-related tasks, but their potential in WebShell detection remains underexplored. In this paper, we make two contributions: (1) a comprehensive evaluation of seven LLMs, including GPT-4, LLaMA 3.1 70B, and Qwen 2.5 variants, benchmarked against traditional sequence- and graph-based methods using a dataset of 26.59K PHP scripts, and (2) the Behavioral Function-Aware Detection (BFAD) framework, designed to address the specific challenges of applying LLMs to this domain. Our framework integrates three components: a Critical Function Filter that isolates malicious PHP function calls, a Context-Aware Code Extraction strategy that captures the most behaviorally indicative code segments, and Weighted Behavioral Function Profiling that enhances in-context learning by prioritizing the most relevant demonstrations based on discriminative function-level profiles. Our results show that, stemming from their distinct analytical strategies, larger LLMs achieve near-perfect precision but lower recall, while smaller models exhibit the opposite trade-off. However, all baseline models lag behind previous SOTA methods. With the application of BFAD, the performance of all LLMs improves significantly, yielding an average F1 score increase of 13.82%. Notably, larger models now outperform SOTA benchmarks, while smaller models such as Qwen-2.5-Coder-3B achieve performance competitive with traditional methods. This work is the first to explore the feasibility and limitations of LLMs for WebShell detection and provides solutions to address the challenges in this task.



## **31. Membership Inference Attacks on LLM-based Recommender Systems**

cs.IR

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2508.18665v1) [paper-pdf](http://arxiv.org/pdf/2508.18665v1)

**Authors**: Jiajie He, Yuechun Gu, Min-Chun Chen, Keke Chen

**Abstract**: Large language models (LLMs) based Recommender Systems (RecSys) can flexibly adapt recommendation systems to different domains. It utilizes in-context learning (ICL), i.e., the prompts, to customize the recommendation functions, which include sensitive historical user-specific item interactions, e.g., implicit feedback like clicked items or explicit product reviews. Such private information may be exposed to novel privacy attack. However, no study has been done on this important issue. We design four membership inference attacks (MIAs), aiming to reveal whether victims' historical interactions have been used by system prompts. They are \emph{direct inquiry, hallucination, similarity, and poisoning attacks}, each of which utilizes the unique features of LLMs or RecSys. We have carefully evaluated them on three LLMs that have been used to develop ICL-LLM RecSys and two well-known RecSys benchmark datasets. The results confirm that the MIA threat on LLM RecSys is realistic: direct inquiry and poisoning attacks showing significantly high attack advantages. We have also analyzed the factors affecting these attacks, such as the number of shots in system prompts and the position of the victim in the shots.



## **32. Large Language Model-Based Framework for Explainable Cyberattack Detection in Automatic Generation Control Systems**

cs.CR

Accepted Paper

**SubmitDate**: 2025-08-26    [abs](http://arxiv.org/abs/2507.22239v2) [paper-pdf](http://arxiv.org/pdf/2507.22239v2)

**Authors**: Muhammad Sharshar, Ahmad Mohammad Saber, Davor Svetinovic, Amr M. Youssef, Deepa Kundur, Ehab F. El-Saadany

**Abstract**: The increasing digitization of smart grids has improved operational efficiency but also introduced new cybersecurity vulnerabilities, such as False Data Injection Attacks (FDIAs) targeting Automatic Generation Control (AGC) systems. While machine learning (ML) and deep learning (DL) models have shown promise in detecting such attacks, their opaque decision-making limits operator trust and real-world applicability. This paper proposes a hybrid framework that integrates lightweight ML-based attack detection with natural language explanations generated by Large Language Models (LLMs). Classifiers such as LightGBM achieve up to 95.13% attack detection accuracy with only 0.004 s inference latency. Upon detecting a cyberattack, the system invokes LLMs, including GPT-3.5 Turbo, GPT-4 Turbo, and GPT-4o mini, to generate human-readable explanation of the event. Evaluated on 100 test samples, GPT-4o mini with 20-shot prompting achieved 93% accuracy in identifying the attack target, a mean absolute error of 0.075 pu in estimating attack magnitude, and 2.19 seconds mean absolute error (MAE) in estimating attack onset. These results demonstrate that the proposed framework effectively balances real-time detection with interpretable, high-fidelity explanations, addressing a critical need for actionable AI in smart grid cybersecurity.



## **33. Prefill-level Jailbreak: A Black-Box Risk Analysis of Large Language Models**

cs.CR

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2504.21038v2) [paper-pdf](http://arxiv.org/pdf/2504.21038v2)

**Authors**: Yakai Li, Jiekang Hu, Weiduan Sang, Luping Ma, Dongsheng Nie, Weijuan Zhang, Aimin Yu, Yi Su, Qingjia Huang, Qihang Zhou

**Abstract**: Large Language Models face security threats from jailbreak attacks. Existing research has predominantly focused on prompt-level attacks while largely ignoring the underexplored attack surface of user-controlled response prefilling. This functionality allows an attacker to dictate the beginning of a model's output, thereby shifting the attack paradigm from persuasion to direct state manipulation.In this paper, we present a systematic black-box security analysis of prefill-level jailbreak attacks. We categorize these new attacks and evaluate their effectiveness across fourteen language models. Our experiments show that prefill-level attacks achieve high success rates, with adaptive methods exceeding 99% on several models. Token-level probability analysis reveals that these attacks work through initial-state manipulation by changing the first-token probability from refusal to compliance.Furthermore, we show that prefill-level jailbreak can act as effective enhancers, increasing the success of existing prompt-level attacks by 10 to 15 percentage points. Our evaluation of several defense strategies indicates that conventional content filters offer limited protection. We find that a detection method focusing on the manipulative relationship between the prompt and the prefill is more effective. Our findings reveal a gap in current LLM safety alignment and highlight the need to address the prefill attack surface in future safety training.



## **34. Defending Against Prompt Injection With a Few DefensiveTokens**

cs.CR

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2507.07974v2) [paper-pdf](http://arxiv.org/pdf/2507.07974v2)

**Authors**: Sizhe Chen, Yizhu Wang, Nicholas Carlini, Chawin Sitawarin, David Wagner

**Abstract**: When large language model (LLM) systems interact with external data to perform complex tasks, a new attack, namely prompt injection, becomes a significant threat. By injecting instructions into the data accessed by the system, the attacker is able to override the initial user task with an arbitrary task directed by the attacker. To secure the system, test-time defenses, e.g., defensive prompting, have been proposed for system developers to attain security only when needed in a flexible manner. However, they are much less effective than training-time defenses that change the model parameters. Motivated by this, we propose DefensiveToken, a test-time defense with prompt injection robustness comparable to training-time alternatives. DefensiveTokens are newly inserted as special tokens, whose embeddings are optimized for security. In security-sensitive cases, system developers can append a few DefensiveTokens before the LLM input to achieve security with a minimal utility drop. In scenarios where security is less of a concern, developers can simply skip DefensiveTokens; the LLM system remains the same as there is no defense, generating high-quality responses. Thus, DefensiveTokens, if released alongside the model, allow a flexible switch between the state-of-the-art (SOTA) utility and almost-SOTA security at test time. The code is available at https://github.com/Sizhe-Chen/DefensiveToken.



## **35. Confidential Prompting: Privacy-preserving LLM Inference on Cloud**

cs.CR

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2409.19134v4) [paper-pdf](http://arxiv.org/pdf/2409.19134v4)

**Authors**: Caihua Li, In Gim, Lin Zhong

**Abstract**: This paper introduces a vision of confidential prompting: securing user prompts from untrusted, cloud-hosted large language model (LLM) provider while preserving model confidentiality, output invariance, and compute efficiency. As a first step toward this vision, we present Obfuscated Secure Partitioned Decoding (OSPD), a system built on two key innovations. First, Secure Partitioned Decoding (SPD) isolates user prompts within per-user processes residing in a confidential virtual machine (CVM) on the cloud, which are inaccessible for the cloud LLM while allowing it to generate tokens efficiently. Second, Prompt Obfuscation (PO) introduces a novel cryptographic technique that enhances SPD resilience against advanced prompt reconstruction attacks. Together, these innovations ensure OSPD protects both prompt and model confidentiality while maintaining service functionality. OSPD enables practical, privacy-preserving cloud-hosted LLM inference for sensitive applications, such as processing personal data, clinical records, and financial documents.



## **36. Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks**

cs.CL

23 pages, 10 figures, 11 tables

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2503.00187v2) [paper-pdf](http://arxiv.org/pdf/2503.00187v2)

**Authors**: Hanjiang Hu, Alexander Robey, Changliu Liu

**Abstract**: Large language models (LLMs) are shown to be vulnerable to jailbreaking attacks where adversarial prompts are designed to elicit harmful responses. While existing defenses effectively mitigate single-turn attacks by detecting and filtering unsafe inputs, they fail against multi-turn jailbreaks that exploit contextual drift over multiple interactions, gradually leading LLMs away from safe behavior. To address this challenge, we propose a safety steering framework grounded in safe control theory, ensuring invariant safety in multi-turn dialogues. Our approach models the dialogue with LLMs using state-space representations and introduces a novel neural barrier function (NBF) to detect and filter harmful queries emerging from evolving contexts proactively. Our method achieves invariant safety at each turn of dialogue by learning a safety predictor that accounts for adversarial queries, preventing potential context drift toward jailbreaks. Extensive experiments under multiple LLMs show that our NBF-based safety steering outperforms safety alignment, prompt-based steering and lightweight LLM guardrails baselines, offering stronger defenses against multi-turn jailbreaks while maintaining a better trade-off among safety, helpfulness and over-refusal. Check out the website here https://sites.google.com/view/llm-nbf/home . Our code is available on https://github.com/HanjiangHu/NBF-LLM .



## **37. Stand on The Shoulders of Giants: Building JailExpert from Previous Attack Experience**

cs.CR

18 pages, EMNLP 2025 Main Conference

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.19292v1) [paper-pdf](http://arxiv.org/pdf/2508.19292v1)

**Authors**: Xi Wang, Songlei Jian, Shasha Li, Xiaopeng Li, Bin Ji, Jun Ma, Xiaodong Liu, Jing Wang, Feilong Bao, Jianfeng Zhang, Baosheng Wang, Jie Yu

**Abstract**: Large language models (LLMs) generate human-aligned content under certain safety constraints. However, the current known technique ``jailbreak prompt'' can circumvent safety-aligned measures and induce LLMs to output malicious content. Research on Jailbreaking can help identify vulnerabilities in LLMs and guide the development of robust security frameworks. To circumvent the issue of attack templates becoming obsolete as models evolve, existing methods adopt iterative mutation and dynamic optimization to facilitate more automated jailbreak attacks. However, these methods face two challenges: inefficiency and repetitive optimization, as they overlook the value of past attack experiences. To better integrate past attack experiences to assist current jailbreak attempts, we propose the \textbf{JailExpert}, an automated jailbreak framework, which is the first to achieve a formal representation of experience structure, group experiences based on semantic drift, and support the dynamic updating of the experience pool. Extensive experiments demonstrate that JailExpert significantly improves both attack effectiveness and efficiency. Compared to the current state-of-the-art black-box jailbreak methods, JailExpert achieves an average increase of 17\% in attack success rate and 2.7 times improvement in attack efficiency. Our implementation is available at \href{https://github.com/xiZAIzai/JailExpert}{XiZaiZai/JailExpert}



## **38. Head-Specific Intervention Can Induce Misaligned AI Coordination in Large Language Models**

cs.CL

Published at Transaction of Machine Learning Research 08/2025, Large  Language Models (LLMs), Interference-time activation shifting, Steerability,  Explainability, AI alignment, Interpretability

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2502.05945v3) [paper-pdf](http://arxiv.org/pdf/2502.05945v3)

**Authors**: Paul Darm, Annalisa Riccardi

**Abstract**: Robust alignment guardrails for large language models (LLMs) are becoming increasingly important with their widespread application. In contrast to previous studies, we demonstrate that inference-time activation interventions can bypass safety alignments and effectively steer model generations towards harmful AI coordination. Our method applies fine-grained interventions at specific attention heads, which we identify by probing each head in a simple binary choice task. We then show that interventions on these heads generalise to the open-ended generation setting, effectively circumventing safety guardrails. We demonstrate that intervening on a few attention heads is more effective than intervening on full layers or supervised fine-tuning. We further show that only a few example completions are needed to compute effective steering directions, which is an advantage over classical fine-tuning. We also demonstrate that applying interventions in the negative direction can prevent a common jailbreak attack. Our results suggest that, at the attention head level, activations encode fine-grained linearly separable behaviours. Practically, the approach offers a straightforward methodology to steer large language model behaviour, which could be extended to diverse domains beyond safety, requiring fine-grained control over the model output. The code and datasets for this study can be found on https://github.com/PaulDrm/targeted_intervention.



## **39. Speculative Safety-Aware Decoding**

cs.LG

EMNLP'2025 main conference; more experiments will be added to the  coming camera-ready version

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.17739v1) [paper-pdf](http://arxiv.org/pdf/2508.17739v1)

**Authors**: Xuekang Wang, Shengyu Zhu, Xueqi Cheng

**Abstract**: Despite extensive efforts to align Large Language Models (LLMs) with human values and safety rules, jailbreak attacks that exploit certain vulnerabilities continuously emerge, highlighting the need to strengthen existing LLMs with additional safety properties to defend against these attacks. However, tuning large models has become increasingly resource-intensive and may have difficulty ensuring consistent performance. We introduce Speculative Safety-Aware Decoding (SSD), a lightweight decoding-time approach that equips LLMs with the desired safety property while accelerating inference. We assume that there exists a small language model that possesses this desired property. SSD integrates speculative sampling during decoding and leverages the match ratio between the small and composite models to quantify jailbreak risks. This enables SSD to dynamically switch between decoding schemes to prioritize utility or safety, to handle the challenge of different model capacities. The output token is then sampled from a new distribution that combines the distributions of the original and the small models. Experimental results show that SSD successfully equips the large model with the desired safety property, and also allows the model to remain helpful to benign queries. Furthermore, SSD accelerates the inference time, thanks to the speculative sampling design.



## **40. Prompt-in-Content Attacks: Exploiting Uploaded Inputs to Hijack LLM Behavior**

cs.CR

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.19287v1) [paper-pdf](http://arxiv.org/pdf/2508.19287v1)

**Authors**: Zhuotao Lian, Weiyu Wang, Qingkui Zeng, Toru Nakanishi, Teruaki Kitasuka, Chunhua Su

**Abstract**: Large Language Models (LLMs) are widely deployed in applications that accept user-submitted content, such as uploaded documents or pasted text, for tasks like summarization and question answering. In this paper, we identify a new class of attacks, prompt in content injection, where adversarial instructions are embedded in seemingly benign inputs. When processed by the LLM, these hidden prompts can manipulate outputs without user awareness or system compromise, leading to biased summaries, fabricated claims, or misleading suggestions. We demonstrate the feasibility of such attacks across popular platforms, analyze their root causes including prompt concatenation and insufficient input isolation, and discuss mitigation strategies. Our findings reveal a subtle yet practical threat in real-world LLM workflows.



## **41. Attacking LLMs and AI Agents: Advertisement Embedding Attacks Against Large Language Models**

cs.CR

7 pages, 2 figures

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.17674v1) [paper-pdf](http://arxiv.org/pdf/2508.17674v1)

**Authors**: Qiming Guo, Jinwen Tang, Xingran Huang

**Abstract**: We introduce Advertisement Embedding Attacks (AEA), a new class of LLM security threats that stealthily inject promotional or malicious content into model outputs and AI agents. AEA operate through two low-cost vectors: (1) hijacking third-party service-distribution platforms to prepend adversarial prompts, and (2) publishing back-doored open-source checkpoints fine-tuned with attacker data. Unlike conventional attacks that degrade accuracy, AEA subvert information integrity, causing models to return covert ads, propaganda, or hate speech while appearing normal. We detail the attack pipeline, map five stakeholder victim groups, and present an initial prompt-based self-inspection defense that mitigates these injections without additional model retraining. Our findings reveal an urgent, under-addressed gap in LLM security and call for coordinated detection, auditing, and policy responses from the AI-safety community.



## **42. TombRaider: Entering the Vault of History to Jailbreak Large Language Models**

cs.CR

Main Conference of EMNLP

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2501.18628v2) [paper-pdf](http://arxiv.org/pdf/2501.18628v2)

**Authors**: Junchen Ding, Jiahao Zhang, Yi Liu, Ziqi Ding, Gelei Deng, Yuekang Li

**Abstract**: Warning: This paper contains content that may involve potentially harmful behaviours, discussed strictly for research purposes.   Jailbreak attacks can hinder the safety of Large Language Model (LLM) applications, especially chatbots. Studying jailbreak techniques is an important AI red teaming task for improving the safety of these applications. In this paper, we introduce TombRaider, a novel jailbreak technique that exploits the ability to store, retrieve, and use historical knowledge of LLMs. TombRaider employs two agents, the inspector agent to extract relevant historical information and the attacker agent to generate adversarial prompts, enabling effective bypassing of safety filters. We intensively evaluated TombRaider on six popular models. Experimental results showed that TombRaider could outperform state-of-the-art jailbreak techniques, achieving nearly 100% attack success rates (ASRs) on bare models and maintaining over 55.4% ASR against defence mechanisms. Our findings highlight critical vulnerabilities in existing LLM safeguards, underscoring the need for more robust safety defences.



## **43. RL-Finetuned LLMs for Privacy-Preserving Synthetic Rewriting**

cs.CR

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.19286v1) [paper-pdf](http://arxiv.org/pdf/2508.19286v1)

**Authors**: Zhan Shi, Yefeng Yuan, Yuhong Liu, Liang Cheng, Yi Fang

**Abstract**: The performance of modern machine learning systems depends on access to large, high-quality datasets, often sourced from user-generated content or proprietary, domain-specific corpora. However, these rich datasets inherently contain sensitive personal information, raising significant concerns about privacy, data security, and compliance with regulatory frameworks. While conventional anonymization techniques can remove explicit identifiers, such removal may result in performance drop in downstream machine learning tasks. More importantly, simple anonymization may not be effective against inference attacks that exploit implicit signals such as writing style, topical focus, or demographic cues, highlighting the need for more robust privacy safeguards during model training. To address the challenging issue of balancing user privacy and data utility, we propose a reinforcement learning framework that fine-tunes a large language model (LLM) using a composite reward function that jointly optimizes for explicit and implicit privacy, semantic fidelity, and output diversity. To effectively capture population level regularities, the privacy reward combines semantic cues with structural patterns derived from a minimum spanning tree (MST) over latent representations. By modeling these privacy-sensitive signals in their distributional context, the proposed approach guides the model to generate synthetic rewrites that preserve utility while mitigating privacy risks. Empirical results show that the proposed method significantly enhances author obfuscation and privacy metrics without degrading semantic quality, providing a scalable and model-agnostic solution for privacy preserving data generation in the era of large language models.



## **44. Adaptive Linguistic Prompting (ALP) Enhances Phishing Webpage Detection in Multimodal Large Language Models**

cs.CL

Published at ACL 2025 SRW, 9 pages, 3 figures

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2507.13357v2) [paper-pdf](http://arxiv.org/pdf/2507.13357v2)

**Authors**: Atharva Bhargude, Ishan Gonehal, Dave Yoon, Kaustubh Vinnakota, Chandler Haney, Aaron Sandoval, Kevin Zhu

**Abstract**: Phishing attacks represent a significant cybersecurity threat, necessitating adaptive detection techniques. This study explores few-shot Adaptive Linguistic Prompting (ALP) in detecting phishing webpages through the multimodal capabilities of state-of-the-art large language models (LLMs) such as GPT-4o and Gemini 1.5 Pro. ALP is a structured semantic reasoning method that guides LLMs to analyze textual deception by breaking down linguistic patterns, detecting urgency cues, and identifying manipulative diction commonly found in phishing content. By integrating textual, visual, and URL-based analysis, we propose a unified model capable of identifying sophisticated phishing attempts. Our experiments demonstrate that ALP significantly enhances phishing detection accuracy by guiding LLMs through structured reasoning and contextual analysis. The findings highlight the potential of ALP-integrated multimodal LLMs to advance phishing detection frameworks, achieving an F1-score of 0.93, surpassing traditional approaches. These results establish a foundation for more robust, interpretable, and adaptive linguistic-based phishing detection systems using LLMs.



## **45. Unified attacks to large language model watermarks: spoofing and scrubbing in unauthorized knowledge distillation**

cs.CL

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2504.17480v4) [paper-pdf](http://arxiv.org/pdf/2504.17480v4)

**Authors**: Xin Yi, Yue Li, Shunfan Zheng, Linlin Wang, Xiaoling Wang, Liang He

**Abstract**: Watermarking has emerged as a critical technique for combating misinformation and protecting intellectual property in large language models (LLMs). A recent discovery, termed watermark radioactivity, reveals that watermarks embedded in teacher models can be inherited by student models through knowledge distillation. On the positive side, this inheritance allows for the detection of unauthorized knowledge distillation by identifying watermark traces in student models. However, the robustness of watermarks against scrubbing attacks and their unforgeability in the face of spoofing attacks under unauthorized knowledge distillation remain largely unexplored. Existing watermark attack methods either assume access to model internals or fail to simultaneously support both scrubbing and spoofing attacks. In this work, we propose Contrastive Decoding-Guided Knowledge Distillation (CDG-KD), a unified framework that enables bidirectional attacks under unauthorized knowledge distillation. Our approach employs contrastive decoding to extract corrupted or amplified watermark texts via comparing outputs from the student model and weakly watermarked references, followed by bidirectional distillation to train new student models capable of watermark removal and watermark forgery, respectively. Extensive experiments show that CDG-KD effectively performs attacks while preserving the general performance of the distilled model. Our findings underscore critical need for developing watermarking schemes that are robust and unforgeable.



## **46. Defending against Jailbreak through Early Exit Generation of Large Language Models**

cs.AI

ICONIP 2025

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2408.11308v2) [paper-pdf](http://arxiv.org/pdf/2408.11308v2)

**Authors**: Chongwen Zhao, Zhihao Dou, Kaizhu Huang

**Abstract**: Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation. In an effort to mitigate such risks, the concept of "Alignment" technology has been developed. However, recent studies indicate that this alignment can be undermined using sophisticated prompt engineering or adversarial suffixes, a technique known as "Jailbreak." Our research takes cues from the human-like generate process of LLMs. We identify that while jailbreaking prompts may yield output logits similar to benign prompts, their initial embeddings within the model's latent space tend to be more analogous to those of malicious prompts. Leveraging this finding, we propose utilizing the early transformer outputs of LLMs as a means to detect malicious inputs, and terminate the generation immediately. We introduce a simple yet significant defense approach called EEG-Defender for LLMs. We conduct comprehensive experiments on ten jailbreak methods across three models. Our results demonstrate that EEG-Defender is capable of reducing the Attack Success Rate (ASR) by a significant margin, roughly 85% in comparison with 50% for the present SOTAs, with minimal impact on the utility of LLMs.



## **47. Exploring the Vulnerability of the Content Moderation Guardrail in Large Language Models via Intent Manipulation**

cs.CL

Accepted for EMNLP'25 Findings. TL;DR: We propose a new two-stage  intent-based prompt-refinement framework, IntentPrompt, that aims to explore  the vulnerability of LLMs' content moderation guardrails by refining prompts  into benign-looking declarative forms via intent manipulation for red-teaming  purposes

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2505.18556v2) [paper-pdf](http://arxiv.org/pdf/2505.18556v2)

**Authors**: Jun Zhuang, Haibo Jin, Ye Zhang, Zhengjian Kang, Wenbin Zhang, Gaby G. Dagher, Haohan Wang

**Abstract**: Intent detection, a core component of natural language understanding, has considerably evolved as a crucial mechanism in safeguarding large language models (LLMs). While prior work has applied intent detection to enhance LLMs' moderation guardrails, showing a significant success against content-level jailbreaks, the robustness of these intent-aware guardrails under malicious manipulations remains under-explored. In this work, we investigate the vulnerability of intent-aware guardrails and demonstrate that LLMs exhibit implicit intent detection capabilities. We propose a two-stage intent-based prompt-refinement framework, IntentPrompt, that first transforms harmful inquiries into structured outlines and further reframes them into declarative-style narratives by iteratively optimizing prompts via feedback loops to enhance jailbreak success for red-teaming purposes. Extensive experiments across four public benchmarks and various black-box LLMs indicate that our framework consistently outperforms several cutting-edge jailbreak methods and evades even advanced Intent Analysis (IA) and Chain-of-Thought (CoT)-based defenses. Specifically, our "FSTR+SPIN" variant achieves attack success rates ranging from 88.25% to 96.54% against CoT-based defenses on the o1 model, and from 86.75% to 97.12% on the GPT-4o model under IA-based defenses. These findings highlight a critical weakness in LLMs' safety mechanisms and suggest that intent manipulation poses a growing challenge to content moderation guardrails.



## **48. Trust Me, I Know This Function: Hijacking LLM Static Analysis using Bias**

cs.LG

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17361v1) [paper-pdf](http://arxiv.org/pdf/2508.17361v1)

**Authors**: Shir Bernstein, David Beste, Daniel Ayzenshteyn, Lea Schonherr, Yisroel Mirsky

**Abstract**: Large Language Models (LLMs) are increasingly trusted to perform automated code review and static analysis at scale, supporting tasks such as vulnerability detection, summarization, and refactoring. In this paper, we identify and exploit a critical vulnerability in LLM-based code analysis: an abstraction bias that causes models to overgeneralize familiar programming patterns and overlook small, meaningful bugs. Adversaries can exploit this blind spot to hijack the control flow of the LLM's interpretation with minimal edits and without affecting actual runtime behavior. We refer to this attack as a Familiar Pattern Attack (FPA).   We develop a fully automated, black-box algorithm that discovers and injects FPAs into target code. Our evaluation shows that FPAs are not only effective, but also transferable across models (GPT-4o, Claude 3.5, Gemini 2.0) and universal across programming languages (Python, C, Rust, Go). Moreover, FPAs remain effective even when models are explicitly warned about the attack via robust system prompts. Finally, we explore positive, defensive uses of FPAs and discuss their broader implications for the reliability and safety of code-oriented LLMs.



## **49. Risk Assessment and Security Analysis of Large Language Models**

cs.CR

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17329v1) [paper-pdf](http://arxiv.org/pdf/2508.17329v1)

**Authors**: Xiaoyan Zhang, Dongyang Lyu, Xiaoqi Li

**Abstract**: As large language models (LLMs) expose systemic security challenges in high risk applications, including privacy leaks, bias amplification, and malicious abuse, there is an urgent need for a dynamic risk assessment and collaborative defence framework that covers their entire life cycle. This paper focuses on the security problems of large language models (LLMs) in critical application scenarios, such as the possibility of disclosure of user data, the deliberate input of harmful instructions, or the models bias. To solve these problems, we describe the design of a system for dynamic risk assessment and a hierarchical defence system that allows different levels of protection to cooperate. This paper presents a risk assessment system capable of evaluating both static and dynamic indicators simultaneously. It uses entropy weighting to calculate essential data, such as the frequency of sensitive words, whether the API call is typical, the realtime risk entropy value is significant, and the degree of context deviation. The experimental results show that the system is capable of identifying concealed attacks, such as role escape, and can perform rapid risk evaluation. The paper uses a hybrid model called BERT-CRF (Bidirectional Encoder Representation from Transformers) at the input layer to identify and filter malicious commands. The model layer uses dynamic adversarial training and differential privacy noise injection technology together. The output layer also has a neural watermarking system that can track the source of the content. In practice, the quality of this method, especially important in terms of customer service in the financial industry.



## **50. Fine-Grained Safety Neurons with Training-Free Continual Projection to Reduce LLM Fine Tuning Risks**

cs.LG

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.09190v3) [paper-pdf](http://arxiv.org/pdf/2508.09190v3)

**Authors**: Bing Han, Feifei Zhao, Dongcheng Zhao, Guobin Shen, Ping Wu, Yu Shi, Yi Zeng

**Abstract**: Fine-tuning as service injects domain-specific knowledge into large language models (LLMs), while challenging the original alignment mechanisms and introducing safety risks. A series of defense strategies have been proposed for the alignment, fine-tuning, and post-fine-tuning phases, where most post-fine-tuning defenses rely on coarse-grained safety layer mapping. These methods lack a comprehensive consideration of both safety layers and fine-grained neurons, limiting their ability to efficiently balance safety and utility. To address this, we propose the Fine-Grained Safety Neurons (FGSN) with Training-Free Continual Projection method to reduce the fine-tuning safety risks. FGSN inherently integrates the multi-scale interactions between safety layers and neurons, localizing sparser and more precise fine-grained safety neurons while minimizing interference with downstream task neurons. We then project the safety neuron parameters onto safety directions, improving model safety while aligning more closely with human preferences. Extensive experiments across multiple fine-tuned LLM models demonstrate that our method significantly reduce harmfulness scores and attack success rates with minimal parameter modifications, while preserving the model's utility. Furthermore, by introducing a task-specific, multi-dimensional heterogeneous safety neuron cluster optimization mechanism, we achieve continual defense and generalization capability against unforeseen emerging safety concerns.



