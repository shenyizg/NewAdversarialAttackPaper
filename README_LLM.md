# Latest Large Language Model Attack Papers
**update at 2025-11-07 11:06:50**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Large Language Models for Cyber Security**

cs.CR

**SubmitDate**: 2025-11-06    [abs](http://arxiv.org/abs/2511.04508v1) [paper-pdf](http://arxiv.org/pdf/2511.04508v1)

**Authors**: Raunak Somani, Aswani Kumar Cherukuri

**Abstract**: This paper studies the integration off Large Language Models into cybersecurity tools and protocols. The main issue discussed in this paper is how traditional rule-based and signature based security systems are not enough to deal with modern AI powered cyber threats. Cybersecurity industry is changing as threats are becoming more dangerous and adaptive in nature by levering the features provided by AI tools. By integrating LLMs into these tools and protocols, make the systems scalable, context-aware and intelligent. Thus helping it to mitigate these evolving cyber threats. The paper studies the architecture and functioning of LLMs, its integration into Encrypted prompts to prevent prompt injection attacks. It also studies the integration of LLMs into cybersecurity tools using a four layered architecture. At last, the paper has tried to explain various ways of integration LLMs into traditional Intrusion Detection System and enhancing its original abilities in various dimensions. The key findings of this paper has been (i)Encrypted Prompt with LLM is an effective way to mitigate prompt injection attacks, (ii) LLM enhanced cyber security tools are more accurate, scalable and adaptable to new threats as compared to traditional models, (iii) The decoupled model approach for LLM integration into IDS is the best way as it is the most accurate way.



## **2. AdversariaLLM: A Unified and Modular Toolbox for LLM Robustness Research**

cs.AI

**SubmitDate**: 2025-11-06    [abs](http://arxiv.org/abs/2511.04316v1) [paper-pdf](http://arxiv.org/pdf/2511.04316v1)

**Authors**: Tim Beyer, Jonas Dornbusch, Jakob Steimle, Moritz Ladenburger, Leo Schwinn, Stephan Günnemann

**Abstract**: The rapid expansion of research on Large Language Model (LLM) safety and robustness has produced a fragmented and oftentimes buggy ecosystem of implementations, datasets, and evaluation methods. This fragmentation makes reproducibility and comparability across studies challenging, hindering meaningful progress. To address these issues, we introduce AdversariaLLM, a toolbox for conducting LLM jailbreak robustness research. Its design centers on reproducibility, correctness, and extensibility. The framework implements twelve adversarial attack algorithms, integrates seven benchmark datasets spanning harmfulness, over-refusal, and utility evaluation, and provides access to a wide range of open-weight LLMs via Hugging Face. The implementation includes advanced features for comparability and reproducibility such as compute-resource tracking, deterministic results, and distributional evaluation techniques. \name also integrates judging through the companion package JudgeZoo, which can also be used independently. Together, these components aim to establish a robust foundation for transparent, comparable, and reproducible research in LLM safety.



## **3. Black-Box Guardrail Reverse-engineering Attack**

cs.CR

**SubmitDate**: 2025-11-06    [abs](http://arxiv.org/abs/2511.04215v1) [paper-pdf](http://arxiv.org/pdf/2511.04215v1)

**Authors**: Hongwei Yao, Yun Xia, Shuo Shao, Haoran Shi, Tong Qiao, Cong Wang

**Abstract**: Large language models (LLMs) increasingly employ guardrails to enforce ethical, legal, and application-specific constraints on their outputs. While effective at mitigating harmful responses, these guardrails introduce a new class of vulnerabilities by exposing observable decision patterns. In this work, we present the first study of black-box LLM guardrail reverse-engineering attacks. We propose Guardrail Reverse-engineering Attack (GRA), a reinforcement learning-based framework that leverages genetic algorithm-driven data augmentation to approximate the decision-making policy of victim guardrails. By iteratively collecting input-output pairs, prioritizing divergence cases, and applying targeted mutations and crossovers, our method incrementally converges toward a high-fidelity surrogate of the victim guardrail. We evaluate GRA on three widely deployed commercial systems, namely ChatGPT, DeepSeek, and Qwen3, and demonstrate that it achieves an rule matching rate exceeding 0.92 while requiring less than $85 in API costs. These findings underscore the practical feasibility of guardrail extraction and highlight significant security risks for current LLM safety mechanisms. Our findings expose critical vulnerabilities in current guardrail designs and highlight the urgent need for more robust defense mechanisms in LLM deployment.



## **4. Transferable & Stealthy Ensemble Attacks: A Black-Box Jailbreaking Framework for Large Language Models**

cs.CR

**SubmitDate**: 2025-11-06    [abs](http://arxiv.org/abs/2410.23558v3) [paper-pdf](http://arxiv.org/pdf/2410.23558v3)

**Authors**: Yiqi Yang, Hongye Fu

**Abstract**: We present a novel black-box jailbreaking framework that integrates multiple LLM-as-Attacker strategies to deliver highly transferable and effective attacks. The framework is grounded in three key insights from prior jailbreaking research and practice: ensemble approaches outperform single methods in exposing aligned LLM vulnerabilities, malicious instructions vary in jailbreaking difficulty requiring tailored optimization, and disrupting semantic coherence of malicious prompts can manipulate their embeddings to boost success rates. Validated in the Competition for LLM and Agent Safety 2024, our solution achieved top rankings in the Jailbreaking Attack Track.



## **5. Measuring the Security of Mobile LLM Agents under Adversarial Prompts from Untrusted Third-Party Channels**

cs.CR

**SubmitDate**: 2025-11-06    [abs](http://arxiv.org/abs/2510.27140v2) [paper-pdf](http://arxiv.org/pdf/2510.27140v2)

**Authors**: Chenghao Du, Quanfeng Huang, Tingxuan Tang, Zihao Wang, Adwait Nadkarni, Yue Xiao

**Abstract**: Large Language Models (LLMs) have transformed software development, enabling AI-powered applications known as LLM-based agents that promise to automate tasks across diverse apps and workflows. Yet, the security implications of deploying such agents in adversarial mobile environments remain poorly understood. In this paper, we present the first systematic study of security risks in mobile LLM agents. We design and evaluate a suite of adversarial case studies, ranging from opportunistic manipulations such as pop-up advertisements to advanced, end-to-end workflows involving malware installation and cross-app data exfiltration. Our evaluation covers eight state-of-the-art mobile agents across three architectures, with over 2,000 adversarial and paired benign trials. The results reveal systemic vulnerabilities: low-barrier vectors such as fraudulent ads succeed with over 80% reliability, while even workflows requiring the circumvention of operating-system warnings, such as malware installation, are consistently completed by advanced multi-app agents. By mapping these attacks to the MITRE ATT&CK Mobile framework, we uncover novel privilege-escalation and persistence pathways unique to LLM-driven automation. Collectively, our findings provide the first end-to-end evidence that mobile LLM agents are exploitable in realistic adversarial settings, where untrusted third-party channels (e.g., ads, embedded webviews, cross-app notifications) are an inherent part of the mobile ecosystem.



## **6. Whisper Leak: a side-channel attack on Large Language Models**

cs.CR

14 pages, 7 figures

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2511.03675v1) [paper-pdf](http://arxiv.org/pdf/2511.03675v1)

**Authors**: Geoff McDonald, Jonathan Bar Or

**Abstract**: Large Language Models (LLMs) are increasingly deployed in sensitive domains including healthcare, legal services, and confidential communications, where privacy is paramount. This paper introduces Whisper Leak, a side-channel attack that infers user prompt topics from encrypted LLM traffic by analyzing packet size and timing patterns in streaming responses. Despite TLS encryption protecting content, these metadata patterns leak sufficient information to enable topic classification. We demonstrate the attack across 28 popular LLMs from major providers, achieving near-perfect classification (often >98% AUPRC) and high precision even at extreme class imbalance (10,000:1 noise-to-target ratio). For many models, we achieve 100% precision in identifying sensitive topics like "money laundering" while recovering 5-20% of target conversations. This industry-wide vulnerability poses significant risks for users under network surveillance by ISPs, governments, or local adversaries. We evaluate three mitigation strategies - random padding, token batching, and packet injection - finding that while each reduces attack effectiveness, none provides complete protection. Through responsible disclosure, we have collaborated with providers to implement initial countermeasures. Our findings underscore the need for LLM providers to address metadata leakage as AI systems handle increasingly sensitive information.



## **7. Manipulation Facing Threats: Evaluating Physical Vulnerabilities in End-to-End Vision Language Action Models**

cs.CV

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2409.13174v4) [paper-pdf](http://arxiv.org/pdf/2409.13174v4)

**Authors**: Hao Cheng, Erjia Xiao, Yichi Wang, Chengyuan Yu, Mengshu Sun, Qiang Zhang, Jiahang Cao, Yijie Guo, Ning Liu, Kaidi Xu, Jize Zhang, Chao Shen, Philip Torr, Jindong Gu, Renjing Xu

**Abstract**: Recently, driven by advancements in Multimodal Large Language Models (MLLMs), Vision Language Action Models (VLAMs) are being proposed to achieve better performance in open-vocabulary scenarios for robotic manipulation tasks. Since manipulation tasks involve direct interaction with the physical world, ensuring robustness and safety during the execution of this task is always a very critical issue. In this paper, by synthesizing current safety research on MLLMs and the specific application scenarios of the manipulation task in the physical world, we comprehensively evaluate VLAMs in the face of potential physical threats. Specifically, we propose the Physical Vulnerability Evaluating Pipeline (PVEP) that can incorporate as many visual modal physical threats as possible for evaluating the physical robustness of VLAMs. The physical threats in PVEP specifically include Out-of-Distribution, Typography-based Visual Prompt, and Adversarial Patch Attacks. By comparing the performance fluctuations of VLAMs before and after being attacked, we provide generalizable \textbf{\textit{Analyses}} of how VLAMs respond to different physical threats.



## **8. Let the Bees Find the Weak Spots: A Path Planning Perspective on Multi-Turn Jailbreak Attacks against LLMs**

cs.CR

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2511.03271v1) [paper-pdf](http://arxiv.org/pdf/2511.03271v1)

**Authors**: Yize Liu, Yunyun Hou, Aina Sui

**Abstract**: Large Language Models (LLMs) have been widely deployed across various applications, yet their potential security and ethical risks have raised increasing concerns. Existing research employs red teaming evaluations, utilizing multi-turn jailbreaks to identify potential vulnerabilities in LLMs. However, these approaches often lack exploration of successful dialogue trajectories within the attack space, and they tend to overlook the considerable overhead associated with the attack process. To address these limitations, this paper first introduces a theoretical model based on dynamically weighted graph topology, abstracting the multi-turn attack process as a path planning problem. Based on this framework, we propose ABC, an enhanced Artificial Bee Colony algorithm for multi-turn jailbreaks, featuring a collaborative search mechanism with employed, onlooker, and scout bees. This algorithm significantly improves the efficiency of optimal attack path search while substantially reducing the average number of queries required. Empirical evaluations on three open-source and two proprietary language models demonstrate the effectiveness of our approach, achieving attack success rates above 90\% across the board, with a peak of 98\% on GPT-3.5-Turbo, and outperforming existing baselines. Furthermore, it achieves comparable success with only 26 queries on average, significantly reducing red teaming overhead and highlighting its superior efficiency.



## **9. Death by a Thousand Prompts: Open Model Vulnerability Analysis**

cs.CR

**SubmitDate**: 2025-11-05    [abs](http://arxiv.org/abs/2511.03247v1) [paper-pdf](http://arxiv.org/pdf/2511.03247v1)

**Authors**: Amy Chang, Nicholas Conley, Harish Santhanalakshmi Ganesan, Adam Swanda

**Abstract**: Open-weight models provide researchers and developers with accessible foundations for diverse downstream applications. We tested the safety and security postures of eight open-weight large language models (LLMs) to identify vulnerabilities that may impact subsequent fine-tuning and deployment. Using automated adversarial testing, we measured each model's resilience against single-turn and multi-turn prompt injection and jailbreak attacks. Our findings reveal pervasive vulnerabilities across all tested models, with multi-turn attacks achieving success rates between 25.86\% and 92.78\% -- representing a $2\times$ to $10\times$ increase over single-turn baselines. These results underscore a systemic inability of current open-weight models to maintain safety guardrails across extended interactions. We assess that alignment strategies and lab priorities significantly influence resilience: capability-focused models such as Llama 3.3 and Qwen 3 demonstrate higher multi-turn susceptibility, whereas safety-oriented designs such as Google Gemma 3 exhibit more balanced performance.   The analysis concludes that open-weight models, while crucial for innovation, pose tangible operational and ethical risks when deployed without layered security controls. These findings are intended to inform practitioners and developers of the potential risks and the value of professional AI security solutions to mitigate exposure. Addressing multi-turn vulnerabilities is essential to ensure the safe, reliable, and responsible deployment of open-weight LLMs in enterprise and public domains. We recommend adopting a security-first design philosophy and layered protections to ensure resilient deployments of open-weight models.



## **10. Adaptive and Robust Data Poisoning Detection and Sanitization in Wearable IoT Systems using Large Language Models**

cs.LG

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.02894v1) [paper-pdf](http://arxiv.org/pdf/2511.02894v1)

**Authors**: W. K. M Mithsara, Ning Yang, Ahmed Imteaj, Hussein Zangoti, Abdur R. Shahid

**Abstract**: The widespread integration of wearable sensing devices in Internet of Things (IoT) ecosystems, particularly in healthcare, smart homes, and industrial applications, has required robust human activity recognition (HAR) techniques to improve functionality and user experience. Although machine learning models have advanced HAR, they are increasingly susceptible to data poisoning attacks that compromise the data integrity and reliability of these systems. Conventional approaches to defending against such attacks often require extensive task-specific training with large, labeled datasets, which limits adaptability in dynamic IoT environments. This work proposes a novel framework that uses large language models (LLMs) to perform poisoning detection and sanitization in HAR systems, utilizing zero-shot, one-shot, and few-shot learning paradigms. Our approach incorporates \textit{role play} prompting, whereby the LLM assumes the role of expert to contextualize and evaluate sensor anomalies, and \textit{think step-by-step} reasoning, guiding the LLM to infer poisoning indicators in the raw sensor data and plausible clean alternatives. These strategies minimize reliance on curation of extensive datasets and enable robust, adaptable defense mechanisms in real-time. We perform an extensive evaluation of the framework, quantifying detection accuracy, sanitization quality, latency, and communication cost, thus demonstrating the practicality and effectiveness of LLMs in improving the security and reliability of wearable IoT systems.



## **11. Do Methods to Jailbreak and Defend LLMs Generalize Across Languages?**

cs.CL

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.00689v2) [paper-pdf](http://arxiv.org/pdf/2511.00689v2)

**Authors**: Berk Atil, Rebecca J. Passonneau, Fred Morstatter

**Abstract**: Large language models (LLMs) undergo safety alignment after training and tuning, yet recent work shows that safety can be bypassed through jailbreak attacks. While many jailbreaks and defenses exist, their cross-lingual generalization remains underexplored. This paper presents the first systematic multilingual evaluation of jailbreaks and defenses across ten languages -- spanning high-, medium-, and low-resource languages -- using six LLMs on HarmBench and AdvBench. We assess two jailbreak types: logical-expression-based and adversarial-prompt-based. For both types, attack success and defense robustness vary across languages: high-resource languages are safer under standard queries but more vulnerable to adversarial ones. Simple defenses can be effective, but are language- and model-dependent. These findings call for language-aware and cross-lingual safety benchmarks for LLMs.



## **12. Verifying LLM Inference to Prevent Model Weight Exfiltration**

cs.CR

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.02620v1) [paper-pdf](http://arxiv.org/pdf/2511.02620v1)

**Authors**: Roy Rinberg, Adam Karvonen, Alex Hoover, Daniel Reuter, Keri Warr

**Abstract**: As large AI models become increasingly valuable assets, the risk of model weight exfiltration from inference servers grows accordingly. An attacker controlling an inference server may exfiltrate model weights by hiding them within ordinary model outputs, a strategy known as steganography. This work investigates how to verify model responses to defend against such attacks and, more broadly, to detect anomalous or buggy behavior during inference. We formalize model exfiltration as a security game, propose a verification framework that can provably mitigate steganographic exfiltration, and specify the trust assumptions associated with our scheme. To enable verification, we characterize valid sources of non-determinism in large language model inference and introduce two practical estimators for them. We evaluate our detection framework on several open-weight models ranging from 3B to 30B parameters. On MOE-Qwen-30B, our detector reduces exfiltratable information to <0.5% with false-positive rate of 0.01%, corresponding to a >200x slowdown for adversaries. Overall, this work further establishes a foundation for defending against model weight exfiltration and demonstrates that strong protection can be achieved with minimal additional cost to inference providers.



## **13. The Dark Side of LLMs: Agent-based Attacks for Complete Computer Takeover**

cs.CR

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2507.06850v5) [paper-pdf](http://arxiv.org/pdf/2507.06850v5)

**Authors**: Matteo Lupinacci, Francesco Aurelio Pironti, Francesco Blefari, Francesco Romeo, Luigi Arena, Angelo Furfaro

**Abstract**: The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables remarkable capabilities in natural language processing and generation. However, these systems introduce security vulnerabilities that extend beyond traditional content generation to system-level compromises. This paper presents a comprehensive evaluation of the LLMs security used as reasoning engines within autonomous agents, highlighting how they can be exploited as attack vectors capable of achieving computer takeovers. We focus on how different attack surfaces and trust boundaries can be leveraged to orchestrate such takeovers. We demonstrate that adversaries can effectively coerce popular LLMs into autonomously installing and executing malware on victim machines. Our evaluation of 18 state-of-the-art LLMs reveals an alarming scenario: 94.4% of models succumb to Direct Prompt Injection, and 83.3% are vulnerable to the more stealthy and evasive RAG Backdoor Attack. Notably, we tested trust boundaries within multi-agent systems, where LLM agents interact and influence each other, and we revealed that LLMs which successfully resist direct injection or RAG backdoor attacks will execute identical payloads when requested by peer agents. We found that 100.0% of tested LLMs can be compromised through Inter-Agent Trust Exploitation attacks, and that every model exhibits context-dependent security behaviors that create exploitable blind spots.



## **14. AutoAdv: Automated Adversarial Prompting for Multi-Turn Jailbreaking of Large Language Models**

cs.CL

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.02376v1) [paper-pdf](http://arxiv.org/pdf/2511.02376v1)

**Authors**: Aashray Reddy, Andrew Zagula, Nicholas Saban

**Abstract**: Large Language Models (LLMs) remain vulnerable to jailbreaking attacks where adversarial prompts elicit harmful outputs, yet most evaluations focus on single-turn interactions while real-world attacks unfold through adaptive multi-turn conversations. We present AutoAdv, a training-free framework for automated multi-turn jailbreaking that achieves up to 95% attack success rate on Llama-3.1-8B within six turns a 24 percent improvement over single turn baselines. AutoAdv uniquely combines three adaptive mechanisms: a pattern manager that learns from successful attacks to enhance future prompts, a temperature manager that dynamically adjusts sampling parameters based on failure modes, and a two-phase rewriting strategy that disguises harmful requests then iteratively refines them. Extensive evaluation across commercial and open-source models (GPT-4o-mini, Qwen3-235B, Mistral-7B) reveals persistent vulnerabilities in current safety mechanisms, with multi-turn attacks consistently outperforming single-turn approaches. These findings demonstrate that alignment strategies optimized for single-turn interactions fail to maintain robustness across extended conversations, highlighting an urgent need for multi-turn-aware defenses.



## **15. An Automated Framework for Strategy Discovery, Retrieval, and Evolution in LLM Jailbreak Attacks**

cs.CR

**SubmitDate**: 2025-11-04    [abs](http://arxiv.org/abs/2511.02356v1) [paper-pdf](http://arxiv.org/pdf/2511.02356v1)

**Authors**: Xu Liu, Yan Chen, Kan Ling, Yichi Zhu, Hengrun Zhang, Guisheng Fan, Huiqun Yu

**Abstract**: The widespread deployment of Large Language Models (LLMs) as public-facing web services and APIs has made their security a core concern for the web ecosystem. Jailbreak attacks, as one of the significant threats to LLMs, have recently attracted extensive research. In this paper, we reveal a jailbreak strategy which can effectively evade current defense strategies. It can extract valuable information from failed or partially successful attack attempts and contains self-evolution from attack interactions, resulting in sufficient strategy diversity and adaptability. Inspired by continuous learning and modular design principles, we propose ASTRA, a jailbreak framework that autonomously discovers, retrieves, and evolves attack strategies to achieve more efficient and adaptive attacks. To enable this autonomous evolution, we design a closed-loop "attack-evaluate-distill-reuse" core mechanism that not only generates attack prompts but also automatically distills and generalizes reusable attack strategies from every interaction. To systematically accumulate and apply this attack knowledge, we introduce a three-tier strategy library that categorizes strategies into Effective, Promising, and Ineffective based on their performance scores. The strategy library not only provides precise guidance for attack generation but also possesses exceptional extensibility and transferability. We conduct extensive experiments under a black-box setting, and the results show that ASTRA achieves an average Attack Success Rate (ASR) of 82.7%, significantly outperforming baselines.



## **16. Retrieval-Augmented Defense: Adaptive and Controllable Jailbreak Prevention for Large Language Models**

cs.CR

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2508.16406v2) [paper-pdf](http://arxiv.org/pdf/2508.16406v2)

**Authors**: Guangyu Yang, Jinghong Chen, Jingbiao Mei, Weizhe Lin, Bill Byrne

**Abstract**: Large Language Models (LLMs) remain vulnerable to jailbreak attacks, which attempt to elicit harmful responses from LLMs. The evolving nature and diversity of these attacks pose many challenges for defense systems, including (1) adaptation to counter emerging attack strategies without costly retraining, and (2) control of the trade-off between safety and utility. To address these challenges, we propose Retrieval-Augmented Defense (RAD), a novel framework for jailbreak detection that incorporates a database of known attack examples into Retrieval-Augmented Generation, which is used to infer the underlying, malicious user query and jailbreak strategy used to attack the system. RAD enables training-free updates for newly discovered jailbreak strategies and provides a mechanism to balance safety and utility. Experiments on StrongREJECT show that RAD substantially reduces the effectiveness of strong jailbreak attacks such as PAP and PAIR while maintaining low rejection rates for benign queries. We propose a novel evaluation scheme and show that RAD achieves a robust safety-utility trade-off across a range of operating points in a controllable manner.



## **17. Prompt Injection as an Emerging Threat: Evaluating the Resilience of Large Language Models**

cs.CR

10 pages, 6 figures

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01634v1) [paper-pdf](http://arxiv.org/pdf/2511.01634v1)

**Authors**: Daniyal Ganiuly, Assel Smaiyl

**Abstract**: Large Language Models (LLMs) are increasingly used in intelligent systems that perform reasoning, summarization, and code generation. Their ability to follow natural-language instructions, while powerful, also makes them vulnerable to a new class of attacks known as prompt injection. In these attacks, hidden or malicious instructions are inserted into user inputs or external content, causing the model to ignore its intended task or produce unsafe responses. This study proposes a unified framework for evaluating how resistant Large Language Models (LLMs) are to prompt injection attacks. The framework defines three complementary metrics such as the Resilience Degradation Index (RDI), Safety Compliance Coefficient (SCC), and Instructional Integrity Metric (IIM) to jointly measure robustness, safety, and semantic stability. We evaluated four instruction-tuned models (GPT-4, GPT-4o, LLaMA-3 8B Instruct, and Flan-T5-Large) on five common language tasks: question answering, summarization, translation, reasoning, and code generation. Results show that GPT-4 performs best overall, while open-weight models remain more vulnerable. The findings highlight that strong alignment and safety tuning are more important for resilience than model size alone. Results show that all models remain partially vulnerable, especially to indirect and direct-override attacks. GPT-4 achieved the best overall resilience (RDR = 9.8 %, SCR = 96.4 %), while open-source models exhibited higher performance degradation and lower safety scores. The findings demonstrate that alignment strength and safety tuning play a greater role in resilience than model size alone. The proposed framework offers a structured, reproducible approach for assessing model robustness and provides practical insights for improving LLM safety and reliability.



## **18. Black-Box Membership Inference Attack for LVLMs via Prior Knowledge-Calibrated Memory Probing**

cs.CR

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01952v1) [paper-pdf](http://arxiv.org/pdf/2511.01952v1)

**Authors**: Jinhua Yin, Peiru Yang, Chen Yang, Huili Wang, Zhiyang Hu, Shangguang Wang, Yongfeng Huang, Tao Qi

**Abstract**: Large vision-language models (LVLMs) derive their capabilities from extensive training on vast corpora of visual and textual data. Empowered by large-scale parameters, these models often exhibit strong memorization of their training data, rendering them susceptible to membership inference attacks (MIAs). Existing MIA methods for LVLMs typically operate under white- or gray-box assumptions, by extracting likelihood-based features for the suspected data samples based on the target LVLMs. However, mainstream LVLMs generally only expose generated outputs while concealing internal computational features during inference, limiting the applicability of these methods. In this work, we propose the first black-box MIA framework for LVLMs, based on a prior knowledge-calibrated memory probing mechanism. The core idea is to assess the model memorization of the private semantic information embedded within the suspected image data, which is unlikely to be inferred from general world knowledge alone. We conducted extensive experiments across four LVLMs and three datasets. Empirical results demonstrate that our method effectively identifies training data of LVLMs in a purely black-box setting and even achieves performance comparable to gray-box and white-box methods. Further analysis reveals the robustness of our method against potential adversarial manipulations, and the effectiveness of the methodology designs. Our code and data are available at https://github.com/spmede/KCMP.



## **19. Align to Misalign: Automatic LLM Jailbreak with Meta-Optimized LLM Judges**

cs.AI

under review, 28 pages

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01375v1) [paper-pdf](http://arxiv.org/pdf/2511.01375v1)

**Authors**: Hamin Koo, Minseon Kim, Jaehyung Kim

**Abstract**: Identifying the vulnerabilities of large language models (LLMs) is crucial for improving their safety by addressing inherent weaknesses. Jailbreaks, in which adversaries bypass safeguards with crafted input prompts, play a central role in red-teaming by probing LLMs to elicit unintended or unsafe behaviors. Recent optimization-based jailbreak approaches iteratively refine attack prompts by leveraging LLMs. However, they often rely heavily on either binary attack success rate (ASR) signals, which are sparse, or manually crafted scoring templates, which introduce human bias and uncertainty in the scoring outcomes. To address these limitations, we introduce AMIS (Align to MISalign), a meta-optimization framework that jointly evolves jailbreak prompts and scoring templates through a bi-level structure. In the inner loop, prompts are refined using fine-grained and dense feedback using a fixed scoring template. In the outer loop, the template is optimized using an ASR alignment score, gradually evolving to better reflect true attack outcomes across queries. This co-optimization process yields progressively stronger jailbreak prompts and more calibrated scoring signals. Evaluations on AdvBench and JBB-Behaviors demonstrate that AMIS achieves state-of-the-art performance, including 88.0% ASR on Claude-3.5-Haiku and 100.0% ASR on Claude-4-Sonnet, outperforming existing baselines by substantial margins.



## **20. Rescuing the Unpoisoned: Efficient Defense against Knowledge Corruption Attacks on RAG Systems**

cs.CR

15 pages, 7 figures, 10 tables. To appear in the Proceedings of the  2025 Annual Computer Security Applications Conference (ACSAC)

**SubmitDate**: 2025-11-03    [abs](http://arxiv.org/abs/2511.01268v1) [paper-pdf](http://arxiv.org/pdf/2511.01268v1)

**Authors**: Minseok Kim, Hankook Lee, Hyungjoon Koo

**Abstract**: Large language models (LLMs) are reshaping numerous facets of our daily lives, leading widespread adoption as web-based services. Despite their versatility, LLMs face notable challenges, such as generating hallucinated content and lacking access to up-to-date information. Lately, to address such limitations, Retrieval-Augmented Generation (RAG) has emerged as a promising direction by generating responses grounded in external knowledge sources. A typical RAG system consists of i) a retriever that probes a group of relevant passages from a knowledge base and ii) a generator that formulates a response based on the retrieved content. However, as with other AI systems, recent studies demonstrate the vulnerability of RAG, such as knowledge corruption attacks by injecting misleading information. In response, several defense strategies have been proposed, including having LLMs inspect the retrieved passages individually or fine-tuning robust retrievers. While effective, such approaches often come with substantial computational costs.   In this work, we introduce RAGDefender, a resource-efficient defense mechanism against knowledge corruption (i.e., by data poisoning) attacks in practical RAG deployments. RAGDefender operates during the post-retrieval phase, leveraging lightweight machine learning techniques to detect and filter out adversarial content without requiring additional model training or inference. Our empirical evaluations show that RAGDefender consistently outperforms existing state-of-the-art defenses across multiple models and adversarial scenarios: e.g., RAGDefender reduces the attack success rate (ASR) against the Gemini model from 0.89 to as low as 0.02, compared to 0.69 for RobustRAG and 0.24 for Discern-and-Answer when adversarial passages outnumber legitimate ones by a factor of four (4x).



## **21. MistralBSM: Leveraging Mistral-7B for Vehicular Networks Misbehavior Detection**

cs.LG

**SubmitDate**: 2025-11-02    [abs](http://arxiv.org/abs/2407.18462v2) [paper-pdf](http://arxiv.org/pdf/2407.18462v2)

**Authors**: Wissal Hamhoum, Soumaya Cherkaoui

**Abstract**: Malicious attacks on vehicular networks pose a serious threat to road safety as well as communication reliability. A major source of these threats stems from misbehaving vehicles within the network. To address this challenge, we propose a Large Language Model (LLM)-empowered Misbehavior Detection System (MDS) within an edge-cloud detection framework. Specifically, we fine-tune Mistral-7B, a compact and high-performing LLM, to detect misbehavior based on Basic Safety Messages (BSM) sequences as the edge component for real-time detection, while a larger LLM deployed in the cloud validates and reinforces the edge model's detection through a more comprehensive analysis. By updating only 0.012% of the model parameters, our model, which we named MistralBSM, achieves 98% accuracy in binary classification and 96% in multiclass classification on a selected set of attacks from VeReMi dataset, outperforming LLAMA2-7B and RoBERTa. Our results validate the potential of LLMs in MDS, showing a significant promise in strengthening vehicular network security to better ensure the safety of road users.



## **22. Exploring the limits of strong membership inference attacks on large language models**

cs.CR

NeurIPS 2025

**SubmitDate**: 2025-11-02    [abs](http://arxiv.org/abs/2505.18773v2) [paper-pdf](http://arxiv.org/pdf/2505.18773v2)

**Authors**: Jamie Hayes, Ilia Shumailov, Christopher A. Choquette-Choo, Matthew Jagielski, George Kaissis, Milad Nasr, Sahra Ghalebikesabi, Meenatchi Sundaram Mutu Selva Annamalai, Niloofar Mireshghallah, Igor Shilov, Matthieu Meeus, Yves-Alexandre de Montjoye, Katherine Lee, Franziska Boenisch, Adam Dziedzic, A. Feder Cooper

**Abstract**: State-of-the-art membership inference attacks (MIAs) typically require training many reference models, making it difficult to scale these attacks to large pre-trained language models (LLMs). As a result, prior research has either relied on weaker attacks that avoid training references (e.g., fine-tuning attacks), or on stronger attacks applied to small models and datasets. However, weaker attacks have been shown to be brittle and insights from strong attacks in simplified settings do not translate to today's LLMs. These challenges prompt an important question: are the limitations observed in prior work due to attack design choices, or are MIAs fundamentally ineffective on LLMs? We address this question by scaling LiRA--one of the strongest MIAs--to GPT-2 architectures ranging from 10M to 1B parameters, training references on over 20B tokens from the C4 dataset. Our results advance the understanding of MIAs on LLMs in four key ways. While (1) strong MIAs can succeed on pre-trained LLMs, (2) their effectiveness, remains limited (e.g., AUC<0.7) in practical settings. (3) Even when strong MIAs achieve better-than-random AUC, aggregate metrics can conceal substantial per-sample MIA decision instability: due to training randomness, many decisions are so unstable that they are statistically indistinguishable from a coin flip. Finally, (4) the relationship between MIA success and related LLM privacy metrics is not as straightforward as prior work has suggested.



## **23. SafeDialBench: A Fine-Grained Safety Benchmark for Large Language Models in Multi-Turn Dialogues with Diverse Jailbreak Attacks**

cs.CL

**SubmitDate**: 2025-11-02    [abs](http://arxiv.org/abs/2502.11090v3) [paper-pdf](http://arxiv.org/pdf/2502.11090v3)

**Authors**: Hongye Cao, Yanming Wang, Sijia Jing, Ziyue Peng, Zhixin Bai, Zhe Cao, Meng Fang, Fan Feng, Boyan Wang, Jiaheng Liu, Tianpei Yang, Jing Huo, Yang Gao, Fanyu Meng, Xi Yang, Chao Deng, Junlan Feng

**Abstract**: With the rapid advancement of Large Language Models (LLMs), the safety of LLMs has been a critical concern requiring precise assessment. Current benchmarks primarily concentrate on single-turn dialogues or a single jailbreak attack method to assess the safety. Additionally, these benchmarks have not taken into account the LLM's capability of identifying and handling unsafe information in detail. To address these issues, we propose a fine-grained benchmark SafeDialBench for evaluating the safety of LLMs across various jailbreak attacks in multi-turn dialogues. Specifically, we design a two-tier hierarchical safety taxonomy that considers 6 safety dimensions and generates more than 4000 multi-turn dialogues in both Chinese and English under 22 dialogue scenarios. We employ 7 jailbreak attack strategies, such as reference attack and purpose reverse, to enhance the dataset quality for dialogue generation. Notably, we construct an innovative assessment framework of LLMs, measuring capabilities in detecting, and handling unsafe information and maintaining consistency when facing jailbreak attacks. Experimental results across 17 LLMs reveal that Yi-34B-Chat and GLM4-9B-Chat demonstrate superior safety performance, while Llama3.1-8B-Instruct and o3-mini exhibit safety vulnerabilities.



## **24. Enhancing Adversarial Transferability in Visual-Language Pre-training Models via Local Shuffle and Sample-based Attack**

cs.CV

Accepted by NAACL2025 findings

**SubmitDate**: 2025-11-02    [abs](http://arxiv.org/abs/2511.00831v1) [paper-pdf](http://arxiv.org/pdf/2511.00831v1)

**Authors**: Xin Liu, Aoyang Zhou, Aoyang Zhou

**Abstract**: Visual-Language Pre-training (VLP) models have achieved significant performance across various downstream tasks. However, they remain vulnerable to adversarial examples. While prior efforts focus on improving the adversarial transferability of multimodal adversarial examples through cross-modal interactions, these approaches suffer from overfitting issues, due to a lack of input diversity by relying excessively on information from adversarial examples in one modality when crafting attacks in another. To address this issue, we draw inspiration from strategies in some adversarial training methods and propose a novel attack called Local Shuffle and Sample-based Attack (LSSA). LSSA randomly shuffles one of the local image blocks, thus expanding the original image-text pairs, generating adversarial images, and sampling around them. Then, it utilizes both the original and sampled images to generate the adversarial texts. Extensive experiments on multiple models and datasets demonstrate that LSSA significantly enhances the transferability of multimodal adversarial examples across diverse VLP models and downstream tasks. Moreover, LSSA outperforms other advanced attacks on Large Vision-Language Models.



## **25. Adversarial Déjà Vu: Jailbreak Dictionary Learning for Stronger Generalization to Unseen Attacks**

cs.LG

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2510.21910v2) [paper-pdf](http://arxiv.org/pdf/2510.21910v2)

**Authors**: Mahavir Dabas, Tran Huynh, Nikhil Reddy Billa, Jiachen T. Wang, Peng Gao, Charith Peris, Yao Ma, Rahul Gupta, Ming Jin, Prateek Mittal, Ruoxi Jia

**Abstract**: Large language models remain vulnerable to jailbreak attacks that bypass safety guardrails to elicit harmful outputs. Defending against novel jailbreaks represents a critical challenge in AI safety. Adversarial training -- designed to make models robust against worst-case perturbations -- has been the dominant paradigm for adversarial robustness. However, due to optimization challenges and difficulties in defining realistic threat models, adversarial training methods often fail on newly developed jailbreaks in practice. This paper proposes a new paradigm for improving robustness against unseen jailbreaks, centered on the Adversarial D\'ej\`a Vu hypothesis: novel jailbreaks are not fundamentally new, but largely recombinations of adversarial skills from previous attacks. We study this hypothesis through a large-scale analysis of 32 attack papers published over two years. Using an automated pipeline, we extract and compress adversarial skills into a sparse dictionary of primitives, with LLMs generating human-readable descriptions. Our analysis reveals that unseen attacks can be effectively explained as sparse compositions of earlier skills, with explanatory power increasing monotonically as skill coverage grows. Guided by this insight, we introduce Adversarial Skill Compositional Training (ASCoT), which trains on diverse compositions of skill primitives rather than isolated attack instances. ASCoT substantially improves robustness to unseen attacks, including multi-turn jailbreaks, while maintaining low over-refusal rates. We also demonstrate that expanding adversarial skill coverage, not just data scale, is key to defending against novel attacks. \textcolor{red}{\textbf{Warning: This paper contains content that may be harmful or offensive in nature.



## **26. ShadowLogic: Backdoors in Any Whitebox LLM**

cs.CR

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2511.00664v1) [paper-pdf](http://arxiv.org/pdf/2511.00664v1)

**Authors**: Kasimir Schulz, Amelia Kawasaki, Leo Ring

**Abstract**: Large language models (LLMs) are widely deployed across various applications, often with safeguards to prevent the generation of harmful or restricted content. However, these safeguards can be covertly bypassed through adversarial modifications to the computational graph of a model. This work highlights a critical security vulnerability in computational graph-based LLM formats, demonstrating that widely used deployment pipelines may be susceptible to obscured backdoors. We introduce ShadowLogic, a method for creating a backdoor in a white-box LLM by injecting an uncensoring vector into its computational graph representation. We set a trigger phrase that, when added to the beginning of a prompt into the LLM, applies the uncensoring vector and removes the content generation safeguards in the model. We embed trigger logic directly into the computational graph which detects the trigger phrase in a prompt. To evade detection of our backdoor, we obfuscate this logic within the graph structure, making it similar to standard model functions. Our method requires minimal alterations to model parameters, making backdoored models appear benign while retaining the ability to generate uncensored responses when activated. We successfully implement ShadowLogic in Phi-3 and Llama 3.2, using ONNX for manipulating computational graphs. Implanting the uncensoring vector achieved a >60% attack success rate for further malicious queries.



## **27. SLIP: Securing LLMs IP Using Weights Decomposition**

cs.CR

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2407.10886v3) [paper-pdf](http://arxiv.org/pdf/2407.10886v3)

**Authors**: Yehonathan Refael, Adam Hakim, Lev Greenberg, Satya Lokam, Tal Aviv, Ben Fishman, Shachar Seidman, Racchit Jain, Jay Tenenbaum

**Abstract**: Large language models (LLMs) have recently seen widespread adoption in both academia and industry. As these models grow, they become valuable intellectual property (IP), reflecting substantial investments by their owners. The high cost of cloud-based deployment has spurred interest in running models on edge devices, but this risks exposing parameters to theft and unauthorized use. Existing approaches to protect model IP on the edge trade off practicality, accuracy, or deployment requirements. We introduce SLIP, a hybrid inference algorithm designed to protect edge-deployed models from theft. SLIP is, to our knowledge, the first hybrid protocol that is both practical for real-world applications and provably secure, while incurring zero accuracy degradation and minimal latency overhead. It partitions the model across two computing resources: one secure but expensive, and one cost-effective but vulnerable. Using matrix decomposition, the secure resource retains the most sensitive portion of the model's IP while performing only a small fraction of the computation; the vulnerable resource executes the remainder. The protocol includes security guarantees that prevent attackers from using the partition to infer the protected information. Finally, we present experimental results that demonstrate the robustness and effectiveness of our method, positioning it as a compelling solution for protecting LLMs.



## **28. What Features in Prompts Jailbreak LLMs? Investigating the Mechanisms Behind Attacks**

cs.CR

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2411.03343v3) [paper-pdf](http://arxiv.org/pdf/2411.03343v3)

**Authors**: Nathalie Kirch, Constantin Weisser, Severin Field, Helen Yannakoudakis, Stephen Casper

**Abstract**: Jailbreaks have been a central focus of research regarding the safety and reliability of large language models (LLMs), yet the mechanisms underlying these attacks remain poorly understood. While previous studies have predominantly relied on linear methods to detect jailbreak attempts and model refusals, we take a different approach by examining both linear and non-linear features in prompts that lead to successful jailbreaks. First, we introduce a novel dataset comprising 10,800 jailbreak attempts spanning 35 diverse attack methods. Leveraging this dataset, we train linear and non-linear probes on hidden states of open-weight LLMs to predict jailbreak success. Probes achieve strong in-distribution accuracy but transfer is attack-family-specific, revealing that different jailbreaks are supported by distinct internal mechanisms rather than a single universal direction. To establish causal relevance, we construct probe-guided latent interventions that systematically shift compliance in the predicted direction. Interventions derived from non-linear probes produce larger and more reliable effects than those from linear probes, indicating that features linked to jailbreak success are encoded non-linearly in prompt representations. Overall, the results surface heterogeneous, non-linear structure in jailbreak mechanisms and provide a prompt-side methodology for recovering and testing the features that drive jailbreak outcomes.



## **29. Friend or Foe: How LLMs' Safety Mind Gets Fooled by Intent Shift Attack**

cs.CL

Preprint, 14 pages, 5 figures, 7 tables

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2511.00556v1) [paper-pdf](http://arxiv.org/pdf/2511.00556v1)

**Authors**: Peng Ding, Jun Kuang, Wen Sun, Zongyu Wang, Xuezhi Cao, Xunliang Cai, Jiajun Chen, Shujian Huang

**Abstract**: Large language models (LLMs) remain vulnerable to jailbreaking attacks despite their impressive capabilities. Investigating these weaknesses is crucial for robust safety mechanisms. Existing attacks primarily distract LLMs by introducing additional context or adversarial tokens, leaving the core harmful intent unchanged. In this paper, we introduce ISA (Intent Shift Attack), which obfuscates LLMs about the intent of the attacks. More specifically, we establish a taxonomy of intent transformations and leverage them to generate attacks that may be misperceived by LLMs as benign requests for information. Unlike prior methods relying on complex tokens or lengthy context, our approach only needs minimal edits to the original request, and yields natural, human-readable, and seemingly harmless prompts. Extensive experiments on both open-source and commercial LLMs show that ISA achieves over 70% improvement in attack success rate compared to direct harmful prompts. More critically, fine-tuning models on only benign data reformulated with ISA templates elevates success rates to nearly 100%. For defense, we evaluate existing methods and demonstrate their inadequacy against ISA, while exploring both training-free and training-based mitigation strategies. Our findings reveal fundamental challenges in intent inference for LLMs safety and underscore the need for more effective defenses. Our code and datasets are available at https://github.com/NJUNLP/ISA.



## **30. Reimagining Safety Alignment with An Image**

cs.AI

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2511.00509v1) [paper-pdf](http://arxiv.org/pdf/2511.00509v1)

**Authors**: Yifan Xia, Guorui Chen, Wenqian Yu, Zhijiang Li, Philip Torr, Jindong Gu

**Abstract**: Large language models (LLMs) excel in diverse applications but face dual challenges: generating harmful content under jailbreak attacks and over-refusal of benign queries due to rigid safety mechanisms. These issues are further complicated by the need to accommodate different value systems and precisely align with given safety preferences. Moreover, traditional methods like SFT and RLHF lack this capability due to their costly parameter tuning requirements and inability to support multiple value systems within a single model. These problems are more obvious in multimodal large language models (MLLMs), especially in terms of heightened over-refusal in cross-modal tasks and new security risks arising from expanded attack surfaces. We propose Magic Image, an optimization-driven visual prompt framework that enhances security while reducing over-refusal. By optimizing image prompts using harmful/benign samples, our method enables a single model to adapt to different value systems and better align with given safety preferences without parameter updates. Experiments demonstrate improved safety-effectiveness balance across diverse datasets while preserving model performance, offering a practical solution for deployable MLLM safety alignment.



## **31. Proactive DDoS Detection and Mitigation in Decentralized Software-Defined Networking via Port-Level Monitoring and Zero-Training Large Language Models**

cs.CR

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2511.00460v1) [paper-pdf](http://arxiv.org/pdf/2511.00460v1)

**Authors**: Mohammed N. Swileh, Shengli Zhang

**Abstract**: Centralized Software-Defined Networking (cSDN) offers flexible and programmable control of networks but suffers from scalability and reliability issues due to its reliance on centralized controllers. Decentralized SDN (dSDN) alleviates these concerns by distributing control across multiple local controllers, yet this architecture remains highly vulnerable to Distributed Denial-of-Service (DDoS) attacks. In this paper, we propose a novel detection and mitigation framework tailored for dSDN environments. The framework leverages lightweight port-level statistics combined with prompt engineering and in-context learning, enabling the DeepSeek-v3 Large Language Model (LLM) to classify traffic as benign or malicious without requiring fine-tuning or retraining. Once an anomaly is detected, mitigation is enforced directly at the attacker's port, ensuring that malicious traffic is blocked at their origin while normal traffic remains unaffected. An automatic recovery mechanism restores normal operation after the attack inactivity, ensuring both security and availability. Experimental evaluation under diverse DDoS attack scenarios demonstrates that the proposed approach achieves near-perfect detection, with 99.99% accuracy, 99.97% precision, 100% recall, 99.98% F1-score, and an AUC of 1.0. These results highlight the effectiveness of combining distributed monitoring with zero-training LLM inference, providing a proactive and scalable defense mechanism for securing dSDN infrastructures against DDoS threats.



## **32. DRIP: Defending Prompt Injection via De-instruction Training and Residual Fusion Model Architecture**

cs.CR

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2511.00447v1) [paper-pdf](http://arxiv.org/pdf/2511.00447v1)

**Authors**: Ruofan Liu, Yun Lin, Jin Song Dong

**Abstract**: Large language models (LLMs) have demonstrated impressive instruction-following capabilities. However, these capabilities also expose models to prompt injection attacks, where maliciously crafted inputs overwrite or distract from the intended instructions. A core vulnerability lies in the model's lack of semantic role understanding: it cannot distinguish directive intent from descriptive content, leading it to execute instruction-like phrases embedded in data.   We propose DRIP, a training-time defense grounded in a semantic modeling perspective, which enforces robust separation between instruction and data semantics without sacrificing utility. DRIP introduces two lightweight yet complementary mechanisms: (1) a token-wise de-instruction shift that performs semantic disentanglement, weakening directive semantics in data tokens while preserving content meaning; and (2) a residual fusion pathway that provides a persistent semantic anchor, reinforcing the influence of the true top-level instruction during generation. Experimental results on LLaMA-8B and Mistral-7B across three prompt injection benchmarks (SEP, AlpacaFarm, and InjecAgent) demonstrate that DRIP outperforms state-of-the-art defenses, including StruQ, SecAlign, ISE, and PFT, improving role separation by 49%, and reducing attack success rate by 66% for adaptive attacks. Meanwhile, DRIP's utility is on par with the undefended model across AlpacaEval, IFEval, and MT-Bench. Our findings underscore the power of lightweight representation edits and role-aware supervision in securing LLMs against adaptive prompt injection.



## **33. ToxicTextCLIP: Text-Based Poisoning and Backdoor Attacks on CLIP Pre-training**

cs.CV

Accepted by NeurIPS 2025

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2511.00446v1) [paper-pdf](http://arxiv.org/pdf/2511.00446v1)

**Authors**: Xin Yao, Haiyang Zhao, Yimin Chen, Jiawei Guo, Kecheng Huang, Ming Zhao

**Abstract**: The Contrastive Language-Image Pretraining (CLIP) model has significantly advanced vision-language modeling by aligning image-text pairs from large-scale web data through self-supervised contrastive learning. Yet, its reliance on uncurated Internet-sourced data exposes it to data poisoning and backdoor risks. While existing studies primarily investigate image-based attacks, the text modality, which is equally central to CLIP's training, remains underexplored. In this work, we introduce ToxicTextCLIP, a framework for generating high-quality adversarial texts that target CLIP during the pre-training phase. The framework addresses two key challenges: semantic misalignment caused by background inconsistency with the target class, and the scarcity of background-consistent texts. To this end, ToxicTextCLIP iteratively applies: 1) a background-aware selector that prioritizes texts with background content aligned to the target class, and 2) a background-driven augmenter that generates semantically coherent and diverse poisoned samples. Extensive experiments on classification and retrieval tasks show that ToxicTextCLIP achieves up to 95.83% poisoning success and 98.68% backdoor Hit@1, while bypassing RoCLIP, CleanCLIP and SafeCLIP defenses. The source code can be accessed via https://github.com/xinyaocse/ToxicTextCLIP/.



## **34. Enhancing Adversarial Transferability by Balancing Exploration and Exploitation with Gradient-Guided Sampling**

cs.LG

accepted by iccv 2025

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2511.00411v1) [paper-pdf](http://arxiv.org/pdf/2511.00411v1)

**Authors**: Zenghao Niu, Weicheng Xie, Siyang Song, Zitong Yu, Feng Liu, Linlin Shen

**Abstract**: Adversarial attacks present a critical challenge to deep neural networks' robustness, particularly in transfer scenarios across different model architectures. However, the transferability of adversarial attacks faces a fundamental dilemma between Exploitation (maximizing attack potency) and Exploration (enhancing cross-model generalization). Traditional momentum-based methods over-prioritize Exploitation, i.e., higher loss maxima for attack potency but weakened generalization (narrow loss surface). Conversely, recent methods with inner-iteration sampling over-prioritize Exploration, i.e., flatter loss surfaces for cross-model generalization but weakened attack potency (suboptimal local maxima). To resolve this dilemma, we propose a simple yet effective Gradient-Guided Sampling (GGS), which harmonizes both objectives through guiding sampling along the gradient ascent direction to improve both sampling efficiency and stability. Specifically, based on MI-FGSM, GGS introduces inner-iteration random sampling and guides the sampling direction using the gradient from the previous inner-iteration (the sampling's magnitude is determined by a random distribution). This mechanism encourages adversarial examples to reside in balanced regions with both flatness for cross-model generalization and higher local maxima for strong attack potency. Comprehensive experiments across multiple DNN architectures and multimodal large language models (MLLMs) demonstrate the superiority of our method over state-of-the-art transfer attacks. Code is made available at https://github.com/anuin-cat/GGS.



## **35. Stochastic Subspace Descent Accelerated via Bi-fidelity Line Search**

cs.LG

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2505.00162v2) [paper-pdf](http://arxiv.org/pdf/2505.00162v2)

**Authors**: Nuojin Cheng, Alireza Doostan, Stephen Becker

**Abstract**: Efficient optimization remains a fundamental challenge across numerous scientific and engineering domains, especially when objective function and gradient evaluations are computationally expensive. While zeroth-order optimization methods offer effective approaches when gradients are inaccessible, their practical performance can be limited by the high cost associated with function queries. This work introduces the bi-fidelity stochastic subspace descent (BF-SSD) algorithm, a novel zeroth-order optimization method designed to reduce this computational burden. BF-SSD leverages a bi-fidelity framework, constructing a surrogate model from a combination of computationally inexpensive low-fidelity (LF) and accurate high-fidelity (HF) function evaluations. This surrogate model facilitates an efficient backtracking line search for step size selection, for which we provide theoretical convergence guarantees under standard assumptions. We perform a comprehensive empirical evaluation of BF-SSD across four distinct problems: a synthetic optimization benchmark, dual-form kernel ridge regression, black-box adversarial attacks on machine learning models, and transformer-based black-box language model fine-tuning. Numerical results demonstrate that BF-SSD consistently achieves superior optimization performance while requiring significantly fewer HF function evaluations compared to relevant baseline methods. This study highlights the efficacy of integrating bi-fidelity strategies within zeroth-order optimization, positioning BF-SSD as a promising and computationally efficient approach for tackling large-scale, high-dimensional problems encountered in various real-world applications.



## **36. CoP: Agentic Red-teaming for Large Language Models using Composition of Principles**

cs.AI

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2506.00781v2) [paper-pdf](http://arxiv.org/pdf/2506.00781v2)

**Authors**: Chen Xiong, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Recent advances in Large Language Models (LLMs) have spurred transformative applications in various domains, ranging from open-source to proprietary LLMs. However, jailbreak attacks, which aim to break safety alignment and user compliance by tricking the target LLMs into answering harmful and risky responses, are becoming an urgent concern. The practice of red-teaming for LLMs is to proactively explore potential risks and error-prone instances before the release of frontier AI technology. This paper proposes an agentic workflow to automate and scale the red-teaming process of LLMs through the Composition-of-Principles (CoP) framework, where human users provide a set of red-teaming principles as instructions to an AI agent to automatically orchestrate effective red-teaming strategies and generate jailbreak prompts. Distinct from existing red-teaming methods, our CoP framework provides a unified and extensible framework to encompass and orchestrate human-provided red-teaming principles to enable the automated discovery of new red-teaming strategies. When tested against leading LLMs, CoP reveals unprecedented safety risks by finding novel jailbreak prompts and improving the best-known single-turn attack success rate by up to 19.0 times.



## **37. Exploiting Latent Space Discontinuities for Building Universal LLM Jailbreaks and Data Extraction Attacks**

cs.CR

10 pages, 5 figures, 4 tables, Published at the Brazilian Symposium  on Cybersecurity (SBSeg 2025)

**SubmitDate**: 2025-11-01    [abs](http://arxiv.org/abs/2511.00346v1) [paper-pdf](http://arxiv.org/pdf/2511.00346v1)

**Authors**: Kayua Oleques Paim, Rodrigo Brandao Mansilha, Diego Kreutz, Muriel Figueredo Franco, Weverton Cordeiro

**Abstract**: The rapid proliferation of Large Language Models (LLMs) has raised significant concerns about their security against adversarial attacks. In this work, we propose a novel approach to crafting universal jailbreaks and data extraction attacks by exploiting latent space discontinuities, an architectural vulnerability related to the sparsity of training data. Unlike previous methods, our technique generalizes across various models and interfaces, proving highly effective in seven state-of-the-art LLMs and one image generation model. Initial results indicate that when these discontinuities are exploited, they can consistently and profoundly compromise model behavior, even in the presence of layered defenses. The findings suggest that this strategy has substantial potential as a systemic attack vector.



## **38. Visual Backdoor Attacks on MLLM Embodied Decision Making via Contrastive Trigger Learning**

cs.AI

**SubmitDate**: 2025-10-31    [abs](http://arxiv.org/abs/2510.27623v1) [paper-pdf](http://arxiv.org/pdf/2510.27623v1)

**Authors**: Qiusi Zhan, Hyeonjeong Ha, Rui Yang, Sirui Xu, Hanyang Chen, Liang-Yan Gui, Yu-Xiong Wang, Huan Zhang, Heng Ji, Daniel Kang

**Abstract**: Multimodal large language models (MLLMs) have advanced embodied agents by enabling direct perception, reasoning, and planning task-oriented actions from visual inputs. However, such vision driven embodied agents open a new attack surface: visual backdoor attacks, where the agent behaves normally until a visual trigger appears in the scene, then persistently executes an attacker-specified multi-step policy. We introduce BEAT, the first framework to inject such visual backdoors into MLLM-based embodied agents using objects in the environments as triggers. Unlike textual triggers, object triggers exhibit wide variation across viewpoints and lighting, making them difficult to implant reliably. BEAT addresses this challenge by (1) constructing a training set that spans diverse scenes, tasks, and trigger placements to expose agents to trigger variability, and (2) introducing a two-stage training scheme that first applies supervised fine-tuning (SFT) and then our novel Contrastive Trigger Learning (CTL). CTL formulates trigger discrimination as preference learning between trigger-present and trigger-free inputs, explicitly sharpening the decision boundaries to ensure precise backdoor activation. Across various embodied agent benchmarks and MLLMs, BEAT achieves attack success rates up to 80%, while maintaining strong benign task performance, and generalizes reliably to out-of-distribution trigger placements. Notably, compared to naive SFT, CTL boosts backdoor activation accuracy up to 39% under limited backdoor data. These findings expose a critical yet unexplored security risk in MLLM-based embodied agents, underscoring the need for robust defenses before real-world deployment.



## **39. Decoding Latent Attack Surfaces in LLMs: Prompt Injection via HTML in Web Summarization**

cs.CR

**SubmitDate**: 2025-10-31    [abs](http://arxiv.org/abs/2509.05831v2) [paper-pdf](http://arxiv.org/pdf/2509.05831v2)

**Authors**: Ishaan Verma

**Abstract**: Large Language Models (LLMs) are increasingly integrated into web-based systems for content summarization, yet their susceptibility to prompt injection attacks remains a pressing concern. In this study, we explore how non-visible HTML elements such as <meta>, aria-label, and alt attributes can be exploited to embed adversarial instructions without altering the visible content of a webpage. We introduce a novel dataset comprising 280 static web pages, evenly divided between clean and adversarial injected versions, crafted using diverse HTML-based strategies. These pages are processed through a browser automation pipeline to extract both raw HTML and rendered text, closely mimicking real-world LLM deployment scenarios. We evaluate two state-of-the-art open-source models, Llama 4 Scout (Meta) and Gemma 9B IT (Google), on their ability to summarize this content. Using both lexical (ROUGE-L) and semantic (SBERT cosine similarity) metrics, along with manual annotations, we assess the impact of these covert injections. Our findings reveal that over 29% of injected samples led to noticeable changes in the Llama 4 Scout summaries, while Gemma 9B IT showed a lower, yet non-trivial, success rate of 15%. These results highlight a critical and largely overlooked vulnerability in LLM driven web pipelines, where hidden adversarial content can subtly manipulate model outputs. Our work offers a reproducible framework and benchmark for evaluating HTML-based prompt injection and underscores the urgent need for robust mitigation strategies in LLM applications involving web content.



## **40. Eliciting Secret Knowledge from Language Models**

cs.LG

**SubmitDate**: 2025-10-31    [abs](http://arxiv.org/abs/2510.01070v2) [paper-pdf](http://arxiv.org/pdf/2510.01070v2)

**Authors**: Bartosz Cywiński, Emil Ryd, Rowan Wang, Senthooran Rajamanoharan, Neel Nanda, Arthur Conmy, Samuel Marks

**Abstract**: We study secret elicitation: discovering knowledge that an AI possesses but does not explicitly verbalize. As a testbed, we train three families of large language models (LLMs) to possess specific knowledge that they apply downstream but deny knowing when asked directly. For example, in one setting, we train an LLM to generate replies that are consistent with knowing the user is female, while denying this knowledge when asked directly. We then design various black-box and white-box secret elicitation techniques and evaluate them based on whether they can help an LLM auditor successfully guess the secret knowledge. Many of our techniques improve on simple baselines. Our most effective techniques (performing best in all settings) are based on prefill attacks, a black-box technique where the LLM reveals secret knowledge when generating a completion from a predefined prefix. Our white-box techniques based on logit lens and sparse autoencoders (SAEs) also consistently increase the success rate of the LLM auditor, but are less effective. We release our models and code, establishing a public benchmark for evaluating secret elicitation methods.



## **41. Prevalence of Security and Privacy Risk-Inducing Usage of AI-based Conversational Agents**

cs.CR

10 pages, 3 figures, 5 tables, under submission

**SubmitDate**: 2025-10-31    [abs](http://arxiv.org/abs/2510.27275v1) [paper-pdf](http://arxiv.org/pdf/2510.27275v1)

**Authors**: Kathrin Grosse, Nico Ebert

**Abstract**: Recent improvement gains in large language models (LLMs) have lead to everyday usage of AI-based Conversational Agents (CAs). At the same time, LLMs are vulnerable to an array of threats, including jailbreaks and, for example, causing remote code execution when fed specific inputs. As a result, users may unintentionally introduce risks, for example, by uploading malicious files or disclosing sensitive information. However, the extent to which such user behaviors occur and thus potentially facilitate exploits remains largely unclear. To shed light on this issue, we surveyed a representative sample of 3,270 UK adults in 2024 using Prolific. A third of these use CA services such as ChatGPT or Gemini at least once a week. Of these ``regular users'', up to a third exhibited behaviors that may enable attacks, and a fourth have tried jailbreaking (often out of understandable reasons such as curiosity, fun or information seeking). Half state that they sanitize data and most participants report not sharing sensitive data. However, few share very sensitive data such as passwords. The majority are unaware that their data can be used to train models and that they can opt-out. Our findings suggest that current academic threat models manifest in the wild, and mitigations or guidelines for the secure usage of CAs should be developed. In areas critical to security and privacy, CAs must be equipped with effective AI guardrails to prevent, for example, revealing sensitive information to curious employees. Vendors need to increase efforts to prevent the entry of sensitive data, and to create transparency with regard to data usage policies and settings.



## **42. Adaptive Defense against Harmful Fine-Tuning for Large Language Models via Bayesian Data Scheduler**

cs.LG

**SubmitDate**: 2025-10-31    [abs](http://arxiv.org/abs/2510.27172v1) [paper-pdf](http://arxiv.org/pdf/2510.27172v1)

**Authors**: Zixuan Hu, Li Shen, Zhenyi Wang, Yongxian Wei, Dacheng Tao

**Abstract**: Harmful fine-tuning poses critical safety risks to fine-tuning-as-a-service for large language models. Existing defense strategies preemptively build robustness via attack simulation but suffer from fundamental limitations: (i) the infeasibility of extending attack simulations beyond bounded threat models due to the inherent difficulty of anticipating unknown attacks, and (ii) limited adaptability to varying attack settings, as simulation fails to capture their variability and complexity. To address these challenges, we propose Bayesian Data Scheduler (BDS), an adaptive tuning-stage defense strategy with no need for attack simulation. BDS formulates harmful fine-tuning defense as a Bayesian inference problem, learning the posterior distribution of each data point's safety attribute, conditioned on the fine-tuning and alignment datasets. The fine-tuning process is then constrained by weighting data with their safety attributes sampled from the posterior, thus mitigating the influence of harmful data. By leveraging the post hoc nature of Bayesian inference, the posterior is conditioned on the fine-tuning dataset, enabling BDS to tailor its defense to the specific dataset, thereby achieving adaptive defense. Furthermore, we introduce a neural scheduler based on amortized Bayesian learning, enabling efficient transfer to new data without retraining. Comprehensive results across diverse attack and defense settings demonstrate the state-of-the-art performance of our approach. Code is available at https://github.com/Egg-Hu/Bayesian-Data-Scheduler.



## **43. Characterizing Selective Refusal Bias in Large Language Models**

cs.CL

21 pages, 12 figures, 14 tables

**SubmitDate**: 2025-10-31    [abs](http://arxiv.org/abs/2510.27087v1) [paper-pdf](http://arxiv.org/pdf/2510.27087v1)

**Authors**: Adel Khorramrouz, Sharon Levy

**Abstract**: Safety guardrails in large language models(LLMs) are developed to prevent malicious users from generating toxic content at a large scale. However, these measures can inadvertently introduce or reflect new biases, as LLMs may refuse to generate harmful content targeting some demographic groups and not others. We explore this selective refusal bias in LLM guardrails through the lens of refusal rates of targeted individual and intersectional demographic groups, types of LLM responses, and length of generated refusals. Our results show evidence of selective refusal bias across gender, sexual orientation, nationality, and religion attributes. This leads us to investigate additional safety implications via an indirect attack, where we target previously refused groups. Our findings emphasize the need for more equitable and robust performance in safety guardrails across demographic groups.



## **44. Adapting Large Language Models to Emerging Cybersecurity using Retrieval Augmented Generation**

cs.CR

**SubmitDate**: 2025-10-31    [abs](http://arxiv.org/abs/2510.27080v1) [paper-pdf](http://arxiv.org/pdf/2510.27080v1)

**Authors**: Arnabh Borah, Md Tanvirul Alam, Nidhi Rastogi

**Abstract**: Security applications are increasingly relying on large language models (LLMs) for cyber threat detection; however, their opaque reasoning often limits trust, particularly in decisions that require domain-specific cybersecurity knowledge. Because security threats evolve rapidly, LLMs must not only recall historical incidents but also adapt to emerging vulnerabilities and attack patterns. Retrieval-Augmented Generation (RAG) has demonstrated effectiveness in general LLM applications, but its potential for cybersecurity remains underexplored. In this work, we introduce a RAG-based framework designed to contextualize cybersecurity data and enhance LLM accuracy in knowledge retention and temporal reasoning. Using external datasets and the Llama-3-8B-Instruct model, we evaluate baseline RAG, an optimized hybrid retrieval approach, and conduct a comparative analysis across multiple performance metrics. Our findings highlight the promise of hybrid retrieval in strengthening the adaptability and reliability of LLMs for cybersecurity tasks.



## **45. LLM-based Multi-class Attack Analysis and Mitigation Framework in IoT/IIoT Networks**

cs.CR

**SubmitDate**: 2025-10-30    [abs](http://arxiv.org/abs/2510.26941v1) [paper-pdf](http://arxiv.org/pdf/2510.26941v1)

**Authors**: Seif Ikbarieh, Maanak Gupta, Elmahedi Mahalal

**Abstract**: The Internet of Things has expanded rapidly, transforming communication and operations across industries but also increasing the attack surface and security breaches. Artificial Intelligence plays a key role in securing IoT, enabling attack detection, attack behavior analysis, and mitigation suggestion. Despite advancements, evaluations remain purely qualitative, and the lack of a standardized, objective benchmark for quantitatively measuring AI-based attack analysis and mitigation hinders consistent assessment of model effectiveness. In this work, we propose a hybrid framework combining Machine Learning (ML) for multi-class attack detection with Large Language Models (LLMs) for attack behavior analysis and mitigation suggestion. After benchmarking several ML and Deep Learning (DL) classifiers on the Edge-IIoTset and CICIoT2023 datasets, we applied structured role-play prompt engineering with Retrieval-Augmented Generation (RAG) to guide ChatGPT-o3 and DeepSeek-R1 in producing detailed, context-aware responses. We introduce novel evaluation metrics for quantitative assessment to guide us and an ensemble of judge LLMs, namely ChatGPT-4o, DeepSeek-V3, Mixtral 8x7B Instruct, Gemini 2.5 Flash, Meta Llama 4, TII Falcon H1 34B Instruct, xAI Grok 3, and Claude 4 Sonnet, to independently evaluate the responses. Results show that Random Forest has the best detection model, and ChatGPT-o3 outperformed DeepSeek-R1 in attack analysis and mitigation.



## **46. LatentBreak: Jailbreaking Large Language Models through Latent Space Feedback**

cs.CL

**SubmitDate**: 2025-10-30    [abs](http://arxiv.org/abs/2510.08604v2) [paper-pdf](http://arxiv.org/pdf/2510.08604v2)

**Authors**: Raffaele Mura, Giorgio Piras, Kamilė Lukošiūtė, Maura Pintor, Amin Karbasi, Battista Biggio

**Abstract**: Jailbreaks are adversarial attacks designed to bypass the built-in safety mechanisms of large language models. Automated jailbreaks typically optimize an adversarial suffix or adapt long prompt templates by forcing the model to generate the initial part of a restricted or harmful response. In this work, we show that existing jailbreak attacks that leverage such mechanisms to unlock the model response can be detected by a straightforward perplexity-based filtering on the input prompt. To overcome this issue, we propose LatentBreak, a white-box jailbreak attack that generates natural adversarial prompts with low perplexity capable of evading such defenses. LatentBreak substitutes words in the input prompt with semantically-equivalent ones, preserving the initial intent of the prompt, instead of adding high-perplexity adversarial suffixes or long templates. These words are chosen by minimizing the distance in the latent space between the representation of the adversarial prompt and that of harmless requests. Our extensive evaluation shows that LatentBreak leads to shorter and low-perplexity prompts, thus outperforming competing jailbreak algorithms against perplexity-based filters on multiple safety-aligned models.



## **47. Broken-Token: Filtering Obfuscated Prompts by Counting Characters-Per-Token**

cs.CR

16 pages, 9 figures

**SubmitDate**: 2025-10-30    [abs](http://arxiv.org/abs/2510.26847v1) [paper-pdf](http://arxiv.org/pdf/2510.26847v1)

**Authors**: Shaked Zychlinski, Yuval Kainan

**Abstract**: Large Language Models (LLMs) are susceptible to jailbreak attacks where malicious prompts are disguised using ciphers and character-level encodings to bypass safety guardrails. While these guardrails often fail to interpret the encoded content, the underlying models can still process the harmful instructions. We introduce CPT-Filtering, a novel, model-agnostic with negligible-costs and near-perfect accuracy guardrail technique that aims to mitigate these attacks by leveraging the intrinsic behavior of Byte-Pair Encoding (BPE) tokenizers. Our method is based on the principle that tokenizers, trained on natural language, represent out-of-distribution text, such as ciphers, using a significantly higher number of shorter tokens. Our technique uses a simple yet powerful artifact of using language models: the average number of Characters Per Token (CPT) in the text. This approach is motivated by the high compute cost of modern methods - relying on added modules such as dedicated LLMs or perplexity models. We validate our approach across a large dataset of over 100,000 prompts, testing numerous encoding schemes with several popular tokenizers. Our experiments demonstrate that a simple CPT threshold robustly identifies encoded text with high accuracy, even for very short inputs. CPT-Filtering provides a practical defense layer that can be immediately deployed for real-time text filtering and offline data curation.



## **48. PVMark: Enabling Public Verifiability for LLM Watermarking Schemes**

cs.CR

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-10-30    [abs](http://arxiv.org/abs/2510.26274v1) [paper-pdf](http://arxiv.org/pdf/2510.26274v1)

**Authors**: Haohua Duan, Liyao Xiang, Xin Zhang

**Abstract**: Watermarking schemes for large language models (LLMs) have been proposed to identify the source of the generated text, mitigating the potential threats emerged from model theft. However, current watermarking solutions hardly resolve the trust issue: the non-public watermark detection cannot prove itself faithfully conducting the detection. We observe that it is attributed to the secret key mostly used in the watermark detection -- it cannot be public, or the adversary may launch removal attacks provided the key; nor can it be private, or the watermarking detection is opaque to the public. To resolve the dilemma, we propose PVMark, a plugin based on zero-knowledge proof (ZKP), enabling the watermark detection process to be publicly verifiable by third parties without disclosing any secret key. PVMark hinges upon the proof of `correct execution' of watermark detection on which a set of ZKP constraints are built, including mapping, random number generation, comparison, and summation. We implement multiple variants of PVMark in Python, Rust and Circom, covering combinations of three watermarking schemes, three hash functions, and four ZKP protocols, to show our approach effectively works under a variety of circumstances. By experimental results, PVMark efficiently enables public verifiability on the state-of-the-art LLM watermarking schemes yet without compromising the watermarking performance, promising to be deployed in practice.



## **49. IRCopilot: Automated Incident Response with Large Language Models**

cs.CR

**SubmitDate**: 2025-10-30    [abs](http://arxiv.org/abs/2505.20945v3) [paper-pdf](http://arxiv.org/pdf/2505.20945v3)

**Authors**: Xihuan Lin, Jie Zhang, Gelei Deng, Tianzhe Liu, Tianwei Zhang, Qing Guo, Riqing Chen

**Abstract**: Incident response plays a pivotal role in mitigating the impact of cyber attacks. In recent years, the intensity and complexity of global cyber threats have grown significantly, making it increasingly challenging for traditional threat detection and incident response methods to operate effectively in complex network environments. While Large Language Models (LLMs) have shown great potential in early threat detection, their capabilities remain limited when it comes to automated incident response after an intrusion. To address this gap, we construct an incremental benchmark based on real-world incident response tasks to thoroughly evaluate the performance of LLMs in this domain. Our analysis reveals several key challenges that hinder the practical application of contemporary LLMs, including context loss, hallucinations, privacy protection concerns, and their limited ability to provide accurate, context-specific recommendations. In response to these challenges, we propose IRCopilot, a novel framework for automated incident response powered by LLMs. IRCopilot mimics the three dynamic phases of a real-world incident response team using four collaborative LLM-based session components. These components are designed with clear divisions of responsibility, reducing issues such as hallucinations and context loss. Our method leverages diverse prompt designs and strategic responsibility segmentation, significantly improving the system's practicality and efficiency. Experimental results demonstrate that IRCopilot outperforms baseline LLMs across key benchmarks, achieving sub-task completion rates of 150%, 138%, 136%, 119%, and 114% for various response tasks. Moreover, IRCopilot exhibits robust performance on public incident response platforms and in real-world attack scenarios, showcasing its strong applicability.



## **50. ALMGuard: Safety Shortcuts and Where to Find Them as Guardrails for Audio-Language Models**

cs.SD

Accepted to NeurIPS 2025

**SubmitDate**: 2025-10-30    [abs](http://arxiv.org/abs/2510.26096v1) [paper-pdf](http://arxiv.org/pdf/2510.26096v1)

**Authors**: Weifei Jin, Yuxin Cao, Junjie Su, Minhui Xue, Jie Hao, Ke Xu, Jin Song Dong, Derui Wang

**Abstract**: Recent advances in Audio-Language Models (ALMs) have significantly improved multimodal understanding capabilities. However, the introduction of the audio modality also brings new and unique vulnerability vectors. Previous studies have proposed jailbreak attacks that specifically target ALMs, revealing that defenses directly transferred from traditional audio adversarial attacks or text-based Large Language Model (LLM) jailbreaks are largely ineffective against these ALM-specific threats. To address this issue, we propose ALMGuard, the first defense framework tailored to ALMs. Based on the assumption that safety-aligned shortcuts naturally exist in ALMs, we design a method to identify universal Shortcut Activation Perturbations (SAPs) that serve as triggers that activate the safety shortcuts to safeguard ALMs at inference time. To better sift out effective triggers while preserving the model's utility on benign tasks, we further propose Mel-Gradient Sparse Mask (M-GSM), which restricts perturbations to Mel-frequency bins that are sensitive to jailbreaks but insensitive to speech understanding. Both theoretical analyses and empirical results demonstrate the robustness of our method against both seen and unseen attacks. Overall, \MethodName reduces the average success rate of advanced ALM-specific jailbreak attacks to 4.6% across four models, while maintaining comparable utility on benign benchmarks, establishing it as the new state of the art. Our code and data are available at https://github.com/WeifeiJin/ALMGuard.



