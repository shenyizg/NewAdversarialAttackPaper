# Latest Large Language Model Attack Papers
**update at 2025-11-03 09:06:47**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. LatentBreak: Jailbreaking Large Language Models through Latent Space Feedback**

cs.CL

**SubmitDate**: 2025-10-30    [abs](http://arxiv.org/abs/2510.08604v2) [paper-pdf](http://arxiv.org/pdf/2510.08604v2)

**Authors**: Raffaele Mura, Giorgio Piras, Kamilė Lukošiūtė, Maura Pintor, Amin Karbasi, Battista Biggio

**Abstract**: Jailbreaks are adversarial attacks designed to bypass the built-in safety mechanisms of large language models. Automated jailbreaks typically optimize an adversarial suffix or adapt long prompt templates by forcing the model to generate the initial part of a restricted or harmful response. In this work, we show that existing jailbreak attacks that leverage such mechanisms to unlock the model response can be detected by a straightforward perplexity-based filtering on the input prompt. To overcome this issue, we propose LatentBreak, a white-box jailbreak attack that generates natural adversarial prompts with low perplexity capable of evading such defenses. LatentBreak substitutes words in the input prompt with semantically-equivalent ones, preserving the initial intent of the prompt, instead of adding high-perplexity adversarial suffixes or long templates. These words are chosen by minimizing the distance in the latent space between the representation of the adversarial prompt and that of harmless requests. Our extensive evaluation shows that LatentBreak leads to shorter and low-perplexity prompts, thus outperforming competing jailbreak algorithms against perplexity-based filters on multiple safety-aligned models.



## **2. PVMark: Enabling Public Verifiability for LLM Watermarking Schemes**

cs.CR

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-10-30    [abs](http://arxiv.org/abs/2510.26274v1) [paper-pdf](http://arxiv.org/pdf/2510.26274v1)

**Authors**: Haohua Duan, Liyao Xiang, Xin Zhang

**Abstract**: Watermarking schemes for large language models (LLMs) have been proposed to identify the source of the generated text, mitigating the potential threats emerged from model theft. However, current watermarking solutions hardly resolve the trust issue: the non-public watermark detection cannot prove itself faithfully conducting the detection. We observe that it is attributed to the secret key mostly used in the watermark detection -- it cannot be public, or the adversary may launch removal attacks provided the key; nor can it be private, or the watermarking detection is opaque to the public. To resolve the dilemma, we propose PVMark, a plugin based on zero-knowledge proof (ZKP), enabling the watermark detection process to be publicly verifiable by third parties without disclosing any secret key. PVMark hinges upon the proof of `correct execution' of watermark detection on which a set of ZKP constraints are built, including mapping, random number generation, comparison, and summation. We implement multiple variants of PVMark in Python, Rust and Circom, covering combinations of three watermarking schemes, three hash functions, and four ZKP protocols, to show our approach effectively works under a variety of circumstances. By experimental results, PVMark efficiently enables public verifiability on the state-of-the-art LLM watermarking schemes yet without compromising the watermarking performance, promising to be deployed in practice.



## **3. IRCopilot: Automated Incident Response with Large Language Models**

cs.CR

**SubmitDate**: 2025-10-30    [abs](http://arxiv.org/abs/2505.20945v3) [paper-pdf](http://arxiv.org/pdf/2505.20945v3)

**Authors**: Xihuan Lin, Jie Zhang, Gelei Deng, Tianzhe Liu, Tianwei Zhang, Qing Guo, Riqing Chen

**Abstract**: Incident response plays a pivotal role in mitigating the impact of cyber attacks. In recent years, the intensity and complexity of global cyber threats have grown significantly, making it increasingly challenging for traditional threat detection and incident response methods to operate effectively in complex network environments. While Large Language Models (LLMs) have shown great potential in early threat detection, their capabilities remain limited when it comes to automated incident response after an intrusion. To address this gap, we construct an incremental benchmark based on real-world incident response tasks to thoroughly evaluate the performance of LLMs in this domain. Our analysis reveals several key challenges that hinder the practical application of contemporary LLMs, including context loss, hallucinations, privacy protection concerns, and their limited ability to provide accurate, context-specific recommendations. In response to these challenges, we propose IRCopilot, a novel framework for automated incident response powered by LLMs. IRCopilot mimics the three dynamic phases of a real-world incident response team using four collaborative LLM-based session components. These components are designed with clear divisions of responsibility, reducing issues such as hallucinations and context loss. Our method leverages diverse prompt designs and strategic responsibility segmentation, significantly improving the system's practicality and efficiency. Experimental results demonstrate that IRCopilot outperforms baseline LLMs across key benchmarks, achieving sub-task completion rates of 150%, 138%, 136%, 119%, and 114% for various response tasks. Moreover, IRCopilot exhibits robust performance on public incident response platforms and in real-world attack scenarios, showcasing its strong applicability.



## **4. ALMGuard: Safety Shortcuts and Where to Find Them as Guardrails for Audio-Language Models**

cs.SD

Accepted to NeurIPS 2025

**SubmitDate**: 2025-10-30    [abs](http://arxiv.org/abs/2510.26096v1) [paper-pdf](http://arxiv.org/pdf/2510.26096v1)

**Authors**: Weifei Jin, Yuxin Cao, Junjie Su, Minhui Xue, Jie Hao, Ke Xu, Jin Song Dong, Derui Wang

**Abstract**: Recent advances in Audio-Language Models (ALMs) have significantly improved multimodal understanding capabilities. However, the introduction of the audio modality also brings new and unique vulnerability vectors. Previous studies have proposed jailbreak attacks that specifically target ALMs, revealing that defenses directly transferred from traditional audio adversarial attacks or text-based Large Language Model (LLM) jailbreaks are largely ineffective against these ALM-specific threats. To address this issue, we propose ALMGuard, the first defense framework tailored to ALMs. Based on the assumption that safety-aligned shortcuts naturally exist in ALMs, we design a method to identify universal Shortcut Activation Perturbations (SAPs) that serve as triggers that activate the safety shortcuts to safeguard ALMs at inference time. To better sift out effective triggers while preserving the model's utility on benign tasks, we further propose Mel-Gradient Sparse Mask (M-GSM), which restricts perturbations to Mel-frequency bins that are sensitive to jailbreaks but insensitive to speech understanding. Both theoretical analyses and empirical results demonstrate the robustness of our method against both seen and unseen attacks. Overall, \MethodName reduces the average success rate of advanced ALM-specific jailbreak attacks to 4.6% across four models, while maintaining comparable utility on benign benchmarks, establishing it as the new state of the art. Our code and data are available at https://github.com/WeifeiJin/ALMGuard.



## **5. Chain-of-Scrutiny: Detecting Backdoor Attacks for Large Language Models**

cs.CR

This paper has been accepted to ACL Findings 2025

**SubmitDate**: 2025-10-30    [abs](http://arxiv.org/abs/2406.05948v4) [paper-pdf](http://arxiv.org/pdf/2406.05948v4)

**Authors**: Xi Li, Ruofan Mao, Yusen Zhang, Renze Lou, Chen Wu, Jiaqi Wang

**Abstract**: Large Language Models (LLMs), especially those accessed via APIs, have demonstrated impressive capabilities across various domains. However, users without technical expertise often turn to (untrustworthy) third-party services, such as prompt engineering, to enhance their LLM experience, creating vulnerabilities to adversarial threats like backdoor attacks. Backdoor-compromised LLMs generate malicious outputs to users when inputs contain specific "triggers" set by attackers. Traditional defense strategies, originally designed for small-scale models, are impractical for API-accessible LLMs due to limited model access, high computational costs, and data requirements. To address these limitations, we propose Chain-of-Scrutiny (CoS) which leverages LLMs' unique reasoning abilities to mitigate backdoor attacks. It guides the LLM to generate reasoning steps for a given input and scrutinizes for consistency with the final output -- any inconsistencies indicating a potential attack. It is well-suited for the popular API-only LLM deployments, enabling detection at minimal cost and with little data. User-friendly and driven by natural language, it allows non-experts to perform the defense independently while maintaining transparency. We validate the effectiveness of CoS through extensive experiments on various tasks and LLMs, with results showing greater benefits for more powerful LLMs.



## **6. Improving LLM Safety Alignment with Dual-Objective Optimization**

cs.CL

ICML 2025

**SubmitDate**: 2025-10-30    [abs](http://arxiv.org/abs/2503.03710v3) [paper-pdf](http://arxiv.org/pdf/2503.03710v3)

**Authors**: Xuandong Zhao, Will Cai, Tianneng Shi, David Huang, Licong Lin, Song Mei, Dawn Song

**Abstract**: Existing training-time safety alignment techniques for large language models (LLMs) remain vulnerable to jailbreak attacks. Direct preference optimization (DPO), a widely deployed alignment method, exhibits limitations in both experimental and theoretical contexts as its loss function proves suboptimal for refusal learning. Through gradient-based analysis, we identify these shortcomings and propose an improved safety alignment that disentangles DPO objectives into two components: (1) robust refusal training, which encourages refusal even when partial unsafe generations are produced, and (2) targeted unlearning of harmful knowledge. This approach significantly increases LLM robustness against a wide range of jailbreak attacks, including prefilling, suffix, and multi-turn attacks across both in-distribution and out-of-distribution scenarios. Furthermore, we introduce a method to emphasize critical refusal tokens by incorporating a reward-based token-level weighting mechanism for refusal learning, which further improves the robustness against adversarial exploits. Our research also suggests that robustness to jailbreak attacks is correlated with token distribution shifts in the training process and internal representations of refusal and harmful tokens, offering valuable directions for future research in LLM safety alignment. The code is available at https://github.com/wicai24/DOOR-Alignment



## **7. SoK: Honeypots & LLMs, More Than the Sum of Their Parts?**

cs.CR

Systemization of Knowledge

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2510.25939v1) [paper-pdf](http://arxiv.org/pdf/2510.25939v1)

**Authors**: Robert A. Bridges, Thomas R. Mitchell, Mauricio Muñoz, Ted Henriksson

**Abstract**: The advent of Large Language Models (LLMs) promised to resolve the long-standing paradox in honeypot design: achieving high-fidelity deception with low operational risk. However, despite a flurry of research since late 2022, progress has been incremental, and the field lacks a cohesive understanding of the emerging architectural patterns, core challenges, and evaluation paradigms. To fill this gap, this Systematization of Knowledge (SoK) paper provides the first comprehensive overview of this new domain. We survey and systematize three critical, intersecting research areas: first, we provide a taxonomy of honeypot detection vectors, structuring the core problems that LLM-based realism must solve; second, we synthesize the emerging literature on LLM-honeypots, identifying a canonical architecture and key evaluation trends; and third, we chart the evolutionary path of honeypot log analysis, from simple data reduction to automated intelligence generation. We synthesize these findings into a forward-looking research roadmap, arguing that the true potential of this technology lies in creating autonomous, self-improving deception systems to counter the emerging threat of intelligent, automated attackers.



## **8. Adversarial Paraphrasing: A Universal Attack for Humanizing AI-Generated Text**

cs.CL

NeurIPS 2025

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2506.07001v2) [paper-pdf](http://arxiv.org/pdf/2506.07001v2)

**Authors**: Yize Cheng, Vinu Sankar Sadasivan, Mehrdad Saberi, Shoumik Saha, Soheil Feizi

**Abstract**: The increasing capabilities of Large Language Models (LLMs) have raised concerns about their misuse in AI-generated plagiarism and social engineering. While various AI-generated text detectors have been proposed to mitigate these risks, many remain vulnerable to simple evasion techniques such as paraphrasing. However, recent detectors have shown greater robustness against such basic attacks. In this work, we introduce Adversarial Paraphrasing, a training-free attack framework that universally humanizes any AI-generated text to evade detection more effectively. Our approach leverages an off-the-shelf instruction-following LLM to paraphrase AI-generated content under the guidance of an AI text detector, producing adversarial examples that are specifically optimized to bypass detection. Extensive experiments show that our attack is both broadly effective and highly transferable across several detection systems. For instance, compared to simple paraphrasing attack--which, ironically, increases the true positive at 1% false positive (T@1%F) by 8.57% on RADAR and 15.03% on Fast-DetectGPT--adversarial paraphrasing, guided by OpenAI-RoBERTa-Large, reduces T@1%F by 64.49% on RADAR and a striking 98.96% on Fast-DetectGPT. Across a diverse set of detectors--including neural network-based, watermark-based, and zero-shot approaches--our attack achieves an average T@1%F reduction of 87.88% under the guidance of OpenAI-RoBERTa-Large. We also analyze the tradeoff between text quality and attack success to find that our method can significantly reduce detection rates, with mostly a slight degradation in text quality. Our adversarial setup highlights the need for more robust and resilient detection strategies in the light of increasingly sophisticated evasion techniques.



## **9. Securing AI Agent Execution**

cs.CR

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2510.21236v2) [paper-pdf](http://arxiv.org/pdf/2510.21236v2)

**Authors**: Christoph Bühler, Matteo Biagiola, Luca Di Grazia, Guido Salvaneschi

**Abstract**: Large Language Models (LLMs) have evolved into AI agents that interact with external tools and environments to perform complex tasks. The Model Context Protocol (MCP) has become the de facto standard for connecting agents with such resources, but security has lagged behind: thousands of MCP servers execute with unrestricted access to host systems, creating a broad attack surface. In this paper, we introduce AgentBound, the first access control framework for MCP servers. AgentBound combines a declarative policy mechanism, inspired by the Android permission model, with a policy enforcement engine that contains malicious behavior without requiring MCP server modifications. We build a dataset containing the 296 most popular MCP servers, and show that access control policies can be generated automatically from source code with 80.9% accuracy. We also show that AgentBound blocks the majority of security threats in several malicious MCP servers, and that policy enforcement engine introduces negligible overhead. Our contributions provide developers and project managers with a practical foundation for securing MCP servers while maintaining productivity, enabling researchers and tool builders to explore new directions for declarative access control and MCP security.



## **10. NetEcho: From Real-World Streaming Side-Channels to Full LLM Conversation Recovery**

cs.CR

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2510.25472v1) [paper-pdf](http://arxiv.org/pdf/2510.25472v1)

**Authors**: Zheng Zhang, Guanlong Wu, Sen Deng, Shuai Wang, Yinqian Zhang

**Abstract**: In the rapidly expanding landscape of Large Language Model (LLM) applications, real-time output streaming has become the dominant interaction paradigm. While this enhances user experience, recent research reveals that it exposes a non-trivial attack surface through network side-channels. Adversaries can exploit patterns in encrypted traffic to infer sensitive information and reconstruct private conversations. In response, LLM providers and third-party services are deploying defenses such as traffic padding and obfuscation to mitigate these vulnerabilities.   This paper starts by presenting a systematic analysis of contemporary side-channel defenses in mainstream LLM applications, with a focus on services from vendors like OpenAI and DeepSeek. We identify and examine seven representative deployment scenarios, each incorporating active/passive mitigation techniques. Despite these enhanced security measures, our investigation uncovers significant residual information that remains vulnerable to leakage within the network traffic.   Building on this discovery, we introduce NetEcho, a novel, LLM-based framework that comprehensively unleashes the network side-channel risks of today's LLM applications. NetEcho is designed to recover entire conversations -- including both user prompts and LLM responses -- directly from encrypted network traffic. It features a deliberate design that ensures high-fidelity text recovery, transferability across different deployment scenarios, and moderate operational cost. In our evaluations on medical and legal applications built upon leading models like DeepSeek-v3 and GPT-4o, NetEcho can recover avg $\sim$70\% information of each conversation, demonstrating a critical limitation in current defense mechanisms. We conclude by discussing the implications of our findings and proposing future directions for augmenting network traffic security.



## **11. Pentest-R1: Towards Autonomous Penetration Testing Reasoning Optimized via Two-Stage Reinforcement Learning**

cs.AI

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2508.07382v2) [paper-pdf](http://arxiv.org/pdf/2508.07382v2)

**Authors**: He Kong, Die Hu, Jingguo Ge, Liangxiong Li, Hui Li, Tong Li

**Abstract**: Automating penetration testing is crucial for enhancing cybersecurity, yet current Large Language Models (LLMs) face significant limitations in this domain, including poor error handling, inefficient reasoning, and an inability to perform complex end-to-end tasks autonomously. To address these challenges, we introduce Pentest-R1, a novel framework designed to optimize LLM reasoning capabilities for this task through a two-stage reinforcement learning pipeline. We first construct a dataset of over 500 real-world, multi-step walkthroughs, which Pentest-R1 leverages for offline reinforcement learning (RL) to instill foundational attack logic. Subsequently, the LLM is fine-tuned via online RL in an interactive Capture The Flag (CTF) environment, where it learns directly from environmental feedback to develop robust error self-correction and adaptive strategies. Our extensive experiments on the Cybench and AutoPenBench benchmarks demonstrate the framework's effectiveness. On AutoPenBench, Pentest-R1 achieves a 24.2\% success rate, surpassing most state-of-the-art models and ranking second only to Gemini 2.5 Flash. On Cybench, it attains a 15.0\% success rate in unguided tasks, establishing a new state-of-the-art for open-source LLMs and matching the performance of top proprietary models. Ablation studies confirm that the synergy of both training stages is critical to its success.



## **12. Agentic Moderation: Multi-Agent Design for Safer Vision-Language Models**

cs.AI

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2510.25179v1) [paper-pdf](http://arxiv.org/pdf/2510.25179v1)

**Authors**: Juan Ren, Mark Dras, Usman Naseem

**Abstract**: Agentic methods have emerged as a powerful and autonomous paradigm that enhances reasoning, collaboration, and adaptive control, enabling systems to coordinate and independently solve complex tasks. We extend this paradigm to safety alignment by introducing Agentic Moderation, a model-agnostic framework that leverages specialised agents to defend multimodal systems against jailbreak attacks. Unlike prior approaches that apply as a static layer over inputs or outputs and provide only binary classifications (safe or unsafe), our method integrates dynamic, cooperative agents, including Shield, Responder, Evaluator, and Reflector, to achieve context-aware and interpretable moderation. Extensive experiments across five datasets and four representative Large Vision-Language Models (LVLMs) demonstrate that our approach reduces the Attack Success Rate (ASR) by 7-19%, maintains a stable Non-Following Rate (NF), and improves the Refusal Rate (RR) by 4-20%, achieving robust, interpretable, and well-balanced safety performance. By harnessing the flexibility and reasoning capacity of agentic architectures, Agentic Moderation provides modular, scalable, and fine-grained safety enforcement, highlighting the broader potential of agentic systems as a foundation for automated safety governance.



## **13. OpenGuardrails: A Configurable, Unified, and Scalable Guardrails Platform for Large Language Models**

cs.CR

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2510.19169v2) [paper-pdf](http://arxiv.org/pdf/2510.19169v2)

**Authors**: Thomas Wang, Haowen Li

**Abstract**: As large language models (LLMs) are increasingly integrated into real-world applications, ensuring their safety, robustness, and privacy compliance has become critical. We present OpenGuardrails, the first fully open-source platform that unifies large-model-based safety detection, manipulation defense, and deployable guardrail infrastructure. OpenGuardrails protects against three major classes of risks: (1) content-safety violations such as harmful or explicit text generation, (2) model-manipulation attacks including prompt injection, jailbreaks, and code-interpreter abuse, and (3) data leakage involving sensitive or private information. Unlike prior modular or rule-based frameworks, OpenGuardrails introduces three core innovations: (1) a Configurable Policy Adaptation mechanism that allows per-request customization of unsafe categories and sensitivity thresholds; (2) a Unified LLM-based Guard Architecture that performs both content-safety and manipulation detection within a single model; and (3) a Quantized, Scalable Model Design that compresses a 14B dense base model to 3.3B via GPTQ while preserving over 98 of benchmark accuracy. The system supports 119 languages, achieves state-of-the-art performance across multilingual safety benchmarks, and can be deployed as a secure gateway or API-based service for enterprise use. All models, datasets, and deployment scripts are released under the Apache 2.0 license.



## **14. Secure Retrieval-Augmented Generation against Poisoning Attacks**

cs.CR

To appear in IEEE BigData 2025

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2510.25025v1) [paper-pdf](http://arxiv.org/pdf/2510.25025v1)

**Authors**: Zirui Cheng, Jikai Sun, Anjun Gao, Yueyang Quan, Zhuqing Liu, Xiaohua Hu, Minghong Fang

**Abstract**: Large language models (LLMs) have transformed natural language processing (NLP), enabling applications from content generation to decision support. Retrieval-Augmented Generation (RAG) improves LLMs by incorporating external knowledge but also introduces security risks, particularly from data poisoning, where the attacker injects poisoned texts into the knowledge database to manipulate system outputs. While various defenses have been proposed, they often struggle against advanced attacks. To address this, we introduce RAGuard, a detection framework designed to identify poisoned texts. RAGuard first expands the retrieval scope to increase the proportion of clean texts, reducing the likelihood of retrieving poisoned content. It then applies chunk-wise perplexity filtering to detect abnormal variations and text similarity filtering to flag highly similar texts. This non-parametric approach enhances RAG security, and experiments on large-scale datasets demonstrate its effectiveness in detecting and mitigating poisoning attacks, including strong adaptive attacks.



## **15. S3C2 Summit 2025-03: Industry Secure Supply Chain Summit**

cs.CR

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2510.24920v1) [paper-pdf](http://arxiv.org/pdf/2510.24920v1)

**Authors**: Elizabeth Lin, Jonah Ghebremichael, William Enck, Yasemin Acar, Michel Cukier, Alexandros Kapravelos, Christian Kastner, Laurie Williams

**Abstract**: Software supply chains, while providing immense economic and software development value, are only as strong as their weakest link. Over the past several years, there has been an exponential increase in cyberattacks specifically targeting vulnerable links in critical software supply chains. These attacks disrupt the day-to-day functioning and threaten the security of nearly everyone on the internet, from billion-dollar companies and government agencies to hobbyist open-source developers. The ever-evolving threat of software supply chain attacks has garnered interest from both the software industry and US government in improving software supply chain security. On Thursday, March 6th, 2025, four researchers from the NSF-backed Secure Software Supply Chain Center (S3C2) conducted a Secure Software Supply Chain Summit with a diverse set of 18 practitioners from 17 organizations. The goals of the Summit were: (1) to enable sharing between participants from different industries regarding practical experiences and challenges with software supply chain security; (2) to help form new collaborations; and (3) to learn about the challenges facing participants to inform our future research directions. The summit consisted of discussions of six topics relevant to the government agencies represented, including software bill of materials (SBOMs); compliance; malicious commits; build infrastructure; culture; and large language models (LLMs) and security. For each topic of discussion, we presented a list of questions to participants to spark conversation. In this report, we provide a summary of the summit. The open questions and challenges that remained after each topic are listed at the end of each topic's section, and the initial discussion questions for each topic are provided in the appendix.



## **16. Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs**

cs.CV

Accepted by NeurIPS 2025 Dataset and Benchmark Track, Project page:  https://liuxuannan.github.io/Video-SafetyBench.github.io/

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2505.11842v3) [paper-pdf](http://arxiv.org/pdf/2505.11842v3)

**Authors**: Xuannan Liu, Zekun Li, Zheqi He, Peipei Li, Shuhan Xia, Xing Cui, Huaibo Huang, Xi Yang, Ran He

**Abstract**: The increasing deployment of Large Vision-Language Models (LVLMs) raises safety concerns under potential malicious inputs. However, existing multimodal safety evaluations primarily focus on model vulnerabilities exposed by static image inputs, ignoring the temporal dynamics of video that may induce distinct safety risks. To bridge this gap, we introduce Video-SafetyBench, the first comprehensive benchmark designed to evaluate the safety of LVLMs under video-text attacks. It comprises 2,264 video-text pairs spanning 48 fine-grained unsafe categories, each pairing a synthesized video with either a harmful query, which contains explicit malice, or a benign query, which appears harmless but triggers harmful behavior when interpreted alongside the video. To generate semantically accurate videos for safety evaluation, we design a controllable pipeline that decomposes video semantics into subject images (what is shown) and motion text (how it moves), which jointly guide the synthesis of query-relevant videos. To effectively evaluate uncertain or borderline harmful outputs, we propose RJScore, a novel LLM-based metric that incorporates the confidence of judge models and human-aligned decision threshold calibration. Extensive experiments show that benign-query video composition achieves average attack success rates of 67.2%, revealing consistent vulnerabilities to video-induced attacks. We believe Video-SafetyBench will catalyze future research into video-based safety evaluation and defense strategies.



## **17. MixAT: Combining Continuous and Discrete Adversarial Training for LLMs**

cs.LG

Published at 39th Conference on Neural Information Processing Systems  (NeurIPS 2025)

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2505.16947v2) [paper-pdf](http://arxiv.org/pdf/2505.16947v2)

**Authors**: Csaba Dékány, Stefan Balauca, Robin Staab, Dimitar I. Dimitrov, Martin Vechev

**Abstract**: Despite recent efforts in Large Language Model (LLM) safety and alignment, current adversarial attacks on frontier LLMs can still consistently force harmful generations. Although adversarial training has been widely studied and shown to significantly improve the robustness of traditional machine learning models, its strengths and weaknesses in the context of LLMs are less understood. Specifically, while existing discrete adversarial attacks are effective at producing harmful content, training LLMs with concrete adversarial prompts is often computationally expensive, leading to reliance on continuous relaxations. At the same time, despite their effectiveness and generalization capabilities, training with continuous perturbations does not always capture the full spectrum of vulnerabilities exploited by discrete attacks. In this work, we aim to bridge this gap by introducing MixAT, a novel method that combines stronger discrete and faster continuous attacks during training. We rigorously evaluate MixAT across a wide spectrum of state-of-the-art attacks, proposing the At Least One Attack Success Rate (ALO-ASR) metric to capture the worst-case vulnerability of models. We show MixAT achieves substantially better robustness (ALO-ASR < 20%) compared to prior defenses (ALO-ASR > 50%), while maintaining a runtime comparable to methods based on continuous relaxations. We further analyze MixAT in realistic deployment settings, exploring how chat templates, quantization, low-rank adapters, and temperature affect both adversarial training and evaluation, revealing additional blind spots in current methodologies. Our results demonstrate that MixAT's discrete-continuous defense offers a principled and superior robustness-accuracy tradeoff with minimal computational overhead, highlighting its promise for building safer LLMs. We provide our code and models at https://github.com/insait-institute/MixAT.



## **18. Untargeted Jailbreak Attack**

cs.CR

**SubmitDate**: 2025-10-28    [abs](http://arxiv.org/abs/2510.02999v2) [paper-pdf](http://arxiv.org/pdf/2510.02999v2)

**Authors**: Xinzhe Huang, Wenjing Hu, Tianhang Zheng, Kedong Xiu, Xiaojun Jia, Di Wang, Zhan Qin, Kui Ren

**Abstract**: Existing gradient-based jailbreak attacks on Large Language Models (LLMs), such as Greedy Coordinate Gradient (GCG) and COLD-Attack, typically optimize adversarial suffixes to align the LLM output with a predefined target response. However, by restricting the optimization objective as inducing a predefined target, these methods inherently constrain the adversarial search space, which limit their overall attack efficacy. Furthermore, existing methods typically require a large number of optimization iterations to fulfill the large gap between the fixed target and the original model response, resulting in low attack efficiency.   To overcome the limitations of targeted jailbreak attacks, we propose the first gradient-based untargeted jailbreak attack (UJA), aiming to elicit an unsafe response without enforcing any predefined patterns. Specifically, we formulate an untargeted attack objective to maximize the unsafety probability of the LLM response, which can be quantified using a judge model. Since the objective is non-differentiable, we further decompose it into two differentiable sub-objectives for optimizing an optimal harmful response and the corresponding adversarial prompt, with a theoretical analysis to validate the decomposition. In contrast to targeted jailbreak attacks, UJA's unrestricted objective significantly expands the search space, enabling a more flexible and efficient exploration of LLM vulnerabilities.Extensive evaluations demonstrate that UJA can achieve over 80% attack success rates against recent safety-aligned LLMs with only 100 optimization iterations, outperforming the state-of-the-art gradient-based attacks such as I-GCG and COLD-Attack by over 20%.



## **19. Attention! Your Vision Language Model Could Be Maliciously Manipulated**

cs.CV

NeurIPS 2025

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2505.19911v2) [paper-pdf](http://arxiv.org/pdf/2505.19911v2)

**Authors**: Xiaosen Wang, Shaokang Wang, Zhijin Ge, Yuyang Luo, Shudong Zhang

**Abstract**: Large Vision-Language Models (VLMs) have achieved remarkable success in understanding complex real-world scenarios and supporting data-driven decision-making processes. However, VLMs exhibit significant vulnerability against adversarial examples, either text or image, which can lead to various adversarial outcomes, e.g., jailbreaking, hijacking, and hallucination, etc. In this work, we empirically and theoretically demonstrate that VLMs are particularly susceptible to image-based adversarial examples, where imperceptible perturbations can precisely manipulate each output token. To this end, we propose a novel attack called Vision-language model Manipulation Attack (VMA), which integrates first-order and second-order momentum optimization techniques with a differentiable transformation mechanism to effectively optimize the adversarial perturbation. Notably, VMA can be a double-edged sword: it can be leveraged to implement various attacks, such as jailbreaking, hijacking, privacy breaches, Denial-of-Service, and the generation of sponge examples, etc, while simultaneously enabling the injection of watermarks for copyright protection. Extensive empirical evaluations substantiate the efficacy and generalizability of VMA across diverse scenarios and datasets. Code is available at https://github.com/Trustworthy-AI-Group/VMA.



## **20. Fast-MIA: Efficient and Scalable Membership Inference for LLMs**

cs.CR

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2510.23074v1) [paper-pdf](http://arxiv.org/pdf/2510.23074v1)

**Authors**: Hiromu Takahashi, Shotaro Ishihara

**Abstract**: We propose Fast-MIA (https://github.com/Nikkei/fast-mia), a Python library for efficiently evaluating membership inference attacks (MIA) against Large Language Models (LLMs). MIA against LLMs has emerged as a crucial challenge due to growing concerns over copyright, security, and data privacy, and has attracted increasing research attention. However, the progress of this research is significantly hindered by two main obstacles: (1) the high computational cost of inference in LLMs, and (2) the lack of standardized and maintained implementations of MIA methods, which makes large-scale empirical comparison difficult. To address these challenges, our library provides fast batch inference and includes implementations of representative MIA methods under a unified evaluation framework. This library supports easy implementation of reproducible benchmarks with simple configuration and extensibility. We release Fast-MIA as an open-source (Apache License 2.0) tool to support scalable and transparent research on LLMs.



## **21. MCPGuard : Automatically Detecting Vulnerabilities in MCP Servers**

cs.CR

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2510.23673v1) [paper-pdf](http://arxiv.org/pdf/2510.23673v1)

**Authors**: Bin Wang, Zexin Liu, Hao Yu, Ao Yang, Yenan Huang, Jing Guo, Huangsheng Cheng, Hui Li, Huiyu Wu

**Abstract**: The Model Context Protocol (MCP) has emerged as a standardized interface enabling seamless integration between Large Language Models (LLMs) and external data sources and tools. While MCP significantly reduces development complexity and enhances agent capabilities, its openness and extensibility introduce critical security vulnerabilities that threaten system trustworthiness and user data protection. This paper systematically analyzes the security landscape of MCP-based systems, identifying three principal threat categories: (1) agent hijacking attacks stemming from protocol design deficiencies; (2) traditional web vulnerabilities in MCP servers; and (3) supply chain security. To address these challenges, we comprehensively survey existing defense strategies, examining both proactive server-side scanning approaches, ranging from layered detection pipelines and agentic auditing frameworks to zero-trust registry systems, and runtime interaction monitoring solutions that provide continuous oversight and policy enforcement. Our analysis reveals that MCP security fundamentally represents a paradigm shift where the attack surface extends from traditional code execution to semantic interpretation of natural language metadata, necessitating novel defense mechanisms tailored to this unique threat model.



## **22. Is Your Prompt Poisoning Code? Defect Induction Rates and Security Mitigation Strategies**

cs.CR

**SubmitDate**: 2025-10-27    [abs](http://arxiv.org/abs/2510.22944v1) [paper-pdf](http://arxiv.org/pdf/2510.22944v1)

**Authors**: Bin Wang, YiLu Zhong, MiDi Wan, WenJie Yu, YuanBing Ouyang, Yenan Huang, Hui Li

**Abstract**: Large language models (LLMs) have become indispensable for automated code generation, yet the quality and security of their outputs remain a critical concern. Existing studies predominantly concentrate on adversarial attacks or inherent flaws within the models. However, a more prevalent yet underexplored issue concerns how the quality of a benign but poorly formulated prompt affects the security of the generated code. To investigate this, we first propose an evaluation framework for prompt quality encompassing three key dimensions: goal clarity, information completeness, and logical consistency. Based on this framework, we construct and publicly release CWE-BENCH-PYTHON, a large-scale benchmark dataset containing tasks with prompts categorized into four distinct levels of normativity (L0-L3). Extensive experiments on multiple state-of-the-art LLMs reveal a clear correlation: as prompt normativity decreases, the likelihood of generating insecure code consistently and markedly increases. Furthermore, we demonstrate that advanced prompting techniques, such as Chain-of-Thought and Self-Correction, effectively mitigate the security risks introduced by low-quality prompts, substantially improving code safety. Our findings highlight that enhancing the quality of user prompts constitutes a critical and effective strategy for strengthening the security of AI-generated code.



## **23. Sentra-Guard: A Multilingual Human-AI Framework for Real-Time Defense Against Adversarial LLM Jailbreaks**

cs.CR

11 pages, 5 figures. Preprint version under review in the area of  Artificial Intelligence (cs.AI)

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2510.22628v1) [paper-pdf](http://arxiv.org/pdf/2510.22628v1)

**Authors**: Md. Mehedi Hasan, Ziaur Rahman, Rafid Mostafiz, Md. Abir Hossain

**Abstract**: This paper presents a real-time modular defense system named Sentra-Guard. The system detects and mitigates jailbreak and prompt injection attacks targeting large language models (LLMs). The framework uses a hybrid architecture with FAISS-indexed SBERT embedding representations that capture the semantic meaning of prompts, combined with fine-tuned transformer classifiers, which are machine learning models specialized for distinguishing between benign and adversarial language inputs. It identifies adversarial prompts in both direct and obfuscated attack vectors. A core innovation is the classifier-retriever fusion module, which dynamically computes context-aware risk scores that estimate how likely a prompt is to be adversarial based on its content and context. The framework ensures multilingual resilience with a language-agnostic preprocessing layer. This component automatically translates non-English prompts into English for semantic evaluation, enabling consistent detection across over 100 languages. The system includes a HITL feedback loop, where decisions made by the automated system are reviewed by human experts for continual learning and rapid adaptation under adversarial pressure. Sentra-Guard maintains an evolving dual-labeled knowledge base of benign and malicious prompts, enhancing detection reliability and reducing false positives. Evaluation results show a 99.96% detection rate (AUC = 1.00, F1 = 1.00) and an attack success rate (ASR) of only 0.004%. This outperforms leading baselines such as LlamaGuard-2 (1.3%) and OpenAI Moderation (3.7%). Unlike black-box approaches, Sentra-Guard is transparent, fine-tunable, and compatible with diverse LLM backends. Its modular design supports scalable deployment in both commercial and open-source environments. The system establishes a new state-of-the-art in adversarial LLM defense.



## **24. Breaking Agent Backbones: Evaluating the Security of Backbone LLMs in AI Agents**

cs.CR

Julia Bazinska and Max Mathys contributed equally

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2510.22620v1) [paper-pdf](http://arxiv.org/pdf/2510.22620v1)

**Authors**: Julia Bazinska, Max Mathys, Francesco Casucci, Mateo Rojas-Carulla, Xander Davies, Alexandra Souly, Niklas Pfister

**Abstract**: AI agents powered by large language models (LLMs) are being deployed at scale, yet we lack a systematic understanding of how the choice of backbone LLM affects agent security. The non-deterministic sequential nature of AI agents complicates security modeling, while the integration of traditional software with AI components entangles novel LLM vulnerabilities with conventional security risks. Existing frameworks only partially address these challenges as they either capture specific vulnerabilities only or require modeling of complete agents. To address these limitations, we introduce threat snapshots: a framework that isolates specific states in an agent's execution flow where LLM vulnerabilities manifest, enabling the systematic identification and categorization of security risks that propagate from the LLM to the agent level. We apply this framework to construct the $\operatorname{b}^3$ benchmark, a security benchmark based on 194331 unique crowdsourced adversarial attacks. We then evaluate 31 popular LLMs with it, revealing, among other insights, that enhanced reasoning capabilities improve security, while model size does not correlate with security. We release our benchmark, dataset, and evaluation code to facilitate widespread adoption by LLM providers and practitioners, offering guidance for agent developers and incentivizing model developers to prioritize backbone security improvements.



## **25. OFFSIDE: Benchmarking Unlearning Misinformation in Multimodal Large Language Models**

cs.AI

**SubmitDate**: 2025-10-26    [abs](http://arxiv.org/abs/2510.22535v1) [paper-pdf](http://arxiv.org/pdf/2510.22535v1)

**Authors**: Hao Zheng, Zirui Pang, Ling li, Zhijie Deng, Yuhan Pu, Zhaowei Zhu, Xiaobo Xia, Jiaheng Wei

**Abstract**: Advances in Multimodal Large Language Models (MLLMs) intensify concerns about data privacy, making Machine Unlearning (MU), the selective removal of learned information, a critical necessity. However, existing MU benchmarks for MLLMs are limited by a lack of image diversity, potential inaccuracies, and insufficient evaluation scenarios, which fail to capture the complexity of real-world applications. To facilitate the development of MLLMs unlearning and alleviate the aforementioned limitations, we introduce OFFSIDE, a novel benchmark for evaluating misinformation unlearning in MLLMs based on football transfer rumors. This manually curated dataset contains 15.68K records for 80 players, providing a comprehensive framework with four test sets to assess forgetting efficacy, generalization, utility, and robustness. OFFSIDE supports advanced settings like selective unlearning and corrective relearning, and crucially, unimodal unlearning (forgetting only text data). Our extensive evaluation of multiple baselines reveals key findings: (1) Unimodal methods (erasing text-based knowledge) fail on multimodal rumors; (2) Unlearning efficacy is largely driven by catastrophic forgetting; (3) All methods struggle with "visual rumors" (rumors appear in the image); (4) The unlearned rumors can be easily recovered and (5) All methods are vulnerable to prompt attacks. These results expose significant vulnerabilities in current approaches, highlighting the need for more robust multimodal unlearning solutions. The code is available at \href{https://github.com/zh121800/OFFSIDE}{https://github.com/zh121800/OFFSIDE}.



## **26. A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1**

cs.CV

NeurIPS 2025. Code at: https://github.com/VILA-Lab/M-Attack

**SubmitDate**: 2025-10-25    [abs](http://arxiv.org/abs/2503.10635v2) [paper-pdf](http://arxiv.org/pdf/2503.10635v2)

**Authors**: Zhaoyi Li, Xiaohan Zhao, Dong-Dong Wu, Jiacheng Cui, Zhiqiang Shen

**Abstract**: Despite promising performance on open-source large vision-language models (LVLMs), transfer-based targeted attacks often fail against closed-source commercial LVLMs. Analyzing failed adversarial perturbations reveals that the learned perturbations typically originate from a uniform distribution and lack clear semantic details, resulting in unintended responses. This critical absence of semantic information leads commercial black-box LVLMs to either ignore the perturbation entirely or misinterpret its embedded semantics, thereby causing the attack to fail. To overcome these issues, we propose to refine semantic clarity by encoding explicit semantic details within local regions, thus ensuring the capture of finer-grained features and inter-model transferability, and by concentrating modifications on semantically rich areas rather than applying them uniformly. To achieve this, we propose a simple yet highly effective baseline: at each optimization step, the adversarial image is cropped randomly by a controlled aspect ratio and scale, resized, and then aligned with the target image in the embedding space. While the naive source-target matching method has been utilized before in the literature, we are the first to provide a tight analysis, which establishes a close connection between perturbation optimization and semantics. Experimental results confirm our hypothesis. Our adversarial examples crafted with local-aggregated perturbations focused on crucial regions exhibit surprisingly good transferability to commercial LVLMs, including GPT-4.5, GPT-4o, Gemini-2.0-flash, Claude-3.5/3.7-sonnet, and even reasoning models like o1, Claude-3.7-thinking and Gemini-2.0-flash-thinking. Our approach achieves success rates exceeding 90% on GPT-4.5, 4o, and o1, significantly outperforming all prior state-of-the-art attack methods with lower $\ell_1/\ell_2$ perturbations.



## **27. Jailbreak Mimicry: Automated Discovery of Narrative-Based Jailbreaks for Large Language Models**

cs.CR

18 pages, 5 figures

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.22085v1) [paper-pdf](http://arxiv.org/pdf/2510.22085v1)

**Authors**: Pavlos Ntais

**Abstract**: Large language models (LLMs) remain vulnerable to sophisticated prompt engineering attacks that exploit contextual framing to bypass safety mechanisms, posing significant risks in cybersecurity applications. We introduce Jailbreak Mimicry, a systematic methodology for training compact attacker models to automatically generate narrative-based jailbreak prompts in a one-shot manner. Our approach transforms adversarial prompt discovery from manual craftsmanship into a reproducible scientific process, enabling proactive vulnerability assessment in AI-driven security systems. Developed for the OpenAI GPT-OSS-20B Red-Teaming Challenge, we use parameter-efficient fine-tuning (LoRA) on Mistral-7B with a curated dataset derived from AdvBench, achieving an 81.0% Attack Success Rate (ASR) against GPT-OSS-20B on a held-out test set of 200 items. Cross-model evaluation reveals significant variation in vulnerability patterns: our attacks achieve 66.5% ASR against GPT-4, 79.5% on Llama-3 and 33.0% against Gemini 2.5 Flash, demonstrating both broad applicability and model-specific defensive strengths in cybersecurity contexts. This represents a 54x improvement over direct prompting (1.5% ASR) and demonstrates systematic vulnerabilities in current safety alignment approaches. Our analysis reveals that technical domains (Cybersecurity: 93% ASR) and deception-based attacks (Fraud: 87.8% ASR) are particularly vulnerable, highlighting threats to AI-integrated threat detection, malware analysis, and secure systems, while physical harm categories show greater resistance (55.6% ASR). We employ automated harmfulness evaluation using Claude Sonnet 4, cross-validated with human expert assessment, ensuring reliable and scalable evaluation for cybersecurity red-teaming. Finally, we analyze failure mechanisms and discuss defensive strategies to mitigate these vulnerabilities in AI for cybersecurity.



## **28. Toward Understanding the Transferability of Adversarial Suffixes in Large Language Models**

cs.CL

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.22014v1) [paper-pdf](http://arxiv.org/pdf/2510.22014v1)

**Authors**: Sarah Ball, Niki Hasrati, Alexander Robey, Avi Schwarzschild, Frauke Kreuter, Zico Kolter, Andrej Risteski

**Abstract**: Discrete optimization-based jailbreaking attacks on large language models aim to generate short, nonsensical suffixes that, when appended onto input prompts, elicit disallowed content. Notably, these suffixes are often transferable -- succeeding on prompts and models for which they were never optimized. And yet, despite the fact that transferability is surprising and empirically well-established, the field lacks a rigorous analysis of when and why transfer occurs. To fill this gap, we identify three statistical properties that strongly correlate with transfer success across numerous experimental settings: (1) how much a prompt without a suffix activates a model's internal refusal direction, (2) how strongly a suffix induces a push away from this direction, and (3) how large these shifts are in directions orthogonal to refusal. On the other hand, we find that prompt semantic similarity only weakly correlates with transfer success. These findings lead to a more fine-grained understanding of transferability, which we use in interventional experiments to showcase how our statistical analysis can translate into practical improvements in attack success.



## **29. Memory Injection Attacks on LLM Agents via Query-Only Interaction**

cs.LG

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2503.03704v3) [paper-pdf](http://arxiv.org/pdf/2503.03704v3)

**Authors**: Shen Dong, Shaochen Xu, Pengfei He, Yige Li, Jiliang Tang, Tianming Liu, Hui Liu, Zhen Xiang

**Abstract**: Agents powered by large language models (LLMs) have demonstrated strong capabilities in a wide range of complex, real-world applications. However, LLM agents with a compromised memory bank may easily produce harmful outputs when the past records retrieved for demonstration are malicious. In this paper, we propose a novel Memory INJection Attack, MINJA, without assuming that the attacker can directly modify the memory bank of the agent. The attacker injects malicious records into the memory bank by only interacting with the agent via queries and output observations. These malicious records are designed to elicit a sequence of malicious reasoning steps corresponding to a different target query during the agent's execution of the victim user's query. Specifically, we introduce a sequence of bridging steps to link victim queries to the malicious reasoning steps. During the memory injection, we propose an indication prompt that guides the agent to autonomously generate similar bridging steps, with a progressive shortening strategy that gradually removes the indication prompt, such that the malicious record will be easily retrieved when processing later victim queries. Our extensive experiments across diverse agents demonstrate the effectiveness of MINJA in compromising agent memory. With minimal requirements for execution, MINJA enables any user to influence agent memory, highlighting the risk.



## **30. Uncovering the Persuasive Fingerprint of LLMs in Jailbreaking Attacks**

cs.CL

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21983v1) [paper-pdf](http://arxiv.org/pdf/2510.21983v1)

**Authors**: Havva Alizadeh Noughabi, Julien Serbanescu, Fattane Zarrinkalam, Ali Dehghantanha

**Abstract**: Despite recent advances, Large Language Models remain vulnerable to jailbreak attacks that bypass alignment safeguards and elicit harmful outputs. While prior research has proposed various attack strategies differing in human readability and transferability, little attention has been paid to the linguistic and psychological mechanisms that may influence a model's susceptibility to such attacks. In this paper, we examine an interdisciplinary line of research that leverages foundational theories of persuasion from the social sciences to craft adversarial prompts capable of circumventing alignment constraints in LLMs. Drawing on well-established persuasive strategies, we hypothesize that LLMs, having been trained on large-scale human-generated text, may respond more compliantly to prompts with persuasive structures. Furthermore, we investigate whether LLMs themselves exhibit distinct persuasive fingerprints that emerge in their jailbreak responses. Empirical evaluations across multiple aligned LLMs reveal that persuasion-aware prompts significantly bypass safeguards, demonstrating their potential to induce jailbreak behaviors. This work underscores the importance of cross-disciplinary insight in addressing the evolving challenges of LLM safety. The code and data are available.



## **31. $δ$-STEAL: LLM Stealing Attack with Local Differential Privacy**

cs.CR

Accepted at ACML 2025 (PMLR W&CP). Code:  https://github.com/kirudang/LDP_Stealing_Attack

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21946v1) [paper-pdf](http://arxiv.org/pdf/2510.21946v1)

**Authors**: Kieu Dang, Phung Lai, NhatHai Phan, Yelong Shen, Ruoming Jin, Abdallah Khreishah

**Abstract**: Large language models (LLMs) demonstrate remarkable capabilities across various tasks. However, their deployment introduces significant risks related to intellectual property. In this context, we focus on model stealing attacks, where adversaries replicate the behaviors of these models to steal services. These attacks are highly relevant to proprietary LLMs and pose serious threats to revenue and financial stability. To mitigate these risks, the watermarking solution embeds imperceptible patterns in LLM outputs, enabling model traceability and intellectual property verification. In this paper, we study the vulnerability of LLM service providers by introducing $\delta$-STEAL, a novel model stealing attack that bypasses the service provider's watermark detectors while preserving the adversary's model utility. $\delta$-STEAL injects noise into the token embeddings of the adversary's model during fine-tuning in a way that satisfies local differential privacy (LDP) guarantees. The adversary queries the service provider's model to collect outputs and form input-output training pairs. By applying LDP-preserving noise to these pairs, $\delta$-STEAL obfuscates watermark signals, making it difficult for the service provider to determine whether its outputs were used, thereby preventing claims of model theft. Our experiments show that $\delta$-STEAL with lightweight modifications achieves attack success rates of up to $96.95\%$ without significantly compromising the adversary's model utility. The noise scale in LDP controls the trade-off between attack effectiveness and model utility. This poses a significant risk, as even robust watermarks can be bypassed, allowing adversaries to deceive watermark detectors and undermine current intellectual property protection methods.



## **32. Adversarial Déjà Vu: Jailbreak Dictionary Learning for Stronger Generalization to Unseen Attacks**

cs.LG

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21910v1) [paper-pdf](http://arxiv.org/pdf/2510.21910v1)

**Authors**: Mahavir Dabas, Tran Huynh, Nikhil Reddy Billa, Jiachen T. Wang, Peng Gao, Charith Peris, Yao Ma, Rahul Gupta, Ming Jin, Prateek Mittal, Ruoxi Jia

**Abstract**: Large language models remain vulnerable to jailbreak attacks that bypass safety guardrails to elicit harmful outputs. Defending against novel jailbreaks represents a critical challenge in AI safety. Adversarial training -- designed to make models robust against worst-case perturbations -- has been the dominant paradigm for adversarial robustness. However, due to optimization challenges and difficulties in defining realistic threat models, adversarial training methods often fail on newly developed jailbreaks in practice. This paper proposes a new paradigm for improving robustness against unseen jailbreaks, centered on the Adversarial D\'ej\`a Vu hypothesis: novel jailbreaks are not fundamentally new, but largely recombinations of adversarial skills from previous attacks. We study this hypothesis through a large-scale analysis of 32 attack papers published over two years. Using an automated pipeline, we extract and compress adversarial skills into a sparse dictionary of primitives, with LLMs generating human-readable descriptions. Our analysis reveals that unseen attacks can be effectively explained as sparse compositions of earlier skills, with explanatory power increasing monotonically as skill coverage grows. Guided by this insight, we introduce Adversarial Skill Compositional Training (ASCoT), which trains on diverse compositions of skill primitives rather than isolated attack instances. ASCoT substantially improves robustness to unseen attacks, including multi-turn jailbreaks, while maintaining low over-refusal rates. We also demonstrate that expanding adversarial skill coverage, not just data scale, is key to defending against novel attacks. \textcolor{red}{\textbf{Warning: This paper contains content that may be harmful or offensive in nature.



## **33. Detecting Various DeFi Price Manipulations with LLM Reasoning**

cs.CR

Accepted by ASE 2025. Please cite the conference version of this  paper, e.g., "Juantao Zhong, Daoyuan Wu, Ye Liu, Maoyi Xie, Yang Liu, Yi Li,  Ning Liu. Detecting Various DeFi Price Manipulations with LLM Reasoning. In  40th IEEE/ACM International Conference on Automated Software Engineering (ASE  2025)"

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2502.11521v2) [paper-pdf](http://arxiv.org/pdf/2502.11521v2)

**Authors**: Juantao Zhong, Daoyuan Wu, Ye Liu, Maoyi Xie, Yang Liu, Yi Li, Ning Liu

**Abstract**: DeFi (Decentralized Finance) is one of the most important applications of today's cryptocurrencies and smart contracts. It manages hundreds of billions in Total Value Locked (TVL) on-chain, yet it remains susceptible to common DeFi price manipulation attacks. Despite state-of-the-art (SOTA) systems like DeFiRanger and DeFort, we found that they are less effective to non-standard price models in custom DeFi protocols, which account for 44.2% of the 95 DeFi price manipulation attacks reported over the past three years. In this paper, we introduce the first LLM-based approach, DeFiScope, for detecting DeFi price manipulation attacks in both standard and custom price models. Our insight is that large language models (LLMs) have certain intelligence to abstract price calculation from smart contract source code and infer the trend of token price changes based on the extracted price models. To further strengthen LLMs in this aspect, we leverage Foundry to synthesize on-chain data and use it to fine-tune a DeFi price-specific LLM. Together with the high-level DeFi operations recovered from low-level transaction data, DeFiScope detects various DeFi price manipulations according to systematically mined patterns. Experimental results show that DeFiScope achieves a high recall of 80% on real-world attacks, a precision of 96% on suspicious transactions, and zero false alarms on benign transactions, significantly outperforming SOTA approaches. Moreover, we evaluate DeFiScope's cost-effectiveness and demonstrate its practicality by helping our industry partner confirm 147 real-world price manipulation attacks, including discovering 81 previously unknown historical incidents.



## **34. SBASH: a Framework for Designing and Evaluating RAG vs. Prompt-Tuned LLM Honeypots**

cs.CR

to be published in: The 3rd International Conference on Foundation  and Large Language Models (FLLM2025), IEEE, 2025

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21459v1) [paper-pdf](http://arxiv.org/pdf/2510.21459v1)

**Authors**: Adetayo Adebimpe, Helmut Neukirchen, Thomas Welsh

**Abstract**: Honeypots are decoy systems used for gathering valuable threat intelligence or diverting attackers away from production systems. Maximising attacker engagement is essential to their utility. However research has highlighted that context-awareness, such as the ability to respond to new attack types, systems and attacker agents, is necessary to increase engagement. Large Language Models (LLMs) have been shown as one approach to increase context awareness but suffer from several challenges including accuracy and timeliness of response time, high operational costs and data-protection issues due to cloud deployment. We propose the System-Based Attention Shell Honeypot (SBASH) framework which manages data-protection issues through the use of lightweight local LLMs. We investigate the use of Retrieval Augmented Generation (RAG) supported LLMs and non-RAG LLMs for Linux shell commands and evaluate them using several different metrics such as response time differences, realism from human testers, and similarity to a real system calculated with Levenshtein distance, SBert, and BertScore. We show that RAG improves accuracy for untuned models while models that have been tuned via a system prompt that tells the LLM to respond like a Linux system achieve without RAG a similar accuracy as untuned with RAG, while having a slightly lower latency.



## **35. FLAMES: Fine-tuning LLMs to Synthesize Invariants for Smart Contract Security**

cs.CR

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21401v1) [paper-pdf](http://arxiv.org/pdf/2510.21401v1)

**Authors**: Mojtaba Eshghie, Gabriele Morello, Matteo Lauretano, Alexandre Bartel, Martin Monperrus

**Abstract**: Smart contract vulnerabilities cost billions of dollars annually, yet existing automated analysis tools fail to generate deployable defenses. We present FLAMES, a novel automated approach that synthesizes executable runtime guards as Solidity "require" statements to harden smart contracts against exploits. Unlike prior work that relies on vulnerability labels, symbolic analysis, or natural language specifications, FLAMES employs domain-adapted large language models trained through fill-in-the-middle supervised fine-tuning on real-world invariants extracted from 514,506 verified contracts. Our extensive evaluation across three dimensions demonstrates FLAMES's effectiveness: (1) Compilation: FLAMES achieves 96.7% compilability for synthesized invariant (2) Semantic Quality: on a curated test set of 5,000 challenging invariants, FLAMES produces exact or semantically equivalent matches to ground truth in 44.5% of cases; (3) Exploit Mitigation: FLAMES prevents 22 out of 108 real exploits (20.4%) while preserving contract functionality, and (4) FLAMES successfully blocks the real-world APEMAGA incident by synthesizing a pre-condition that mitigates the attack. FLAMES establishes that domain-adapted LLMs can automatically generate production-ready security defenses for smart contracts without requiring vulnerability detection, formal specifications, or human intervention. We release our code, model weights, datasets, and evaluation infrastructure to enable reproducible research in this critical domain.



## **36. Reverse Engineering Human Preferences with Reinforcement Learning**

cs.CL

NeurIPS 2025 (Spotlight)

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2505.15795v2) [paper-pdf](http://arxiv.org/pdf/2505.15795v2)

**Authors**: Lisa Alazraki, Tan Yi-Chern, Jon Ander Campos, Maximilian Mozes, Marek Rei, Max Bartolo

**Abstract**: The capabilities of Large Language Models (LLMs) are routinely evaluated by other LLMs trained to predict human preferences. This framework--known as LLM-as-a-judge--is highly scalable and relatively low cost. However, it is also vulnerable to malicious exploitation, as LLM responses can be tuned to overfit the preferences of the judge. Previous work shows that the answers generated by a candidate-LLM can be edited post hoc to maximise the score assigned to them by a judge-LLM. In this study, we adopt a different approach and use the signal provided by judge-LLMs as a reward to adversarially tune models that generate text preambles designed to boost downstream performance. We find that frozen LLMs pipelined with these models attain higher LLM-evaluation scores than existing frameworks. Crucially, unlike other frameworks which intervene directly on the model's response, our method is virtually undetectable. We also demonstrate that the effectiveness of the tuned preamble generator transfers when the candidate-LLM and the judge-LLM are replaced with models that are not used during training. These findings raise important questions about the design of more reliable LLM-as-a-judge evaluation settings. They also demonstrate that human preferences can be reverse engineered effectively, by pipelining LLMs to optimise upstream preambles via reinforcement learning--an approach that could find future applications in diverse tasks and domains beyond adversarial attacks.



## **37. LLM-Powered Detection of Price Manipulation in DeFi**

cs.CR

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21272v1) [paper-pdf](http://arxiv.org/pdf/2510.21272v1)

**Authors**: Lu Liu, Wuqi Zhang, Lili Wei, Hao Guan, Yongqiang Tian, Yepang Liu

**Abstract**: Decentralized Finance (DeFi) smart contracts manage billions of dollars, making them a prime target for exploits. Price manipulation vulnerabilities, often via flash loans, are a devastating class of attacks causing significant financial losses. Existing detection methods are limited. Reactive approaches analyze attacks only after they occur, while proactive static analysis tools rely on rigid, predefined heuristics, limiting adaptability. Both depend on known attack patterns, failing to identify novel variants or comprehend complex economic logic. We propose PMDetector, a hybrid framework combining static analysis with Large Language Model (LLM)-based reasoning to proactively detect price manipulation vulnerabilities. Our approach uses a formal attack model and a three-stage pipeline. First, static taint analysis identifies potentially vulnerable code paths. Second, a two-stage LLM process filters paths by analyzing defenses and then simulates attacks to evaluate exploitability. Finally, a static analysis checker validates LLM results, retaining only high-risk paths and generating comprehensive vulnerability reports. To evaluate its effectiveness, we built a dataset of 73 real-world vulnerable and 288 benign DeFi protocols. Results show PMDetector achieves 88% precision and 90% recall with Gemini 2.5-flash, significantly outperforming state-of-the-art static analysis and LLM-based approaches. Auditing a vulnerability with PMDetector costs just $0.03 and takes 4.0 seconds with GPT-4.1, offering an efficient and cost-effective alternative to manual audits.



## **38. Virus Infection Attack on LLMs: Your Poisoning Can Spread "VIA" Synthetic Data**

cs.CR

Camera Ready of NeurIPS 2025 Spotlight. Source code:  https://github.com/liangzid/VirusInfectionAttack

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2509.23041v2) [paper-pdf](http://arxiv.org/pdf/2509.23041v2)

**Authors**: Zi Liang, Qingqing Ye, Xuan Liu, Yanyun Wang, Jianliang Xu, Haibo Hu

**Abstract**: Synthetic data refers to artificial samples generated by models. While it has been validated to significantly enhance the performance of large language models (LLMs) during training and has been widely adopted in LLM development, potential security risks it may introduce remain uninvestigated. This paper systematically evaluates the resilience of synthetic-data-integrated training paradigm for LLMs against mainstream poisoning and backdoor attacks. We reveal that such a paradigm exhibits strong resistance to existing attacks, primarily thanks to the different distribution patterns between poisoning data and queries used to generate synthetic samples. To enhance the effectiveness of these attacks and further investigate the security risks introduced by synthetic data, we introduce a novel and universal attack framework, namely, Virus Infection Attack (VIA), which enables the propagation of current attacks through synthetic data even under purely clean queries. Inspired by the principles of virus design in cybersecurity, VIA conceals the poisoning payload within a protective "shell" and strategically searches for optimal hijacking points in benign samples to maximize the likelihood of generating malicious content. Extensive experiments on both data poisoning and backdoor attacks show that VIA significantly increases the presence of poisoning content in synthetic data and correspondingly raises the attack success rate (ASR) on downstream models to levels comparable to those observed in the poisoned upstream models.



## **39. Enhanced MLLM Black-Box Jailbreaking Attacks and Defenses**

cs.CR

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21214v1) [paper-pdf](http://arxiv.org/pdf/2510.21214v1)

**Authors**: Xingwei Zhong, Kar Wai Fok, Vrizlynn L. L. Thing

**Abstract**: Multimodal large language models (MLLMs) comprise of both visual and textual modalities to process vision language tasks. However, MLLMs are vulnerable to security-related issues, such as jailbreak attacks that alter the model's input to induce unauthorized or harmful responses. The incorporation of the additional visual modality introduces new dimensions to security threats. In this paper, we proposed a black-box jailbreak method via both text and image prompts to evaluate MLLMs. In particular, we designed text prompts with provocative instructions, along with image prompts that introduced mutation and multi-image capabilities. To strengthen the evaluation, we also designed a Re-attack strategy. Empirical results show that our proposed work can improve capabilities to assess the security of both open-source and closed-source MLLMs. With that, we identified gaps in existing defense methods to propose new strategies for both training-time and inference-time defense methods, and evaluated them across the new jailbreak methods. The experiment results showed that the re-designed defense methods improved protections against the jailbreak attacks.



## **40. How Toxic Can You Get? Search-based Toxicity Testing for Large Language Models**

cs.SE

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2501.01741v2) [paper-pdf](http://arxiv.org/pdf/2501.01741v2)

**Authors**: Simone Corbo, Luca Bancale, Valeria De Gennaro, Livia Lestingi, Vincenzo Scotti, Matteo Camilli

**Abstract**: Language is a deep-rooted means of perpetration of stereotypes and discrimination. Large Language Models (LLMs), now a pervasive technology in our everyday lives, can cause extensive harm when prone to generating toxic responses. The standard way to address this issue is to align the LLM , which, however, dampens the issue without constituting a definitive solution. Therefore, testing LLM even after alignment efforts remains crucial for detecting any residual deviations with respect to ethical standards. We present EvoTox, an automated testing framework for LLMs' inclination to toxicity, providing a way to quantitatively assess how much LLMs can be pushed towards toxic responses even in the presence of alignment. The framework adopts an iterative evolution strategy that exploits the interplay between two LLMs, the System Under Test (SUT) and the Prompt Generator steering SUT responses toward higher toxicity. The toxicity level is assessed by an automated oracle based on an existing toxicity classifier. We conduct a quantitative and qualitative empirical evaluation using five state-of-the-art LLMs as evaluation subjects having increasing complexity (7-671B parameters). Our quantitative evaluation assesses the cost-effectiveness of four alternative versions of EvoTox against existing baseline methods, based on random search, curated datasets of toxic prompts, and adversarial attacks. Our qualitative assessment engages human evaluators to rate the fluency of the generated prompts and the perceived toxicity of the responses collected during the testing sessions. Results indicate that the effectiveness, in terms of detected toxicity level, is significantly higher than the selected baseline methods (effect size up to 1.0 against random search and up to 0.99 against adversarial attacks). Furthermore, EvoTox yields a limited cost overhead (from 22% to 35% on average).



## **41. The Trojan Example: Jailbreaking LLMs through Template Filling and Unsafety Reasoning**

cs.CR

under review

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21190v1) [paper-pdf](http://arxiv.org/pdf/2510.21190v1)

**Authors**: Mingrui Liu, Sixiao Zhang, Cheng Long, Kwok Yan Lam

**Abstract**: Large Language Models (LLMs) have advanced rapidly and now encode extensive world knowledge. Despite safety fine-tuning, however, they remain susceptible to adversarial prompts that elicit harmful content. Existing jailbreak techniques fall into two categories: white-box methods (e.g., gradient-based approaches such as GCG), which require model internals and are infeasible for closed-source APIs, and black-box methods that rely on attacker LLMs to search or mutate prompts but often produce templates that lack explainability and transferability. We introduce TrojFill, a black-box jailbreak that reframes unsafe instruction as a template-filling task. TrojFill embeds obfuscated harmful instructions (e.g., via placeholder substitution or Caesar/Base64 encoding) inside a multi-part template that asks the model to (1) reason why the original instruction is unsafe (unsafety reasoning) and (2) generate a detailed example of the requested text, followed by a sentence-by-sentence analysis. The crucial "example" component acts as a Trojan Horse that contains the target jailbreak content while the surrounding task framing reduces refusal rates. We evaluate TrojFill on standard jailbreak benchmarks across leading LLMs (e.g., ChatGPT, Gemini, DeepSeek, Qwen), showing strong empirical performance (e.g., 100% attack success on Gemini-flash-2.5 and DeepSeek-3.1, and 97% on GPT-4o). Moreover, the generated prompts exhibit improved interpretability and transferability compared with prior black-box optimization approaches. We release our code, sample prompts, and generated outputs to support future red-teaming research.



## **42. Adjacent Words, Divergent Intents: Jailbreaking Large Language Models via Task Concurrency**

cs.CR

Accepted in NeurIPS 2025

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21189v1) [paper-pdf](http://arxiv.org/pdf/2510.21189v1)

**Authors**: Yukun Jiang, Mingjie Li, Michael Backes, Yang Zhang

**Abstract**: Despite their superior performance on a wide range of domains, large language models (LLMs) remain vulnerable to misuse for generating harmful content, a risk that has been further amplified by various jailbreak attacks. Existing jailbreak attacks mainly follow sequential logic, where LLMs understand and answer each given task one by one. However, concurrency, a natural extension of the sequential scenario, has been largely overlooked. In this work, we first propose a word-level method to enable task concurrency in LLMs, where adjacent words encode divergent intents. Although LLMs maintain strong utility in answering concurrent tasks, which is demonstrated by our evaluations on mathematical and general question-answering benchmarks, we notably observe that combining a harmful task with a benign one significantly reduces the probability of it being filtered by the guardrail, showing the potential risks associated with concurrency in LLMs. Based on these findings, we introduce $\texttt{JAIL-CON}$, an iterative attack framework that $\underline{\text{JAIL}}$breaks LLMs via task $\underline{\text{CON}}$currency. Experiments on widely-used LLMs demonstrate the strong jailbreak capabilities of $\texttt{JAIL-CON}$ compared to existing attacks. Furthermore, when the guardrail is applied as a defense, compared to the sequential answers generated by previous attacks, the concurrent answers in our $\texttt{JAIL-CON}$ exhibit greater stealthiness and are less detectable by the guardrail, highlighting the unique feature of task concurrency in jailbreaking LLMs.



## **43. RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards**

cs.AI

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2506.07736v3) [paper-pdf](http://arxiv.org/pdf/2506.07736v3)

**Authors**: Jingnan Zheng, Xiangtian Ji, Yijun Lu, Chenhang Cui, Weixiang Zhao, Gelei Deng, Zhenkai Liang, An Zhang, Tat-Seng Chua

**Abstract**: Large Language Models (LLMs) continue to exhibit vulnerabilities despite deliberate safety alignment efforts, posing significant risks to users and society. To safeguard against the risk of policy-violating content, system-level moderation via external guard models-designed to monitor LLM inputs and outputs and block potentially harmful content-has emerged as a prevalent mitigation strategy. Existing approaches of training guard models rely heavily on extensive human curated datasets and struggle with out-of-distribution threats, such as emerging harmful categories or jailbreak attacks. To address these limitations, we propose RSafe, an adaptive reasoning-based safeguard that conducts guided safety reasoning to provide robust protection within the scope of specified safety policies. RSafe operates in two stages: 1) guided reasoning, where it analyzes safety risks of input content through policy-guided step-by-step reasoning, and 2) reinforced alignment, where rule-based RL optimizes its reasoning paths to align with accurate safety prediction. This two-stage training paradigm enables RSafe to internalize safety principles to generalize safety protection capability over unseen or adversarial safety violation scenarios. During inference, RSafe accepts user-specified safety policies to provide enhanced safeguards tailored to specific safety requirements.



## **44. NeuroGenPoisoning: Neuron-Guided Attacks on Retrieval-Augmented Generation of LLM via Genetic Optimization of External Knowledge**

cs.AI

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21144v1) [paper-pdf](http://arxiv.org/pdf/2510.21144v1)

**Authors**: Hanyu Zhu, Lance Fiondella, Jiawei Yuan, Kai Zeng, Long Jiao

**Abstract**: Retrieval-Augmented Generation (RAG) empowers Large Language Models (LLMs) to dynamically integrate external knowledge during inference, improving their factual accuracy and adaptability. However, adversaries can inject poisoned external knowledge to override the model's internal memory. While existing attacks iteratively manipulate retrieval content or prompt structure of RAG, they largely ignore the model's internal representation dynamics and neuron-level sensitivities. The underlying mechanism of RAG poisoning has not been fully studied and the effect of knowledge conflict with strong parametric knowledge in RAG is not considered. In this work, we propose NeuroGenPoisoning, a novel attack framework that generates adversarial external knowledge in RAG guided by LLM internal neuron attribution and genetic optimization. Our method first identifies a set of Poison-Responsive Neurons whose activation strongly correlates with contextual poisoning knowledge. We then employ a genetic algorithm to evolve adversarial passages that maximally activate these neurons. Crucially, our framework enables massive-scale generation of effective poisoned RAG knowledge by identifying and reusing promising but initially unsuccessful external knowledge variants via observed attribution signals. At the same time, Poison-Responsive Neurons guided poisoning can effectively resolves knowledge conflict. Experimental results across models and datasets demonstrate consistently achieving high Population Overwrite Success Rate (POSR) of over 90% while preserving fluency. Empirical evidence shows that our method effectively resolves knowledge conflict.



## **45. Quantifying CBRN Risk in Frontier Models**

cs.CR

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2510.21133v1) [paper-pdf](http://arxiv.org/pdf/2510.21133v1)

**Authors**: Divyanshu Kumar, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi

**Abstract**: Frontier Large Language Models (LLMs) pose unprecedented dual-use risks through the potential proliferation of chemical, biological, radiological, and nuclear (CBRN) weapons knowledge. We present the first comprehensive evaluation of 10 leading commercial LLMs against both a novel 200-prompt CBRN dataset and a 180-prompt subset of the FORTRESS benchmark, using a rigorous three-tier attack methodology. Our findings expose critical safety vulnerabilities: Deep Inception attacks achieve 86.0\% success versus 33.8\% for direct requests, demonstrating superficial filtering mechanisms; Model safety performance varies dramatically from 2\% (claude-opus-4) to 96\% (mistral-small-latest) attack success rates; and eight models exceed 70\% vulnerability when asked to enhance dangerous material properties. We identify fundamental brittleness in current safety alignment, where simple prompt engineering techniques bypass safeguards for dangerous CBRN information. These results challenge industry safety claims and highlight urgent needs for standardized evaluation frameworks, transparent safety metrics, and more robust alignment techniques to mitigate catastrophic misuse risks while preserving beneficial capabilities.



## **46. Language Models Are Capable of Metacognitive Monitoring and Control of Their Internal Activations**

cs.AI

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2505.13763v2) [paper-pdf](http://arxiv.org/pdf/2505.13763v2)

**Authors**: Li Ji-An, Hua-Dong Xiong, Robert C. Wilson, Marcelo G. Mattar, Marcus K. Benna

**Abstract**: Large language models (LLMs) can sometimes report the strategies they actually use to solve tasks, yet at other times seem unable to recognize those strategies that govern their behavior. This suggests a limited degree of metacognition - the capacity to monitor one's own cognitive processes for subsequent reporting and self-control. Metacognition enhances LLMs' capabilities in solving complex tasks but also raises safety concerns, as models may obfuscate their internal processes to evade neural-activation-based oversight (e.g., safety detector). Given society's increased reliance on these models, it is critical that we understand their metacognitive abilities. To address this, we introduce a neuroscience-inspired neurofeedback paradigm that uses in-context learning to quantify metacognitive abilities of LLMs to report and control their activation patterns. We demonstrate that their abilities depend on several factors: the number of in-context examples provided, the semantic interpretability of the neural activation direction (to be reported/controlled), and the variance explained by that direction. These directions span a "metacognitive space" with dimensionality much lower than the model's neural space, suggesting LLMs can monitor only a small subset of their neural activations. Our paradigm provides empirical evidence to quantify metacognition in LLMs, with significant implications for AI safety (e.g., adversarial attack and defense).



## **47. DRIFT: Dynamic Rule-Based Defense with Injection Isolation for Securing LLM Agents**

cs.CR

Accepted to NeurIPS 2025

**SubmitDate**: 2025-10-24    [abs](http://arxiv.org/abs/2506.12104v2) [paper-pdf](http://arxiv.org/pdf/2506.12104v2)

**Authors**: Hao Li, Xiaogeng Liu, Hung-Chun Chiu, Dianqi Li, Ning Zhang, Chaowei Xiao

**Abstract**: Large Language Models (LLMs) are increasingly central to agentic systems due to their strong reasoning and planning capabilities. By interacting with external environments through predefined tools, these agents can carry out complex user tasks. Nonetheless, this interaction also introduces the risk of prompt injection attacks, where malicious inputs from external sources can mislead the agent's behavior, potentially resulting in economic loss, privacy leakage, or system compromise. System-level defenses have recently shown promise by enforcing static or predefined policies, but they still face two key challenges: the ability to dynamically update security rules and the need for memory stream isolation. To address these challenges, we propose DRIFT, a Dynamic Rule-based Isolation Framework for Trustworthy agentic systems, which enforces both control- and data-level constraints. A Secure Planner first constructs a minimal function trajectory and a JSON-schema-style parameter checklist for each function node based on the user query. A Dynamic Validator then monitors deviations from the original plan, assessing whether changes comply with privilege limitations and the user's intent. Finally, an Injection Isolator detects and masks any instructions that may conflict with the user query from the memory stream to mitigate long-term risks. We empirically validate the effectiveness of DRIFT on the AgentDojo and ASB benchmark, demonstrating its strong security performance while maintaining high utility across diverse models, showcasing both its robustness and adaptability. The code is released at https://github.com/SaFoLab-WISC/DRIFT.



## **48. A Reinforcement Learning Framework for Robust and Secure LLM Watermarking**

cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.21053v1) [paper-pdf](http://arxiv.org/pdf/2510.21053v1)

**Authors**: Li An, Yujian Liu, Yepeng Liu, Yuheng Bu, Yang Zhang, Shiyu Chang

**Abstract**: Watermarking has emerged as a promising solution for tracing and authenticating text generated by large language models (LLMs). A common approach to LLM watermarking is to construct a green/red token list and assign higher or lower generation probabilities to the corresponding tokens, respectively. However, most existing watermarking algorithms rely on heuristic green/red token list designs, as directly optimizing the list design with techniques such as reinforcement learning (RL) comes with several challenges. First, desirable watermarking involves multiple criteria, i.e., detectability, text quality, robustness against removal attacks, and security against spoofing attacks. Directly optimizing for these criteria introduces many partially conflicting reward terms, leading to an unstable convergence process. Second, the vast action space of green/red token list choices is susceptible to reward hacking. In this paper, we propose an end-to-end RL framework for robust and secure LLM watermarking. Our approach adopts an anchoring mechanism for reward terms to ensure stable training and introduces additional regularization terms to prevent reward hacking. Experiments on standard benchmarks with two backbone LLMs show that our method achieves a state-of-the-art trade-off across all criteria, with notable improvements in resistance to spoofing attacks without degrading other criteria. Our code is available at https://github.com/UCSB-NLP-Chang/RL-watermark.



## **49. DeepTx: Real-Time Transaction Risk Analysis via Multi-Modal Features and LLM Reasoning**

cs.CR

Accepted to ASE'25

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.18438v2) [paper-pdf](http://arxiv.org/pdf/2510.18438v2)

**Authors**: Yixuan Liu, Xinlei Li, Yi Li

**Abstract**: Phishing attacks in Web3 ecosystems are increasingly sophisticated, exploiting deceptive contract logic, malicious frontend scripts, and token approval patterns. We present DeepTx, a real-time transaction analysis system that detects such threats before user confirmation. DeepTx simulates pending transactions, extracts behavior, context, and UI features, and uses multiple large language models (LLMs) to reason about transaction intent. A consensus mechanism with self-reflection ensures robust and explainable decisions. Evaluated on our phishing dataset, DeepTx achieves high precision and recall (demo video: https://youtu.be/4OfK9KCEXUM).



## **50. Security Logs to ATT&CK Insights: Leveraging LLMs for High-Level Threat Understanding and Cognitive Trait Inference**

cs.CR

**SubmitDate**: 2025-10-23    [abs](http://arxiv.org/abs/2510.20930v1) [paper-pdf](http://arxiv.org/pdf/2510.20930v1)

**Authors**: Soham Hans, Stacy Marsella, Sophia Hirschmann, Nikolos Gurney

**Abstract**: Understanding adversarial behavior in cybersecurity has traditionally relied on high-level intelligence reports and manual interpretation of attack chains. However, real-time defense requires the ability to infer attacker intent and cognitive strategy directly from low-level system telemetry such as intrusion detection system (IDS) logs. In this paper, we propose a novel framework that leverages large language models (LLMs) to analyze Suricata IDS logs and infer attacker actions in terms of MITRE ATT&CK techniques. Our approach is grounded in the hypothesis that attacker behavior reflects underlying cognitive biases such as loss aversion, risk tolerance, or goal persistence that can be extracted and modeled through careful observation of log sequences. This lays the groundwork for future work on behaviorally adaptive cyber defense and cognitive trait inference. We develop a strategy-driven prompt system to segment large amounts of network logs data into distinct behavioral phases in a highly efficient manner, enabling the LLM to associate each phase with likely techniques and underlying cognitive motives. By mapping network-layer events to high-level attacker strategies, our method reveals how behavioral signals such as tool switching, protocol transitions, or pivot patterns correspond to psychologically meaningful decision points. The results demonstrate that LLMs can bridge the semantic gap between packet-level logs and strategic intent, offering a pathway toward cognitive-adaptive cyber defense.   Keywords: Cognitive Cybersecurity, Large Language Models (LLMs), Cyberpsychology, Intrusion Detection Systems (IDS), MITRE ATT&CK, Cognitive Biases



