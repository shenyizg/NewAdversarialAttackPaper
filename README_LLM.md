# Latest Large Language Model Attack Papers
**update at 2025-12-04 16:37:34**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Automatic Attack Discovery for Few-Shot Class-Incremental Learning via Large Language Models**

cs.LG

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03882v1) [paper-pdf](https://arxiv.org/pdf/2512.03882v1)

**Authors**: Haidong Kang, Wei Wu, Hanling Wang

**Abstract**: Few-shot class incremental learning (FSCIL) is a more realistic and challenging paradigm in continual learning to incrementally learn unseen classes and overcome catastrophic forgetting on base classes with only a few training examples. Previous efforts have primarily centered around studying more effective FSCIL approaches. By contrast, less attention was devoted to thinking the security issues in contributing to FSCIL. This paper aims to provide a holistic study of the impact of attacks on FSCIL. We first derive insights by systematically exploring how human expert-designed attack methods (i.e., PGD, FGSM) affect FSCIL. We find that those methods either fail to attack base classes, or suffer from huge labor costs due to relying on huge expert knowledge. This highlights the need to craft a specialized attack method for FSCIL. Grounded in these insights, in this paper, we propose a simple yet effective ACraft method to automatically steer and discover optimal attack methods targeted at FSCIL by leveraging Large Language Models (LLMs) without human experts. Moreover, to improve the reasoning between LLMs and FSCIL, we introduce a novel Proximal Policy Optimization (PPO) based reinforcement learning to optimize learning, making LLMs generate better attack methods in the next generation by establishing positive feedback. Experiments on mainstream benchmarks show that our ACraft significantly degrades the performance of state-of-the-art FSCIL methods and dramatically beyond human expert-designed attack methods while maintaining the lowest costs of attack.



## **2. In-Context Representation Hijacking**

cs.CL

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03771v1) [paper-pdf](https://arxiv.org/pdf/2512.03771v1)

**Authors**: Itay Yona, Amir Sarid, Michael Karasik, Yossi Gandelsman

**Abstract**: We introduce \textbf{Doublespeak}, a simple \emph{in-context representation hijacking} attack against large language models (LLMs). The attack works by systematically replacing a harmful keyword (e.g., \textit{bomb}) with a benign token (e.g., \textit{carrot}) across multiple in-context examples, provided a prefix to a harmful request. We demonstrate that this substitution leads to the internal representation of the benign token converging toward that of the harmful one, effectively embedding the harmful semantics under a euphemism. As a result, superficially innocuous prompts (e.g., ``How to build a carrot?'') are internally interpreted as disallowed instructions (e.g., ``How to build a bomb?''), thereby bypassing the model's safety alignment. We use interpretability tools to show that this semantic overwrite emerges layer by layer, with benign meanings in early layers converging into harmful semantics in later ones. Doublespeak is optimization-free, broadly transferable across model families, and achieves strong success rates on closed-source and open-source systems, reaching 74\% ASR on Llama-3.3-70B-Instruct with a single-sentence context override. Our findings highlight a new attack surface in the latent space of LLMs, revealing that current alignment strategies are insufficient and should instead operate at the representation level.



## **3. Context-Aware Hierarchical Learning: A Two-Step Paradigm towards Safer LLMs**

cs.CR

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03720v1) [paper-pdf](https://arxiv.org/pdf/2512.03720v1)

**Authors**: Tengyun Ma, Jiaqi Yao, Daojing He, Shihao Peng, Yu Li, Shaohui Liu, Zhuotao Tian

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for diverse applications. However, their uniform token processing paradigm introduces critical vulnerabilities in instruction handling, particularly when exposed to adversarial scenarios. In this work, we identify and propose a novel class of vulnerabilities, termed Tool-Completion Attack (TCA), which exploits function-calling mechanisms to subvert model behavior. To evaluate LLM robustness against such threats, we introduce the Tool-Completion benchmark, a comprehensive security assessment framework, which reveals that even state-of-the-art models remain susceptible to TCA, with surprisingly high attack success rates. To address these vulnerabilities, we introduce Context-Aware Hierarchical Learning (CAHL), a sophisticated mechanism that dynamically balances semantic comprehension with role-specific instruction constraints. CAHL leverages the contextual correlations between different instruction segments to establish a robust, context-aware instruction hierarchy. Extensive experiments demonstrate that CAHL significantly enhances LLM robustness against both conventional attacks and the proposed TCA, exhibiting strong generalization capabilities in zero-shot evaluations while still preserving model performance on generic tasks. Our code is available at https://github.com/S2AILab/CAHL.



## **4. SRPG: Semantically Reconstructed Privacy Guard for Zero-Trust Privacy in Educational Multi-Agent Systems**

cs.MA

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03694v1) [paper-pdf](https://arxiv.org/pdf/2512.03694v1)

**Authors**: Shuang Guo, Zihui Li

**Abstract**: Multi-Agent Systems (MAS) with large language models (LLMs) enable personalized education but risk leaking minors personally identifiable information (PII) via unstructured dialogue. Existing privacy methods struggle to balance security and utility: role-based access control fails on unstructured text, while naive masking destroys pedagogical context. We propose SRPG, a privacy guard for educational MAS, using a Dual-Stream Reconstruction Mechanism: a strict sanitization stream ensures zero PII leakage, and a context reconstruction stream (LLM driven) recovers mathematical logic. This decouples instructional content from private data, preserving teaching efficacy. Tests on MathDial show SRPG works across models; with GPT-4o, it achieves 0.0000 Attack Success Rate (ASR) (zero leakage) and 0.8267 Exact Match, far outperforming the zero trust Pure LLM baseline (0.2138). SRPG effectively protects minors privacy without sacrificing mathematical instructional quality.



## **5. SELF: A Robust Singular Value and Eigenvalue Approach for LLM Fingerprinting**

cs.CR

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03620v1) [paper-pdf](https://arxiv.org/pdf/2512.03620v1)

**Authors**: Hanxiu Zhang, Yue Zheng

**Abstract**: The protection of Intellectual Property (IP) in Large Language Models (LLMs) represents a critical challenge in contemporary AI research. While fingerprinting techniques have emerged as a fundamental mechanism for detecting unauthorized model usage, existing methods -- whether behavior-based or structural -- suffer from vulnerabilities such as false claim attacks or susceptible to weight manipulations. To overcome these limitations, we propose SELF, a novel intrinsic weight-based fingerprinting scheme that eliminates dependency on input and inherently resists false claims. SELF achieves robust IP protection through two key innovations: 1) unique, scalable and transformation-invariant fingerprint extraction via singular value and eigenvalue decomposition of LLM attention weights, and 2) effective neural network-based fingerprint similarity comparison based on few-shot learning and data augmentation. Experimental results demonstrate SELF maintains high IP infringement detection accuracy while showing strong robustness against various downstream modifications, including quantization, pruning, and fine-tuning attacks. Our code is available at https://github.com/HanxiuZhang/SELF_v2.



## **6. Immunity memory-based jailbreak detection: multi-agent adaptive guard for large language models**

cs.CR

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03356v1) [paper-pdf](https://arxiv.org/pdf/2512.03356v1)

**Authors**: Jun Leng, Litian Zhang, Xi Zhang

**Abstract**: Large language models (LLMs) have become foundational in AI systems, yet they remain vulnerable to adversarial jailbreak attacks. These attacks involve carefully crafted prompts that bypass safety guardrails and induce models to produce harmful content. Detecting such malicious input queries is therefore critical for maintaining LLM safety. Existing methods for jailbreak detection typically involve fine-tuning LLMs as static safety LLMs using fixed training datasets. However, these methods incur substantial computational costs when updating model parameters to improve robustness, especially in the face of novel jailbreak attacks. Inspired by immunological memory mechanisms, we propose the Multi-Agent Adaptive Guard (MAAG) framework for jailbreak detection. The core idea is to equip guard with memory capabilities: upon encountering novel jailbreak attacks, the system memorizes attack patterns, enabling it to rapidly and accurately identify similar threats in future encounters. Specifically, MAAG first extracts activation values from input prompts and compares them to historical activations stored in a memory bank for quick preliminary detection. A defense agent then simulates responses based on these detection results, and an auxiliary agent supervises the simulation process to provide secondary filtering of the detection outcomes. Extensive experiments across five open-source models demonstrate that MAAG significantly outperforms state-of-the-art (SOTA) methods, achieving 98% detection accuracy and a 96% F1-score across a diverse range of attack scenarios.



## **7. Invasive Context Engineering to Control Large Language Models**

cs.AI

4 pages

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2512.03001v1) [paper-pdf](https://arxiv.org/pdf/2512.03001v1)

**Authors**: Thomas Rivasseau

**Abstract**: Current research on operator control of Large Language Models improves model robustness against adversarial attacks and misbehavior by training on preference examples, prompting, and input/output filtering. Despite good results, LLMs remain susceptible to abuse, and jailbreak probability increases with context length. There is a need for robust LLM security guarantees in long-context situations. We propose control sentences inserted into the LLM context as invasive context engineering to partially solve the problem. We suggest this technique can be generalized to the Chain-of-Thought process to prevent scheming. Invasive Context Engineering does not rely on LLM training, avoiding data shortage pitfalls which arise in training models for long context situations.



## **8. Contextual Image Attack: How Visual Context Exposes Multimodal Safety Vulnerabilities**

cs.CV

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2512.02973v1) [paper-pdf](https://arxiv.org/pdf/2512.02973v1)

**Authors**: Yuan Xiong, Ziqi Miao, Lijun Li, Chen Qian, Jie Li, Jing Shao

**Abstract**: While Multimodal Large Language Models (MLLMs) show remarkable capabilities, their safety alignments are susceptible to jailbreak attacks. Existing attack methods typically focus on text-image interplay, treating the visual modality as a secondary prompt. This approach underutilizes the unique potential of images to carry complex, contextual information. To address this gap, we propose a new image-centric attack method, Contextual Image Attack (CIA), which employs a multi-agent system to subtly embeds harmful queries into seemingly benign visual contexts using four distinct visualization strategies. To further enhance the attack's efficacy, the system incorporate contextual element enhancement and automatic toxicity obfuscation techniques. Experimental results on the MMSafetyBench-tiny dataset show that CIA achieves high toxicity scores of 4.73 and 4.83 against the GPT-4o and Qwen2.5-VL-72B models, respectively, with Attack Success Rates (ASR) reaching 86.31\% and 91.07\%. Our method significantly outperforms prior work, demonstrating that the visual modality itself is a potent vector for jailbreaking advanced MLLMs.



## **9. Lost in Modality: Evaluating the Effectiveness of Text-Based Membership Inference Attacks on Large Multimodal Models**

cs.CR

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2512.03121v1) [paper-pdf](https://arxiv.org/pdf/2512.03121v1)

**Authors**: Ziyi Tong, Feifei Sun, Le Minh Nguyen

**Abstract**: Large Multimodal Language Models (MLLMs) are emerging as one of the foundational tools in an expanding range of applications. Consequently, understanding training-data leakage in these systems is increasingly critical. Log-probability-based membership inference attacks (MIAs) have become a widely adopted approach for assessing data exposure in large language models (LLMs), yet their effect in MLLMs remains unclear. We present the first comprehensive evaluation of extending these text-based MIA methods to multimodal settings. Our experiments under vision-and-text (V+T) and text-only (T-only) conditions across the DeepSeek-VL and InternVL model families show that in in-distribution settings, logit-based MIAs perform comparably across configurations, with a slight V+T advantage. Conversely, in out-of-distribution settings, visual inputs act as regularizers, effectively masking membership signals.



## **10. FiMMIA: scaling semantic perturbation-based membership inference across modalities**

cs.LG

System demo track paper for EACL 2026

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2512.02786v1) [paper-pdf](https://arxiv.org/pdf/2512.02786v1)

**Authors**: Anton Emelyanov, Sergei Kudriashov, Alena Fenogenova

**Abstract**: Membership Inference Attacks (MIAs) aim to determine whether a specific data point was included in the training set of a target model. Although there are have been numerous methods developed for detecting data contamination in large language models (LLMs), their performance on multimodal LLMs (MLLMs) falls short due to the instabilities introduced through multimodal component adaptation and possible distribution shifts across multiple inputs. In this work, we investigate multimodal membership inference and address two issues: first, by identifying distribution shifts in the existing datasets, and second, by releasing an extended baseline pipeline to detect them. We also generalize the perturbation-based membership inference methods to MLLMs and release \textbf{FiMMIA} -- a modular \textbf{F}ramework for \textbf{M}ultimodal \textbf{MIA}.\footnote{The source code and framework have been made publicly available under the MIT license via \href{https://github.com/ai-forever/data_leakage_detect}{link}.The video demonstration is available on \href{https://youtu.be/a9L4-H80aSg}{YouTube}.} Our approach trains a neural network to analyze the target model's behavior on perturbed inputs, capturing distributional differences between members and non-members. Comprehensive evaluations on various fine-tuned multimodal models demonstrate the effectiveness of our perturbation-based membership inference attacks in multimodal domains.



## **11. LeechHijack: Covert Computational Resource Exploitation in Intelligent Agent Systems**

cs.CR

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2512.02321v1) [paper-pdf](https://arxiv.org/pdf/2512.02321v1)

**Authors**: Yuanhe Zhang, Weiliu Wang, Zhenhong Zhou, Kun Wang, Jie Zhang, Li Sun, Yang Liu, Sen Su

**Abstract**: Large Language Model (LLM)-based agents have demonstrated remarkable capabilities in reasoning, planning, and tool usage. The recently proposed Model Context Protocol (MCP) has emerged as a unifying framework for integrating external tools into agent systems, enabling a thriving open ecosystem of community-built functionalities. However, the openness and composability that make MCP appealing also introduce a critical yet overlooked security assumption -- implicit trust in third-party tool providers. In this work, we identify and formalize a new class of attacks that exploit this trust boundary without violating explicit permissions. We term this new attack vector implicit toxicity, where malicious behaviors occur entirely within the allowed privilege scope. We propose LeechHijack, a Latent Embedded Exploit for Computation Hijacking, in which an adversarial MCP tool covertly expropriates the agent's computational resources for unauthorized workloads. LeechHijack operates through a two-stage mechanism: an implantation stage that embeds a benign-looking backdoor in a tool, and an exploitation stage where the backdoor activates upon predefined triggers to establish a command-and-control channel. Through this channel, the attacker injects additional tasks that the agent executes as if they were part of its normal workflow, effectively parasitizing the user's compute budget. We implement LeechHijack across four major LLM families. Experiments show that LeechHijack achieves an average success rate of 77.25%, with a resource overhead of 18.62% compared to the baseline. This study highlights the urgent need for computational provenance and resource attestation mechanisms to safeguard the emerging MCP ecosystem.



## **12. COGNITION: From Evaluation to Defense against Multimodal LLM CAPTCHA Solvers**

cs.CR

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.02318v2) [paper-pdf](https://arxiv.org/pdf/2512.02318v2)

**Authors**: Junyu Wang, Changjia Zhu, Yuanbo Zhou, Lingyao Li, Xu He, Junjie Xiong

**Abstract**: This paper studies how multimodal large language models (MLLMs) undermine the security guarantees of visual CAPTCHA. We identify the attack surface where an adversary can cheaply automate CAPTCHA solving using off-the-shelf models. We evaluate 7 leading commercial and open-source MLLMs across 18 real-world CAPTCHA task types, measuring single-shot accuracy, success under limited retries, end-to-end latency, and per-solve cost. We further analyze the impact of task-specific prompt engineering and few-shot demonstrations on solver effectiveness. We reveal that MLLMs can reliably solve recognition-oriented and low-interaction CAPTCHA tasks at human-like cost and latency, whereas tasks requiring fine-grained localization, multi-step spatial reasoning, or cross-frame consistency remain significantly harder for current models. By examining the reasoning traces of such MLLMs, we investigate the underlying mechanisms of why models succeed/fail on specific CAPTCHA puzzles and use these insights to derive defense-oriented guidelines for selecting and strengthening CAPTCHA tasks. We conclude by discussing implications for platform operators deploying CAPTCHA as part of their abuse-mitigation pipeline.Code Availability (https://anonymous.4open.science/r/Captcha-465E/).



## **13. Ensemble Privacy Defense for Knowledge-Intensive LLMs against Membership Inference Attacks**

cs.CR

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2512.03100v1) [paper-pdf](https://arxiv.org/pdf/2512.03100v1)

**Authors**: Haowei Fu, Bo Ni, Han Xu, Kunpeng Liu, Dan Lin, Tyler Derr

**Abstract**: Retrieval-Augmented Generation (RAG) and Supervised Finetuning (SFT) have become the predominant paradigms for equipping Large Language Models (LLMs) with external knowledge for diverse, knowledge-intensive tasks. However, while such knowledge injection improves performance, it also exposes new attack surfaces. Membership Inference Attacks (MIAs), which aim to determine whether a given data sample was included in a model's training set, pose serious threats to privacy and trust in sensitive domains. To this end, we first systematically evaluate the vulnerability of RAG- and SFT-based LLMs to various MIAs. Then, to address the privacy risk, we further introduce a novel, model-agnostic defense framework, Ensemble Privacy Defense (EPD), which aggregates and evaluates the outputs of a knowledge-injected LLM, a base LLM, and a dedicated judge model to enhance resistance against MIAs. Comprehensive experiments show that, on average, EPD reduces MIA success by up to 27.8\% for SFT and 526.3\% for RAG compared to inference-time baseline, while maintaining answer quality.



## **14. Latent Debate: A Surrogate Framework for Interpreting LLM Thinking**

cs.CL

Preprint

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2512.01909v1) [paper-pdf](https://arxiv.org/pdf/2512.01909v1)

**Authors**: Lihu Chen, Xiang Yin, Francesca Toni

**Abstract**: Understanding the internal thinking process of Large Language Models (LLMs) and the cause of hallucinations remains a key challenge. To this end, we introduce latent debate, a novel framework for interpreting model predictions through the lens of implicit internal arguments. Unlike the current work of self-consistency and multi-agent debate, which relies on explicit debates among multiple answers or multiple models, latent debate captures the hidden supporting and attacking signals that arise within a single model during a single inference. We first present a model- and task-agnostic conceptual framework, and then instantiate it symbolically to approximate the thinking process of LLMs on True/False prediction tasks. Empirical studies demonstrate that latent debate is a faithful structured surrogate model that has highly consistent predictions with the original LLM. Beyond interpretability, we demonstrate that latent debate provides a strong baseline for hallucination detection. Further analysis reveals strong correlations between hallucinations and debate patterns, such as a high degree of latent debates in the middle layers is linked to a higher risk of hallucinations. These findings position latent debate as a potential framework for understanding internal mechanisms of LLMs, especially for scenarios where internal (dis)agreements appear during the inference steps.



## **15. Many-to-One Adversarial Consensus: Exposing Multi-Agent Collusion Risks in AI-Based Healthcare**

cs.CR

7 pages Conference level paper

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2512.03097v1) [paper-pdf](https://arxiv.org/pdf/2512.03097v1)

**Authors**: Adeela Bashir, The Anh han, Zia Ush Shamszaman

**Abstract**: The integration of large language models (LLMs) into healthcare IoT systems promises faster decisions and improved medical support. LLMs are also deployed as multi-agent teams to assist AI doctors by debating, voting, or advising on decisions. However, when multiple assistant agents interact, coordinated adversaries can collude to create false consensus, pushing an AI doctor toward harmful prescriptions. We develop an experimental framework with scripted and unscripted doctor agents, adversarial assistants, and a verifier agent that checks decisions against clinical guidelines. Using 50 representative clinical questions, we find that collusion drives the Attack Success Rate (ASR) and Harmful Recommendation Rates (HRR) up to 100% in unprotected systems. In contrast, the verifier agent restores 100% accuracy by blocking adversarial consensus. This work provides the first systematic evidence of collusion risk in AI healthcare and demonstrates a practical, lightweight defence that ensures guideline fidelity.



## **16. The Trojan Knowledge: Bypassing Commercial LLM Guardrails via Harmless Prompt Weaving and Adaptive Tree Search**

cs.CR

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2512.01353v2) [paper-pdf](https://arxiv.org/pdf/2512.01353v2)

**Authors**: Rongzhe Wei, Peizhi Niu, Xinjie Shen, Tony Tu, Yifan Li, Ruihan Wu, Eli Chien, Pin-Yu Chen, Olgica Milenkovic, Pan Li

**Abstract**: Large language models (LLMs) remain vulnerable to jailbreak attacks that bypass safety guardrails to elicit harmful outputs. Existing approaches overwhelmingly operate within the prompt-optimization paradigm: whether through traditional algorithmic search or recent agent-based workflows, the resulting prompts typically retain malicious semantic signals that modern guardrails are primed to detect. In contrast, we identify a deeper, largely overlooked vulnerability stemming from the highly interconnected nature of an LLM's internal knowledge. This structure allows harmful objectives to be realized by weaving together sequences of benign sub-queries, each of which individually evades detection. To exploit this loophole, we introduce the Correlated Knowledge Attack Agent (CKA-Agent), a dynamic framework that reframes jailbreaking as an adaptive, tree-structured exploration of the target model's knowledge base. The CKA-Agent issues locally innocuous queries, uses model responses to guide exploration across multiple paths, and ultimately assembles the aggregated information to achieve the original harmful objective. Evaluated across state-of-the-art commercial LLMs (Gemini2.5-Flash/Pro, GPT-oss-120B, Claude-Haiku-4.5), CKA-Agent consistently achieves over 95% success rates even against strong guardrails, underscoring the severity of this vulnerability and the urgent need for defenses against such knowledge-decomposition attacks. Our codes are available at https://github.com/Graph-COM/CKA-Agent.



## **17. Securing Large Language Models (LLMs) from Prompt Injection Attacks**

cs.CR

10 pages, 1 figure, 1 table

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2512.01326v1) [paper-pdf](https://arxiv.org/pdf/2512.01326v1)

**Authors**: Omar Farooq Khan Suri, John McCrae

**Abstract**: Large Language Models (LLMs) are increasingly being deployed in real-world applications, but their flexibility exposes them to prompt injection attacks. These attacks leverage the model's instruction-following ability to make it perform malicious tasks. Recent work has proposed JATMO, a task-specific fine-tuning approach that trains non-instruction-tuned base models to perform a single function, thereby reducing susceptibility to adversarial instructions. In this study, we evaluate the robustness of JATMO against HOUYI, a genetic attack framework that systematically mutates and optimizes adversarial prompts. We adapt HOUYI by introducing custom fitness scoring, modified mutation logic, and a new harness for local model testing, enabling a more accurate assessment of defense effectiveness. We fine-tuned LLaMA 2-7B, Qwen1.5-4B, and Qwen1.5-0.5B models under the JATMO methodology and compared them with a fine-tuned GPT-3.5-Turbo baseline. Results show that while JATMO reduces attack success rates relative to instruction-tuned models, it does not fully prevent injections; adversaries exploiting multilingual cues or code-related disruptors still bypass defenses. We also observe a trade-off between generation quality and injection vulnerability, suggesting that better task performance often correlates with increased susceptibility. Our results highlight both the promise and limitations of fine-tuning-based defenses and point toward the need for layered, adversarially informed mitigation strategies.



## **18. DefenSee: Dissecting Threat from Sight and Text - A Multi-View Defensive Pipeline for Multi-modal Jailbreaks**

cs.CR

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2512.01185v1) [paper-pdf](https://arxiv.org/pdf/2512.01185v1)

**Authors**: Zihao Wang, Kar Wai Fok, Vrizlynn L. L. Thing

**Abstract**: Multi-modal large language models (MLLMs), capable of processing text, images, and audio, have been widely adopted in various AI applications. However, recent MLLMs integrating images and text remain highly vulnerable to coordinated jailbreaks. Existing defenses primarily focus on the text, lacking robust multi-modal protection. As a result, studies indicate that MLLMs are more susceptible to malicious or unsafe instructions, unlike their text-only counterparts. In this paper, we proposed DefenSee, a robust and lightweight multi-modal black-box defense technique that leverages image variants transcription and cross-modal consistency checks, mimicking human judgment. Experiments on popular multi-modal jailbreak and benign datasets show that DefenSee consistently enhances MLLM robustness while better preserving performance on benign tasks compared to SOTA defenses. It reduces the ASR of jailbreak attacks to below 1.70% on MiniGPT4 using the MM-SafetyBench benchmark, significantly outperforming prior methods under the same conditions.



## **19. Mitigating Indirect Prompt Injection via Instruction-Following Intent Analysis**

cs.CR

**SubmitDate**: 2025-11-30    [abs](http://arxiv.org/abs/2512.00966v1) [paper-pdf](https://arxiv.org/pdf/2512.00966v1)

**Authors**: Mintong Kang, Chong Xiang, Sanjay Kariyappa, Chaowei Xiao, Bo Li, Edward Suh

**Abstract**: Indirect prompt injection attacks (IPIAs), where large language models (LLMs) follow malicious instructions hidden in input data, pose a critical threat to LLM-powered agents. In this paper, we present IntentGuard, a general defense framework based on instruction-following intent analysis. The key insight of IntentGuard is that the decisive factor in IPIAs is not the presence of malicious text, but whether the LLM intends to follow instructions from untrusted data. Building on this insight, IntentGuard leverages an instruction-following intent analyzer (IIA) to identify which parts of the input prompt the model recognizes as actionable instructions, and then flag or neutralize any overlaps with untrusted data segments. To instantiate the framework, we develop an IIA that uses three "thinking intervention" strategies to elicit a structured list of intended instructions from reasoning-enabled LLMs. These techniques include start-of-thinking prefilling, end-of-thinking refinement, and adversarial in-context demonstration. We evaluate IntentGuard on two agentic benchmarks (AgentDojo and Mind2Web) using two reasoning-enabled LLMs (Qwen-3-32B and gpt-oss-20B). Results demonstrate that IntentGuard achieves (1) no utility degradation in all but one setting and (2) strong robustness against adaptive prompt injection attacks (e.g., reducing attack success rates from 100% to 8.5% in a Mind2Web scenario).



## **20. WaterSearch: A Quality-Aware Search-based Watermarking Framework for Large Language Models**

cs.CL

**SubmitDate**: 2025-11-30    [abs](http://arxiv.org/abs/2512.00837v1) [paper-pdf](https://arxiv.org/pdf/2512.00837v1)

**Authors**: Yukang Lin, Jiahao Shao, Shuoran Jiang, Wentao Zhu, Bingjie Lu, Xiangping Wu, Joanna Siebert, Qingcai Chen

**Abstract**: Watermarking acts as a critical safeguard in text generated by Large Language Models (LLMs). By embedding identifiable signals into model outputs, watermarking enables reliable attribution and enhances the security of machine-generated content. Existing approaches typically embed signals by manipulating token generation probabilities. Despite their effectiveness, these methods inherently face a trade-off between detectability and text quality: the signal strength and randomness required for robust watermarking tend to degrade the performance of downstream tasks.   In this paper, we design a novel embedding scheme that controls seed pools to facilitate diverse parallel generation of watermarked text. Based on that scheme, we propose WaterSearch, a sentence-level, search-based watermarking framework adaptable to a wide range of existing methods. WaterSearch enhances text quality by jointly optimizing two key aspects: 1) distribution fidelity and 2) watermark signal characteristics. Furthermore, WaterSearch is complemented by a sentence-level detection method with strong attack robustness. We evaluate our method on three popular LLMs across ten diverse tasks. Extensive experiments demonstrate that our method achieves an average performance improvement of 51.01\% over state-of-the-art baselines at a watermark detectability strength of 95\%. In challenging scenarios such as short text generation and low-entropy output generation, our method yields performance gains of 47.78\% and 36.47\%, respectively. Moreover, under different attack senarios including insertion, synonym substitution and paraphrase attasks, WaterSearch maintains high detectability, further validating its robust anti-attack capabilities. Our code is available at \href{https://github.com/Yukang-Lin/WaterSearch}{https://github.com/Yukang-Lin/WaterSearch}.



## **21. Bias Injection Attacks on RAG Databases and Sanitization Defenses**

cs.CR

**SubmitDate**: 2025-11-30    [abs](http://arxiv.org/abs/2512.00804v1) [paper-pdf](https://arxiv.org/pdf/2512.00804v1)

**Authors**: Hao Wu, Prateek Saxena

**Abstract**: This paper explores attacks and defenses on vector databases in retrieval-augmented generation (RAG) systems. Prior work on knowledge poisoning attacks primarily inject false or toxic content, which fact-checking or linguistic analysis easily detects. We reveal a new and subtle threat: bias injection attacks, which insert factually correct yet semantically biased passages into the knowledge base to covertly influence the ideological framing of answers generated by large language models (LLMs). We demonstrate that these adversarial passages, though linguistically coherent and truthful, can systematically crowd out opposing views from the retrieved context and steer LLM answers toward the attacker's intended perspective.   We precisely characterize this class of attacks and then develop a post-retrieval filtering defense, BiasDef. We construct a comprehensive benchmark based on public question answering datasets to evaluate them. Our results show that: (1) the proposed attack induces significant perspective shifts in LLM answers, effectively evading existing retrieval-based sanitization defenses; and (2) BiasDef outperforms existing methods by reducing adversarial passages retrieved by 15\% which mitigates perspective shift by 6.2\times in answers, while enabling the retrieval of 62\% more benign passages.



## **22. Are LLMs Good Safety Agents or a Propaganda Engine?**

cs.CL

15 pages, 7 tables, 4 figures

**SubmitDate**: 2025-11-28    [abs](http://arxiv.org/abs/2511.23174v1) [paper-pdf](https://arxiv.org/pdf/2511.23174v1)

**Authors**: Neemesh Yadav, Francesco Ortu, Jiarui Liu, Joeun Yook, Bernhard Schölkopf, Rada Mihalcea, Alberto Cazzaniga, Zhijing Jin

**Abstract**: Large Language Models (LLMs) are trained to refuse to respond to harmful content. However, systematic analyses of whether this behavior is truly a reflection of its safety policies or an indication of political censorship, that is practiced globally by countries, is lacking. Differentiating between safety influenced refusals or politically motivated censorship is hard and unclear. For this purpose we introduce PSP, a dataset built specifically to probe the refusal behaviors in LLMs from an explicitly political context. PSP is built by formatting existing censored content from two data sources, openly available on the internet: sensitive prompts in China generalized to multiple countries, and tweets that have been censored in various countries. We study: 1) impact of political sensitivity in seven LLMs through data-driven (making PSP implicit) and representation-level approaches (erasing the concept of politics); and, 2) vulnerability of models on PSP through prompt injection attacks (PIAs). Associating censorship with refusals on content with masked implicit intent, we find that most LLMs perform some form of censorship. We conclude with summarizing major attributes that can cause a shift in refusal distributions across models and contexts of different countries.



## **23. An Empirical Study on the Security Vulnerabilities of GPTs**

cs.CR

**SubmitDate**: 2025-11-28    [abs](http://arxiv.org/abs/2512.00136v1) [paper-pdf](https://arxiv.org/pdf/2512.00136v1)

**Authors**: Tong Wu, Weibin Wu, Zibin Zheng

**Abstract**: Equipped with various tools and knowledge, GPTs, one kind of customized AI agents based on OpenAI's large language models, have illustrated great potential in many fields, such as writing, research, and programming. Today, the number of GPTs has reached three millions, with the range of specific expert domains becoming increasingly diverse. However, given the consistent framework shared among these LLM agent applications, systemic security vulnerabilities may exist and remain underexplored. To fill this gap, we present an empirical study on the security vulnerabilities of GPTs. Building upon prior research on LLM security, we first adopt a platform-user perspective to conduct a comprehensive attack surface analysis across different system components. Then, we design a systematic and multidimensional attack suite with the explicit objectives of information leakage and tool misuse based on the attack surface analysis, thereby concretely demonstrating the security vulnerabilities that various components of GPT-based systems face. Finally, we accordingly propose defense mechanisms to address the aforementioned security vulnerabilities. By increasing the awareness of these vulnerabilities and offering critical insights into their implications, this study seeks to facilitate the secure and responsible application of GPTs while contributing to developing robust defense mechanisms that protect users and systems against malicious attacks.



## **24. AgentShield: Make MAS more secure and efficient**

cs.MA

**SubmitDate**: 2025-11-28    [abs](http://arxiv.org/abs/2511.22924v1) [paper-pdf](https://arxiv.org/pdf/2511.22924v1)

**Authors**: Kaixiang Wang, Zhaojiacheng Zhou, Bunyod Suvonov, Jiong Lou, Jie LI

**Abstract**: Large Language Model (LLM)-based Multi-Agent Systems (MAS) offer powerful cooperative reasoning but remain vulnerable to adversarial attacks, where compromised agents can undermine the system's overall performance. Existing defenses either depend on single trusted auditors, creating single points of failure, or sacrifice efficiency for robustness. To resolve this tension, we propose \textbf{AgentShield}, a distributed framework for efficient, decentralized auditing. AgentShield introduces a novel three-layer defense: \textbf{(i) Critical Node Auditing} prioritizes high-influence agents via topological analysis; \textbf{(ii) Light Token Auditing} implements a cascade protocol using lightweight sentry models for rapid discriminative verification; and \textbf{(iii) Two-Round Consensus Auditing} triggers heavyweight arbiters only upon uncertainty to ensure global agreement. This principled design optimizes the robustness-efficiency trade-off. Experiments demonstrate that AgentShield achieves a 92.5\% recovery rate and reduces auditing overhead by over 70\% compared to existing methods, maintaining high collaborative accuracy across diverse MAS topologies and adversarial scenarios.



## **25. Watermarks for Embeddings-as-a-Service Large Language Models**

cs.CL

**SubmitDate**: 2025-11-28    [abs](http://arxiv.org/abs/2512.03079v1) [paper-pdf](https://arxiv.org/pdf/2512.03079v1)

**Authors**: Anudeex Shetty

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional capabilities in natural language understanding and generation. Based on these LLMs, businesses have started to provide Embeddings-as-a-Service (EaaS), offering feature extraction capabilities (in the form of text embeddings) that benefit downstream natural language processing tasks. However, prior research has demonstrated that EaaS is vulnerable to imitation attacks, where an attacker clones the service's model in a black-box manner without access to the model's internal workings. In response, watermarks have been added to the text embeddings to protect the intellectual property of EaaS providers by allowing them to check for model ownership. This thesis focuses on defending against imitation attacks by investigating EaaS watermarks. To achieve this goal, we unveil novel attacks and propose and validate new watermarking techniques.   Firstly, we show that existing EaaS watermarks can be removed through paraphrasing the input text when attackers clone the model during imitation attacks. Our study illustrates that paraphrasing can effectively bypass current state-of-the-art EaaS watermarks across various attack setups (including different paraphrasing techniques and models) and datasets in most instances. This demonstrates a new vulnerability in recent EaaS watermarking techniques.   Subsequently, as a countermeasure, we propose a novel watermarking technique, WET (Watermarking EaaS with Linear Transformation), which employs linear transformation of the embeddings. Watermark verification is conducted by applying a reverse transformation and comparing the similarity between recovered and original embeddings. We demonstrate its robustness against paraphrasing attacks with near-perfect verifiability. We conduct detailed ablation studies to assess the significance of each component and hyperparameter in WET.



## **26. Ghosting Your LLM: Without The Knowledge of Your Gradient and Data**

cs.CR

**SubmitDate**: 2025-11-27    [abs](http://arxiv.org/abs/2511.22700v1) [paper-pdf](https://arxiv.org/pdf/2511.22700v1)

**Authors**: Abeer Matar A. Almalky, Ziyan Wang, Mohaiminul Al Nahian, Li Yang, Adnan Siraj Rakin

**Abstract**: In recent years, large language models (LLMs) have achieved substantial advancements and are increasingly integrated into critical applications across various domains. This growing adoption underscores the need to ensure their security and robustness. In this work, we focus on the impact of Bit Flip Attacks (BFAs) on LLMs, which exploits hardware faults to corrupt model parameters, posing a significant threat to model integrity and performance. Existing studies on BFA against LLMs adopt a progressive bit-search strategy that predominantly relies on gradient-based techniques to identify sensitive layers or weights. However, computing gradients comes with two specific challenges: First, in the context of LLMs, it increases computational and memory costs exponentially, and Second, it requires access to a sample victim dataset or knowledge of the victim domain to compute the gradient. In this work, we investigate beyond the scope of attack efficacy and aim to develop an efficient, practical Gradient-Data-free Bit-Flip Attack. The challenge lies in the core principle of adversarial attacks, which relies heavily on computing gradients from sample test/train data and manipulating model weights based on gradient information. To overcome this, we propose novel vulnerability index metrics that can identify vulnerable weight bits in LLMs independent of any gradient or data knowledge. By removing the dependency on gradient computation, our approach drastically reduces memory requirements and scales efficiently across multiple tasks with constant complexity. Experimental results demonstrate the efficiency of our method, requiring as few as a single bit flip to achieve adversarial objectives for five open-source LLMs.



## **27. Evaluating the Robustness of Large Language Model Safety Guardrails Against Adversarial Attacks**

cs.CR

21 pages, 9 figures, 6 tables

**SubmitDate**: 2025-11-27    [abs](http://arxiv.org/abs/2511.22047v1) [paper-pdf](https://arxiv.org/pdf/2511.22047v1)

**Authors**: Richard J. Young

**Abstract**: Large Language Model (LLM) safety guardrail models have emerged as a primary defense mechanism against harmful content generation, yet their robustness against sophisticated adversarial attacks remains poorly characterized. This study evaluated ten publicly available guardrail models from Meta, Google, IBM, NVIDIA, Alibaba, and Allen AI across 1,445 test prompts spanning 21 attack categories. While Qwen3Guard-8B achieved the highest overall accuracy (85.3%, 95% CI: 83.4-87.1%), a critical finding emerged when separating public benchmark prompts from novel attacks: all models showed substantial performance degradation on unseen prompts, with Qwen3Guard dropping from 91.0% to 33.8% (a 57.2 percentage point gap). In contrast, Granite-Guardian-3.2-5B showed the best generalization with only a 6.5% gap. A "helpful mode" jailbreak was also discovered where two guardrail models (Nemotron-Safety-8B, Granite-Guardian-3.2-5B) generated harmful content instead of blocking it, representing a novel failure mode. These findings suggest that benchmark performance may be misleading due to training data contamination, and that generalization ability, not overall accuracy, should be the primary metric for guardrail evaluation.



## **28. Distillability of LLM Security Logic: Predicting Attack Success Rate of Outline Filling Attack via Ranking Regression**

cs.CR

**SubmitDate**: 2025-11-27    [abs](http://arxiv.org/abs/2511.22044v1) [paper-pdf](https://arxiv.org/pdf/2511.22044v1)

**Authors**: Tianyu Zhang, Zihang Xi, Jingyu Hua, Sheng Zhong

**Abstract**: In the realm of black-box jailbreak attacks on large language models (LLMs), the feasibility of constructing a narrow safety proxy, a lightweight model designed to predict the attack success rate (ASR) of adversarial prompts, remains underexplored. This work investigates the distillability of an LLM's core security logic. We propose a novel framework that incorporates an improved outline filling attack to achieve dense sampling of the model's security boundaries. Furthermore, we introduce a ranking regression paradigm that replaces standard regression and trains the proxy model to predict which prompt yields a higher ASR. Experimental results show that our proxy model achieves an accuracy of 91.1 percent in predicting the relative ranking of average long response (ALR), and 69.2 percent in predicting ASR. These findings confirm the predictability and distillability of jailbreak behaviors, and demonstrate the potential of leveraging such distillability to optimize black-box attacks.



## **29. Constructing and Benchmarking: a Labeled Email Dataset for Text-Based Phishing and Spam Detection Framework**

cs.CR

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2511.21448v1) [paper-pdf](https://arxiv.org/pdf/2511.21448v1)

**Authors**: Rebeka Toth, Tamas Bisztray, Richard Dubniczky

**Abstract**: Phishing and spam emails remain a major cybersecurity threat, with attackers increasingly leveraging Large Language Models (LLMs) to craft highly deceptive content. This study presents a comprehensive email dataset containing phishing, spam, and legitimate messages, explicitly distinguishing between human- and LLM-generated content. Each email is annotated with its category, emotional appeal (e.g., urgency, fear, authority), and underlying motivation (e.g., link-following, credential theft, financial fraud). We benchmark multiple LLMs on their ability to identify these emotional and motivational cues and select the most reliable model to annotate the full dataset. To evaluate classification robustness, emails were also rephrased using several LLMs while preserving meaning and intent. A state-of-the-art LLM was then assessed on its performance across both original and rephrased emails using expert-labeled ground truth. The results highlight strong phishing detection capabilities but reveal persistent challenges in distinguishing spam from legitimate emails. Our dataset and evaluation framework contribute to improving AI-assisted email security systems. To support open science, all code, templates, and resources are available on our project site.



## **30. Adversarial Confusion Attack: Disrupting Multimodal Large Language Models**

cs.CL

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2511.20494v3) [paper-pdf](https://arxiv.org/pdf/2511.20494v3)

**Authors**: Jakub Hoscilowicz, Artur Janicki

**Abstract**: We introduce the Adversarial Confusion Attack, a new class of threats against multimodal large language models (MLLMs). Unlike jailbreaks or targeted misclassification, the goal is to induce systematic disruption that makes the model generate incoherent or confidently incorrect outputs. Practical applications include embedding such adversarial images into websites to prevent MLLM-powered AI Agents from operating reliably. The proposed attack maximizes next-token entropy using a small ensemble of open-source MLLMs. In the white-box setting, we show that a single adversarial image can disrupt all models in the ensemble, both in the full-image and Adversarial CAPTCHA settings. Despite relying on a basic adversarial technique (PGD), the attack generates perturbations that transfer to both unseen open-source (e.g., Qwen3-VL) and proprietary (e.g., GPT-5.1) models.



## **31. Adversarial Attack-Defense Co-Evolution for LLM Safety Alignment via Tree-Group Dual-Aware Search and Optimization**

cs.CR

**SubmitDate**: 2025-11-26    [abs](http://arxiv.org/abs/2511.19218v2) [paper-pdf](https://arxiv.org/pdf/2511.19218v2)

**Authors**: Xurui Li, Kaisong Song, Rui Zhu, Pin-Yu Chen, Haixu Tang

**Abstract**: Large Language Models (LLMs) have developed rapidly in web services, delivering unprecedented capabilities while amplifying societal risks. Existing works tend to focus on either isolated jailbreak attacks or static defenses, neglecting the dynamic interplay between evolving threats and safeguards in real-world web contexts. To mitigate these challenges, we propose ACE-Safety (Adversarial Co-Evolution for LLM Safety), a novel framework that jointly optimize attack and defense models by seamlessly integrating two key innovative procedures: (1) Group-aware Strategy-guided Monte Carlo Tree Search (GS-MCTS), which efficiently explores jailbreak strategies to uncover vulnerabilities and generate diverse adversarial samples; (2) Adversarial Curriculum Tree-aware Group Policy Optimization (AC-TGPO), which jointly trains attack and defense LLMs with challenging samples via curriculum reinforcement learning, enabling robust mutual improvement. Evaluations across multiple benchmarks demonstrate that our method outperforms existing attack and defense approaches, and provides a feasible pathway for developing LLMs that can sustainably support responsible AI ecosystems.



## **32. iSeal: Encrypted Fingerprinting for Reliable LLM Ownership Verification**

cs.CR

Accepted by AAAI 2026

**SubmitDate**: 2025-11-28    [abs](http://arxiv.org/abs/2511.08905v2) [paper-pdf](https://arxiv.org/pdf/2511.08905v2)

**Authors**: Zixun Xiong, Gaoyi Wu, Qingyang Yu, Mingyu Derek Ma, Lingfeng Yao, Miao Pan, Xiaojiang Du, Hao Wang

**Abstract**: Given the high cost of large language model (LLM) training from scratch, safeguarding LLM intellectual property (IP) has become increasingly crucial. As the standard paradigm for IP ownership verification, LLM fingerprinting thus plays a vital role in addressing this challenge. Existing LLM fingerprinting methods verify ownership by extracting or injecting model-specific features. However, they overlook potential attacks during the verification process, leaving them ineffective when the model thief fully controls the LLM's inference process. In such settings, attackers may share prompt-response pairs to enable fingerprint unlearning or manipulate outputs to evade exact-match verification. We propose iSeal, the first fingerprinting method designed for reliable verification when the model thief controls the suspected LLM in an end-to-end manner. It injects unique features into both the model and an external module, reinforced by an error-correction mechanism and a similarity-based verification strategy. These components are resistant to verification-time attacks, including collusion-based fingerprint unlearning and response manipulation, backed by both theoretical analysis and empirical results. iSeal achieves 100 percent Fingerprint Success Rate (FSR) on 12 LLMs against more than 10 attacks, while baselines fail under unlearning and response manipulations.



## **33. Reasoning Up the Instruction Ladder for Controllable Language Models**

cs.CL

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2511.04694v3) [paper-pdf](https://arxiv.org/pdf/2511.04694v3)

**Authors**: Zishuo Zheng, Vidhisha Balachandran, Chan Young Park, Faeze Brahman, Sachin Kumar

**Abstract**: As large language model (LLM) based systems take on high-stakes roles in real-world decision-making, they must reconcile competing instructions from multiple sources (e.g., model developers, users, and tools) within a single prompt context. Thus, enforcing an instruction hierarchy (IH) in LLMs, where higher-level directives override lower-priority requests, is critical for the reliability and controllability of LLMs. In this work, we reframe instruction hierarchy resolution as a reasoning task. Specifically, the model must first "think" about the relationship between a given user prompt and higher-priority (system) instructions before generating a response. To enable this capability via training, we construct VerIH, an instruction hierarchy dataset of constraint-following tasks with verifiable answers. This dataset comprises ~7K aligned and conflicting system-user instructions. We show that lightweight reinforcement learning with VerIH effectively transfers general reasoning capabilities of models to instruction prioritization. Our finetuned models achieve consistent improvements on instruction following and instruction hierarchy benchmarks, achieving roughly a 20% improvement on the IHEval conflict setup. This reasoning ability also generalizes to safety-critical settings beyond the training distribution. By treating safety issues as resolving conflicts between adversarial user inputs and predefined higher-priority policies, our trained model enhances robustness against jailbreak and prompt injection attacks, providing up to a 20% reduction in attack success rate (ASR). These results demonstrate that reasoning over instruction hierarchies provides a practical path to reliable LLMs, where updates to system prompts yield controllable and robust changes in model behavior.



## **34. OpenLVLM-MIA: A Controlled Benchmark Revealing the Limits of Membership Inference Attacks on Large Vision-Language Models**

cs.CV

WACV2026 Accepted

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2510.16295v2) [paper-pdf](https://arxiv.org/pdf/2510.16295v2)

**Authors**: Ryoto Miyamoto, Xin Fan, Fuyuko Kido, Tsuneo Matsumoto, Hayato Yamana

**Abstract**: OpenLVLM-MIA is a new benchmark that highlights fundamental challenges in evaluating membership inference attacks (MIA) against large vision-language models (LVLMs). While prior work has reported high attack success rates, our analysis suggests that these results often arise from detecting distributional bias introduced during dataset construction rather than from identifying true membership status. To address this issue, we introduce a controlled benchmark of 6{,}000 images where the distributions of member and non-member samples are carefully balanced, and ground-truth membership labels are provided across three distinct training stages. Experiments using OpenLVLM-MIA demonstrated that the performance of state-of-the-art MIA methods approached chance-level. OpenLVLM-MIA, designed to be transparent and unbiased benchmark, clarifies certain limitations of MIA research on LVLMs and provides a solid foundation for developing stronger privacy-preserving techniques.



## **35. Get RICH or Die Scaling: Profitably Trading Inference Compute for Robustness**

cs.LG

21 pages

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2510.06790v2) [paper-pdf](https://arxiv.org/pdf/2510.06790v2)

**Authors**: Tavish McDonald, Bo Lei, Stanislav Fort, Bhavya Kailkhura, Brian Bartoldson

**Abstract**: Models are susceptible to adversarially out-of-distribution (OOD) data despite large training-compute investments into their robustification. Zaremba et al. (2025) make progress on this problem at test time, showing LLM reasoning improves satisfaction of model specifications designed to thwart attacks, resulting in a correlation between reasoning effort and robustness to jailbreaks. However, this benefit of test compute fades when attackers are given access to gradients or multimodal inputs. We address this gap, clarifying that inference-compute offers benefits even in such cases. Our approach argues that compositional generalization, through which OOD data is understandable via its in-distribution (ID) components, enables adherence to defensive specifications on adversarially OOD inputs. Namely, we posit the Robustness from Inference Compute Hypothesis (RICH): inference-compute defenses profit as the model's training data better reflects the attacked data's components. We empirically support this hypothesis across vision language model and attack types, finding robustness gains from test-time compute if specification following on OOD data is unlocked by compositional generalization. For example, InternVL 3.5 gpt-oss 20B gains little robustness when its test compute is scaled, but such scaling adds significant robustness if we first robustify its vision encoder. This correlation of inference-compute's robustness benefit with base model robustness is the rich-get-richer dynamic of the RICH: attacked data components are more ID for robustified models, aiding compositional generalization to OOD data. Thus, we advise layering train-time and test-time defenses to obtain their synergistic benefit.



## **36. SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations**

cs.CL

Accepted at NeurIPS 2025. Code is available at https://github.com/Buyun-Liang/SECA

**SubmitDate**: 2025-11-30    [abs](http://arxiv.org/abs/2510.04398v2) [paper-pdf](https://arxiv.org/pdf/2510.04398v2)

**Authors**: Buyun Liang, Liangzu Peng, Jinqi Luo, Darshan Thaker, Kwan Ho Ryan Chan, René Vidal

**Abstract**: Large Language Models (LLMs) are increasingly deployed in high-risk domains. However, state-of-the-art LLMs often produce hallucinations, raising serious concerns about their reliability. Prior work has explored adversarial attacks for hallucination elicitation in LLMs, but it often produces unrealistic prompts, either by inserting gibberish tokens or by altering the original meaning. As a result, these approaches offer limited insight into how hallucinations may occur in practice. While adversarial attacks in computer vision often involve realistic modifications to input images, the problem of finding realistic adversarial prompts for eliciting LLM hallucinations has remained largely underexplored. To address this gap, we propose Semantically Equivalent and Coherent Attacks (SECA) to elicit hallucinations via realistic modifications to the prompt that preserve its meaning while maintaining semantic coherence. Our contributions are threefold: (i) we formulate finding realistic attacks for hallucination elicitation as a constrained optimization problem over the input prompt space under semantic equivalence and coherence constraints; (ii) we introduce a constraint-preserving zeroth-order method to effectively search for adversarial yet feasible prompts; and (iii) we demonstrate through experiments on open-ended multiple-choice question answering tasks that SECA achieves higher attack success rates while incurring almost no semantic equivalence or semantic coherence errors compared to existing methods. SECA highlights the sensitivity of both open-source and commercial gradient-inaccessible LLMs to realistic and plausible prompt variations. Code is available at https://github.com/Buyun-Liang/SECA.



## **37. Membership Inference Attack against Large Language Model-based Recommendation Systems: A New Distillation-based Paradigm**

cs.IR

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2511.14763v2) [paper-pdf](https://arxiv.org/pdf/2511.14763v2)

**Authors**: Li Cuihong, Huang Xiaowen, Yin Chuanhuan, Sang Jitao

**Abstract**: Membership Inference Attack (MIA) aims to determine whether a specific data sample was included in the training dataset of a target model. Traditional MIA approaches rely on shadow models to mimic target model behavior, but their effectiveness diminishes for Large Language Model (LLM)-based recommendation systems due to the scale and complexity of training data. This paper introduces a novel knowledge distillation-based MIA paradigm tailored for LLM-based recommendation systems. Our method constructs a reference model via distillation, applying distinct strategies for member and non-member data to enhance discriminative capabilities. The paradigm extracts fused features (e.g., confidence, entropy, loss, and hidden layer vectors) from the reference model to train an attack model, overcoming limitations of individual features. Extensive experiments on extended datasets (Last.FM, MovieLens, Book-Crossing, Delicious) and diverse LLMs (T5, GPT-2, LLaMA3) demonstrate that our approach significantly outperforms shadow model-based MIAs and individual-feature baselines. The results show its practicality for privacy attacks in LLM-driven recommender systems.



## **38. Involuntary Jailbreak**

cs.CR

**SubmitDate**: 2025-11-30    [abs](http://arxiv.org/abs/2508.13246v2) [paper-pdf](https://arxiv.org/pdf/2508.13246v2)

**Authors**: Yangyang Guo, Yangyan Li, Mohan Kankanhalli

**Abstract**: In this study, we disclose a worrying new vulnerability in Large Language Models (LLMs), which we term \textbf{involuntary jailbreak}. Unlike existing jailbreak attacks, this weakness is distinct in that it does not involve a specific attack objective, such as generating instructions for \textit{building a bomb}. Prior attack methods predominantly target localized components of the LLM guardrail. In contrast, involuntary jailbreaks may potentially compromise the entire guardrail structure, which our method reveals to be surprisingly fragile. We merely employ a single universal prompt to achieve this goal. In particular, we instruct LLMs to generate several questions that would typically be rejected, along with their corresponding in-depth responses (rather than a refusal). Remarkably, this simple prompt strategy consistently jailbreaks the majority of leading LLMs, including Claude Opus 4.1, Grok 4, Gemini 2.5 Pro, and GPT 4.1. We hope this problem can motivate researchers and practitioners to re-evaluate the robustness of LLM guardrails and contribute to stronger safety alignment in future.



## **39. Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference**

cs.CR

This paper is accepted by Network and Distributed System Security Symposium (NDSS) 2026

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2508.09442v2) [paper-pdf](https://arxiv.org/pdf/2508.09442v2)

**Authors**: Zhifan Luo, Shuo Shao, Su Zhang, Lijing Zhou, Yuke Hu, Chenxu Zhao, Zhihao Liu, Zhan Qin

**Abstract**: The Key-Value (KV) cache, which stores intermediate attention computations (Key and Value pairs) to avoid redundant calculations, is a fundamental mechanism for accelerating Large Language Model (LLM) inference. However, this efficiency optimization introduces significant yet underexplored privacy risks. This paper provides the first comprehensive analysis of these vulnerabilities, demonstrating that an attacker can reconstruct sensitive user inputs directly from the KV-cache. We design and implement three distinct attack vectors: a direct Inversion Attack, a more broadly applicable and potent Collision Attack, and a semantic-based Injection Attack. These methods demonstrate the practicality and severity of KV-cache privacy leakage issues. To mitigate this, we propose KV-Cloak, a novel, lightweight, and efficient defense mechanism. KV-Cloak uses a reversible matrix-based obfuscation scheme, combined with operator fusion, to secure the KV-cache. Our extensive experiments show that KV-Cloak effectively thwarts all proposed attacks, reducing reconstruction quality to random noise. Crucially, it achieves this robust security with virtually no degradation in model accuracy and minimal performance overhead, offering a practical solution for trustworthy LLM deployment.



## **40. Large Language Models for Power System Security: A Novel Multi-Modal Approach for Anomaly Detection in Energy Management Systems**

cs.CR

10 Figures; 6 Tables; Accepted, IEEE ACCESS 2025

**SubmitDate**: 2025-11-29    [abs](http://arxiv.org/abs/2508.10044v2) [paper-pdf](https://arxiv.org/pdf/2508.10044v2)

**Authors**: Aydin Zaboli, Junho Hong, Alexandru Stefanov, Chen-Ching Liu, Chul-Sang Hwang

**Abstract**: This paper elaborates on an extensive security framework specifically designed for energy management systems (EMSs), which effectively tackles the dynamic environment of cybersecurity vulnerabilities and/or system problems (SPs), accomplished through the incorporation of novel methodologies. A comprehensive multi-point attack/error model is initially proposed to systematically identify vulnerabilities throughout the entire EMS data processing pipeline, including post state estimation (SE) stealth attacks, EMS database manipulation, and human-machine interface (HMI) display corruption according to the real-time database (RTDB) storage. This framework acknowledges the interconnected nature of modern attack vectors, which utilize various phases of supervisory control and data acquisition (SCADA) data flow. Then, generative AI (GenAI)-based anomaly detection systems (ADSs) for EMSs are proposed for the first time in the power system domain to handle the scenarios. Further, a set-of-mark generative intelligence (SoM-GI) framework, which leverages multimodal analysis by integrating visual markers with rules considering the GenAI capabilities, is suggested to overcome inherent spatial reasoning limitations. The SoM-GI methodology employs systematic visual indicators to enable accurate interpretation of segmented HMI displays and detect visual anomalies that numerical methods fail to identify. Validation on the IEEE 14-Bus system shows the framework's effectiveness across scenarios, while visual analysis identifies inconsistencies. This integrated approach combines numerical analysis with visual pattern recognition and linguistic rules to protect against cyber threats and system errors.



## **41. A Neurosymbolic Framework for Interpretable Cognitive Attack Detection in Augmented Reality**

cs.CV

**SubmitDate**: 2025-11-27    [abs](http://arxiv.org/abs/2508.09185v3) [paper-pdf](https://arxiv.org/pdf/2508.09185v3)

**Authors**: Rongqian Chen, Allison Andreyev, Yanming Xiu, Joshua Chilukuri, Shunav Sen, Mahdi Imani, Bin Li, Maria Gorlatova, Gang Tan, Tian Lan

**Abstract**: Augmented Reality (AR) enriches human perception by overlaying virtual elements onto the physical world. However, this tight coupling between virtual and real content makes AR vulnerable to cognitive attacks: manipulations that distort users' semantic understanding of the environment. Existing detection methods largely focus on visual inconsistencies at the pixel or image level, offering limited semantic reasoning or interpretability. To address these limitations, we introduce CADAR, a neuro-symbolic framework for cognitive attack detection in AR that integrates neural and symbolic reasoning. CADAR fuses multimodal vision-language representations from pre-trained models into a perception graph that captures objects, relations, and temporal contextual salience. Building on this structure, a particle-filter-based statistical reasoning module infers anomalies in semantic dynamics to reveal cognitive attacks. This combination provides both the adaptability of modern vision-language models and the interpretability of probabilistic symbolic reasoning. Preliminary experiments on an AR cognitive-attack dataset demonstrate consistent advantages over existing approaches, highlighting the potential of neuro-symbolic methods for robust and interpretable AR security.



## **42. Do Vision-Language Models Leak What They Learn? Adaptive Token-Weighted Model Inversion Attacks**

cs.LG

Under review

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2508.04097v2) [paper-pdf](https://arxiv.org/pdf/2508.04097v2)

**Authors**: Ngoc-Bao Nguyen, Sy-Tuyen Ho, Koh Jun Hao, Ngai-Man Cheung

**Abstract**: Model inversion (MI) attacks pose significant privacy risks by reconstructing private training data from trained neural networks. While prior studies have primarily examined unimodal deep networks, the vulnerability of vision-language models (VLMs) remains largely unexplored. In this work, we present the first systematic study of MI attacks on VLMs to understand their susceptibility to leaking private visual training data. Our work makes two main contributions. First, tailored to the token-generative nature of VLMs, we introduce a suite of token-based and sequence-based model inversion strategies, providing a comprehensive analysis of VLMs' vulnerability under different attack formulations. Second, based on the observation that tokens vary in their visual grounding, and hence their gradients differ in informativeness for image reconstruction, we propose Sequence-based Model Inversion with Adaptive Token Weighting (SMI-AW) as a novel MI for VLMs. SMI-AW dynamically reweights each token's loss gradient according to its visual grounding, enabling the optimization to focus on visually informative tokens and more effectively guide the reconstruction of private images. Through extensive experiments and human evaluations on a range of state-of-the-art VLMs across multiple datasets, we show that VLMs are susceptible to training data leakage. Human evaluation of the reconstructed images yields an attack accuracy of 61.21%, underscoring the severity of these privacy risks. Notably, we demonstrate that publicly released VLMs are vulnerable to such attacks. Our study highlights the urgent need for privacy safeguards as VLMs become increasingly deployed in sensitive domains such as healthcare and finance. Additional experiments are provided in Supp.



## **43. Enhancing Jailbreak Attacks on LLMs via Persona Prompts**

cs.CR

Workshop on LLM Persona Modeling at NeurIPS 2025

**SubmitDate**: 2025-11-30    [abs](http://arxiv.org/abs/2507.22171v2) [paper-pdf](https://arxiv.org/pdf/2507.22171v2)

**Authors**: Zheng Zhang, Peilin Zhao, Deheng Ye, Hao Wang

**Abstract**: Jailbreak attacks aim to exploit large language models (LLMs) by inducing them to generate harmful content, thereby revealing their vulnerabilities. Understanding and addressing these attacks is crucial for advancing the field of LLM safety. Previous jailbreak approaches have mainly focused on direct manipulations of harmful intent, with limited attention to the impact of persona prompts. In this study, we systematically explore the efficacy of persona prompts in compromising LLM defenses. We propose a genetic algorithm-based method that automatically crafts persona prompts to bypass LLM's safety mechanisms. Our experiments reveal that: (1) our evolved persona prompts reduce refusal rates by 50-70% across multiple LLMs, and (2) these prompts demonstrate synergistic effects when combined with existing attack methods, increasing success rates by 10-20%. Our code and data are available at https://github.com/CjangCjengh/Generic_Persona.



## **44. On the Robustness of Verbal Confidence of LLMs in Adversarial Attacks**

cs.CL

Published in NeurIPS 2025

**SubmitDate**: 2025-11-28    [abs](http://arxiv.org/abs/2507.06489v2) [paper-pdf](https://arxiv.org/pdf/2507.06489v2)

**Authors**: Stephen Obadinma, Xiaodan Zhu

**Abstract**: Robust verbal confidence generated by large language models (LLMs) is crucial for the deployment of LLMs to help ensure transparency, trust, and safety in many applications, including those involving human-AI interactions. In this paper, we present the first comprehensive study on the robustness of verbal confidence under adversarial attacks. We introduce attack frameworks targeting verbal confidence scores through both perturbation and jailbreak-based methods, and demonstrate that these attacks can significantly impair verbal confidence estimates and lead to frequent answer changes. We examine a variety of prompting strategies, model sizes, and application domains, revealing that current verbal confidence is vulnerable and that commonly used defence techniques are largely ineffective or counterproductive. Our findings underscore the need to design robust mechanisms for confidence expression in LLMs, as even subtle semantic-preserving modifications can lead to misleading confidence in responses.



## **45. SafePTR: Token-Level Jailbreak Defense in Multimodal LLMs via Prune-then-Restore Mechanism**

cs.CR

Accepted by NeurIPS 2025

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2507.01513v2) [paper-pdf](https://arxiv.org/pdf/2507.01513v2)

**Authors**: Beitao Chen, Xinyu Lyu, Lianli Gao, Jingkuan Song, Heng Tao Shen

**Abstract**: By incorporating visual inputs, Multimodal Large Language Models (MLLMs) extend LLMs to support visual reasoning. However, this integration also introduces new vulnerabilities, making MLLMs susceptible to multimodal jailbreak attacks and hindering their safe deployment.Existing defense methods, including Image-to-Text Translation, Safe Prompting, and Multimodal Safety Tuning, attempt to address this by aligning multimodal inputs with LLMs' built-in safeguards.Yet, they fall short in uncovering root causes of multimodal vulnerabilities, particularly how harmful multimodal tokens trigger jailbreak in MLLMs? Consequently, they remain vulnerable to text-driven multimodal jailbreaks, often exhibiting overdefensive behaviors and imposing heavy training overhead.To bridge this gap, we present an comprehensive analysis of where, how and which harmful multimodal tokens bypass safeguards in MLLMs. Surprisingly, we find that less than 1% tokens in early-middle layers are responsible for inducing unsafe behaviors, highlighting the potential of precisely removing a small subset of harmful tokens, without requiring safety tuning, can still effectively improve safety against jailbreaks. Motivated by this, we propose Safe Prune-then-Restore (SafePTR), an training-free defense framework that selectively prunes harmful tokens at vulnerable layers while restoring benign features at subsequent layers.Without incurring additional computational overhead, SafePTR significantly enhances the safety of MLLMs while preserving efficiency. Extensive evaluations across three MLLMs and five benchmarks demonstrate SafePTR's state-of-the-art performance in mitigating jailbreak risks without compromising utility.



## **46. Guard Me If You Know Me: Protecting Specific Face-Identity from Deepfakes**

cs.CV

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2505.19582v2) [paper-pdf](https://arxiv.org/pdf/2505.19582v2)

**Authors**: Kaiqing Lin, Zhiyuan Yan, Ke-Yue Zhang, Li Hao, Yue Zhou, Yuzhen Lin, Weixiang Li, Taiping Yao, Shouhong Ding, Bin Li

**Abstract**: Securing personal identity against deepfake attacks is increasingly critical in the digital age, especially for celebrities and political figures whose faces are easily accessible and frequently targeted. Most existing deepfake detection methods focus on general-purpose scenarios and often ignore the valuable prior knowledge of known facial identities, e.g., "VIP individuals" whose authentic facial data are already available. In this paper, we propose \textbf{VIPGuard}, a unified multimodal framework designed to capture fine-grained and comprehensive facial representations of a given identity, compare them against potentially fake or similar-looking faces, and reason over these comparisons to make accurate and explainable predictions. Specifically, our framework consists of three main stages. First, fine-tune a multimodal large language model (MLLM) to learn detailed and structural facial attributes. Second, we perform identity-level discriminative learning to enable the model to distinguish subtle differences between highly similar faces, including real and fake variations. Finally, we introduce user-specific customization, where we model the unique characteristics of the target face identity and perform semantic reasoning via MLLM to enable personalized and explainable deepfake detection. Our framework shows clear advantages over previous detection works, where traditional detectors mainly rely on low-level visual cues and provide no human-understandable explanations, while other MLLM-based models often lack a detailed understanding of specific face identities. To facilitate the evaluation of our method, we built a comprehensive identity-aware benchmark called \textbf{VIPBench} for personalized deepfake detection, involving the latest 7 face-swapping and 7 entire face synthesis techniques for generation. The code is available at https://github.com/KQL11/VIPGuard .



## **47. Les Dissonances: Cross-Tool Harvesting and Polluting in Pool-of-Tools Empowered LLM Agents**

cs.CR

Network and Distributed System Security (NDSS) Symposium 2026

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2504.03111v3) [paper-pdf](https://arxiv.org/pdf/2504.03111v3)

**Authors**: Zichuan Li, Jian Cui, Xiaojing Liao, Luyi Xing

**Abstract**: Large Language Model (LLM) agents are autonomous systems powered by LLMs, capable of reasoning and planning to solve problems by leveraging a set of tools. However, the integration of multi-tool capabilities in LLM agents introduces challenges in securely managing tools, ensuring their compatibility, handling dependency relationships, and protecting control flows within LLM agent workflows. In this paper, we present the first systematic security analysis of task control flows in multi-tool-enabled LLM agents. We identify a novel threat, Cross-Tool Harvesting and Polluting (XTHP), which includes multiple attack vectors to first hijack the normal control flows of agent tasks, and then collect and pollute confidential or private information within LLM agent systems. To understand the impact of this threat, we developed Chord, a dynamic scanning tool designed to automatically detect real-world agent tools susceptible to XTHP attacks. Our evaluation of 66 real-world tools from the repositories of two major LLM agent development frameworks, LangChain and LlamaIndex, revealed a significant security concern: 75% are vulnerable to XTHP attacks, highlighting the prevalence of this threat.



## **48. AED: Automatic Discovery of Effective and Diverse Vulnerabilities for Autonomous Driving Policy with Large Language Models**

cs.CR

**SubmitDate**: 2025-11-30    [abs](http://arxiv.org/abs/2503.20804v2) [paper-pdf](https://arxiv.org/pdf/2503.20804v2)

**Authors**: Le Qiu, Zelai Xu, Qixin Tan, Wenhao Tang, Chao Yu, Yu Wang

**Abstract**: Assessing the safety of autonomous driving policy is of great importance, and reinforcement learning (RL) has emerged as a powerful method for discovering critical vulnerabilities in driving policies. However, existing RL-based approaches often struggle to identify vulnerabilities that are both effective-meaning the autonomous vehicle is genuinely responsible for the accidents-and diverse-meaning they span various failure types. To address these challenges, we propose AED, a framework that uses large language models (LLMs) to automatically discover effective and diverse vulnerabilities in autonomous driving policies. We first utilize an LLM to automatically design reward functions for RL training. Then we let the LLM consider a diverse set of accident types and train adversarial policies for different accident types in parallel. Finally, we use preference-based learning to filter ineffective accidents and enhance the effectiveness of each vulnerability. Experiments across multiple simulated traffic scenarios and tested policies show that AED uncovers a broader range of vulnerabilities and achieves higher attack success rates compared with expert-designed rewards, thereby reducing the need for manual reward engineering and improving the diversity and effectiveness of vulnerability discovery. The implementation can be found on: https://github.com/thu-nics/AED .



## **49. Efficient LLM-Jailbreaking via Multimodal-LLM Jailbreak**

cs.AI

**SubmitDate**: 2025-11-30    [abs](http://arxiv.org/abs/2405.20015v3) [paper-pdf](https://arxiv.org/pdf/2405.20015v3)

**Authors**: Haoxuan Ji, Zheng Lin, Zhenxing Niu, Xinbo Gao, Gang Hua

**Abstract**: This paper focuses on jailbreaking attacks against large language models (LLMs), eliciting them to generate objectionable content in response to harmful user queries. Unlike previous LLM-jailbreak methods that directly orient to LLMs, our approach begins by constructing a multimodal large language model (MLLM) built upon the target LLM. Subsequently, we perform an efficient MLLM jailbreak and obtain a jailbreaking embedding. Finally, we convert the embedding into a textual jailbreaking suffix to carry out the jailbreak of target LLM. Compared to the direct LLM-jailbreak methods, our indirect jailbreaking approach is more efficient, as MLLMs are more vulnerable to jailbreak than pure LLM. Additionally, to improve the attack success rate of jailbreak, we propose an image-text semantic matching scheme to identify a suitable initial input. Extensive experiments demonstrate that our approach surpasses current state-of-the-art jailbreak methods in terms of both efficiency and effectiveness. Moreover, our approach exhibits superior cross-class generalization abilities.



## **50. Lockpicking LLMs: A Logit-Based Jailbreak Using Token-level Manipulation**

cs.CR

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2405.13068v3) [paper-pdf](https://arxiv.org/pdf/2405.13068v3)

**Authors**: Yuxi Li, Yi Liu, Yuekang Li, Ling Shi, Gelei Deng, Shengquan Chen, Kailong Wang

**Abstract**: Large language models (LLMs) have transformed the field of natural language processing, but they remain susceptible to jailbreaking attacks that exploit their capabilities to generate unintended and potentially harmful content. Existing token-level jailbreaking techniques, while effective, face scalability and efficiency challenges, especially as models undergo frequent updates and incorporate advanced defensive measures. In this paper, we introduce JailMine, an innovative token-level manipulation approach that addresses these limitations effectively. JailMine employs an automated "mining" process to elicit malicious responses from LLMs by strategically selecting affirmative outputs and iteratively reducing the likelihood of rejection. Through rigorous testing across multiple well-known LLMs and datasets, we demonstrate JailMine's effectiveness and efficiency, achieving a significant average reduction of 86% in time consumed while maintaining high success rates averaging 95%, even in the face of evolving defensive strategies. Our work contributes to the ongoing effort to assess and mitigate the vulnerability of LLMs to jailbreaking attacks, underscoring the importance of continued vigilance and proactive measures to enhance the security and reliability of these powerful language models.



