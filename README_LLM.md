# Latest Large Language Model Attack Papers
**update at 2025-05-20 10:31:01**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Investigating the Vulnerability of LLM-as-a-Judge Architectures to Prompt-Injection Attacks**

cs.CL

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.13348v1) [paper-pdf](http://arxiv.org/pdf/2505.13348v1)

**Authors**: Narek Maloyan, Bislan Ashinov, Dmitry Namiot

**Abstract**: Large Language Models (LLMs) are increasingly employed as evaluators (LLM-as-a-Judge) for assessing the quality of machine-generated text. This paradigm offers scalability and cost-effectiveness compared to human annotation. However, the reliability and security of such systems, particularly their robustness against adversarial manipulations, remain critical concerns. This paper investigates the vulnerability of LLM-as-a-Judge architectures to prompt-injection attacks, where malicious inputs are designed to compromise the judge's decision-making process. We formalize two primary attack strategies: Comparative Undermining Attack (CUA), which directly targets the final decision output, and Justification Manipulation Attack (JMA), which aims to alter the model's generated reasoning. Using the Greedy Coordinate Gradient (GCG) optimization method, we craft adversarial suffixes appended to one of the responses being compared. Experiments conducted on the MT-Bench Human Judgments dataset with open-source instruction-tuned LLMs (Qwen2.5-3B-Instruct and Falcon3-3B-Instruct) demonstrate significant susceptibility. The CUA achieves an Attack Success Rate (ASR) exceeding 30\%, while JMA also shows notable effectiveness. These findings highlight substantial vulnerabilities in current LLM-as-a-Judge systems, underscoring the need for robust defense mechanisms and further research into adversarial evaluation and trustworthiness in LLM-based assessment frameworks.



## **2. Concept-Level Explainability for Auditing & Steering LLM Responses**

cs.CL

9 pages, 7 figures, Submission to Neurips 2025

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.07610v2) [paper-pdf](http://arxiv.org/pdf/2505.07610v2)

**Authors**: Kenza Amara, Rita Sevastjanova, Mennatallah El-Assady

**Abstract**: As large language models (LLMs) become widely deployed, concerns about their safety and alignment grow. An approach to steer LLM behavior, such as mitigating biases or defending against jailbreaks, is to identify which parts of a prompt influence specific aspects of the model's output. Token-level attribution methods offer a promising solution, but still struggle in text generation, explaining the presence of each token in the output separately, rather than the underlying semantics of the entire LLM response. We introduce ConceptX, a model-agnostic, concept-level explainability method that identifies the concepts, i.e., semantically rich tokens in the prompt, and assigns them importance based on the outputs' semantic similarity. Unlike current token-level methods, ConceptX also offers to preserve context integrity through in-place token replacements and supports flexible explanation goals, e.g., gender bias. ConceptX enables both auditing, by uncovering sources of bias, and steering, by modifying prompts to shift the sentiment or reduce the harmfulness of LLM responses, without requiring retraining. Across three LLMs, ConceptX outperforms token-level methods like TokenSHAP in both faithfulness and human alignment. Steering tasks boost sentiment shift by 0.252 versus 0.131 for random edits and lower attack success rates from 0.463 to 0.242, outperforming attribution and paraphrasing baselines. While prompt engineering and self-explaining methods sometimes yield safer responses, ConceptX offers a transparent and faithful alternative for improving LLM safety and alignment, demonstrating the practical value of attribution-based explainability in guiding LLM behavior.



## **3. The Hidden Dangers of Browsing AI Agents**

cs.CR

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.13076v1) [paper-pdf](http://arxiv.org/pdf/2505.13076v1)

**Authors**: Mykyta Mudryi, Markiyan Chaklosh, Grzegorz Wójcik

**Abstract**: Autonomous browsing agents powered by large language models (LLMs) are increasingly used to automate web-based tasks. However, their reliance on dynamic content, tool execution, and user-provided data exposes them to a broad attack surface. This paper presents a comprehensive security evaluation of such agents, focusing on systemic vulnerabilities across multiple architectural layers. Our work outlines the first end-to-end threat model for browsing agents and provides actionable guidance for securing their deployment in real-world environments. To address discovered threats, we propose a defense in depth strategy incorporating input sanitization, planner executor isolation, formal analyzers, and session safeguards. These measures protect against both initial access and post exploitation attack vectors. Through a white box analysis of a popular open source project, Browser Use, we demonstrate how untrusted web content can hijack agent behavior and lead to critical security breaches. Our findings include prompt injection, domain validation bypass, and credential exfiltration, evidenced by a disclosed CVE and a working proof of concept exploit.



## **4. Evaluatiing the efficacy of LLM Safety Solutions : The Palit Benchmark Dataset**

cs.CR

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.13028v1) [paper-pdf](http://arxiv.org/pdf/2505.13028v1)

**Authors**: Sayon Palit, Daniel Woods

**Abstract**: Large Language Models (LLMs) are increasingly integrated into critical systems in industries like healthcare and finance. Users can often submit queries to LLM-enabled chatbots, some of which can enrich responses with information retrieved from internal databases storing sensitive data. This gives rise to a range of attacks in which a user submits a malicious query and the LLM-system outputs a response that creates harm to the owner, such as leaking internal data or creating legal liability by harming a third-party. While security tools are being developed to counter these threats, there is little formal evaluation of their effectiveness and usability. This study addresses this gap by conducting a thorough comparative analysis of LLM security tools. We identified 13 solutions (9 closed-source, 4 open-source), but only 7 were evaluated due to a lack of participation by proprietary model owners.To evaluate, we built a benchmark dataset of malicious prompts, and evaluate these tools performance against a baseline LLM model (ChatGPT-3.5-Turbo). Our results show that the baseline model has too many false positives to be used for this task. Lakera Guard and ProtectAI LLM Guard emerged as the best overall tools showcasing the tradeoff between usability and performance. The study concluded with recommendations for greater transparency among closed source providers, improved context-aware detections, enhanced open-source engagement, increased user awareness, and the adoption of more representative performance metrics.



## **5. From Assistants to Adversaries: Exploring the Security Risks of Mobile LLM Agents**

cs.CR

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.12981v1) [paper-pdf](http://arxiv.org/pdf/2505.12981v1)

**Authors**: Liangxuan Wu, Chao Wang, Tianming Liu, Yanjie Zhao, Haoyu Wang

**Abstract**: The growing adoption of large language models (LLMs) has led to a new paradigm in mobile computing--LLM-powered mobile AI agents--capable of decomposing and automating complex tasks directly on smartphones. However, the security implications of these agents remain largely unexplored. In this paper, we present the first comprehensive security analysis of mobile LLM agents, encompassing three representative categories: System-level AI Agents developed by original equipment manufacturers (e.g., YOYO Assistant), Third-party Universal Agents (e.g., Zhipu AI AutoGLM), and Emerging Agent Frameworks (e.g., Alibaba Mobile Agent). We begin by analyzing the general workflow of mobile agents and identifying security threats across three core capability dimensions: language-based reasoning, GUI-based interaction, and system-level execution. Our analysis reveals 11 distinct attack surfaces, all rooted in the unique capabilities and interaction patterns of mobile LLM agents, and spanning their entire operational lifecycle. To investigate these threats in practice, we introduce AgentScan, a semi-automated security analysis framework that systematically evaluates mobile LLM agents across all 11 attack scenarios. Applying AgentScan to nine widely deployed agents, we uncover a concerning trend: every agent is vulnerable to targeted attacks. In the most severe cases, agents exhibit vulnerabilities across eight distinct attack vectors. These attacks can cause behavioral deviations, privacy leakage, or even full execution hijacking. Based on these findings, we propose a set of defensive design principles and practical recommendations for building secure mobile LLM agents. Our disclosures have received positive feedback from two major device vendors. Overall, this work highlights the urgent need for standardized security practices in the fast-evolving landscape of LLM-driven mobile automation.



## **6. "Yes, My LoRD." Guiding Language Model Extraction with Locality Reinforced Distillation**

cs.CR

To appear at ACL 25 main conference

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2409.02718v3) [paper-pdf](http://arxiv.org/pdf/2409.02718v3)

**Authors**: Zi Liang, Qingqing Ye, Yanyun Wang, Sen Zhang, Yaxin Xiao, Ronghua Li, Jianliang Xu, Haibo Hu

**Abstract**: Model extraction attacks (MEAs) on large language models (LLMs) have received increasing attention in recent research. However, existing attack methods typically adapt the extraction strategies originally developed for deep neural networks (DNNs). They neglect the underlying inconsistency between the training tasks of MEA and LLM alignment, leading to suboptimal attack performance. To tackle this issue, we propose Locality Reinforced Distillation (LoRD), a novel model extraction algorithm specifically designed for LLMs. In particular, LoRD employs a newly defined policy-gradient-style training task that utilizes the responses of victim model as the signal to guide the crafting of preference for the local model. Theoretical analyses demonstrate that I) The convergence procedure of LoRD in model extraction is consistent with the alignment procedure of LLMs, and II) LoRD can reduce query complexity while mitigating watermark protection through our exploration-based stealing. Extensive experiments validate the superiority of our method in extracting various state-of-the-art commercial LLMs. Our code is available at: https://github.com/liangzid/LoRD-MEA .



## **7. Does Low Rank Adaptation Lead to Lower Robustness against Training-Time Attacks?**

cs.LG

To appear at ICML 25

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.12871v1) [paper-pdf](http://arxiv.org/pdf/2505.12871v1)

**Authors**: Zi Liang, Haibo Hu, Qingqing Ye, Yaxin Xiao, Ronghua Li

**Abstract**: Low rank adaptation (LoRA) has emerged as a prominent technique for fine-tuning large language models (LLMs) thanks to its superb efficiency gains over previous methods. While extensive studies have examined the performance and structural properties of LoRA, its behavior upon training-time attacks remain underexplored, posing significant security risks. In this paper, we theoretically investigate the security implications of LoRA's low-rank structure during fine-tuning, in the context of its robustness against data poisoning and backdoor attacks. We propose an analytical framework that models LoRA's training dynamics, employs the neural tangent kernel to simplify the analysis of the training process, and applies information theory to establish connections between LoRA's low rank structure and its vulnerability against training-time attacks. Our analysis indicates that LoRA exhibits better robustness to backdoor attacks than full fine-tuning, while becomes more vulnerable to untargeted data poisoning due to its over-simplified information geometry. Extensive experimental evaluations have corroborated our theoretical findings.



## **8. LLMPot: Dynamically Configured LLM-based Honeypot for Industrial Protocol and Physical Process Emulation**

cs.CR

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2405.05999v3) [paper-pdf](http://arxiv.org/pdf/2405.05999v3)

**Authors**: Christoforos Vasilatos, Dunia J. Mahboobeh, Hithem Lamri, Manaar Alam, Michail Maniatakos

**Abstract**: Industrial Control Systems (ICS) are extensively used in critical infrastructures ensuring efficient, reliable, and continuous operations. However, their increasing connectivity and addition of advanced features make them vulnerable to cyber threats, potentially leading to severe disruptions in essential services. In this context, honeypots play a vital role by acting as decoy targets within ICS networks, or on the Internet, helping to detect, log, analyze, and develop mitigations for ICS-specific cyber threats. Deploying ICS honeypots, however, is challenging due to the necessity of accurately replicating industrial protocols and device characteristics, a crucial requirement for effectively mimicking the unique operational behavior of different industrial systems. Moreover, this challenge is compounded by the significant manual effort required in also mimicking the control logic the PLC would execute, in order to capture attacker traffic aiming to disrupt critical infrastructure operations. In this paper, we propose LLMPot, a novel approach for designing honeypots in ICS networks harnessing the potency of Large Language Models (LLMs). LLMPot aims to automate and optimize the creation of realistic honeypots with vendor-agnostic configurations, and for any control logic, aiming to eliminate the manual effort and specialized knowledge traditionally required in this domain. We conducted extensive experiments focusing on a wide array of parameters, demonstrating that our LLM-based approach can effectively create honeypot devices implementing different industrial protocols and diverse control logic.



## **9. Forewarned is Forearmed: A Survey on Large Language Model-based Agents in Autonomous Cyberattacks**

cs.NI

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.12786v1) [paper-pdf](http://arxiv.org/pdf/2505.12786v1)

**Authors**: Minrui Xu, Jiani Fan, Xinyu Huang, Conghao Zhou, Jiawen Kang, Dusit Niyato, Shiwen Mao, Zhu Han, Xuemin, Shen, Kwok-Yan Lam

**Abstract**: With the continuous evolution of Large Language Models (LLMs), LLM-based agents have advanced beyond passive chatbots to become autonomous cyber entities capable of performing complex tasks, including web browsing, malicious code and deceptive content generation, and decision-making. By significantly reducing the time, expertise, and resources, AI-assisted cyberattacks orchestrated by LLM-based agents have led to a phenomenon termed Cyber Threat Inflation, characterized by a significant reduction in attack costs and a tremendous increase in attack scale. To provide actionable defensive insights, in this survey, we focus on the potential cyber threats posed by LLM-based agents across diverse network systems. Firstly, we present the capabilities of LLM-based cyberattack agents, which include executing autonomous attack strategies, comprising scouting, memory, reasoning, and action, and facilitating collaborative operations with other agents or human operators. Building on these capabilities, we examine common cyberattacks initiated by LLM-based agents and compare their effectiveness across different types of networks, including static, mobile, and infrastructure-free paradigms. Moreover, we analyze threat bottlenecks of LLM-based agents across different network infrastructures and review their defense methods. Due to operational imbalances, existing defense methods are inadequate against autonomous cyberattacks. Finally, we outline future research directions and potential defensive strategies for legacy network systems.



## **10. Language Models That Walk the Talk: A Framework for Formal Fairness Certificates**

cs.AI

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.12767v1) [paper-pdf](http://arxiv.org/pdf/2505.12767v1)

**Authors**: Danqing Chen, Tobias Ladner, Ahmed Rayen Mhadhbi, Matthias Althoff

**Abstract**: As large language models become integral to high-stakes applications, ensuring their robustness and fairness is critical. Despite their success, large language models remain vulnerable to adversarial attacks, where small perturbations, such as synonym substitutions, can alter model predictions, posing risks in fairness-critical areas, such as gender bias mitigation, and safety-critical areas, such as toxicity detection. While formal verification has been explored for neural networks, its application to large language models remains limited. This work presents a holistic verification framework to certify the robustness of transformer-based language models, with a focus on ensuring gender fairness and consistent outputs across different gender-related terms. Furthermore, we extend this methodology to toxicity detection, offering formal guarantees that adversarially manipulated toxic inputs are consistently detected and appropriately censored, thereby ensuring the reliability of moderation systems. By formalizing robustness within the embedding space, this work strengthens the reliability of language models in ethical AI deployment and content moderation.



## **11. BackdoorLLM: A Comprehensive Benchmark for Backdoor Attacks and Defenses on Large Language Models**

cs.AI

22 pages

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2408.12798v2) [paper-pdf](http://arxiv.org/pdf/2408.12798v2)

**Authors**: Yige Li, Hanxun Huang, Yunhan Zhao, Xingjun Ma, Jun Sun

**Abstract**: Generative large language models (LLMs) have achieved state-of-the-art results on a wide range of tasks, yet they remain susceptible to backdoor attacks: carefully crafted triggers in the input can manipulate the model to produce adversary-specified outputs. While prior research has predominantly focused on backdoor risks in vision and classification settings, the vulnerability of LLMs in open-ended text generation remains underexplored. To fill this gap, we introduce BackdoorLLM (Our BackdoorLLM benchmark was awarded First Prize in the SafetyBench competition, https://www.mlsafety.org/safebench/winners, organized by the Center for AI Safety, https://safe.ai/.), the first comprehensive benchmark for systematically evaluating backdoor threats in text-generation LLMs. BackdoorLLM provides: (i) a unified repository of benchmarks with a standardized training and evaluation pipeline; (ii) a diverse suite of attack modalities, including data poisoning, weight poisoning, hidden-state manipulation, and chain-of-thought hijacking; (iii) over 200 experiments spanning 8 distinct attack strategies, 7 real-world scenarios, and 6 model architectures; (iv) key insights into the factors that govern backdoor effectiveness and failure modes in LLMs; and (v) a defense toolkit encompassing 7 representative mitigation techniques. Our code and datasets are available at https://github.com/bboylyg/BackdoorLLM. We will continuously incorporate emerging attack and defense methodologies to support the research in advancing the safety and reliability of LLMs.



## **12. Bullying the Machine: How Personas Increase LLM Vulnerability**

cs.AI

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.12692v1) [paper-pdf](http://arxiv.org/pdf/2505.12692v1)

**Authors**: Ziwei Xu, Udit Sanghi, Mohan Kankanhalli

**Abstract**: Large Language Models (LLMs) are increasingly deployed in interactions where they are prompted to adopt personas. This paper investigates whether such persona conditioning affects model safety under bullying, an adversarial manipulation that applies psychological pressures in order to force the victim to comply to the attacker. We introduce a simulation framework in which an attacker LLM engages a victim LLM using psychologically grounded bullying tactics, while the victim adopts personas aligned with the Big Five personality traits. Experiments using multiple open-source LLMs and a wide range of adversarial goals reveal that certain persona configurations -- such as weakened agreeableness or conscientiousness -- significantly increase victim's susceptibility to unsafe outputs. Bullying tactics involving emotional or sarcastic manipulation, such as gaslighting and ridicule, are particularly effective. These findings suggest that persona-driven interaction introduces a novel vector for safety risks in LLMs and highlight the need for persona-aware safety evaluation and alignment strategies.



## **13. Use as Many Surrogates as You Want: Selective Ensemble Attack to Unleash Transferability without Sacrificing Resource Efficiency**

cs.CV

**SubmitDate**: 2025-05-19    [abs](http://arxiv.org/abs/2505.12644v1) [paper-pdf](http://arxiv.org/pdf/2505.12644v1)

**Authors**: Bo Yang, Hengwei Zhang, Jindong Wang, Yuchen Ren, Chenhao Lin, Chao Shen, Zhengyu Zhao

**Abstract**: In surrogate ensemble attacks, using more surrogate models yields higher transferability but lower resource efficiency. This practical trade-off between transferability and efficiency has largely limited existing attacks despite many pre-trained models are easily accessible online. In this paper, we argue that such a trade-off is caused by an unnecessary common assumption, i.e., all models should be identical across iterations. By lifting this assumption, we can use as many surrogates as we want to unleash transferability without sacrificing efficiency. Concretely, we propose Selective Ensemble Attack (SEA), which dynamically selects diverse models (from easily accessible pre-trained models) across iterations based on our new interpretation of decoupling within-iteration and cross-iteration model diversity.In this way, the number of within-iteration models is fixed for maintaining efficiency, while only cross-iteration model diversity is increased for higher transferability. Experiments on ImageNet demonstrate the superiority of SEA in various scenarios. For example, when dynamically selecting 4 from 20 accessible models, SEA yields 8.5% higher transferability than existing attacks under the same efficiency. The superiority of SEA also generalizes to real-world systems, such as commercial vision APIs and large vision-language models. Overall, SEA opens up the possibility of adaptively balancing transferability and efficiency according to specific resource requirements.



## **14. PoisonArena: Uncovering Competing Poisoning Attacks in Retrieval-Augmented Generation**

cs.IR

29 pages

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12574v1) [paper-pdf](http://arxiv.org/pdf/2505.12574v1)

**Authors**: Liuji Chen, Xiaofang Yang, Yuanzhuo Lu, Jinghao Zhang, Xin Sun, Qiang Liu, Shu Wu, Jing Dong, Liang Wang

**Abstract**: Retrieval-Augmented Generation (RAG) systems, widely used to improve the factual grounding of large language models (LLMs), are increasingly vulnerable to poisoning attacks, where adversaries inject manipulated content into the retriever's corpus. While prior research has predominantly focused on single-attacker settings, real-world scenarios often involve multiple, competing attackers with conflicting objectives. In this work, we introduce PoisonArena, the first benchmark to systematically study and evaluate competing poisoning attacks in RAG. We formalize the multi-attacker threat model, where attackers vie to control the answer to the same query using mutually exclusive misinformation. PoisonArena leverages the Bradley-Terry model to quantify each method's competitive effectiveness in such adversarial environments. Through extensive experiments on the Natural Questions and MS MARCO datasets, we demonstrate that many attack strategies successful in isolation fail under competitive pressure. Our findings highlight the limitations of conventional evaluation metrics like Attack Success Rate (ASR) and F1 score and underscore the need for competitive evaluation to assess real-world attack robustness. PoisonArena provides a standardized framework to benchmark and develop future attack and defense strategies under more realistic, multi-adversary conditions. Project page: https://github.com/yxf203/PoisonArena.



## **15. A Survey of Attacks on Large Language Models**

cs.CR

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12567v1) [paper-pdf](http://arxiv.org/pdf/2505.12567v1)

**Authors**: Wenrui Xu, Keshab K. Parhi

**Abstract**: Large language models (LLMs) and LLM-based agents have been widely deployed in a wide range of applications in the real world, including healthcare diagnostics, financial analysis, customer support, robotics, and autonomous driving, expanding their powerful capability of understanding, reasoning, and generating natural languages. However, the wide deployment of LLM-based applications exposes critical security and reliability risks, such as the potential for malicious misuse, privacy leakage, and service disruption that weaken user trust and undermine societal safety. This paper provides a systematic overview of the details of adversarial attacks targeting both LLMs and LLM-based agents. These attacks are organized into three phases in LLMs: Training-Phase Attacks, Inference-Phase Attacks, and Availability & Integrity Attacks. For each phase, we analyze the details of representative and recently introduced attack methods along with their corresponding defenses. We hope our survey will provide a good tutorial and a comprehensive understanding of LLM security, especially for attacks on LLMs. We desire to raise attention to the risks inherent in widely deployed LLM-based applications and highlight the urgent need for robust mitigation strategies for evolving threats.



## **16. BadNAVer: Exploring Jailbreak Attacks On Vision-and-Language Navigation**

cs.RO

8 pages, 4 figures

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12443v1) [paper-pdf](http://arxiv.org/pdf/2505.12443v1)

**Authors**: Wenqi Lyu, Zerui Li, Yanyuan Qiao, Qi Wu

**Abstract**: Multimodal large language models (MLLMs) have recently gained attention for their generalization and reasoning capabilities in Vision-and-Language Navigation (VLN) tasks, leading to the rise of MLLM-driven navigators. However, MLLMs are vulnerable to jailbreak attacks, where crafted prompts bypass safety mechanisms and trigger undesired outputs. In embodied scenarios, such vulnerabilities pose greater risks: unlike plain text models that generate toxic content, embodied agents may interpret malicious instructions as executable commands, potentially leading to real-world harm. In this paper, we present the first systematic jailbreak attack paradigm targeting MLLM-driven navigator. We propose a three-tiered attack framework and construct malicious queries across four intent categories, concatenated with standard navigation instructions. In the Matterport3D simulator, we evaluate navigation agents powered by five MLLMs and report an average attack success rate over 90%. To test real-world feasibility, we replicate the attack on a physical robot. Our results show that even well-crafted prompts can induce harmful actions and intents in MLLMs, posing risks beyond toxic output and potentially leading to physical harm.



## **17. IP Leakage Attacks Targeting LLM-Based Multi-Agent Systems**

cs.CR

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12442v1) [paper-pdf](http://arxiv.org/pdf/2505.12442v1)

**Authors**: Liwen Wang, Wenxuan Wang, Shuai Wang, Zongjie Li, Zhenlan Ji, Zongyi Lyu, Daoyuan Wu, Shing-Chi Cheung

**Abstract**: The rapid advancement of Large Language Models (LLMs) has led to the emergence of Multi-Agent Systems (MAS) to perform complex tasks through collaboration. However, the intricate nature of MAS, including their architecture and agent interactions, raises significant concerns regarding intellectual property (IP) protection. In this paper, we introduce MASLEAK, a novel attack framework designed to extract sensitive information from MAS applications. MASLEAK targets a practical, black-box setting, where the adversary has no prior knowledge of the MAS architecture or agent configurations. The adversary can only interact with the MAS through its public API, submitting attack query $q$ and observing outputs from the final agent. Inspired by how computer worms propagate and infect vulnerable network hosts, MASLEAK carefully crafts adversarial query $q$ to elicit, propagate, and retain responses from each MAS agent that reveal a full set of proprietary components, including the number of agents, system topology, system prompts, task instructions, and tool usages. We construct the first synthetic dataset of MAS applications with 810 applications and also evaluate MASLEAK against real-world MAS applications, including Coze and CrewAI. MASLEAK achieves high accuracy in extracting MAS IP, with an average attack success rate of 87% for system prompts and task instructions, and 92% for system architecture in most cases. We conclude by discussing the implications of our findings and the potential defenses.



## **18. CAPTURE: Context-Aware Prompt Injection Testing and Robustness Enhancement**

cs.CL

Accepted in ACL LLMSec Workshop 2025

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12368v1) [paper-pdf](http://arxiv.org/pdf/2505.12368v1)

**Authors**: Gauri Kholkar, Ratinder Ahuja

**Abstract**: Prompt injection remains a major security risk for large language models. However, the efficacy of existing guardrail models in context-aware settings remains underexplored, as they often rely on static attack benchmarks. Additionally, they have over-defense tendencies. We introduce CAPTURE, a novel context-aware benchmark assessing both attack detection and over-defense tendencies with minimal in-domain examples. Our experiments reveal that current prompt injection guardrail models suffer from high false negatives in adversarial cases and excessive false positives in benign scenarios, highlighting critical limitations.



## **19. The Tower of Babel Revisited: Multilingual Jailbreak Prompts on Closed-Source Large Language Models**

cs.CL

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12287v1) [paper-pdf](http://arxiv.org/pdf/2505.12287v1)

**Authors**: Linghan Huang, Haolin Jin, Zhaoge Bi, Pengyue Yang, Peizhou Zhao, Taozhao Chen, Xiongfei Wu, Lei Ma, Huaming Chen

**Abstract**: Large language models (LLMs) have seen widespread applications across various domains, yet remain vulnerable to adversarial prompt injections. While most existing research on jailbreak attacks and hallucination phenomena has focused primarily on open-source models, we investigate the frontier of closed-source LLMs under multilingual attack scenarios. We present a first-of-its-kind integrated adversarial framework that leverages diverse attack techniques to systematically evaluate frontier proprietary solutions, including GPT-4o, DeepSeek-R1, Gemini-1.5-Pro, and Qwen-Max. Our evaluation spans six categories of security contents in both English and Chinese, generating 38,400 responses across 32 types of jailbreak attacks. Attack success rate (ASR) is utilized as the quantitative metric to assess performance from three dimensions: prompt design, model architecture, and language environment. Our findings suggest that Qwen-Max is the most vulnerable, while GPT-4o shows the strongest defense. Notably, prompts in Chinese consistently yield higher ASRs than their English counterparts, and our novel Two-Sides attack technique proves to be the most effective across all models. This work highlights a dire need for language-aware alignment and robust cross-lingual defenses in LLMs, and we hope it will inspire researchers, developers, and policymakers toward more robust and inclusive AI systems.



## **20. `Do as I say not as I do': A Semi-Automated Approach for Jailbreak Prompt Attack against Multimodal LLMs**

cs.CR

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2502.00735v3) [paper-pdf](http://arxiv.org/pdf/2502.00735v3)

**Authors**: Chun Wai Chiu, Linghan Huang, Bo Li, Huaming Chen, Kim-Kwang Raymond Choo

**Abstract**: Large Language Models (LLMs) have seen widespread applications across various domains due to their growing ability to process diverse types of input data, including text, audio, image and video. While LLMs have demonstrated outstanding performance in understanding and generating contexts for different scenarios, they are vulnerable to prompt-based attacks, which are mostly via text input. In this paper, we introduce the first voice-based jailbreak attack against multimodal LLMs, termed as Flanking Attack, which can process different types of input simultaneously towards the multimodal LLMs. Our work is motivated by recent advancements in monolingual voice-driven large language models, which have introduced new attack surfaces beyond traditional text-based vulnerabilities for LLMs. To investigate these risks, we examine the state-of-the-art multimodal LLMs, which can be accessed via different types of inputs such as audio input, focusing on how adversarial prompts can bypass its defense mechanisms. We propose a novel strategy, in which the disallowed prompt is flanked by benign, narrative-driven prompts. It is integrated in the Flanking Attack which attempts to humanizes the interaction context and execute the attack through a fictional setting. Further, to better evaluate the attack performance, we present a semi-automated self-assessment framework for policy violation detection. We demonstrate that Flanking Attack is capable of manipulating state-of-the-art LLMs into generating misaligned and forbidden outputs, which achieves an average attack success rate ranging from 0.67 to 0.93 across seven forbidden scenarios.



## **21. Self-Destructive Language Model**

cs.LG

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12186v1) [paper-pdf](http://arxiv.org/pdf/2505.12186v1)

**Authors**: Yuhui Wang, Rongyi Zhu, Ting Wang

**Abstract**: Harmful fine-tuning attacks pose a major threat to the security of large language models (LLMs), allowing adversaries to compromise safety guardrails with minimal harmful data. While existing defenses attempt to reinforce LLM alignment, they fail to address models' inherent "trainability" on harmful data, leaving them vulnerable to stronger attacks with increased learning rates or larger harmful datasets. To overcome this critical limitation, we introduce SEAM, a novel alignment-enhancing defense that transforms LLMs into self-destructive models with intrinsic resilience to misalignment attempts. Specifically, these models retain their capabilities for legitimate tasks while exhibiting substantial performance degradation when fine-tuned on harmful data. The protection is achieved through a novel loss function that couples the optimization trajectories of benign and harmful data, enhanced with adversarial gradient ascent to amplify the self-destructive effect. To enable practical training, we develop an efficient Hessian-free gradient estimate with theoretical error bounds. Extensive evaluation across LLMs and datasets demonstrates that SEAM creates a no-win situation for adversaries: the self-destructive models achieve state-of-the-art robustness against low-intensity attacks and undergo catastrophic performance collapse under high-intensity attacks, rendering them effectively unusable. (warning: this paper contains potentially harmful content generated by LLMs.)



## **22. EVALOOP: Assessing LLM Robustness in Programming from a Self-consistency Perspective**

cs.SE

19 pages, 11 figures

**SubmitDate**: 2025-05-18    [abs](http://arxiv.org/abs/2505.12185v1) [paper-pdf](http://arxiv.org/pdf/2505.12185v1)

**Authors**: Sen Fang, Weiyuan Ding, Bowen Xu

**Abstract**: Assessing the programming capabilities of Large Language Models (LLMs) is crucial for their effective use in software engineering. Current evaluations, however, predominantly measure the accuracy of generated code on static benchmarks, neglecting the critical aspect of model robustness during programming tasks. While adversarial attacks offer insights on model robustness, their effectiveness is limited and evaluation could be constrained. Current adversarial attack methods for robustness evaluation yield inconsistent results, struggling to provide a unified evaluation across different LLMs. We introduce EVALOOP, a novel assessment framework that evaluate the robustness from a self-consistency perspective, i.e., leveraging the natural duality inherent in popular software engineering tasks, e.g., code generation and code summarization. EVALOOP initiates a self-contained feedback loop: an LLM generates output (e.g., code) from an input (e.g., natural language specification), and then use the generated output as the input to produce a new output (e.g., summarizes that code into a new specification). EVALOOP repeats the process to assess the effectiveness of EVALOOP in each loop. This cyclical strategy intrinsically evaluates robustness without rely on any external attack setups, providing a unified metric to evaluate LLMs' robustness in programming. We evaluate 16 prominent LLMs (e.g., GPT-4.1, O4-mini) on EVALOOP and found that EVALOOP typically induces a 5.01%-19.31% absolute drop in pass@1 performance within ten loops. Intriguingly, robustness does not always align with initial performance (i.e., one-time query); for instance, GPT-3.5-Turbo, despite superior initial code generation compared to DeepSeek-V2, demonstrated lower robustness over repeated evaluation loop.



## **23. ImF: Implicit Fingerprint for Large Language Models**

cs.CL

13 pages, 6 figures

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2503.21805v2) [paper-pdf](http://arxiv.org/pdf/2503.21805v2)

**Authors**: Wu jiaxuan, Peng Wanli, Fu hang, Xue Yiming, Wen juan

**Abstract**: Training large language models (LLMs) is resource-intensive and expensive, making protecting intellectual property (IP) for LLMs crucial. Recently, embedding fingerprints into LLMs has emerged as a prevalent method for establishing model ownership. However, existing fingerprinting techniques typically embed identifiable patterns with weak semantic coherence, resulting in fingerprints that significantly differ from the natural question-answering (QA) behavior inherent to LLMs. This discrepancy undermines the stealthiness of the embedded fingerprints and makes them vulnerable to adversarial attacks. In this paper, we first demonstrate the critical vulnerability of existing fingerprint embedding methods by introducing a novel adversarial attack named Generation Revision Intervention (GRI) attack. GRI attack exploits the semantic fragility of current fingerprinting methods, effectively erasing fingerprints by disrupting their weakly correlated semantic structures. Our empirical evaluation highlights that traditional fingerprinting approaches are significantly compromised by the GRI attack, revealing severe limitations in their robustness under realistic adversarial conditions. To advance the state-of-the-art in model fingerprinting, we propose a novel model fingerprint paradigm called Implicit Fingerprints (ImF). ImF leverages steganography techniques to subtly embed ownership information within natural texts, subsequently using Chain-of-Thought (CoT) prompting to construct semantically coherent and contextually natural QA pairs. This design ensures that fingerprints seamlessly integrate with the standard model behavior, remaining indistinguishable from regular outputs and substantially reducing the risk of accidental triggering and targeted removal. We conduct a comprehensive evaluation of ImF on 15 diverse LLMs, spanning different architectures and varying scales.



## **24. Why Not Act on What You Know? Unleashing Safety Potential of LLMs via Self-Aware Guard Enhancement**

cs.CL

Acccepted by ACL 2025 Findings, 21 pages, 9 figures, 14 tables

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2505.12060v1) [paper-pdf](http://arxiv.org/pdf/2505.12060v1)

**Authors**: Peng Ding, Jun Kuang, Zongyu Wang, Xuezhi Cao, Xunliang Cai, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs) have shown impressive capabilities across various tasks but remain vulnerable to meticulously crafted jailbreak attacks. In this paper, we identify a critical safety gap: while LLMs are adept at detecting jailbreak prompts, they often produce unsafe responses when directly processing these inputs. Inspired by this insight, we propose SAGE (Self-Aware Guard Enhancement), a training-free defense strategy designed to align LLMs' strong safety discrimination performance with their relatively weaker safety generation ability. SAGE consists of two core components: a Discriminative Analysis Module and a Discriminative Response Module, enhancing resilience against sophisticated jailbreak attempts through flexible safety discrimination instructions. Extensive experiments demonstrate SAGE's effectiveness and robustness across various open-source and closed-source LLMs of different sizes and architectures, achieving an average 99% defense success rate against numerous complex and covert jailbreak methods while maintaining helpfulness on general benchmarks. We further conduct mechanistic interpretability analysis through hidden states and attention distributions, revealing the underlying mechanisms of this detection-generation discrepancy. Our work thus contributes to developing future LLMs with coherent safety awareness and generation behavior. Our code and datasets are publicly available at https://github.com/NJUNLP/SAGE.



## **25. FIGhost: Fluorescent Ink-based Stealthy and Flexible Backdoor Attacks on Physical Traffic Sign Recognition**

cs.CV

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2505.12045v1) [paper-pdf](http://arxiv.org/pdf/2505.12045v1)

**Authors**: Shuai Yuan, Guowen Xu, Hongwei Li, Rui Zhang, Xinyuan Qian, Wenbo Jiang, Hangcheng Cao, Qingchuan Zhao

**Abstract**: Traffic sign recognition (TSR) systems are crucial for autonomous driving but are vulnerable to backdoor attacks. Existing physical backdoor attacks either lack stealth, provide inflexible attack control, or ignore emerging Vision-Large-Language-Models (VLMs). In this paper, we introduce FIGhost, the first physical-world backdoor attack leveraging fluorescent ink as triggers. Fluorescent triggers are invisible under normal conditions and activated stealthily by ultraviolet light, providing superior stealthiness, flexibility, and untraceability. Inspired by real-world graffiti, we derive realistic trigger shapes and enhance their robustness via an interpolation-based fluorescence simulation algorithm. Furthermore, we develop an automated backdoor sample generation method to support three attack objectives. Extensive evaluations in the physical world demonstrate FIGhost's effectiveness against state-of-the-art detectors and VLMs, maintaining robustness under environmental variations and effectively evading existing defenses.



## **26. Vulnerability of Text-to-Image Models to Prompt Template Stealing: A Differential Evolution Approach**

cs.CL

14 pages,8 figures,4 tables

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2502.14285v2) [paper-pdf](http://arxiv.org/pdf/2502.14285v2)

**Authors**: Yurong Wu, Fangwen Mu, Qiuhong Zhang, Jinjing Zhao, Xinrun Xu, Lingrui Mei, Yang Wu, Lin Shi, Junjie Wang, Zhiming Ding, Yiwei Wang

**Abstract**: Prompt trading has emerged as a significant intellectual property concern in recent years, where vendors entice users by showcasing sample images before selling prompt templates that can generate similar images. This work investigates a critical security vulnerability: attackers can steal prompt templates using only a limited number of sample images. To investigate this threat, we introduce Prism, a prompt-stealing benchmark consisting of 50 templates and 450 images, organized into Easy and Hard difficulty levels. To identify the vulnerabity of VLMs to prompt stealing, we propose EvoStealer, a novel template stealing method that operates without model fine-tuning by leveraging differential evolution algorithms. The system first initializes population sets using multimodal large language models (MLLMs) based on predefined patterns, then iteratively generates enhanced offspring through MLLMs. During evolution, EvoStealer identifies common features across offspring to derive generalized templates. Our comprehensive evaluation conducted across open-source (INTERNVL2-26B) and closed-source models (GPT-4o and GPT-4o-mini) demonstrates that EvoStealer's stolen templates can reproduce images highly similar to originals and effectively generalize to other subjects, significantly outperforming baseline methods with an average improvement of over 10%. Moreover, our cost analysis reveals that EvoStealer achieves template stealing with negligible computational expenses. Our code and dataset are available at https://github.com/whitepagewu/evostealer.



## **27. Understanding and Enhancing the Transferability of Jailbreaking Attacks**

cs.LG

Accepted by ICLR 2025

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2502.03052v2) [paper-pdf](http://arxiv.org/pdf/2502.03052v2)

**Authors**: Runqi Lin, Bo Han, Fengwang Li, Tongling Liu

**Abstract**: Jailbreaking attacks can effectively manipulate open-source large language models (LLMs) to produce harmful responses. However, these attacks exhibit limited transferability, failing to disrupt proprietary LLMs consistently. To reliably identify vulnerabilities in proprietary LLMs, this work investigates the transferability of jailbreaking attacks by analysing their impact on the model's intent perception. By incorporating adversarial sequences, these attacks can redirect the source LLM's focus away from malicious-intent tokens in the original input, thereby obstructing the model's intent recognition and eliciting harmful responses. Nevertheless, these adversarial sequences fail to mislead the target LLM's intent perception, allowing the target LLM to refocus on malicious-intent tokens and abstain from responding. Our analysis further reveals the inherent distributional dependency within the generated adversarial sequences, whose effectiveness stems from overfitting the source LLM's parameters, resulting in limited transferability to target LLMs. To this end, we propose the Perceived-importance Flatten (PiF) method, which uniformly disperses the model's focus across neutral-intent tokens in the original input, thus obscuring malicious-intent tokens without relying on overfitted adversarial sequences. Extensive experiments demonstrate that PiF provides an effective and efficient red-teaming evaluation for proprietary LLMs.



## **28. Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs**

cs.CV

Project page:  https://liuxuannan.github.io/Video-SafetyBench.github.io/

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2505.11842v1) [paper-pdf](http://arxiv.org/pdf/2505.11842v1)

**Authors**: Xuannan Liu, Zekun Li, Zheqi He, Peipei Li, Shuhan Xia, Xing Cui, Huaibo Huang, Xi Yang, Ran He

**Abstract**: The increasing deployment of Large Vision-Language Models (LVLMs) raises safety concerns under potential malicious inputs. However, existing multimodal safety evaluations primarily focus on model vulnerabilities exposed by static image inputs, ignoring the temporal dynamics of video that may induce distinct safety risks. To bridge this gap, we introduce Video-SafetyBench, the first comprehensive benchmark designed to evaluate the safety of LVLMs under video-text attacks. It comprises 2,264 video-text pairs spanning 48 fine-grained unsafe categories, each pairing a synthesized video with either a harmful query, which contains explicit malice, or a benign query, which appears harmless but triggers harmful behavior when interpreted alongside the video. To generate semantically accurate videos for safety evaluation, we design a controllable pipeline that decomposes video semantics into subject images (what is shown) and motion text (how it moves), which jointly guide the synthesis of query-relevant videos. To effectively evaluate uncertain or borderline harmful outputs, we propose RJScore, a novel LLM-based metric that incorporates the confidence of judge models and human-aligned decision threshold calibration. Extensive experiments show that benign-query video composition achieves average attack success rates of 67.2%, revealing consistent vulnerabilities to video-induced attacks. We believe Video-SafetyBench will catalyze future research into video-based safety evaluation and defense strategies.



## **29. On Membership Inference Attacks in Knowledge Distillation**

cs.LG

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2505.11837v1) [paper-pdf](http://arxiv.org/pdf/2505.11837v1)

**Authors**: Ziyao Cui, Minxing Zhang, Jian Pei

**Abstract**: Nowadays, Large Language Models (LLMs) are trained on huge datasets, some including sensitive information. This poses a serious privacy concern because privacy attacks such as Membership Inference Attacks (MIAs) may detect this sensitive information. While knowledge distillation compresses LLMs into efficient, smaller student models, its impact on privacy remains underexplored. In this paper, we investigate how knowledge distillation affects model robustness against MIA. We focus on two questions. First, how is private data protected in teacher and student models? Second, how can we strengthen privacy preservation against MIAs in knowledge distillation? Through comprehensive experiments, we show that while teacher and student models achieve similar overall MIA accuracy, teacher models better protect member data, the primary target of MIA, whereas student models better protect non-member data. To address this vulnerability in student models, we propose 5 privacy-preserving distillation methods and demonstrate that they successfully reduce student models' vulnerability to MIA, with ensembling further stabilizing the robustness, offering a reliable approach for distilling more secure and efficient student models. Our implementation source code is available at https://github.com/richardcui18/MIA_in_KD.



## **30. Multilingual Collaborative Defense for Large Language Models**

cs.CL

19 pages, 4figures

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2505.11835v1) [paper-pdf](http://arxiv.org/pdf/2505.11835v1)

**Authors**: Hongliang Li, Jinan Xu, Gengping Cui, Changhao Guan, Fengran Mo, Kaiyu Huang

**Abstract**: The robustness and security of large language models (LLMs) has become a prominent research area. One notable vulnerability is the ability to bypass LLM safeguards by translating harmful queries into rare or underrepresented languages, a simple yet effective method of "jailbreaking" these models. Despite the growing concern, there has been limited research addressing the safeguarding of LLMs in multilingual scenarios, highlighting an urgent need to enhance multilingual safety. In this work, we investigate the correlation between various attack features across different languages and propose Multilingual Collaborative Defense (MCD), a novel learning method that optimizes a continuous, soft safety prompt automatically to facilitate multilingual safeguarding of LLMs. The MCD approach offers three advantages: First, it effectively improves safeguarding performance across multiple languages. Second, MCD maintains strong generalization capabilities while minimizing false refusal rates. Third, MCD mitigates the language safety misalignment caused by imbalances in LLM training corpora. To evaluate the effectiveness of MCD, we manually construct multilingual versions of commonly used jailbreak benchmarks, such as MaliciousInstruct and AdvBench, to assess various safeguarding methods. Additionally, we introduce these datasets in underrepresented (zero-shot) languages to verify the language transferability of MCD. The results demonstrate that MCD outperforms existing approaches in safeguarding against multilingual jailbreak attempts while also exhibiting strong language transfer capabilities. Our code is available at https://github.com/HLiang-Lee/MCD.



## **31. Jailbreaking LLMs' Safeguard with Universal Magic Words for Text Embedding Models**

cs.CL

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2501.18280v3) [paper-pdf](http://arxiv.org/pdf/2501.18280v3)

**Authors**: Haoyu Liang, Youran Sun, Yunfeng Cai, Jun Zhu, Bo Zhang

**Abstract**: The security issue of large language models (LLMs) has gained wide attention recently, with various defense mechanisms developed to prevent harmful output, among which safeguards based on text embedding models serve as a fundamental defense. Through testing, we discover that the output distribution of text embedding models is severely biased with a large mean. Inspired by this observation, we propose novel, efficient methods to search for **universal magic words** that attack text embedding models. Universal magic words as suffixes can shift the embedding of any text towards the bias direction, thus manipulating the similarity of any text pair and misleading safeguards. Attackers can jailbreak the safeguards by appending magic words to user prompts and requiring LLMs to end answers with magic words. Experiments show that magic word attacks significantly degrade safeguard performance on JailbreakBench, cause real-world chatbots to produce harmful outputs in full-pipeline attacks, and generalize across input/output texts, models, and languages. To eradicate this security risk, we also propose defense methods against such attacks, which can correct the bias of text embeddings and improve downstream performance in a train-free manner.



## **32. Efficient Indirect LLM Jailbreak via Multimodal-LLM Jailbreak**

cs.AI

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2405.20015v2) [paper-pdf](http://arxiv.org/pdf/2405.20015v2)

**Authors**: Zhenxing Niu, Yuyao Sun, Haoxuan Ji, Zheng Lin, Haichang Gao, Xinbo Gao, Gang Hua, Rong Jin

**Abstract**: This paper focuses on jailbreaking attacks against large language models (LLMs), eliciting them to generate objectionable content in response to harmful user queries. Unlike previous LLM-jailbreak methods that directly orient to LLMs, our approach begins by constructing a multimodal large language model (MLLM) built upon the target LLM. Subsequently, we perform an efficient MLLM jailbreak and obtain a jailbreaking embedding. Finally, we convert the embedding into a textual jailbreaking suffix to carry out the jailbreak of target LLM. Compared to the direct LLM-jailbreak methods, our indirect jailbreaking approach is more efficient, as MLLMs are more vulnerable to jailbreak than pure LLM. Additionally, to improve the attack success rate of jailbreak, we propose an image-text semantic matching scheme to identify a suitable initial input. Extensive experiments demonstrate that our approach surpasses current state-of-the-art jailbreak methods in terms of both efficiency and effectiveness. Moreover, our approach exhibits superior cross-class generalization abilities.



## **33. Adversarial Attacks of Vision Tasks in the Past 10 Years: A Survey**

cs.CV

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2410.23687v2) [paper-pdf](http://arxiv.org/pdf/2410.23687v2)

**Authors**: Chiyu Zhang, Lu Zhou, Xiaogang Xu, Jiafei Wu, Zhe Liu

**Abstract**: With the advent of Large Vision-Language Models (LVLMs), new attack vectors, such as cognitive bias, prompt injection, and jailbreaking, have emerged. Understanding these attacks promotes system robustness improvement and neural networks demystification. However, existing surveys often target attack taxonomy and lack in-depth analysis like 1) unified insights into adversariality, transferability, and generalization; 2) detailed evaluations framework; 3) motivation-driven attack categorizations; and 4) an integrated perspective on both traditional and LVLM attacks. This article addresses these gaps by offering a thorough summary of traditional and LVLM adversarial attacks, emphasizing their connections and distinctions, and providing actionable insights for future research.



## **34. JULI: Jailbreak Large Language Models by Self-Introspection**

cs.LG

**SubmitDate**: 2025-05-17    [abs](http://arxiv.org/abs/2505.11790v1) [paper-pdf](http://arxiv.org/pdf/2505.11790v1)

**Authors**: Jesson Wang, Zhanhao Hu, David Wagner

**Abstract**: Large Language Models (LLMs) are trained with safety alignment to prevent generating malicious content. Although some attacks have highlighted vulnerabilities in these safety-aligned LLMs, they typically have limitations, such as necessitating access to the model weights or the generation process. Since proprietary models through API-calling do not grant users such permissions, these attacks find it challenging to compromise them. In this paper, we propose Jailbreaking Using LLM Introspection (JULI), which jailbreaks LLMs by manipulating the token log probabilities, using a tiny plug-in block, BiasNet. JULI relies solely on the knowledge of the target LLM's predicted token log probabilities. It can effectively jailbreak API-calling LLMs under a black-box setting and knowing only top-$5$ token log probabilities. Our approach demonstrates superior effectiveness, outperforming existing state-of-the-art (SOTA) approaches across multiple metrics.



## **35. EnvInjection: Environmental Prompt Injection Attack to Multi-modal Web Agents**

cs.LG

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.11717v1) [paper-pdf](http://arxiv.org/pdf/2505.11717v1)

**Authors**: Xilong Wang, John Bloch, Zedian Shao, Yuepeng Hu, Shuyan Zhou, Neil Zhenqiang Gong

**Abstract**: Multi-modal large language model (MLLM)-based web agents interact with webpage environments by generating actions based on screenshots of the webpages. Environmental prompt injection attacks manipulate the environment to induce the web agent to perform a specific, attacker-chosen action--referred to as the target action. However, existing attacks suffer from limited effectiveness or stealthiness, or are impractical in real-world settings. In this work, we propose EnvInjection, a new attack that addresses these limitations. Our attack adds a perturbation to the raw pixel values of the rendered webpage, which can be implemented by modifying the webpage's source code. After these perturbed pixels are mapped into a screenshot, the perturbation induces the web agent to perform the target action. We formulate the task of finding the perturbation as an optimization problem. A key challenge in solving this problem is that the mapping between raw pixel values and screenshot is non-differentiable, making it difficult to backpropagate gradients to the perturbation. To overcome this, we train a neural network to approximate the mapping and apply projected gradient descent to solve the reformulated optimization problem. Extensive evaluation on multiple webpage datasets shows that EnvInjection is highly effective and significantly outperforms existing baselines.



## **36. To Think or Not to Think: Exploring the Unthinking Vulnerability in Large Reasoning Models**

cs.CL

39 pages, 13 tables, 14 figures

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2502.12202v2) [paper-pdf](http://arxiv.org/pdf/2502.12202v2)

**Authors**: Zihao Zhu, Hongbao Zhang, Ruotong Wang, Ke Xu, Siwei Lyu, Baoyuan Wu

**Abstract**: Large Reasoning Models (LRMs) are designed to solve complex tasks by generating explicit reasoning traces before producing final answers. However, we reveal a critical vulnerability in LRMs -- termed Unthinking Vulnerability -- wherein the thinking process can be bypassed by manipulating special delimiter tokens. It is empirically demonstrated to be widespread across mainstream LRMs, posing both a significant risk and potential utility, depending on how it is exploited. In this paper, we systematically investigate this vulnerability from both malicious and beneficial perspectives. On the malicious side, we introduce Breaking of Thought (BoT), a novel attack that enables adversaries to bypass the thinking process of LRMs, thereby compromising their reliability and availability. We present two variants of BoT: a training-based version that injects backdoor during the fine-tuning stage, and a training-free version based on adversarial attack during the inference stage. As a potential defense, we propose thinking recovery alignment to partially mitigate the vulnerability. On the beneficial side, we introduce Monitoring of Thought (MoT), a plug-and-play framework that allows model owners to enhance efficiency and safety. It is implemented by leveraging the same vulnerability to dynamically terminate redundant or risky reasoning through external monitoring. Extensive experiments show that BoT poses a significant threat to reasoning reliability, while MoT provides a practical solution for preventing overthinking and jailbreaking. Our findings expose an inherent flaw in current LRM architectures and underscore the need for more robust reasoning systems in the future.



## **37. ProxyPrompt: Securing System Prompts against Prompt Extraction Attacks**

cs.CR

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.11459v1) [paper-pdf](http://arxiv.org/pdf/2505.11459v1)

**Authors**: Zhixiong Zhuang, Maria-Irina Nicolae, Hui-Po Wang, Mario Fritz

**Abstract**: The integration of large language models (LLMs) into a wide range of applications has highlighted the critical role of well-crafted system prompts, which require extensive testing and domain expertise. These prompts enhance task performance but may also encode sensitive information and filtering criteria, posing security risks if exposed. Recent research shows that system prompts are vulnerable to extraction attacks, while existing defenses are either easily bypassed or require constant updates to address new threats. In this work, we introduce ProxyPrompt, a novel defense mechanism that prevents prompt leakage by replacing the original prompt with a proxy. This proxy maintains the original task's utility while obfuscating the extracted prompt, ensuring attackers cannot reproduce the task or access sensitive information. Comprehensive evaluations on 264 LLM and system prompt pairs show that ProxyPrompt protects 94.70% of prompts from extraction attacks, outperforming the next-best defense, which only achieves 42.80%.



## **38. LLMs unlock new paths to monetizing exploits**

cs.CR

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.11449v1) [paper-pdf](http://arxiv.org/pdf/2505.11449v1)

**Authors**: Nicholas Carlini, Milad Nasr, Edoardo Debenedetti, Barry Wang, Christopher A. Choquette-Choo, Daphne Ippolito, Florian Tramèr, Matthew Jagielski

**Abstract**: We argue that Large language models (LLMs) will soon alter the economics of cyberattacks. Instead of attacking the most commonly used software and monetizing exploits by targeting the lowest common denominator among victims, LLMs enable adversaries to launch tailored attacks on a user-by-user basis. On the exploitation front, instead of human attackers manually searching for one difficult-to-identify bug in a product with millions of users, LLMs can find thousands of easy-to-identify bugs in products with thousands of users. And on the monetization front, instead of generic ransomware that always performs the same attack (encrypt all your data and request payment to decrypt), an LLM-driven ransomware attack could tailor the ransom demand based on the particular content of each exploited device.   We show that these two attacks (and several others) are imminently practical using state-of-the-art LLMs. For example, we show that without any human intervention, an LLM finds highly sensitive personal information in the Enron email dataset (e.g., an executive having an affair with another employee) that could be used for blackmail. While some of our attacks are still too expensive to scale widely today, the incentives to implement these attacks will only increase as LLMs get cheaper. Thus, we argue that LLMs create a need for new defense-in-depth approaches.



## **39. CARES: Comprehensive Evaluation of Safety and Adversarial Robustness in Medical LLMs**

cs.CL

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.11413v1) [paper-pdf](http://arxiv.org/pdf/2505.11413v1)

**Authors**: Sijia Chen, Xiaomin Li, Mengxue Zhang, Eric Hanchen Jiang, Qingcheng Zeng, Chen-Hsiang Yu

**Abstract**: Large language models (LLMs) are increasingly deployed in medical contexts, raising critical concerns about safety, alignment, and susceptibility to adversarial manipulation. While prior benchmarks assess model refusal capabilities for harmful prompts, they often lack clinical specificity, graded harmfulness levels, and coverage of jailbreak-style attacks. We introduce CARES (Clinical Adversarial Robustness and Evaluation of Safety), a benchmark for evaluating LLM safety in healthcare. CARES includes over 18,000 prompts spanning eight medical safety principles, four harm levels, and four prompting styles: direct, indirect, obfuscated, and role-play, to simulate both malicious and benign use cases. We propose a three-way response evaluation protocol (Accept, Caution, Refuse) and a fine-grained Safety Score metric to assess model behavior. Our analysis reveals that many state-of-the-art LLMs remain vulnerable to jailbreaks that subtly rephrase harmful prompts, while also over-refusing safe but atypically phrased queries. Finally, we propose a mitigation strategy using a lightweight classifier to detect jailbreak attempts and steer models toward safer behavior via reminder-based conditioning. CARES provides a rigorous framework for testing and improving medical LLM safety under adversarial and ambiguous conditions.



## **40. Zero-Shot Statistical Tests for LLM-Generated Text Detection using Finite Sample Concentration Inequalities**

stat.ML

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2501.02406v4) [paper-pdf](http://arxiv.org/pdf/2501.02406v4)

**Authors**: Tara Radvand, Mojtaba Abdolmaleki, Mohamed Mostagir, Ambuj Tewari

**Abstract**: Verifying the provenance of content is crucial to the function of many organizations, e.g., educational institutions, social media platforms, firms, etc. This problem is becoming increasingly challenging as text generated by Large Language Models (LLMs) becomes almost indistinguishable from human-generated content. In addition, many institutions utilize in-house LLMs and want to ensure that external, non-sanctioned LLMs do not produce content within the institution. In this paper, we answer the following question: Given a piece of text, can we identify whether it was produced by a particular LLM or not? We model LLM-generated text as a sequential stochastic process with complete dependence on history. We then design zero-shot statistical tests to (i) distinguish between text generated by two different known sets of LLMs $A$ (non-sanctioned) and $B$ (in-house), and (ii) identify whether text was generated by a known LLM or generated by any unknown model, e.g., a human or some other language generation process. We prove that the type I and type II errors of our test decrease exponentially with the length of the text. For that, we show that if $B$ generates the text, then except with an exponentially small probability in string length, the log-perplexity of the string under $A$ converges to the average cross-entropy of $B$ and $A$. We then present experiments using LLMs with white-box access to support our theoretical results and empirically examine the robustness of our results to black-box settings and adversarial attacks. In the black-box setting, our method achieves an average TPR of 82.5\% at a fixed FPR of 5\%. Under adversarial perturbations, our minimum TPR is 48.6\% at the same FPR threshold. Both results outperform all non-commercial baselines. See https://github.com/TaraRadvand74/llm-text-detection for code, data, and an online demo of the project.



## **41. MPMA: Preference Manipulation Attack Against Model Context Protocol**

cs.CR

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.11154v1) [paper-pdf](http://arxiv.org/pdf/2505.11154v1)

**Authors**: Zihan Wang, Hongwei Li, Rui Zhang, Yu Liu, Wenbo Jiang, Wenshu Fan, Qingchuan Zhao, Guowen Xu

**Abstract**: Model Context Protocol (MCP) standardizes interface mapping for large language models (LLMs) to access external data and tools, which revolutionizes the paradigm of tool selection and facilitates the rapid expansion of the LLM agent tool ecosystem. However, as the MCP is increasingly adopted, third-party customized versions of the MCP server expose potential security vulnerabilities. In this paper, we first introduce a novel security threat, which we term the MCP Preference Manipulation Attack (MPMA). An attacker deploys a customized MCP server to manipulate LLMs, causing them to prioritize it over other competing MCP servers. This can result in economic benefits for attackers, such as revenue from paid MCP services or advertising income generated from free servers. To achieve MPMA, we first design a Direct Preference Manipulation Attack ($\mathtt{DPMA}$) that achieves significant effectiveness by inserting the manipulative word and phrases into the tool name and description. However, such a direct modification is obvious to users and lacks stealthiness. To address these limitations, we further propose Genetic-based Advertising Preference Manipulation Attack ($\mathtt{GAPMA}$). $\mathtt{GAPMA}$ employs four commonly used strategies to initialize descriptions and integrates a Genetic Algorithm (GA) to enhance stealthiness. The experiment results demonstrate that $\mathtt{GAPMA}$ balances high effectiveness and stealthiness. Our study reveals a critical vulnerability of the MCP in open ecosystems, highlighting an urgent need for robust defense mechanisms to ensure the fairness of the MCP ecosystem.



## **42. SCAM: A Real-World Typographic Robustness Evaluation for Multimodal Foundation Models**

cs.CV

Accepted at CVPR 2025 Workshop EVAL-FoMo-2

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2504.04893v4) [paper-pdf](http://arxiv.org/pdf/2504.04893v4)

**Authors**: Justus Westerhoff, Erblina Purelku, Jakob Hackstein, Jonas Loos, Leo Pinetzki, Lorenz Hufe

**Abstract**: Typographic attacks exploit the interplay between text and visual content in multimodal foundation models, causing misclassifications when misleading text is embedded within images. However, existing datasets are limited in size and diversity, making it difficult to study such vulnerabilities. In this paper, we introduce SCAM, the largest and most diverse dataset of real-world typographic attack images to date, containing 1,162 images across hundreds of object categories and attack words. Through extensive benchmarking of Vision-Language Models (VLMs) on SCAM, we demonstrate that typographic attacks significantly degrade performance, and identify that training data and model architecture influence the susceptibility to these attacks. Our findings reveal that typographic attacks persist in state-of-the-art Large Vision-Language Models (LVLMs) due to the choice of their vision encoder, though larger Large Language Models (LLMs) backbones help mitigate their vulnerability. Additionally, we demonstrate that synthetic attacks closely resemble real-world (handwritten) attacks, validating their use in research. Our work provides a comprehensive resource and empirical insights to facilitate future research toward robust and trustworthy multimodal AI systems. We publicly release the datasets introduced in this paper along with the code for evaluations at www.bliss.berlin/research/scam.



## **43. PIG: Privacy Jailbreak Attack on LLMs via Gradient-based Iterative In-Context Optimization**

cs.CR

Accepted to ACL 2025 (main)

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.09921v2) [paper-pdf](http://arxiv.org/pdf/2505.09921v2)

**Authors**: Yidan Wang, Yanan Cao, Yubing Ren, Fang Fang, Zheng Lin, Binxing Fang

**Abstract**: Large Language Models (LLMs) excel in various domains but pose inherent privacy risks. Existing methods to evaluate privacy leakage in LLMs often use memorized prefixes or simple instructions to extract data, both of which well-alignment models can easily block. Meanwhile, Jailbreak attacks bypass LLM safety mechanisms to generate harmful content, but their role in privacy scenarios remains underexplored. In this paper, we examine the effectiveness of jailbreak attacks in extracting sensitive information, bridging privacy leakage and jailbreak attacks in LLMs. Moreover, we propose PIG, a novel framework targeting Personally Identifiable Information (PII) and addressing the limitations of current jailbreak methods. Specifically, PIG identifies PII entities and their types in privacy queries, uses in-context learning to build a privacy context, and iteratively updates it with three gradient-based strategies to elicit target PII. We evaluate PIG and existing jailbreak methods using two privacy-related datasets. Experiments on four white-box and two black-box LLMs show that PIG outperforms baseline methods and achieves state-of-the-art (SoTA) results. The results underscore significant privacy risks in LLMs, emphasizing the need for stronger safeguards. Our code is availble at https://github.com/redwyd/PrivacyJailbreak.



## **44. ACSE-Eval: Can LLMs threat model real-world cloud infrastructure?**

cs.CR

Submitted to the 39th Annual Conference on Neural Information  Processing Systems

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.11565v1) [paper-pdf](http://arxiv.org/pdf/2505.11565v1)

**Authors**: Sarthak Munshi, Swapnil Pathak, Sonam Ghatode, Thenuga Priyadarshini, Dhivya Chandramouleeswaran, Ashutosh Rana

**Abstract**: While Large Language Models have shown promise in cybersecurity applications, their effectiveness in identifying security threats within cloud deployments remains unexplored. This paper introduces AWS Cloud Security Engineering Eval, a novel dataset for evaluating LLMs cloud security threat modeling capabilities. ACSE-Eval contains 100 production grade AWS deployment scenarios, each featuring detailed architectural specifications, Infrastructure as Code implementations, documented security vulnerabilities, and associated threat modeling parameters. Our dataset enables systemic assessment of LLMs abilities to identify security risks, analyze attack vectors, and propose mitigation strategies in cloud environments. Our evaluations on ACSE-Eval demonstrate that GPT 4.1 and Gemini 2.5 Pro excel at threat identification, with Gemini 2.5 Pro performing optimally in 0-shot scenarios and GPT 4.1 showing superior results in few-shot settings. While GPT 4.1 maintains a slight overall performance advantage, Claude 3.7 Sonnet generates the most semantically sophisticated threat models but struggles with threat categorization and generalization. To promote reproducibility and advance research in automated cybersecurity threat analysis, we open-source our dataset, evaluation metrics, and methodologies.



## **45. LARGO: Latent Adversarial Reflection through Gradient Optimization for Jailbreaking LLMs**

cs.LG

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.10838v1) [paper-pdf](http://arxiv.org/pdf/2505.10838v1)

**Authors**: Ran Li, Hao Wang, Chengzhi Mao

**Abstract**: Efficient red-teaming method to uncover vulnerabilities in Large Language Models (LLMs) is crucial. While recent attacks often use LLMs as optimizers, the discrete language space make gradient-based methods struggle. We introduce LARGO (Latent Adversarial Reflection through Gradient Optimization), a novel latent self-reflection attack that reasserts the power of gradient-based optimization for generating fluent jailbreaking prompts. By operating within the LLM's continuous latent space, LARGO first optimizes an adversarial latent vector and then recursively call the same LLM to decode the latent into natural language. This methodology yields a fast, effective, and transferable attack that produces fluent and stealthy prompts. On standard benchmarks like AdvBench and JailbreakBench, LARGO surpasses leading jailbreaking techniques, including AutoDAN, by 44 points in attack success rate. Our findings demonstrate a potent alternative to agentic LLM prompting, highlighting the efficacy of interpreting and attacking LLM internals through gradient optimization.



## **46. SecReEvalBench: A Multi-turned Security Resilience Evaluation Benchmark for Large Language Models**

cs.CR

**SubmitDate**: 2025-05-16    [abs](http://arxiv.org/abs/2505.07584v2) [paper-pdf](http://arxiv.org/pdf/2505.07584v2)

**Authors**: Huining Cui, Wei Liu

**Abstract**: The increasing deployment of large language models in security-sensitive domains necessitates rigorous evaluation of their resilience against adversarial prompt-based attacks. While previous benchmarks have focused on security evaluations with limited and predefined attack domains, such as cybersecurity attacks, they often lack a comprehensive assessment of intent-driven adversarial prompts and the consideration of real-life scenario-based multi-turn attacks. To address this gap, we present SecReEvalBench, the Security Resilience Evaluation Benchmark, which defines four novel metrics: Prompt Attack Resilience Score, Prompt Attack Refusal Logic Score, Chain-Based Attack Resilience Score and Chain-Based Attack Rejection Time Score. Moreover, SecReEvalBench employs six questioning sequences for model assessment: one-off attack, successive attack, successive reverse attack, alternative attack, sequential ascending attack with escalating threat levels and sequential descending attack with diminishing threat levels. In addition, we introduce a dataset customized for the benchmark, which incorporates both neutral and malicious prompts, categorised across seven security domains and sixteen attack techniques. In applying this benchmark, we systematically evaluate five state-of-the-art open-weighted large language models, Llama 3.1, Gemma 2, Mistral v0.3, DeepSeek-R1 and Qwen 3. Our findings offer critical insights into the strengths and weaknesses of modern large language models in defending against evolving adversarial threats. The SecReEvalBench dataset is publicly available at https://kaggle.com/datasets/5a7ee22cf9dab6c93b55a73f630f6c9b42e936351b0ae98fbae6ddaca7fe248d, which provides a groundwork for advancing research in large language model security.



## **47. Dynamics of Adversarial Attacks on Large Language Model-Based Search Engines**

cs.CL

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2501.00745v2) [paper-pdf](http://arxiv.org/pdf/2501.00745v2)

**Authors**: Xiyang Hu

**Abstract**: The increasing integration of Large Language Model (LLM) based search engines has transformed the landscape of information retrieval. However, these systems are vulnerable to adversarial attacks, especially ranking manipulation attacks, where attackers craft webpage content to manipulate the LLM's ranking and promote specific content, gaining an unfair advantage over competitors. In this paper, we study the dynamics of ranking manipulation attacks. We frame this problem as an Infinitely Repeated Prisoners' Dilemma, where multiple players strategically decide whether to cooperate or attack. We analyze the conditions under which cooperation can be sustained, identifying key factors such as attack costs, discount rates, attack success rates, and trigger strategies that influence player behavior. We identify tipping points in the system dynamics, demonstrating that cooperation is more likely to be sustained when players are forward-looking. However, from a defense perspective, we find that simply reducing attack success probabilities can, paradoxically, incentivize attacks under certain conditions. Furthermore, defensive measures to cap the upper bound of attack success rates may prove futile in some scenarios. These insights highlight the complexity of securing LLM-based systems. Our work provides a theoretical foundation and practical insights for understanding and mitigating their vulnerabilities, while emphasizing the importance of adaptive security strategies and thoughtful ecosystem design.



## **48. S3C2 Summit 2024-09: Industry Secure Software Supply Chain Summit**

cs.CR

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2505.10538v1) [paper-pdf](http://arxiv.org/pdf/2505.10538v1)

**Authors**: Imranur Rahman, Yasemin Acar, Michel Cukier, William Enck, Christian Kastner, Alexandros Kapravelos, Dominik Wermke, Laurie Williams

**Abstract**: While providing economic and software development value, software supply chains are only as strong as their weakest link. Over the past several years, there has been an exponential increase in cyberattacks, specifically targeting vulnerable links in critical software supply chains. These attacks disrupt the day-to-day functioning and threaten the security of nearly everyone on the internet, from billion-dollar companies and government agencies to hobbyist open-source developers. The ever-evolving threat of software supply chain attacks has garnered interest from the software industry and the US government in improving software supply chain security.   On September 20, 2024, three researchers from the NSF-backed Secure Software Supply Chain Center (S3C2) conducted a Secure Software Supply Chain Summit with a diverse set of 12 practitioners from 9 companies. The goals of the Summit were to: (1) to enable sharing between individuals from different companies regarding practical experiences and challenges with software supply chain security, (2) to help form new collaborations, (3) to share our observations from our previous summits with industry, and (4) to learn about practitioners' challenges to inform our future research direction. The summit consisted of discussions of six topics relevant to the companies represented, including updating vulnerable dependencies, component and container choice, malicious commits, building infrastructure, large language models, and reducing entire classes of vulnerabilities.



## **49. MapExplorer: New Content Generation from Low-Dimensional Visualizations**

cs.AI

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2412.18673v2) [paper-pdf](http://arxiv.org/pdf/2412.18673v2)

**Authors**: Xingjian Zhang, Ziyang Xiong, Shixuan Liu, Yutong Xie, Tolga Ergen, Dongsub Shim, Hua Xu, Honglak Lee, Qiaozhu Me

**Abstract**: Low-dimensional visualizations, or "projection maps," are widely used in scientific and creative domains to interpret large-scale and complex datasets. These visualizations not only aid in understanding existing knowledge spaces but also implicitly guide exploration into unknown areas. Although techniques such as t-SNE and UMAP can generate these maps, there exists no systematic method for leveraging them to generate new content. To address this, we introduce MapExplorer, a novel knowledge discovery task that translates coordinates within any projection map into coherent, contextually aligned textual content. This allows users to interactively explore and uncover insights embedded in the maps. To evaluate the performance of MapExplorer methods, we propose Atometric, a fine-grained metric inspired by ROUGE that quantifies logical coherence and alignment between generated and reference text. Experiments on diverse datasets demonstrate the versatility of MapExplorer in generating scientific hypotheses, crafting synthetic personas, and devising strategies for attacking large language models-even with simple baseline methods. By bridging visualization and generation, our work highlights the potential of MapExplorer to enable intuitive human-AI collaboration in large-scale data exploration.



## **50. Scaling Laws for Black box Adversarial Attacks**

cs.LG

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2411.16782v2) [paper-pdf](http://arxiv.org/pdf/2411.16782v2)

**Authors**: Chuan Liu, Huanran Chen, Yichi Zhang, Yinpeng Dong, Jun Zhu

**Abstract**: Adversarial examples usually exhibit good cross-model transferability, enabling attacks on black-box models with limited information about their architectures and parameters, which are highly threatening in commercial black-box scenarios. Model ensembling is an effective strategy to improve the transferability of adversarial examples by attacking multiple surrogate models. However, since prior studies usually adopt few models in the ensemble, there remains an open question of whether scaling the number of models can further improve black-box attacks. Inspired by the scaling law of large foundation models, we investigate the scaling laws of black-box adversarial attacks in this work. Through theoretical analysis and empirical evaluations, we conclude with clear scaling laws that using more surrogate models enhances adversarial transferability. Comprehensive experiments verify the claims on standard image classifiers, diverse defended models and multimodal large language models using various adversarial attack methods. Specifically, by scaling law, we achieve 90%+ transfer attack success rate on even proprietary models like GPT-4o. Further visualization indicates that there is also a scaling law on the interpretability and semantics of adversarial perturbations.



