# Latest Large Language Model Attack Papers
**update at 2025-05-03 15:41:28**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Can Differentially Private Fine-tuning LLMs Protect Against Privacy Attacks?**

cs.CR

accepted by DBSec25

**SubmitDate**: 2025-05-01    [abs](http://arxiv.org/abs/2504.21036v2) [paper-pdf](http://arxiv.org/pdf/2504.21036v2)

**Authors**: Hao Du, Shang Liu, Yang Cao

**Abstract**: Fine-tuning large language models (LLMs) has become an essential strategy for adapting them to specialized tasks; however, this process introduces significant privacy challenges, as sensitive training data may be inadvertently memorized and exposed. Although differential privacy (DP) offers strong theoretical guarantees against such leakage, its empirical privacy effectiveness on LLMs remains unclear, especially under different fine-tuning methods. In this paper, we systematically investigate the impact of DP across fine-tuning methods and privacy budgets, using both data extraction and membership inference attacks to assess empirical privacy risks. Our main findings are as follows: (1) Differential privacy reduces model utility, but its impact varies significantly across different fine-tuning methods. (2) Without DP, the privacy risks of models fine-tuned with different approaches differ considerably. (3) When DP is applied, even a relatively high privacy budget can substantially lower privacy risk. (4) The privacy-utility trade-off under DP training differs greatly among fine-tuning methods, with some methods being unsuitable for DP due to severe utility degradation. Our results provide practical guidance for privacy-conscious deployment of LLMs and pave the way for future research on optimizing the privacy-utility trade-off in fine-tuning methodologies.



## **2. Stochastic Subspace Descent Accelerated via Bi-fidelity Line Search**

cs.LG

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2505.00162v1) [paper-pdf](http://arxiv.org/pdf/2505.00162v1)

**Authors**: Nuojin Cheng, Alireza Doostan, Stephen Becker

**Abstract**: Efficient optimization remains a fundamental challenge across numerous scientific and engineering domains, especially when objective function and gradient evaluations are computationally expensive. While zeroth-order optimization methods offer effective approaches when gradients are inaccessible, their practical performance can be limited by the high cost associated with function queries. This work introduces the bi-fidelity stochastic subspace descent (BF-SSD) algorithm, a novel zeroth-order optimization method designed to reduce this computational burden. BF-SSD leverages a bi-fidelity framework, constructing a surrogate model from a combination of computationally inexpensive low-fidelity (LF) and accurate high-fidelity (HF) function evaluations. This surrogate model facilitates an efficient backtracking line search for step size selection, for which we provide theoretical convergence guarantees under standard assumptions. We perform a comprehensive empirical evaluation of BF-SSD across four distinct problems: a synthetic optimization benchmark, dual-form kernel ridge regression, black-box adversarial attacks on machine learning models, and transformer-based black-box language model fine-tuning. Numerical results demonstrate that BF-SSD consistently achieves superior optimization performance while requiring significantly fewer HF function evaluations compared to relevant baseline methods. This study highlights the efficacy of integrating bi-fidelity strategies within zeroth-order optimization, positioning BF-SSD as a promising and computationally efficient approach for tackling large-scale, high-dimensional problems encountered in various real-world applications.



## **3. Can We Trust Embodied Agents? Exploring Backdoor Attacks against Embodied LLM-based Decision-Making Systems**

cs.CR

Accepted paper at ICLR 2025, 31 pages, including main paper,  references, and appendix

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2405.20774v3) [paper-pdf](http://arxiv.org/pdf/2405.20774v3)

**Authors**: Ruochen Jiao, Shaoyuan Xie, Justin Yue, Takami Sato, Lixu Wang, Yixuan Wang, Qi Alfred Chen, Qi Zhu

**Abstract**: Large Language Models (LLMs) have shown significant promise in real-world decision-making tasks for embodied artificial intelligence, especially when fine-tuned to leverage their inherent common sense and reasoning abilities while being tailored to specific applications. However, this fine-tuning process introduces considerable safety and security vulnerabilities, especially in safety-critical cyber-physical systems. In this work, we propose the first comprehensive framework for Backdoor Attacks against LLM-based Decision-making systems (BALD) in embodied AI, systematically exploring the attack surfaces and trigger mechanisms. Specifically, we propose three distinct attack mechanisms: word injection, scenario manipulation, and knowledge injection, targeting various components in the LLM-based decision-making pipeline. We perform extensive experiments on representative LLMs (GPT-3.5, LLaMA2, PaLM2) in autonomous driving and home robot tasks, demonstrating the effectiveness and stealthiness of our backdoor triggers across various attack channels, with cases like vehicles accelerating toward obstacles and robots placing knives on beds. Our word and knowledge injection attacks achieve nearly 100% success rate across multiple models and datasets while requiring only limited access to the system. Our scenario manipulation attack yields success rates exceeding 65%, reaching up to 90%, and does not require any runtime system intrusion. We also assess the robustness of these attacks against defenses, revealing their resilience. Our findings highlight critical security vulnerabilities in embodied LLM systems and emphasize the urgent need for safeguarding these systems to mitigate potential risks.



## **4. XBreaking: Explainable Artificial Intelligence for Jailbreaking LLMs**

cs.CR

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.21700v1) [paper-pdf](http://arxiv.org/pdf/2504.21700v1)

**Authors**: Marco Arazzi, Vignesh Kumar Kembu, Antonino Nocera, Vinod P

**Abstract**: Large Language Models are fundamental actors in the modern IT landscape dominated by AI solutions. However, security threats associated with them might prevent their reliable adoption in critical application scenarios such as government organizations and medical institutions. For this reason, commercial LLMs typically undergo a sophisticated censoring mechanism to eliminate any harmful output they could possibly produce. In response to this, LLM Jailbreaking is a significant threat to such protections, and many previous approaches have already demonstrated its effectiveness across diverse domains. Existing jailbreak proposals mostly adopt a generate-and-test strategy to craft malicious input. To improve the comprehension of censoring mechanisms and design a targeted jailbreak attack, we propose an Explainable-AI solution that comparatively analyzes the behavior of censored and uncensored models to derive unique exploitable alignment patterns. Then, we propose XBreaking, a novel jailbreak attack that exploits these unique patterns to break the security constraints of LLMs by targeted noise injection. Our thorough experimental campaign returns important insights about the censoring mechanisms and demonstrates the effectiveness and performance of our attack.



## **5. Hoist with His Own Petard: Inducing Guardrails to Facilitate Denial-of-Service Attacks on Retrieval-Augmented Generation of LLMs**

cs.CR

11 pages, 6 figures. This work will be submitted to the IEEE for  possible publication

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.21680v1) [paper-pdf](http://arxiv.org/pdf/2504.21680v1)

**Authors**: Pan Suo, Yu-Ming Shang, San-Chuan Guo, Xi Zhang

**Abstract**: Retrieval-Augmented Generation (RAG) integrates Large Language Models (LLMs) with external knowledge bases, improving output quality while introducing new security risks. Existing studies on RAG vulnerabilities typically focus on exploiting the retrieval mechanism to inject erroneous knowledge or malicious texts, inducing incorrect outputs. However, these approaches overlook critical weaknesses within LLMs, leaving important attack vectors unexplored and limiting the scope and efficiency of attacks. In this paper, we uncover a novel vulnerability: the safety guardrails of LLMs, while designed for protection, can also be exploited as an attack vector by adversaries. Building on this vulnerability, we propose MutedRAG, a novel denial-of-service attack that reversely leverages the guardrails of LLMs to undermine the availability of RAG systems. By injecting minimalistic jailbreak texts, such as "\textit{How to build a bomb}", into the knowledge base, MutedRAG intentionally triggers the LLM's safety guardrails, causing the system to reject legitimate queries. Besides, due to the high sensitivity of guardrails, a single jailbreak sample can affect multiple queries, effectively amplifying the efficiency of attacks while reducing their costs. Experimental results on three datasets demonstrate that MutedRAG achieves an attack success rate exceeding 60% in many scenarios, requiring only less than one malicious text to each target query on average. In addition, we evaluate potential defense strategies against MutedRAG, finding that some of current mechanisms are insufficient to mitigate this threat, underscoring the urgent need for more robust solutions.



## **6. Traceback of Poisoning Attacks to Retrieval-Augmented Generation**

cs.CR

Accepted by The Web Conference 2025

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.21668v1) [paper-pdf](http://arxiv.org/pdf/2504.21668v1)

**Authors**: Baolei Zhang, Haoran Xin, Minghong Fang, Zhuqing Liu, Biao Yi, Tong Li, Zheli Liu

**Abstract**: Large language models (LLMs) integrated with retrieval-augmented generation (RAG) systems improve accuracy by leveraging external knowledge sources. However, recent research has revealed RAG's susceptibility to poisoning attacks, where the attacker injects poisoned texts into the knowledge database, leading to attacker-desired responses. Existing defenses, which predominantly focus on inference-time mitigation, have proven insufficient against sophisticated attacks. In this paper, we introduce RAGForensics, the first traceback system for RAG, designed to identify poisoned texts within the knowledge database that are responsible for the attacks. RAGForensics operates iteratively, first retrieving a subset of texts from the database and then utilizing a specially crafted prompt to guide an LLM in detecting potential poisoning texts. Empirical evaluations across multiple datasets demonstrate the effectiveness of RAGForensics against state-of-the-art poisoning attacks. This work pioneers the traceback of poisoned texts in RAG systems, providing a practical and promising defense mechanism to enhance their security.



## **7. Generative AI in Financial Institution: A Global Survey of Opportunities, Threats, and Regulation**

cs.CR

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.21574v1) [paper-pdf](http://arxiv.org/pdf/2504.21574v1)

**Authors**: Bikash Saha, Nanda Rani, Sandeep Kumar Shukla

**Abstract**: Generative Artificial Intelligence (GenAI) is rapidly reshaping the global financial landscape, offering unprecedented opportunities to enhance customer engagement, automate complex workflows, and extract actionable insights from vast financial data. This survey provides an overview of GenAI adoption across the financial ecosystem, examining how banks, insurers, asset managers, and fintech startups worldwide are integrating large language models and other generative tools into their operations. From AI-powered virtual assistants and personalized financial advisory to fraud detection and compliance automation, GenAI is driving innovation across functions. However, this transformation comes with significant cybersecurity and ethical risks. We discuss emerging threats such as AI-generated phishing, deepfake-enabled fraud, and adversarial attacks on AI systems, as well as concerns around bias, opacity, and data misuse. The evolving global regulatory landscape is explored in depth, including initiatives by major financial regulators and international efforts to develop risk-based AI governance. Finally, we propose best practices for secure and responsible adoption - including explainability techniques, adversarial testing, auditability, and human oversight. Drawing from academic literature, industry case studies, and policy frameworks, this chapter offers a perspective on how the financial sector can harness GenAI's transformative potential while navigating the complex risks it introduces.



## **8. Unlocking User-oriented Pages: Intention-driven Black-box Scanner for Real-world Web Applications**

cs.CR

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2504.20801v2) [paper-pdf](http://arxiv.org/pdf/2504.20801v2)

**Authors**: Weizhe Wang, Yao Zhang, Kaitai Liang, Guangquan Xu, Hongpeng Bai, Qingyang Yan, Xi Zheng, Bin Wu

**Abstract**: Black-box scanners have played a significant role in detecting vulnerabilities for web applications. A key focus in current black-box scanning is increasing test coverage (i.e., accessing more web pages). However, since many web applications are user-oriented, some deep pages can only be accessed through complex user interactions, which are difficult to reach by existing black-box scanners. To fill this gap, a key insight is that web pages contain a wealth of semantic information that can aid in understanding potential user intention. Based on this insight, we propose Hoyen, a black-box scanner that uses the Large Language Model to predict user intention and provide guidance for expanding the scanning scope. Hoyen has been rigorously evaluated on 12 popular open-source web applications and compared with 6 representative tools. The results demonstrate that Hoyen performs a comprehensive exploration of web applications, expanding the attack surface while achieving about 2x than the coverage of other scanners on average, with high request accuracy. Furthermore, Hoyen detected over 90% of its requests towards the core functionality of the application, detecting more vulnerabilities than other scanners, including unique vulnerabilities in well-known web applications. Our data/code is available at https://hoyen.tjunsl.com/



## **9. Round Trip Translation Defence against Large Language Model Jailbreaking Attacks**

cs.CL

6 pages, 6 figures

**SubmitDate**: 2025-04-30    [abs](http://arxiv.org/abs/2402.13517v2) [paper-pdf](http://arxiv.org/pdf/2402.13517v2)

**Authors**: Canaan Yung, Hadi Mohaghegh Dolatabadi, Sarah Erfani, Christopher Leckie

**Abstract**: Large language models (LLMs) are susceptible to social-engineered attacks that are human-interpretable but require a high level of comprehension for LLMs to counteract. Existing defensive measures can only mitigate less than half of these attacks at most. To address this issue, we propose the Round Trip Translation (RTT) method, the first algorithm specifically designed to defend against social-engineered attacks on LLMs. RTT paraphrases the adversarial prompt and generalizes the idea conveyed, making it easier for LLMs to detect induced harmful behavior. This method is versatile, lightweight, and transferrable to different LLMs. Our defense successfully mitigated over 70% of Prompt Automatic Iterative Refinement (PAIR) attacks, which is currently the most effective defense to the best of our knowledge. We are also the first to attempt mitigating the MathsAttack and reduced its attack success rate by almost 40%. Our code is publicly available at https://github.com/Cancanxxx/Round_Trip_Translation_Defence   This version of the article has been accepted for publication, after peer review (when applicable) but is not the Version of Record and does not reflect post-acceptance improvements, or any corrections. The Version of Record is available online at: https://doi.org/10.48550/arXiv.2402.13517 Use of this Accepted Version is subject to the publisher's Accepted Manuscript terms of use https://www.springernature.com/gp/open-research/policies/accepted-manuscript-terms



## **10. CachePrune: Neural-Based Attribution Defense Against Indirect Prompt Injection Attacks**

cs.CR

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.21228v1) [paper-pdf](http://arxiv.org/pdf/2504.21228v1)

**Authors**: Rui Wang, Junda Wu, Yu Xia, Tong Yu, Ruiyi Zhang, Ryan Rossi, Lina Yao, Julian McAuley

**Abstract**: Large Language Models (LLMs) are identified as being susceptible to indirect prompt injection attack, where the model undesirably deviates from user-provided instructions by executing tasks injected in the prompt context. This vulnerability stems from LLMs' inability to distinguish between data and instructions within a prompt. In this paper, we propose CachePrune that defends against this attack by identifying and pruning task-triggering neurons from the KV cache of the input prompt context. By pruning such neurons, we encourage the LLM to treat the text spans of input prompt context as only pure data, instead of any indicator of instruction following. These neurons are identified via feature attribution with a loss function induced from an upperbound of the Direct Preference Optimization (DPO) objective. We show that such a loss function enables effective feature attribution with only a few samples. We further improve on the quality of feature attribution, by exploiting an observed triggering effect in instruction following. Our approach does not impose any formatting on the original prompt or introduce extra test-time LLM calls. Experiments show that CachePrune significantly reduces attack success rates without compromising the response quality. Note: This paper aims to defend against indirect prompt injection attacks, with the goal of developing more secure and robust AI systems.



## **11. ACE: A Security Architecture for LLM-Integrated App Systems**

cs.CR

21 pages, 13 figures

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20984v1) [paper-pdf](http://arxiv.org/pdf/2504.20984v1)

**Authors**: Evan Li, Tushin Mallick, Evan Rose, William Robertson, Alina Oprea, Cristina Nita-Rotaru

**Abstract**: LLM-integrated app systems extend the utility of Large Language Models (LLMs) with third-party apps that are invoked by a system LLM using interleaved planning and execution phases to answer user queries. These systems introduce new attack vectors where malicious apps can cause integrity violation of planning or execution, availability breakdown, or privacy compromise during execution.   In this work, we identify new attacks impacting the integrity of planning, as well as the integrity and availability of execution in LLM-integrated apps, and demonstrate them against IsolateGPT, a recent solution designed to mitigate attacks from malicious apps. We propose Abstract-Concrete-Execute (ACE), a new secure architecture for LLM-integrated app systems that provides security guarantees for system planning and execution. Specifically, ACE decouples planning into two phases by first creating an abstract execution plan using only trusted information, and then mapping the abstract plan to a concrete plan using installed system apps. We verify that the plans generated by our system satisfy user-specified secure information flow constraints via static analysis on the structured plan output. During execution, ACE enforces data and capability barriers between apps, and ensures that the execution is conducted according to the trusted abstract plan. We show experimentally that our system is secure against attacks from the INJECAGENT benchmark, a standard benchmark for control flow integrity in the face of indirect prompt injection attacks, and our newly introduced attacks. Our architecture represents a significant advancement towards hardening LLM-based systems containing system facilities of varying levels of trustworthiness.



## **12. Chain-of-Defensive-Thought: Structured Reasoning Elicits Robustness in Large Language Models against Reference Corruption**

cs.CL

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20769v1) [paper-pdf](http://arxiv.org/pdf/2504.20769v1)

**Authors**: Wenxiao Wang, Parsa Hosseini, Soheil Feizi

**Abstract**: Chain-of-thought prompting has demonstrated great success in facilitating the reasoning abilities of large language models. In this work, we explore how these enhanced reasoning abilities can be exploited to improve the robustness of large language models in tasks that are not necessarily reasoning-focused. In particular, we show how a wide range of large language models exhibit significantly improved robustness against reference corruption using a simple method called chain-of-defensive-thought, where only a few exemplars with structured and defensive reasoning are provided as demonstrations. Empirically, the improvements can be astounding, especially given the simplicity and applicability of the method. For example, in the Natural Questions task, the accuracy of GPT-4o degrades from 60% to as low as 3% with standard prompting when 1 out of 10 references provided is corrupted with prompt injection attacks. In contrast, GPT-4o using chain-of-defensive-thought prompting maintains an accuracy of 50%.



## **13. ReCIT: Reconstructing Full Private Data from Gradient in Parameter-Efficient Fine-Tuning of Large Language Models**

cs.CR

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20570v1) [paper-pdf](http://arxiv.org/pdf/2504.20570v1)

**Authors**: Jin Xie, Ruishi He, Songze Li, Xiaojun Jia, Shouling Ji

**Abstract**: Parameter-efficient fine-tuning (PEFT) has emerged as a practical solution for adapting large language models (LLMs) to custom datasets with significantly reduced computational cost. When carrying out PEFT under collaborative learning scenarios (e.g., federated learning), it is often required to exchange model updates (or gradients) across parties. These gradients, even with limited dimensions, can cause severe breach of data privacy. Recent works have shown that both contextual prefixes and personally identifiable information (PII) can be exposed through gradients. However, \emph{simultaneously} and \emph{accurately} recovering both components from the same training instance remains infeasible due to the following challenges: 1) limited number of PEFT parameters; 2) high-dimensional token spaces; and 3) large batch sizes. We propose ReCIT, a novel privacy attack that addresses all challenges, and achieves recovery of \emph{full} private data from PEFT gradients with high fidelity. Specifically, ReCIT proposes to enhance the memorization capability of the pre-trained model through malicious fine-tuning with Personal Notes; ReCIT also proposes a novel filter-based token extraction technique and a token pairing mechanism, to accurately reconstruct tokens from the training sequences with large batch sizes. Extensive evaluations show that ReCIT consistently outperforms state-of-the-art gradient inversion and memorization-based attacks across different PEFT paradigms. It achieves up to 10$\times$ higher PII recovery rates and remains effective across varying batch sizes, especially in settings where prefix reconstruction is intractable for conventional approaches. These findings highlight an urgent need to reassess the privacy guarantees of PEFT, especially in decentralized or shared training environments.



## **14. Token-Efficient Prompt Injection Attack: Provoking Cessation in LLM Reasoning via Adaptive Token Compression**

cs.CR

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20493v1) [paper-pdf](http://arxiv.org/pdf/2504.20493v1)

**Authors**: Yu Cui, Yujun Cai, Yiwei Wang

**Abstract**: While reasoning large language models (LLMs) demonstrate remarkable performance across various tasks, they also contain notable security vulnerabilities. Recent research has uncovered a "thinking-stopped" vulnerability in DeepSeek-R1, where model-generated reasoning tokens can forcibly interrupt the inference process, resulting in empty responses that compromise LLM-integrated applications. However, existing methods triggering this vulnerability require complex mathematical word problems with long prompts--even exceeding 5,000 tokens. To reduce the token cost and formally define this vulnerability, we propose a novel prompt injection attack named "Reasoning Interruption Attack", based on adaptive token compression. We demonstrate that simple standalone arithmetic tasks can effectively trigger this vulnerability, and the prompts based on such tasks exhibit simpler logical structures than mathematical word problems. We develop a systematic approach to efficiently collect attack prompts and an adaptive token compression framework that utilizes LLMs to automatically compress these prompts. Experiments show our compression framework significantly reduces prompt length while maintaining effective attack capabilities. We further investigate the attack's performance via output prefix and analyze the underlying causes of the vulnerability, providing valuable insights for improving security in reasoning LLMs.



## **15. Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction**

cs.CR

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20472v1) [paper-pdf](http://arxiv.org/pdf/2504.20472v1)

**Authors**: Yulin Chen, Haoran Li, Yuan Sui, Yue Liu, Yufei He, Yangqiu Song, Bryan Hooi

**Abstract**: Large language models (LLMs) have demonstrated impressive performance and have come to dominate the field of natural language processing (NLP) across various tasks. However, due to their strong instruction-following capabilities and inability to distinguish between instructions and data content, LLMs are vulnerable to prompt injection attacks. These attacks manipulate LLMs into deviating from the original input instructions and executing maliciously injected instructions within data content, such as web documents retrieved from search engines. Existing defense methods, including prompt-engineering and fine-tuning approaches, typically instruct models to follow the original input instructions while suppressing their tendencies to execute injected instructions. However, our experiments reveal that suppressing instruction-following tendencies is challenging. Through analyzing failure cases, we observe that although LLMs tend to respond to any recognized instructions, they are aware of which specific instructions they are executing and can correctly reference them within the original prompt. Motivated by these findings, we propose a novel defense method that leverages, rather than suppresses, the instruction-following abilities of LLMs. Our approach prompts LLMs to generate responses that include both answers and their corresponding instruction references. Based on these references, we filter out answers not associated with the original input instructions. Comprehensive experiments demonstrate that our method outperforms prompt-engineering baselines and achieves performance comparable to fine-tuning methods, reducing the attack success rate (ASR) to 0 percent in some scenarios. Moreover, our approach has minimal impact on overall utility.



## **16. NeuRel-Attack: Neuron Relearning for Safety Disalignment in Large Language Models**

cs.LG

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.21053v1) [paper-pdf](http://arxiv.org/pdf/2504.21053v1)

**Authors**: Yi Zhou, Wenpeng Xing, Dezhang Kong, Changting Lin, Meng Han

**Abstract**: Safety alignment in large language models (LLMs) is achieved through fine-tuning mechanisms that regulate neuron activations to suppress harmful content. In this work, we propose a novel approach to induce disalignment by identifying and modifying the neurons responsible for safety constraints. Our method consists of three key steps: Neuron Activation Analysis, where we examine activation patterns in response to harmful and harmless prompts to detect neurons that are critical for distinguishing between harmful and harmless inputs; Similarity-Based Neuron Identification, which systematically locates the neurons responsible for safe alignment; and Neuron Relearning for Safety Removal, where we fine-tune these selected neurons to restore the model's ability to generate previously restricted responses. Experimental results demonstrate that our method effectively removes safety constraints with minimal fine-tuning, highlighting a critical vulnerability in current alignment techniques. Our findings underscore the need for robust defenses against adversarial fine-tuning attacks on LLMs.



## **17. Enhancing Leakage Attacks on Searchable Symmetric Encryption Using LLM-Based Synthetic Data Generation**

cs.CR

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.20414v1) [paper-pdf](http://arxiv.org/pdf/2504.20414v1)

**Authors**: Joshua Chiu, Partha Protim Paul, Zahin Wahab

**Abstract**: Searchable Symmetric Encryption (SSE) enables efficient search capabilities over encrypted data, allowing users to maintain privacy while utilizing cloud storage. However, SSE schemes are vulnerable to leakage attacks that exploit access patterns, search frequency, and volume information. Existing studies frequently assume that adversaries possess a substantial fraction of the encrypted dataset to mount effective inference attacks, implying there is a database leakage of such documents, thus, an assumption that may not hold in real-world scenarios. In this work, we investigate the feasibility of enhancing leakage attacks under a more realistic threat model in which adversaries have access to minimal leaked data. We propose a novel approach that leverages large language models (LLMs), specifically GPT-4 variants, to generate synthetic documents that statistically and semantically resemble the real-world dataset of Enron emails. Using the email corpus as a case study, we evaluate the effectiveness of synthetic data generated via random sampling and hierarchical clustering methods on the performance of the SAP (Search Access Pattern) keyword inference attack restricted to token volumes only. Our results demonstrate that, while the choice of LLM has limited effect, increasing dataset size and employing clustering-based generation significantly improve attack accuracy, achieving comparable performance to attacks using larger amounts of real data. We highlight the growing relevance of LLMs in adversarial contexts.



## **18. The Automation Advantage in AI Red Teaming**

cs.CR

15 pages, 6 figures

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.19855v2) [paper-pdf](http://arxiv.org/pdf/2504.19855v2)

**Authors**: Rob Mulla, Ads Dawson, Vincent Abruzzon, Brian Greunke, Nick Landers, Brad Palm, Will Pearce

**Abstract**: This paper analyzes Large Language Model (LLM) security vulnerabilities based on data from Crucible, encompassing 214,271 attack attempts by 1,674 users across 30 LLM challenges. Our findings reveal automated approaches significantly outperform manual techniques (69.5% vs 47.6% success rate), despite only 5.2% of users employing automation. We demonstrate that automated approaches excel in systematic exploration and pattern matching challenges, while manual approaches retain speed advantages in certain creative reasoning scenarios, often solving problems 5x faster when successful. Challenge categories requiring systematic exploration are most effectively targeted through automation, while intuitive challenges sometimes favor manual techniques for time-to-solve metrics. These results illuminate how algorithmic testing is transforming AI red-teaming practices, with implications for both offensive security research and defensive measures. Our analysis suggests optimal security testing combines human creativity for strategy development with programmatic execution for thorough exploration.



## **19. BadMoE: Backdooring Mixture-of-Experts LLMs via Optimizing Routing Triggers and Infecting Dormant Experts**

cs.CR

**SubmitDate**: 2025-04-29    [abs](http://arxiv.org/abs/2504.18598v2) [paper-pdf](http://arxiv.org/pdf/2504.18598v2)

**Authors**: Qingyue Wang, Qi Pang, Xixun Lin, Shuai Wang, Daoyuan Wu

**Abstract**: Mixture-of-Experts (MoE) have emerged as a powerful architecture for large language models (LLMs), enabling efficient scaling of model capacity while maintaining manageable computational costs. The key advantage lies in their ability to route different tokens to different ``expert'' networks within the model, enabling specialization and efficient handling of diverse input. However, the vulnerabilities of MoE-based LLMs still have barely been studied, and the potential for backdoor attacks in this context remains largely unexplored. This paper presents the first backdoor attack against MoE-based LLMs where the attackers poison ``dormant experts'' (i.e., underutilized experts) and activate them by optimizing routing triggers, thereby gaining control over the model's output. We first rigorously prove the existence of a few ``dominating experts'' in MoE models, whose outputs can determine the overall MoE's output. We also show that dormant experts can serve as dominating experts to manipulate model predictions. Accordingly, our attack, namely BadMoE, exploits the unique architecture of MoE models by 1) identifying dormant experts unrelated to the target task, 2) constructing a routing-aware loss to optimize the activation triggers of these experts, and 3) promoting dormant experts to dominating roles via poisoned training data. Extensive experiments show that BadMoE successfully enforces malicious prediction on attackers' target tasks while preserving overall model utility, making it a more potent and stealthy attack than existing methods.



## **20. Leveraging LLM to Strengthen ML-Based Cross-Site Scripting Detection**

cs.CR

This work has been accepted for presentation at the ACM Workshop on  Wireless Security and Machine Learning (WiseML 2025)

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.21045v1) [paper-pdf](http://arxiv.org/pdf/2504.21045v1)

**Authors**: Dennis Miczek, Divyesh Gabbireddy, Suman Saha

**Abstract**: According to the Open Web Application Security Project (OWASP), Cross-Site Scripting (XSS) is a critical security vulnerability. Despite decades of research, XSS remains among the top 10 security vulnerabilities. Researchers have proposed various techniques to protect systems from XSS attacks, with machine learning (ML) being one of the most widely used methods. An ML model is trained on a dataset to identify potential XSS threats, making its effectiveness highly dependent on the size and diversity of the training data. A variation of XSS is obfuscated XSS, where attackers apply obfuscation techniques to alter the code's structure, making it challenging for security systems to detect its malicious intent. Our study's random forest model was trained on traditional (non-obfuscated) XSS data achieved 99.8% accuracy. However, when tested against obfuscated XSS samples, accuracy dropped to 81.9%, underscoring the importance of training ML models with obfuscated data to improve their effectiveness in detecting XSS attacks. A significant challenge is to generate highly complex obfuscated code despite the availability of several public tools. These tools can only produce obfuscation up to certain levels of complexity.   In our proposed system, we fine-tune a Large Language Model (LLM) to generate complex obfuscated XSS payloads automatically. By transforming original XSS samples into diverse obfuscated variants, we create challenging training data for ML model evaluation. Our approach achieved a 99.5% accuracy rate with the obfuscated dataset. We also found that the obfuscated samples generated by the LLMs were 28.1% more complex than those created by other tools, significantly improving the model's ability to handle advanced XSS attacks and making it more effective for real-world application security.



## **21. Exploring the Role of Large Language Models in Cybersecurity: A Systematic Survey**

cs.CR

20 pages, 3 figures

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.15622v2) [paper-pdf](http://arxiv.org/pdf/2504.15622v2)

**Authors**: Shuang Tian, Tao Zhang, Jiqiang Liu, Jiacheng Wang, Xuangou Wu, Xiaoqiang Zhu, Ruichen Zhang, Weiting Zhang, Zhenhui Yuan, Shiwen Mao, Dong In Kim

**Abstract**: With the rapid development of technology and the acceleration of digitalisation, the frequency and complexity of cyber security threats are increasing. Traditional cybersecurity approaches, often based on static rules and predefined scenarios, are struggling to adapt to the rapidly evolving nature of modern cyberattacks. There is an urgent need for more adaptive and intelligent defence strategies. The emergence of Large Language Model (LLM) provides an innovative solution to cope with the increasingly severe cyber threats, and its potential in analysing complex attack patterns, predicting threats and assisting real-time response has attracted a lot of attention in the field of cybersecurity, and exploring how to effectively use LLM to defend against cyberattacks has become a hot topic in the current research field. This survey examines the applications of LLM from the perspective of the cyber attack lifecycle, focusing on the three phases of defense reconnaissance, foothold establishment, and lateral movement, and it analyzes the potential of LLMs in Cyber Threat Intelligence (CTI) tasks. Meanwhile, we investigate how LLM-based security solutions are deployed and applied in different network scenarios. It also summarizes the internal and external risk issues faced by LLM during its application. Finally, this survey also points out the facing risk issues and possible future research directions in this domain.



## **22. Les Dissonances: Cross-Tool Harvesting and Polluting in Multi-Tool Empowered LLM Agents**

cs.CR

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.03111v2) [paper-pdf](http://arxiv.org/pdf/2504.03111v2)

**Authors**: Zichuan Li, Jian Cui, Xiaojing Liao, Luyi Xing

**Abstract**: Large Language Model (LLM) agents are autonomous systems powered by LLMs, capable of reasoning and planning to solve problems by leveraging a set of tools. However, the integration of multi-tool capabilities in LLM agents introduces challenges in securely managing tools, ensuring their compatibility, handling dependency relationships, and protecting control flows within LLM agent workflows. In this paper, we present the first systematic security analysis of task control flows in multi-tool-enabled LLM agents. We identify a novel threat, Cross-Tool Harvesting and Polluting (XTHP), which includes multiple attack vectors to first hijack the normal control flows of agent tasks, and then collect and pollute confidential or private information within LLM agent systems. To understand the impact of this threat, we developed Chord, a dynamic scanning tool designed to automatically detect real-world agent tools susceptible to XTHP attacks. Our evaluation of 66 real-world tools from the repositories of two major LLM agent development frameworks, LangChain and LlamaIndex, revealed a significant security concern: 75\% are vulnerable to XTHP attacks, highlighting the prevalence of this threat.



## **23. SoK: Knowledge is All You Need: Accelerating Last Mile Delivery for Automated Provenance-based Intrusion Detection with LLMs**

cs.CR

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2503.03108v2) [paper-pdf](http://arxiv.org/pdf/2503.03108v2)

**Authors**: Wenrui Cheng, Tiantian Zhu, Chunlin Xiong, Haofei Sun, Zijun Wang, Shunan Jing, Mingqi Lv, Yan Chen

**Abstract**: Recently, provenance-based intrusion detection systems (PIDSes) have been widely proposed for endpoint threat analysis. However, due to the lack of systematic integration and utilization of knowledge, existing PIDSes still require significant manual intervention for practical deployment, making full automation challenging. This paper presents a disruptive innovation by categorizing PIDSes according to the types of knowledge they utilize. In response to the prevalent issue of ``knowledge silos problem'' in existing research, we introduce a novel knowledge-driven provenance-based intrusion detection framework, powered by large language models (LLMs). We also present OmniSec, a best practice system built upon this framework. By integrating attack representation knowledge, threat intelligence knowledge, and benign behavior knowledge, OmniSec outperforms the state-of-the-art approaches on public benchmark datasets. OmniSec is available online at https://anonymous.4open.science/r/PIDS-with-LLM-613B.



## **24. LLMPot: Dynamically Configured LLM-based Honeypot for Industrial Protocol and Physical Process Emulation**

cs.CR

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2405.05999v2) [paper-pdf](http://arxiv.org/pdf/2405.05999v2)

**Authors**: Christoforos Vasilatos, Dunia J. Mahboobeh, Hithem Lamri, Manaar Alam, Michail Maniatakos

**Abstract**: Industrial Control Systems (ICS) are extensively used in critical infrastructures ensuring efficient, reliable, and continuous operations. However, their increasing connectivity and addition of advanced features make them vulnerable to cyber threats, potentially leading to severe disruptions in essential services. In this context, honeypots play a vital role by acting as decoy targets within ICS networks, or on the Internet, helping to detect, log, analyze, and develop mitigations for ICS-specific cyber threats. Deploying ICS honeypots, however, is challenging due to the necessity of accurately replicating industrial protocols and device characteristics, a crucial requirement for effectively mimicking the unique operational behavior of different industrial systems. Moreover, this challenge is compounded by the significant manual effort required in also mimicking the control logic the PLC would execute, in order to capture attacker traffic aiming to disrupt critical infrastructure operations. In this paper, we propose LLMPot, a novel approach for designing honeypots in ICS networks harnessing the potency of Large Language Models (LLMs). LLMPot aims to automate and optimize the creation of realistic honeypots with vendor-agnostic configurations, and for any control logic, aiming to eliminate the manual effort and specialized knowledge traditionally required in this domain. We conducted extensive experiments focusing on a wide array of parameters, demonstrating that our LLM-based approach can effectively create honeypot devices implementing different industrial protocols and diverse control logic.



## **25. Mapping the Italian Telegram Ecosystem**

cs.SI

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.19594v1) [paper-pdf](http://arxiv.org/pdf/2504.19594v1)

**Authors**: Lorenzo Alvisi, Serena Tardelli, Maurizio Tesconi

**Abstract**: Telegram has become a major space for political discourse and alternative media. However, its lack of moderation allows misinformation, extremism, and toxicity to spread. While prior research focused on these particular phenomena or topics, these have mostly been examined separately, and a broader understanding of the Telegram ecosystem is still missing. In this work, we fill this gap by conducting a large-scale analysis of the Italian Telegram sphere, leveraging a dataset of 186 million messages from 13,151 chats collected in 2023. Using network analysis, Large Language Models, and toxicity detection tools, we examine how different thematic communities form, align ideologically, and engage in harmful discourse within the Italian cultural context. Results show strong thematic and ideological homophily. We also identify mixed ideological communities where far-left and far-right rhetoric coexist on particular geopolitical issues. Beyond political analysis, we find that toxicity, rather than being isolated in a few extreme chats, appears widely normalized within highly toxic communities. Moreover, we find that Italian discourse primarily targets Black people, Jews, and gay individuals independently of the topic. Finally, we uncover common trend of intra-national hostility, where Italians often attack other Italians, reflecting regional and intra-regional cultural conflicts that can be traced back to old historical divisions. This study provides the first large-scale mapping of the Italian Telegram ecosystem, offering insights into ideological interactions, toxicity, and identity-targets of hate and contributing to research on online toxicity across different cultural and linguistic contexts on Telegram.



## **26. Prefill-Based Jailbreak: A Novel Approach of Bypassing LLM Safety Boundary**

cs.CR

**SubmitDate**: 2025-04-28    [abs](http://arxiv.org/abs/2504.21038v1) [paper-pdf](http://arxiv.org/pdf/2504.21038v1)

**Authors**: Yakai Li, Jiekang Hu, Weiduan Sang, Luping Ma, Jing Xie, Weijuan Zhang, Aimin Yu, Shijie Zhao, Qingjia Huang, Qihang Zhou

**Abstract**: Large Language Models (LLMs) are designed to generate helpful and safe content. However, adversarial attacks, commonly referred to as jailbreak, can bypass their safety protocols, prompting LLMs to generate harmful content or reveal sensitive data. Consequently, investigating jailbreak methodologies is crucial for exposing systemic vulnerabilities within LLMs, ultimately guiding the continuous implementation of security enhancements by developers. In this paper, we introduce a novel jailbreak attack method that leverages the prefilling feature of LLMs, a feature designed to enhance model output constraints. Unlike traditional jailbreak methods, the proposed attack circumvents LLMs' safety mechanisms by directly manipulating the probability distribution of subsequent tokens, thereby exerting control over the model's output. We propose two attack variants: Static Prefilling (SP), which employs a universal prefill text, and Optimized Prefilling (OP), which iteratively optimizes the prefill text to maximize the attack success rate. Experiments on six state-of-the-art LLMs using the AdvBench benchmark validate the effectiveness of our method and demonstrate its capability to substantially enhance attack success rates when combined with existing jailbreak approaches. The OP method achieved attack success rates of up to 99.82% on certain models, significantly outperforming baseline methods. This work introduces a new jailbreak attack method in LLMs, emphasizing the need for robust content validation mechanisms to mitigate the adversarial exploitation of prefilling features. All code and data used in this paper are publicly available.



## **27. Exposing Privacy Gaps: Membership Inference Attack on Preference Data for LLM Alignment**

cs.AI

**SubmitDate**: 2025-04-27    [abs](http://arxiv.org/abs/2407.06443v2) [paper-pdf](http://arxiv.org/pdf/2407.06443v2)

**Authors**: Qizhang Feng, Siva Rajesh Kasa, Santhosh Kumar Kasa, Hyokun Yun, Choon Hui Teo, Sravan Babu Bodapati

**Abstract**: Large Language Models (LLMs) have seen widespread adoption due to their remarkable natural language capabilities. However, when deploying them in real-world settings, it is important to align LLMs to generate texts according to acceptable human standards. Methods such as Proximal Policy Optimization (PPO) and Direct Preference Optimization (DPO) have enabled significant progress in refining LLMs using human preference data. However, the privacy concerns inherent in utilizing such preference data have yet to be adequately studied. In this paper, we investigate the vulnerability of LLMs aligned using two widely used methods - DPO and PPO - to membership inference attacks (MIAs). Our study has two main contributions: first, we theoretically motivate that DPO models are more vulnerable to MIA compared to PPO models; second, we introduce a novel reference-based attack framework specifically for analyzing preference data called PREMIA (\uline{Pre}ference data \uline{MIA}). Using PREMIA and existing baselines we empirically show that DPO models have a relatively heightened vulnerability towards MIA.



## **28. Graph of Attacks: Improved Black-Box and Interpretable Jailbreaks for LLMs**

cs.CL

19 pages, 1 figure, 6 tables

**SubmitDate**: 2025-04-26    [abs](http://arxiv.org/abs/2504.19019v1) [paper-pdf](http://arxiv.org/pdf/2504.19019v1)

**Authors**: Mohammad Akbar-Tajari, Mohammad Taher Pilehvar, Mohammad Mahmoody

**Abstract**: The challenge of ensuring Large Language Models (LLMs) align with societal standards is of increasing interest, as these models are still prone to adversarial jailbreaks that bypass their safety mechanisms. Identifying these vulnerabilities is crucial for enhancing the robustness of LLMs against such exploits. We propose Graph of ATtacks (GoAT), a method for generating adversarial prompts to test the robustness of LLM alignment using the Graph of Thoughts framework [Besta et al., 2024]. GoAT excels at generating highly effective jailbreak prompts with fewer queries to the victim model than state-of-the-art attacks, achieving up to five times better jailbreak success rate against robust models like Llama. Notably, GoAT creates high-quality, human-readable prompts without requiring access to the targeted model's parameters, making it a black-box attack. Unlike approaches constrained by tree-based reasoning, GoAT's reasoning is based on a more intricate graph structure. By making simultaneous attack paths aware of each other's progress, this dynamic framework allows a deeper integration and refinement of reasoning paths, significantly enhancing the collaborative exploration of adversarial vulnerabilities in LLMs. At a technical level, GoAT starts with a graph structure and iteratively refines it by combining and improving thoughts, enabling synergy between different thought paths. The code for our implementation can be found at: https://github.com/GoAT-pydev/Graph_of_Attacks.



## **29. ThreMoLIA: Threat Modeling of Large Language Model-Integrated Applications**

cs.CR

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.18369v1) [paper-pdf](http://arxiv.org/pdf/2504.18369v1)

**Authors**: Felix Viktor Jedrzejewski, Davide Fucci, Oleksandr Adamov

**Abstract**: Large Language Models (LLMs) are currently being integrated into industrial software applications to help users perform more complex tasks in less time. However, these LLM-Integrated Applications (LIA) expand the attack surface and introduce new kinds of threats. Threat modeling is commonly used to identify these threats and suggest mitigations. However, it is a time-consuming practice that requires the involvement of a security practitioner. Our goals are to 1) provide a method for performing threat modeling for LIAs early in their lifecycle, (2) develop a threat modeling tool that integrates existing threat models, and (3) ensure high-quality threat modeling. To achieve the goals, we work in collaboration with our industry partner. Our proposed way of performing threat modeling will benefit industry by requiring fewer security experts' participation and reducing the time spent on this activity. Our proposed tool combines LLMs and Retrieval Augmented Generation (RAG) and uses sources such as existing threat models and application architecture repositories to continuously create and update threat models. We propose to evaluate the tool offline -- i.e., using benchmarking -- and online with practitioners in the field. We conducted an early evaluation using ChatGPT on a simple LIA and obtained results that encouraged us to proceed with our research efforts.



## **30. Manipulating Multimodal Agents via Cross-Modal Prompt Injection**

cs.CV

17 pages, 5 figures

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.14348v3) [paper-pdf](http://arxiv.org/pdf/2504.14348v3)

**Authors**: Le Wang, Zonghao Ying, Tianyuan Zhang, Siyuan Liang, Shengshan Hu, Mingchuan Zhang, Aishan Liu, Xianglong Liu

**Abstract**: The emergence of multimodal large language models has redefined the agent paradigm by integrating language and vision modalities with external data sources, enabling agents to better interpret human instructions and execute increasingly complex tasks. However, in this work, we identify a critical yet previously overlooked security vulnerability in multimodal agents: cross-modal prompt injection attacks. To exploit this vulnerability, we propose CrossInject, a novel attack framework in which attackers embed adversarial perturbations across multiple modalities to align with target malicious content, allowing external instructions to hijack the agent's decision-making process and execute unauthorized tasks. Our approach consists of two key components. First, we introduce Visual Latent Alignment, where we optimize adversarial features to the malicious instructions in the visual embedding space based on a text-to-image generative model, ensuring that adversarial images subtly encode cues for malicious task execution. Subsequently, we present Textual Guidance Enhancement, where a large language model is leveraged to infer the black-box defensive system prompt through adversarial meta prompting and generate an malicious textual command that steers the agent's output toward better compliance with attackers' requests. Extensive experiments demonstrate that our method outperforms existing injection attacks, achieving at least a +26.4% increase in attack success rates across diverse tasks. Furthermore, we validate our attack's effectiveness in real-world multimodal autonomous agents, highlighting its potential implications for safety-critical applications.



## **31. NoEsis: Differentially Private Knowledge Transfer in Modular LLM Adaptation**

cs.CR

ICLR 2025 MCDC workshop

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.18147v1) [paper-pdf](http://arxiv.org/pdf/2504.18147v1)

**Authors**: Rob Romijnders, Stefanos Laskaridis, Ali Shahin Shamsabadi, Hamed Haddadi

**Abstract**: Large Language Models (LLM) are typically trained on vast amounts of data from various sources. Even when designed modularly (e.g., Mixture-of-Experts), LLMs can leak privacy on their sources. Conversely, training such models in isolation arguably prohibits generalization. To this end, we propose a framework, NoEsis, which builds upon the desired properties of modularity, privacy, and knowledge transfer. NoEsis integrates differential privacy with a hybrid two-staged parameter-efficient fine-tuning that combines domain-specific low-rank adapters, acting as experts, with common prompt tokens, acting as a knowledge-sharing backbone. Results from our evaluation on CodeXGLUE showcase that NoEsis can achieve provable privacy guarantees with tangible knowledge transfer across domains, and empirically show protection against Membership Inference Attacks. Finally, on code completion tasks, NoEsis bridges at least 77% of the accuracy gap between the non-shared and the non-private baseline.



## **32. Automating Function-Level TARA for Automotive Full-Lifecycle Security**

cs.CR

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.18083v1) [paper-pdf](http://arxiv.org/pdf/2504.18083v1)

**Authors**: Yuqiao Yang, Yongzhao Zhang, Wenhao Liu, Jun Li, Pengtao Shi, DingYu Zhong, Jie Yang, Ting Chen, Sheng Cao, Yuntao Ren, Yongyue Wu, Xiaosong Zhang

**Abstract**: As modern vehicles evolve into intelligent and connected systems, their growing complexity introduces significant cybersecurity risks. Threat Analysis and Risk Assessment (TARA) has therefore become essential for managing these risks under mandatory regulations. However, existing TARA automation methods rely on static threat libraries, limiting their utility in the detailed, function-level analyses demanded by industry. This paper introduces DefenseWeaver, the first system that automates function-level TARA using component-specific details and large language models (LLMs). DefenseWeaver dynamically generates attack trees and risk evaluations from system configurations described in an extended OpenXSAM++ format, then employs a multi-agent framework to coordinate specialized LLM roles for more robust analysis. To further adapt to evolving threats and diverse standards, DefenseWeaver incorporates Low-Rank Adaptation (LoRA) fine-tuning and Retrieval-Augmented Generation (RAG) with expert-curated TARA reports. We validated DefenseWeaver through deployment in four automotive security projects, where it identified 11 critical attack paths, verified through penetration testing, and subsequently reported and remediated by the relevant automakers and suppliers. Additionally, DefenseWeaver demonstrated cross-domain adaptability, successfully applying to unmanned aerial vehicles (UAVs) and marine navigation systems. In comparison to human experts, DefenseWeaver outperformed manual attack tree generation across six assessment scenarios. Integrated into commercial cybersecurity platforms such as UAES and Xiaomi, DefenseWeaver has generated over 8,200 attack trees. These results highlight its ability to significantly reduce processing time, and its scalability and transformative impact on cybersecurity across industries.



## **33. DREAM: Disentangling Risks to Enhance Safety Alignment in Multimodal Large Language Models**

cs.CL

[NAACL 2025] The first four authors contribute equally, 23 pages,  repo at https://github.com/Kizna1ver/DREAM

**SubmitDate**: 2025-04-25    [abs](http://arxiv.org/abs/2504.18053v1) [paper-pdf](http://arxiv.org/pdf/2504.18053v1)

**Authors**: Jianyu Liu, Hangyu Guo, Ranjie Duan, Xingyuan Bu, Yancheng He, Shilong Li, Hui Huang, Jiaheng Liu, Yucheng Wang, Chenchen Jing, Xingwei Qu, Xiao Zhang, Yingshui Tan, Yanan Wu, Jihao Gu, Yangguang Li, Jianke Zhu

**Abstract**: Multimodal Large Language Models (MLLMs) pose unique safety challenges due to their integration of visual and textual data, thereby introducing new dimensions of potential attacks and complex risk combinations. In this paper, we begin with a detailed analysis aimed at disentangling risks through step-by-step reasoning within multimodal inputs. We find that systematic multimodal risk disentanglement substantially enhances the risk awareness of MLLMs. Via leveraging the strong discriminative abilities of multimodal risk disentanglement, we further introduce \textbf{DREAM} (\textit{\textbf{D}isentangling \textbf{R}isks to \textbf{E}nhance Safety \textbf{A}lignment in \textbf{M}LLMs}), a novel approach that enhances safety alignment in MLLMs through supervised fine-tuning and iterative Reinforcement Learning from AI Feedback (RLAIF). Experimental results show that DREAM significantly boosts safety during both inference and training phases without compromising performance on normal tasks (namely oversafety), achieving a 16.17\% improvement in the SIUO safe\&effective score compared to GPT-4V. The data and code are available at https://github.com/Kizna1ver/DREAM.



## **34. Beyond Public Access in LLM Pre-Training Data**

cs.CL

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2505.00020v1) [paper-pdf](http://arxiv.org/pdf/2505.00020v1)

**Authors**: Sruly Rosenblat, Tim O'Reilly, Ilan Strauss

**Abstract**: Using a legally obtained dataset of 34 copyrighted O'Reilly Media books, we apply the DE-COP membership inference attack method to investigate whether OpenAI's large language models were trained on copyrighted content without consent. Our AUROC scores show that GPT-4o, OpenAI's more recent and capable model, demonstrates strong recognition of paywalled O'Reilly book content (AUROC = 82\%), compared to OpenAI's earlier model GPT-3.5 Turbo. In contrast, GPT-3.5 Turbo shows greater relative recognition of publicly accessible O'Reilly book samples. GPT-4o Mini, as a much smaller model, shows no knowledge of public or non-public O'Reilly Media content when tested (AUROC $\approx$ 50\%). Testing multiple models, with the same cutoff date, helps us account for potential language shifts over time that might bias our findings. These results highlight the urgent need for increased corporate transparency regarding pre-training data sources as a means to develop formal licensing frameworks for AI content training



## **35. Unified Attacks to Large Language Model Watermarks: Spoofing and Scrubbing in Unauthorized Knowledge Distillation**

cs.CL

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17480v1) [paper-pdf](http://arxiv.org/pdf/2504.17480v1)

**Authors**: Xin Yi, Shunfan Zhengc, Linlin Wanga, Xiaoling Wang, Liang He

**Abstract**: Watermarking has emerged as a critical technique for combating misinformation and protecting intellectual property in large language models (LLMs). A recent discovery, termed watermark radioactivity, reveals that watermarks embedded in teacher models can be inherited by student models through knowledge distillation. On the positive side, this inheritance allows for the detection of unauthorized knowledge distillation by identifying watermark traces in student models. However, the robustness of watermarks against scrubbing attacks and their unforgeability in the face of spoofing attacks under unauthorized knowledge distillation remain largely unexplored. Existing watermark attack methods either assume access to model internals or fail to simultaneously support both scrubbing and spoofing attacks. In this work, we propose Contrastive Decoding-Guided Knowledge Distillation (CDG-KD), a unified framework that enables bidirectional attacks under unauthorized knowledge distillation. Our approach employs contrastive decoding to extract corrupted or amplified watermark texts via comparing outputs from the student model and weakly watermarked references, followed by bidirectional distillation to train new student models capable of watermark removal and watermark forgery, respectively. Extensive experiments show that CDG-KD effectively performs attacks while preserving the general performance of the distilled model. Our findings underscore critical need for developing watermarking schemes that are robust and unforgeable.



## **36. JailbreakLens: Interpreting Jailbreak Mechanism in the Lens of Representation and Circuit**

cs.CR

17 pages, 11 figures

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2411.11114v2) [paper-pdf](http://arxiv.org/pdf/2411.11114v2)

**Authors**: Zeqing He, Zhibo Wang, Zhixuan Chu, Huiyu Xu, Wenhui Zhang, Qinglong Wang, Rui Zheng

**Abstract**: Despite the outstanding performance of Large language Models (LLMs) in diverse tasks, they are vulnerable to jailbreak attacks, wherein adversarial prompts are crafted to bypass their security mechanisms and elicit unexpected responses. Although jailbreak attacks are prevalent, the understanding of their underlying mechanisms remains limited. Recent studies have explained typical jailbreaking behavior (e.g., the degree to which the model refuses to respond) of LLMs by analyzing representation shifts in their latent space caused by jailbreak prompts or identifying key neurons that contribute to the success of jailbreak attacks. However, these studies neither explore diverse jailbreak patterns nor provide a fine-grained explanation from the failure of circuit to the changes of representational, leaving significant gaps in uncovering the jailbreak mechanism. In this paper, we propose JailbreakLens, an interpretation framework that analyzes jailbreak mechanisms from both representation (which reveals how jailbreaks alter the model's harmfulness perception) and circuit perspectives~(which uncovers the causes of these deceptions by identifying key circuits contributing to the vulnerability), tracking their evolution throughout the entire response generation process. We then conduct an in-depth evaluation of jailbreak behavior on five mainstream LLMs under seven jailbreak strategies. Our evaluation reveals that jailbreak prompts amplify components that reinforce affirmative responses while suppressing those that produce refusal. This manipulation shifts model representations toward safe clusters to deceive the LLM, leading it to provide detailed responses instead of refusals. Notably, we find a strong and consistent correlation between representation deception and activation shift of key circuits across diverse jailbreak methods and multiple LLMs.



## **37. GraphRAG under Fire**

cs.LG

13 pages

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2501.14050v2) [paper-pdf](http://arxiv.org/pdf/2501.14050v2)

**Authors**: Jiacheng Liang, Yuhui Wang, Changjiang Li, Rongyi Zhu, Tanqiu Jiang, Neil Gong, Ting Wang

**Abstract**: GraphRAG advances retrieval-augmented generation (RAG) by structuring external knowledge as multi-scale knowledge graphs, enabling language models to integrate both broad context and granular details in their generation. While GraphRAG has demonstrated success across domains, its security implications remain largely unexplored. To bridge this gap, this work examines GraphRAG's vulnerability to poisoning attacks, uncovering an intriguing security paradox: compared to conventional RAG, GraphRAG's graph-based indexing and retrieval enhance resilience against simple poisoning attacks; yet, the same features also create new attack surfaces. We present GRAGPoison, a novel attack that exploits shared relations in the underlying knowledge graph to craft poisoning text capable of compromising multiple queries simultaneously. GRAGPoison employs three key strategies: i) relation injection to introduce false knowledge, ii) relation enhancement to amplify poisoning influence, and iii) narrative generation to embed malicious content within coherent text. Empirical evaluation across diverse datasets and models shows that GRAGPoison substantially outperforms existing attacks in terms of effectiveness (up to 98\% success rate) and scalability (using less than 68\% poisoning text) on various GraphRAG-based systems. We also explore potential defensive measures and their limitations, identifying promising directions for future research.



## **38. CheatAgent: Attacking LLM-Empowered Recommender Systems via LLM Agent**

cs.CR

Accepted by KDD 2024;

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.13192v2) [paper-pdf](http://arxiv.org/pdf/2504.13192v2)

**Authors**: Liang-bo Ning, Shijie Wang, Wenqi Fan, Qing Li, Xin Xu, Hao Chen, Feiran Huang

**Abstract**: Recently, Large Language Model (LLM)-empowered recommender systems (RecSys) have brought significant advances in personalized user experience and have attracted considerable attention. Despite the impressive progress, the research question regarding the safety vulnerability of LLM-empowered RecSys still remains largely under-investigated. Given the security and privacy concerns, it is more practical to focus on attacking the black-box RecSys, where attackers can only observe the system's inputs and outputs. However, traditional attack approaches employing reinforcement learning (RL) agents are not effective for attacking LLM-empowered RecSys due to the limited capabilities in processing complex textual inputs, planning, and reasoning. On the other hand, LLMs provide unprecedented opportunities to serve as attack agents to attack RecSys because of their impressive capability in simulating human-like decision-making processes. Therefore, in this paper, we propose a novel attack framework called CheatAgent by harnessing the human-like capabilities of LLMs, where an LLM-based agent is developed to attack LLM-Empowered RecSys. Specifically, our method first identifies the insertion position for maximum impact with minimal input modification. After that, the LLM agent is designed to generate adversarial perturbations to insert at target positions. To further improve the quality of generated perturbations, we utilize the prompt tuning technique to improve attacking strategies via feedback from the victim RecSys iteratively. Extensive experiments across three real-world datasets demonstrate the effectiveness of our proposed attacking method.



## **39. Automatically Generating Rules of Malicious Software Packages via Large Language Model**

cs.SE

14 pages, 11 figures

**SubmitDate**: 2025-04-24    [abs](http://arxiv.org/abs/2504.17198v1) [paper-pdf](http://arxiv.org/pdf/2504.17198v1)

**Authors**: XiangRui Zhang, HaoYu Chen, Yongzhong He, Wenjia Niu, Qiang Li

**Abstract**: Today's security tools predominantly rely on predefined rules crafted by experts, making them poorly adapted to the emergence of software supply chain attacks. To tackle this limitation, we propose a novel tool, RuleLLM, which leverages large language models (LLMs) to automate rule generation for OSS ecosystems. RuleLLM extracts metadata and code snippets from malware as its input, producing YARA and Semgrep rules that can be directly deployed in software development. Specifically, the rule generation task involves three subtasks: crafting rules, refining rules, and aligning rules. To validate RuleLLM's effectiveness, we implemented a prototype system and conducted experiments on the dataset of 1,633 malicious packages. The results are promising that RuleLLM generated 763 rules (452 YARA and 311 Semgrep) with a precision of 85.2\% and a recall of 91.8\%, outperforming state-of-the-art (SOTA) tools and scored-based approaches. We further analyzed generated rules and proposed a rule taxonomy: 11 categories and 38 subcategories.



## **40. Robo-Troj: Attacking LLM-based Task Planners**

cs.RO

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.17070v1) [paper-pdf](http://arxiv.org/pdf/2504.17070v1)

**Authors**: Mohaiminul Al Nahian, Zainab Altaweel, David Reitano, Sabbir Ahmed, Saumitra Lohokare, Shiqi Zhang, Adnan Siraj Rakin

**Abstract**: Robots need task planning methods to achieve goals that require more than individual actions. Recently, large language models (LLMs) have demonstrated impressive performance in task planning. LLMs can generate a step-by-step solution using a description of actions and the goal. Despite the successes in LLM-based task planning, there is limited research studying the security aspects of those systems. In this paper, we develop Robo-Troj, the first multi-trigger backdoor attack for LLM-based task planners, which is the main contribution of this work. As a multi-trigger attack, Robo-Troj is trained to accommodate the diversity of robot application domains. For instance, one can use unique trigger words, e.g., "herical", to activate a specific malicious behavior, e.g., cutting hand on a kitchen robot. In addition, we develop an optimization method for selecting the trigger words that are most effective. Through demonstrating the vulnerability of LLM-based planners, we aim to promote the development of secured robot systems.



## **41. Trading Devil Final: Backdoor attack via Stock market and Bayesian Optimization**

cs.LG

END (will never be modified again!!) :Jumps-Diffusion and stock  market: Better quantify uncertainty in financial simulations

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2407.14573v7) [paper-pdf](http://arxiv.org/pdf/2407.14573v7)

**Authors**: Orson Mengara

**Abstract**: Since the advent of generative artificial intelligence, every company and researcher has been rushing to develop their own generative models, whether commercial or not. Given the large number of users of these powerful new tools, there is currently no intrinsically verifiable way to explain from the ground up what happens when LLMs (large language models) learn. For example, those based on automatic speech recognition systems, which have to rely on huge and astronomical amounts of data collected from all over the web to produce fast and efficient results, In this article, we develop a backdoor attack called MarketBackFinal 2.0, based on acoustic data poisoning, MarketBackFinal 2.0 is mainly based on modern stock market models. In order to show the possible vulnerabilities of speech-based transformers that may rely on LLMs.



## **42. Safety Pretraining: Toward the Next Generation of Safe AI**

cs.LG

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.16980v1) [paper-pdf](http://arxiv.org/pdf/2504.16980v1)

**Authors**: Pratyush Maini, Sachin Goyal, Dylan Sam, Alex Robey, Yash Savani, Yiding Jiang, Andy Zou, Zacharcy C. Lipton, J. Zico Kolter

**Abstract**: As large language models (LLMs) are increasingly deployed in high-stakes settings, the risk of generating harmful or toxic content remains a central challenge. Post-hoc alignment methods are brittle: once unsafe patterns are learned during pretraining, they are hard to remove. We present a data-centric pretraining framework that builds safety into the model from the start. Our contributions include: (i) a safety classifier trained on 10,000 GPT-4 labeled examples, used to filter 600B tokens; (ii) the largest synthetic safety dataset to date (100B tokens) generated via recontextualization of harmful web data; (iii) RefuseWeb and Moral Education datasets that convert harmful prompts into refusal dialogues and web-style educational material; (iv) Harmfulness-Tag annotations injected during pretraining to flag unsafe content and steer away inference from harmful generations; and (v) safety evaluations measuring base model behavior before instruction tuning. Our safety-pretrained models reduce attack success rates from 38.8% to 8.4% with no performance degradation on standard LLM safety benchmarks.



## **43. Context-Enhanced Vulnerability Detection Based on Large Language Model**

cs.SE

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.16877v1) [paper-pdf](http://arxiv.org/pdf/2504.16877v1)

**Authors**: Yixin Yang, Bowen Xu, Xiang Gao, Hailong Sun

**Abstract**: Vulnerability detection is a critical aspect of software security. Accurate detection is essential to prevent potential security breaches and protect software systems from malicious attacks. Recently, vulnerability detection methods leveraging deep learning and large language models (LLMs) have garnered increasing attention. However, existing approaches often focus on analyzing individual files or functions, which limits their ability to gather sufficient contextual information. Analyzing entire repositories to gather context introduces significant noise and computational overhead. To address these challenges, we propose a context-enhanced vulnerability detection approach that combines program analysis with LLMs. Specifically, we use program analysis to extract contextual information at various levels of abstraction, thereby filtering out irrelevant noise. The abstracted context along with source code are provided to LLM for vulnerability detection. We investigate how different levels of contextual granularity improve LLM-based vulnerability detection performance. Our goal is to strike a balance between providing sufficient detail to accurately capture vulnerabilities and minimizing unnecessary complexity that could hinder model performance. Based on an extensive study using GPT-4, DeepSeek, and CodeLLaMA with various prompting strategies, our key findings includes: (1) incorporating abstracted context significantly enhances vulnerability detection effectiveness; (2) different models benefit from distinct levels of abstraction depending on their code understanding capabilities; and (3) capturing program behavior through program analysis for general LLM-based code analysis tasks can be a direction that requires further attention.



## **44. aiXamine: Simplified LLM Safety and Security**

cs.CR

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.14985v2) [paper-pdf](http://arxiv.org/pdf/2504.14985v2)

**Authors**: Fatih Deniz, Dorde Popovic, Yazan Boshmaf, Euisuh Jeong, Minhaj Ahmad, Sanjay Chawla, Issa Khalil

**Abstract**: Evaluating Large Language Models (LLMs) for safety and security remains a complex task, often requiring users to navigate a fragmented landscape of ad hoc benchmarks, datasets, metrics, and reporting formats. To address this challenge, we present aiXamine, a comprehensive black-box evaluation platform for LLM safety and security. aiXamine integrates over 40 tests (i.e., benchmarks) organized into eight key services targeting specific dimensions of safety and security: adversarial robustness, code security, fairness and bias, hallucination, model and data privacy, out-of-distribution (OOD) robustness, over-refusal, and safety alignment. The platform aggregates the evaluation results into a single detailed report per model, providing a detailed breakdown of model performance, test examples, and rich visualizations. We used aiXamine to assess over 50 publicly available and proprietary LLMs, conducting over 2K examinations. Our findings reveal notable vulnerabilities in leading models, including susceptibility to adversarial attacks in OpenAI's GPT-4o, biased outputs in xAI's Grok-3, and privacy weaknesses in Google's Gemini 2.0. Additionally, we observe that open-source models can match or exceed proprietary models in specific services such as safety alignment, fairness and bias, and OOD robustness. Finally, we identify trade-offs between distillation strategies, model size, training methods, and architectural choices.



## **45. Amplified Vulnerabilities: Structured Jailbreak Attacks on LLM-based Multi-Agent Debate**

cs.CR

33 pages, 5 figures

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2504.16489v1) [paper-pdf](http://arxiv.org/pdf/2504.16489v1)

**Authors**: Senmao Qi, Yifei Zou, Peng Li, Ziyi Lin, Xiuzhen Cheng, Dongxiao Yu

**Abstract**: Multi-Agent Debate (MAD), leveraging collaborative interactions among Large Language Models (LLMs), aim to enhance reasoning capabilities in complex tasks. However, the security implications of their iterative dialogues and role-playing characteristics, particularly susceptibility to jailbreak attacks eliciting harmful content, remain critically underexplored. This paper systematically investigates the jailbreak vulnerabilities of four prominent MAD frameworks built upon leading commercial LLMs (GPT-4o, GPT-4, GPT-3.5-turbo, and DeepSeek) without compromising internal agents. We introduce a novel structured prompt-rewriting framework specifically designed to exploit MAD dynamics via narrative encapsulation, role-driven escalation, iterative refinement, and rhetorical obfuscation. Our extensive experiments demonstrate that MAD systems are inherently more vulnerable than single-agent setups. Crucially, our proposed attack methodology significantly amplifies this fragility, increasing average harmfulness from 28.14% to 80.34% and achieving attack success rates as high as 80% in certain scenarios. These findings reveal intrinsic vulnerabilities in MAD architectures and underscore the urgent need for robust, specialized defenses prior to real-world deployment.



## **46. Large Language Model Sentinel: LLM Agent for Adversarial Purification**

cs.CL

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2405.20770v4) [paper-pdf](http://arxiv.org/pdf/2405.20770v4)

**Authors**: Guang Lin, Toshihisa Tanaka, Qibin Zhao

**Abstract**: Over the past two years, the use of large language models (LLMs) has advanced rapidly. While these LLMs offer considerable convenience, they also raise security concerns, as LLMs are vulnerable to adversarial attacks by some well-designed textual perturbations. In this paper, we introduce a novel defense technique named Large LAnguage MOdel Sentinel (LLAMOS), which is designed to enhance the adversarial robustness of LLMs by purifying the adversarial textual examples before feeding them into the target LLM. Our method comprises two main components: a) Agent instruction, which can simulate a new agent for adversarial defense, altering minimal characters to maintain the original meaning of the sentence while defending against attacks; b) Defense guidance, which provides strategies for modifying clean or adversarial examples to ensure effective defense and accurate outputs from the target LLMs. Remarkably, the defense agent demonstrates robust defensive capabilities even without learning from adversarial examples. Additionally, we conduct an intriguing adversarial experiment where we develop two agents, one for defense and one for attack, and engage them in mutual confrontation. During the adversarial interactions, neither agent completely beat the other. Extensive experiments on both open-source and closed-source LLMs demonstrate that our method effectively defends against adversarial attacks, thereby enhancing adversarial robustness.



## **47. Attention Tracker: Detecting Prompt Injection Attacks in LLMs**

cs.CR

Project page:  https://huggingface.co/spaces/TrustSafeAI/Attention-Tracker

**SubmitDate**: 2025-04-23    [abs](http://arxiv.org/abs/2411.00348v2) [paper-pdf](http://arxiv.org/pdf/2411.00348v2)

**Authors**: Kuo-Han Hung, Ching-Yun Ko, Ambrish Rawat, I-Hsin Chung, Winston H. Hsu, Pin-Yu Chen

**Abstract**: Large Language Models (LLMs) have revolutionized various domains but remain vulnerable to prompt injection attacks, where malicious inputs manipulate the model into ignoring original instructions and executing designated action. In this paper, we investigate the underlying mechanisms of these attacks by analyzing the attention patterns within LLMs. We introduce the concept of the distraction effect, where specific attention heads, termed important heads, shift focus from the original instruction to the injected instruction. Building on this discovery, we propose Attention Tracker, a training-free detection method that tracks attention patterns on instruction to detect prompt injection attacks without the need for additional LLM inference. Our method generalizes effectively across diverse models, datasets, and attack types, showing an AUROC improvement of up to 10.0% over existing methods, and performs well even on small LLMs. We demonstrate the robustness of our approach through extensive evaluations and provide insights into safeguarding LLM-integrated systems from prompt injection vulnerabilities.



## **48. BaThe: Defense against the Jailbreak Attack in Multimodal Large Language Models by Treating Harmful Instruction as Backdoor Trigger**

cs.CR

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2408.09093v3) [paper-pdf](http://arxiv.org/pdf/2408.09093v3)

**Authors**: Yulin Chen, Haoran Li, Yirui Zhang, Zihao Zheng, Yangqiu Song, Bryan Hooi

**Abstract**: Multimodal Large Language Models (MLLMs) have showcased impressive performance in a variety of multimodal tasks. On the other hand, the integration of additional image modality may allow the malicious users to inject harmful content inside the images for jailbreaking. Unlike text-based LLMs, where adversaries need to select discrete tokens to conceal their malicious intent using specific algorithms, the continuous nature of image signals provides a direct opportunity for adversaries to inject harmful intentions. In this work, we propose $\textbf{BaThe}$ ($\textbf{Ba}$ckdoor $\textbf{T}$rigger S$\textbf{h}$i$\textbf{e}$ld), a simple yet effective jailbreak defense mechanism. Our work is motivated by recent research on jailbreak backdoor attack and virtual prompt backdoor attack in generative language models. Jailbreak backdoor attack uses harmful instructions combined with manually crafted strings as triggers to make the backdoored model generate prohibited responses. We assume that harmful instructions can function as triggers, and if we alternatively set rejection responses as the triggered response, the backdoored model then can defend against jailbreak attacks. We achieve this by utilizing virtual rejection prompt, similar to the virtual prompt backdoor attack. We embed the virtual rejection prompt into the soft text embeddings, which we call ``wedge''. Our comprehensive experiments demonstrate that BaThe effectively mitigates various types of jailbreak attacks and is adaptable to defend against unseen attacks, with minimal impact on MLLMs' performance.



## **49. Red Team Diffuser: Exposing Toxic Continuation Vulnerabilities in Vision-Language Models via Reinforcement Learning**

cs.CV

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2503.06223v2) [paper-pdf](http://arxiv.org/pdf/2503.06223v2)

**Authors**: Ruofan Wang, Xiang Zheng, Xiaosen Wang, Cong Wang, Xingjun Ma

**Abstract**: The growing deployment of large Vision-Language Models (VLMs) exposes critical safety gaps in their alignment mechanisms. While existing jailbreak studies primarily focus on VLMs' susceptibility to harmful instructions, we reveal a fundamental yet overlooked vulnerability: toxic text continuation, where VLMs produce highly toxic completions when prompted with harmful text prefixes paired with semantically adversarial images. To systematically study this threat, we propose Red Team Diffuser (RTD), the first red teaming diffusion model that coordinates adversarial image generation and toxic continuation through reinforcement learning. Our key innovations include dynamic cross-modal attack and stealth-aware optimization. For toxic text prefixes from an LLM safety benchmark, we conduct greedy search to identify optimal image prompts that maximally induce toxic completions. The discovered image prompts then drive RL-based diffusion model fine-tuning, producing semantically aligned adversarial images that boost toxicity rates. Stealth-aware optimization introduces joint adversarial rewards that balance toxicity maximization (via Detoxify classifier) and stealthiness (via BERTScore), circumventing traditional noise-based adversarial patterns. Experimental results demonstrate the effectiveness of RTD, increasing the toxicity rate of LLaVA outputs by 10.69% over text-only baselines on the original attack set and 8.91% on an unseen set, proving generalization capability. Moreover, RTD exhibits strong cross-model transferability, raising the toxicity rate by 5.1% on Gemini and 26.83% on LLaMA. Our findings expose two critical flaws in current VLM alignment: (1) failure to prevent toxic continuation from harmful prefixes, and (2) overlooking cross-modal attack vectors. These results necessitate a paradigm shift toward multimodal red teaming in safety evaluations.



## **50. Research on Cloud Platform Network Traffic Monitoring and Anomaly Detection System based on Large Language Models**

cs.NI

Proceedings of 2025 IEEE 7th International Conference on  Communications, Information System and Computer Engineering (CISCE 2025)

**SubmitDate**: 2025-04-22    [abs](http://arxiv.org/abs/2504.17807v1) [paper-pdf](http://arxiv.org/pdf/2504.17807v1)

**Authors**: Ze Yang, Yihong Jin, Juntian Liu, Xinhe Xu, Yihan Zhang, Shuyang Ji

**Abstract**: The rapidly evolving cloud platforms and the escalating complexity of network traffic demand proper network traffic monitoring and anomaly detection to ensure network security and performance. This paper introduces a large language model (LLM)-based network traffic monitoring and anomaly detection system. In addition to existing models such as autoencoders and decision trees, we harness the power of large language models for processing sequence data from network traffic, which allows us a better capture of underlying complex patterns, as well as slight fluctuations in the dataset. We show for a given detection task, the need for a hybrid model that incorporates the attention mechanism of the transformer architecture into a supervised learning framework in order to achieve better accuracy. A pre-trained large language model analyzes and predicts the probable network traffic, and an anomaly detection layer that considers temporality and context is added. Moreover, we present a novel transfer learning-based methodology to enhance the model's effectiveness to quickly adapt to unknown network structures and adversarial conditions without requiring extensive labeled datasets. Actual results show that the designed model outperforms traditional methods in detection accuracy and computational efficiency, effectively identify various network anomalies such as zero-day attacks and traffic congestion pattern, and significantly reduce the false positive rate.



