# Latest Large Language Model Attack Papers
**update at 2025-05-28 14:52:07**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. Adversarial Attacks against Closed-Source MLLMs via Feature Optimal Alignment**

cs.CV

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21494v1) [paper-pdf](http://arxiv.org/pdf/2505.21494v1)

**Authors**: Xiaojun Jia, Sensen Gao, Simeng Qin, Tianyu Pang, Chao Du, Yihao Huang, Xinfeng Li, Yiming Li, Bo Li, Yang Liu

**Abstract**: Multimodal large language models (MLLMs) remain vulnerable to transferable adversarial examples. While existing methods typically achieve targeted attacks by aligning global features-such as CLIP's [CLS] token-between adversarial and target samples, they often overlook the rich local information encoded in patch tokens. This leads to suboptimal alignment and limited transferability, particularly for closed-source models. To address this limitation, we propose a targeted transferable adversarial attack method based on feature optimal alignment, called FOA-Attack, to improve adversarial transfer capability. Specifically, at the global level, we introduce a global feature loss based on cosine similarity to align the coarse-grained features of adversarial samples with those of target samples. At the local level, given the rich local representations within Transformers, we leverage clustering techniques to extract compact local patterns to alleviate redundant local features. We then formulate local feature alignment between adversarial and target samples as an optimal transport (OT) problem and propose a local clustering optimal transport loss to refine fine-grained feature alignment. Additionally, we propose a dynamic ensemble model weighting strategy to adaptively balance the influence of multiple models during adversarial example generation, thereby further improving transferability. Extensive experiments across various models demonstrate the superiority of the proposed method, outperforming state-of-the-art methods, especially in transferring to closed-source MLLMs. The code is released at https://github.com/jiaxiaojunQAQ/FOA-Attack.



## **2. GUARD:Dual-Agent based Backdoor Defense on Chain-of-Thought in Neural Code Generation**

cs.SE

under review

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21425v1) [paper-pdf](http://arxiv.org/pdf/2505.21425v1)

**Authors**: Naizhu Jin, Zhong Li, Tian Zhang, Qingkai Zeng

**Abstract**: With the widespread application of large language models in code generation, recent studies demonstrate that employing additional Chain-of-Thought generation models can significantly enhance code generation performance by providing explicit reasoning steps. However, as external components, CoT models are particularly vulnerable to backdoor attacks, which existing defense mechanisms often fail to detect effectively. To address this challenge, we propose GUARD, a novel dual-agent defense framework specifically designed to counter CoT backdoor attacks in neural code generation. GUARD integrates two core components: GUARD-Judge, which identifies suspicious CoT steps and potential triggers through comprehensive analysis, and GUARD-Repair, which employs a retrieval-augmented generation approach to regenerate secure CoT steps for identified anomalies. Experimental results show that GUARD effectively mitigates attacks while maintaining generation quality, advancing secure code generation systems.



## **3. Breaking the Ceiling: Exploring the Potential of Jailbreak Attacks through Expanding Strategy Space**

cs.CR

19 pages, 20 figures, accepted by ACL 2025, Findings

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21277v1) [paper-pdf](http://arxiv.org/pdf/2505.21277v1)

**Authors**: Yao Huang, Yitong Sun, Shouwei Ruan, Yichi Zhang, Yinpeng Dong, Xingxing Wei

**Abstract**: Large Language Models (LLMs), despite advanced general capabilities, still suffer from numerous safety risks, especially jailbreak attacks that bypass safety protocols. Understanding these vulnerabilities through black-box jailbreak attacks, which better reflect real-world scenarios, offers critical insights into model robustness. While existing methods have shown improvements through various prompt engineering techniques, their success remains limited against safety-aligned models, overlooking a more fundamental problem: the effectiveness is inherently bounded by the predefined strategy spaces. However, expanding this space presents significant challenges in both systematically capturing essential attack patterns and efficiently navigating the increased complexity. To better explore the potential of expanding the strategy space, we address these challenges through a novel framework that decomposes jailbreak strategies into essential components based on the Elaboration Likelihood Model (ELM) theory and develops genetic-based optimization with intention evaluation mechanisms. To be striking, our experiments reveal unprecedented jailbreak capabilities by expanding the strategy space: we achieve over 90% success rate on Claude-3.5 where prior methods completely fail, while demonstrating strong cross-model transferability and surpassing specialized safeguard models in evaluation accuracy. The code is open-sourced at: https://github.com/Aries-iai/CL-GSO.



## **4. JavaSith: A Client-Side Framework for Analyzing Potentially Malicious Extensions in Browsers, VS Code, and NPM Packages**

cs.CR

28 pages , 11 figures

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21263v1) [paper-pdf](http://arxiv.org/pdf/2505.21263v1)

**Authors**: Avihay Cohen

**Abstract**: Modern software supply chains face an increasing threat from malicious code hidden in trusted components such as browser extensions, IDE extensions, and open-source packages. This paper introduces JavaSith, a novel client-side framework for analyzing potentially malicious extensions in web browsers, Visual Studio Code (VSCode), and Node's NPM packages. JavaSith combines a runtime sandbox that emulates browser/Node.js extension APIs (with a ``time machine'' to accelerate time-based triggers) with static analysis and a local large language model (LLM) to assess risk from code and metadata. We present the design and architecture of JavaSith, including techniques for intercepting extension behavior over simulated time and extracting suspicious patterns. Through case studies on real-world attacks (such as a supply-chain compromise of a Chrome extension and malicious VSCode extensions installing cryptominers), we demonstrate how JavaSith can catch stealthy malicious behaviors that evade traditional detection. We evaluate the framework's effectiveness and discuss its limitations and future enhancements. JavaSith's client-side approach empowers end-users/organizations to vet extensions and packages before trustingly integrating them into their environments.



## **5. SHE-LoRA: Selective Homomorphic Encryption for Federated Tuning with Heterogeneous LoRA**

cs.CR

24 pages, 13 figures

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.21051v1) [paper-pdf](http://arxiv.org/pdf/2505.21051v1)

**Authors**: Jianmin Liu, Li Yan, Borui Li, Lei Yu, Chao Shen

**Abstract**: Federated fine-tuning of large language models (LLMs) is critical for improving their performance in handling domain-specific tasks. However, prior work has shown that clients' private data can actually be recovered via gradient inversion attacks. Existing privacy preservation techniques against such attacks typically entail performance degradation and high costs, making them ill-suited for clients with heterogeneous data distributions and device capabilities. In this paper, we propose SHE-LoRA, which integrates selective homomorphic encryption (HE) and low-rank adaptation (LoRA) to enable efficient and privacy-preserving federated tuning of LLMs in cross-device environment. Heterogeneous clients adaptively select partial model parameters for homomorphic encryption based on parameter sensitivity assessment, with the encryption subset obtained via negotiation. To ensure accurate model aggregation, we design a column-aware secure aggregation method and customized reparameterization techniques to align the aggregation results with the heterogeneous device capabilities of clients. Extensive experiments demonstrate that SHE-LoRA maintains performance comparable to non-private baselines, achieves strong resistance to the state-of-the-art attacks, and significantly reduces communication overhead by 94.901\% and encryption computation overhead by 99.829\%, compared to baseline. Our code is accessible at https://anonymous.4open.science/r/SHE-LoRA-8D84.



## **6. BitHydra: Towards Bit-flip Inference Cost Attack against Large Language Models**

cs.CR

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.16670v2) [paper-pdf](http://arxiv.org/pdf/2505.16670v2)

**Authors**: Xiaobei Yan, Yiming Li, Zhaoxin Fan, Han Qiu, Tianwei Zhang

**Abstract**: Large language models (LLMs) have shown impressive capabilities across a wide range of applications, but their ever-increasing size and resource demands make them vulnerable to inference cost attacks, where attackers induce victim LLMs to generate the longest possible output content. In this paper, we revisit existing inference cost attacks and reveal that these methods can hardly produce large-scale malicious effects since they are self-targeting, where attackers are also the users and therefore have to execute attacks solely through the inputs, whose generated content will be charged by LLMs and can only directly influence themselves. Motivated by these findings, this paper introduces a new type of inference cost attacks (dubbed 'bit-flip inference cost attack') that target the victim model itself rather than its inputs. Specifically, we design a simple yet effective method (dubbed 'BitHydra') to effectively flip critical bits of model parameters. This process is guided by a loss function designed to suppress <EOS> token's probability with an efficient critical bit search algorithm, thus explicitly defining the attack objective and enabling effective optimization. We evaluate our method on 11 LLMs ranging from 1.5B to 14B parameters under both int8 and float16 settings. Experimental results demonstrate that with just 4 search samples and as few as 3 bit flips, BitHydra can force 100% of test prompts to reach the maximum generation length (e.g., 2048 tokens) on representative LLMs such as LLaMA3, highlighting its efficiency, scalability, and strong transferability across unseen inputs.



## **7. IRCopilot: Automated Incident Response with Large Language Models**

cs.CR

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20945v1) [paper-pdf](http://arxiv.org/pdf/2505.20945v1)

**Authors**: Xihuan Lin, Jie Zhang, Gelei Deng, Tianzhe Liu, Xiaolong Liu, Changcai Yang, Tianwei Zhang, Qing Guo, Riqing Chen

**Abstract**: Incident response plays a pivotal role in mitigating the impact of cyber attacks. In recent years, the intensity and complexity of global cyber threats have grown significantly, making it increasingly challenging for traditional threat detection and incident response methods to operate effectively in complex network environments. While Large Language Models (LLMs) have shown great potential in early threat detection, their capabilities remain limited when it comes to automated incident response after an intrusion. To address this gap, we construct an incremental benchmark based on real-world incident response tasks to thoroughly evaluate the performance of LLMs in this domain. Our analysis reveals several key challenges that hinder the practical application of contemporary LLMs, including context loss, hallucinations, privacy protection concerns, and their limited ability to provide accurate, context-specific recommendations. In response to these challenges, we propose IRCopilot, a novel framework for automated incident response powered by LLMs. IRCopilot mimics the three dynamic phases of a real-world incident response team using four collaborative LLM-based session components. These components are designed with clear divisions of responsibility, reducing issues such as hallucinations and context loss. Our method leverages diverse prompt designs and strategic responsibility segmentation, significantly improving the system's practicality and efficiency. Experimental results demonstrate that IRCopilot outperforms baseline LLMs across key benchmarks, achieving sub-task completion rates of 150%, 138%, 136%, 119%, and 114% for various response tasks. Moreover, IRCopilot exhibits robust performance on public incident response platforms and in real-world attack scenarios, showcasing its strong applicability.



## **8. Concealment of Intent: A Game-Theoretic Analysis**

cs.CL

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20841v1) [paper-pdf](http://arxiv.org/pdf/2505.20841v1)

**Authors**: Xinbo Wu, Abhishek Umrawal, Lav R. Varshney

**Abstract**: As large language models (LLMs) grow more capable, concerns about their safe deployment have also grown. Although alignment mechanisms have been introduced to deter misuse, they remain vulnerable to carefully designed adversarial prompts. In this work, we present a scalable attack strategy: intent-hiding adversarial prompting, which conceals malicious intent through the composition of skills. We develop a game-theoretic framework to model the interaction between such attacks and defense systems that apply both prompt and response filtering. Our analysis identifies equilibrium points and reveals structural advantages for the attacker. To counter these threats, we propose and analyze a defense mechanism tailored to intent-hiding attacks. Empirically, we validate the attack's effectiveness on multiple real-world LLMs across a range of malicious behaviors, demonstrating clear advantages over existing adversarial prompting techniques.



## **9. MedSentry: Understanding and Mitigating Safety Risks in Medical LLM Multi-Agent Systems**

cs.MA

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20824v1) [paper-pdf](http://arxiv.org/pdf/2505.20824v1)

**Authors**: Kai Chen, Taihang Zhen, Hewei Wang, Kailai Liu, Xinfeng Li, Jing Huo, Tianpei Yang, Jinfeng Xu, Wei Dong, Yang Gao

**Abstract**: As large language models (LLMs) are increasingly deployed in healthcare, ensuring their safety, particularly within collaborative multi-agent configurations, is paramount. In this paper we introduce MedSentry, a benchmark comprising 5 000 adversarial medical prompts spanning 25 threat categories with 100 subthemes. Coupled with this dataset, we develop an end-to-end attack-defense evaluation pipeline to systematically analyze how four representative multi-agent topologies (Layers, SharedPool, Centralized, and Decentralized) withstand attacks from 'dark-personality' agents. Our findings reveal critical differences in how these architectures handle information contamination and maintain robust decision-making, exposing their underlying vulnerability mechanisms. For instance, SharedPool's open information sharing makes it highly susceptible, whereas Decentralized architectures exhibit greater resilience thanks to inherent redundancy and isolation. To mitigate these risks, we propose a personality-scale detection and correction mechanism that identifies and rehabilitates malicious agents, restoring system safety to near-baseline levels. MedSentry thus furnishes both a rigorous evaluation framework and practical defense strategies that guide the design of safer LLM-based multi-agent systems in medical domains.



## **10. TrojanStego: Your Language Model Can Secretly Be A Steganographic Privacy Leaking Agent**

cs.CL

9 pages, 5 figures

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20118v2) [paper-pdf](http://arxiv.org/pdf/2505.20118v2)

**Authors**: Dominik Meier, Jan Philip Wahle, Paul Röttger, Terry Ruas, Bela Gipp

**Abstract**: As large language models (LLMs) become integrated into sensitive workflows, concerns grow over their potential to leak confidential information. We propose TrojanStego, a novel threat model in which an adversary fine-tunes an LLM to embed sensitive context information into natural-looking outputs via linguistic steganography, without requiring explicit control over inference inputs. We introduce a taxonomy outlining risk factors for compromised LLMs, and use it to evaluate the risk profile of the threat. To implement TrojanStego, we propose a practical encoding scheme based on vocabulary partitioning learnable by LLMs via fine-tuning. Experimental results show that compromised models reliably transmit 32-bit secrets with 87% accuracy on held-out prompts, reaching over 97% accuracy using majority voting across three generations. Further, they maintain high utility, can evade human detection, and preserve coherence. These results highlight a new class of LLM data exfiltration attacks that are passive, covert, practical, and dangerous.



## **11. Improved Representation Steering for Language Models**

cs.CL

46 pages, 23 figures, preprint

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.20809v1) [paper-pdf](http://arxiv.org/pdf/2505.20809v1)

**Authors**: Zhengxuan Wu, Qinan Yu, Aryaman Arora, Christopher D. Manning, Christopher Potts

**Abstract**: Steering methods for language models (LMs) seek to provide fine-grained and interpretable control over model generations by variously changing model inputs, weights, or representations to adjust behavior. Recent work has shown that adjusting weights or representations is often less effective than steering by prompting, for instance when wanting to introduce or suppress a particular concept. We demonstrate how to improve representation steering via our new Reference-free Preference Steering (RePS), a bidirectional preference-optimization objective that jointly does concept steering and suppression. We train three parameterizations of RePS and evaluate them on AxBench, a large-scale model steering benchmark. On Gemma models with sizes ranging from 2B to 27B, RePS outperforms all existing steering methods trained with a language modeling objective and substantially narrows the gap with prompting -- while promoting interpretability and minimizing parameter count. In suppression, RePS matches the language-modeling objective on Gemma-2 and outperforms it on the larger Gemma-3 variants while remaining resilient to prompt-based jailbreaking attacks that defeat prompting. Overall, our results suggest that RePS provides an interpretable and robust alternative to prompting for both steering and suppression.



## **12. Forewarned is Forearmed: A Survey on Large Language Model-based Agents in Autonomous Cyberattacks**

cs.NI

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.12786v2) [paper-pdf](http://arxiv.org/pdf/2505.12786v2)

**Authors**: Minrui Xu, Jiani Fan, Xinyu Huang, Conghao Zhou, Jiawen Kang, Dusit Niyato, Shiwen Mao, Zhu Han, Xuemin, Shen, Kwok-Yan Lam

**Abstract**: With the continuous evolution of Large Language Models (LLMs), LLM-based agents have advanced beyond passive chatbots to become autonomous cyber entities capable of performing complex tasks, including web browsing, malicious code and deceptive content generation, and decision-making. By significantly reducing the time, expertise, and resources, AI-assisted cyberattacks orchestrated by LLM-based agents have led to a phenomenon termed Cyber Threat Inflation, characterized by a significant reduction in attack costs and a tremendous increase in attack scale. To provide actionable defensive insights, in this survey, we focus on the potential cyber threats posed by LLM-based agents across diverse network systems. Firstly, we present the capabilities of LLM-based cyberattack agents, which include executing autonomous attack strategies, comprising scouting, memory, reasoning, and action, and facilitating collaborative operations with other agents or human operators. Building on these capabilities, we examine common cyberattacks initiated by LLM-based agents and compare their effectiveness across different types of networks, including static, mobile, and infrastructure-free paradigms. Moreover, we analyze threat bottlenecks of LLM-based agents across different network infrastructures and review their defense methods. Due to operational imbalances, existing defense methods are inadequate against autonomous cyberattacks. Finally, we outline future research directions and potential defensive strategies for legacy network systems.



## **13. $C^3$-Bench: The Things Real Disturbing LLM based Agent in Multi-Tasking**

cs.AI

**SubmitDate**: 2025-05-27    [abs](http://arxiv.org/abs/2505.18746v2) [paper-pdf](http://arxiv.org/pdf/2505.18746v2)

**Authors**: Peijie Yu, Yifan Yang, Jinjian Li, Zelong Zhang, Haorui Wang, Xiao Feng, Feng Zhang

**Abstract**: Agents based on large language models leverage tools to modify environments, revolutionizing how AI interacts with the physical world. Unlike traditional NLP tasks that rely solely on historical dialogue for responses, these agents must consider more complex factors, such as inter-tool relationships, environmental feedback and previous decisions, when making choices. Current research typically evaluates agents via multi-turn dialogues. However, it overlooks the influence of these critical factors on agent behavior. To bridge this gap, we present an open-source and high-quality benchmark $C^3$-Bench. This benchmark integrates attack concepts and applies univariate analysis to pinpoint key elements affecting agent robustness. In concrete, we design three challenges: navigate complex tool relationships, handle critical hidden information and manage dynamic decision paths. Complementing these challenges, we introduce fine-grained metrics, innovative data collection algorithms and reproducible evaluation methods. Extensive experiments are conducted on 49 mainstream agents, encompassing general fast-thinking, slow-thinking and domain-specific models. We observe that agents have significant shortcomings in handling tool dependencies, long context information dependencies and frequent policy-type switching. In essence, $C^3$-Bench aims to expose model vulnerabilities through these challenges and drive research into the interpretability of agent performance. The benchmark is publicly available at https://github.com/yupeijei1997/C3-Bench.



## **14. Beyond the Tip of Efficiency: Uncovering the Submerged Threats of Jailbreak Attacks in Small Language Models**

cs.CR

Accepted to ACL 2025 findings

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2502.19883v3) [paper-pdf](http://arxiv.org/pdf/2502.19883v3)

**Authors**: Sibo Yi, Tianshuo Cong, Xinlei He, Qi Li, Jiaxing Song

**Abstract**: Small language models (SLMs) have become increasingly prominent in the deployment on edge devices due to their high efficiency and low computational cost. While researchers continue to advance the capabilities of SLMs through innovative training strategies and model compression techniques, the security risks of SLMs have received considerably less attention compared to large language models (LLMs).To fill this gap, we provide a comprehensive empirical study to evaluate the security performance of 13 state-of-the-art SLMs under various jailbreak attacks. Our experiments demonstrate that most SLMs are quite susceptible to existing jailbreak attacks, while some of them are even vulnerable to direct harmful prompts.To address the safety concerns, we evaluate several representative defense methods and demonstrate their effectiveness in enhancing the security of SLMs. We further analyze the potential security degradation caused by different SLM techniques including architecture compression, quantization, knowledge distillation, and so on. We expect that our research can highlight the security challenges of SLMs and provide valuable insights to future work in developing more robust and secure SLMs.



## **15. Capability-Based Scaling Laws for LLM Red-Teaming**

cs.AI

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.20162v1) [paper-pdf](http://arxiv.org/pdf/2505.20162v1)

**Authors**: Alexander Panfilov, Paul Kassianik, Maksym Andriushchenko, Jonas Geiping

**Abstract**: As large language models grow in capability and agency, identifying vulnerabilities through red-teaming becomes vital for safe deployment. However, traditional prompt-engineering approaches may prove ineffective once red-teaming turns into a weak-to-strong problem, where target models surpass red-teamers in capabilities. To study this shift, we frame red-teaming through the lens of the capability gap between attacker and target. We evaluate more than 500 attacker-target pairs using LLM-based jailbreak attacks that mimic human red-teamers across diverse families, sizes, and capability levels. Three strong trends emerge: (i) more capable models are better attackers, (ii) attack success drops sharply once the target's capability exceeds the attacker's, and (iii) attack success rates correlate with high performance on social science splits of the MMLU-Pro benchmark. From these trends, we derive a jailbreaking scaling law that predicts attack success for a fixed target based on attacker-target capability gap. These findings suggest that fixed-capability attackers (e.g., humans) may become ineffective against future models, increasingly capable open-source models amplify risks for existing systems, and model providers must accurately measure and control models' persuasive and manipulative abilities to limit their effectiveness as attackers.



## **16. PandaGuard: Systematic Evaluation of LLM Safety against Jailbreaking Attacks**

cs.CR

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.13862v3) [paper-pdf](http://arxiv.org/pdf/2505.13862v3)

**Authors**: Guobin Shen, Dongcheng Zhao, Linghao Feng, Xiang He, Jihang Wang, Sicheng Shen, Haibo Tong, Yiting Dong, Jindong Li, Xiang Zheng, Yi Zeng

**Abstract**: Large language models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial prompts known as jailbreaks, which can bypass safety alignment and elicit harmful outputs. Despite growing efforts in LLM safety research, existing evaluations are often fragmented, focused on isolated attack or defense techniques, and lack systematic, reproducible analysis. In this work, we introduce PandaGuard, a unified and modular framework that models LLM jailbreak safety as a multi-agent system comprising attackers, defenders, and judges. Our framework implements 19 attack methods and 12 defense mechanisms, along with multiple judgment strategies, all within a flexible plugin architecture supporting diverse LLM interfaces, multiple interaction modes, and configuration-driven experimentation that enhances reproducibility and practical deployment. Built on this framework, we develop PandaBench, a comprehensive benchmark that evaluates the interactions between these attack/defense methods across 49 LLMs and various judgment approaches, requiring over 3 billion tokens to execute. Our extensive evaluation reveals key insights into model vulnerabilities, defense cost-performance trade-offs, and judge consistency. We find that no single defense is optimal across all dimensions and that judge disagreement introduces nontrivial variance in safety assessments. We release the code, configurations, and evaluation results to support transparent and reproducible research in LLM safety.



## **17. Crabs: Consuming Resource via Auto-generation for LLM-DoS Attack under Black-box Settings**

cs.CL

22 pages, 8 figures, 11 tables

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2412.13879v4) [paper-pdf](http://arxiv.org/pdf/2412.13879v4)

**Authors**: Yuanhe Zhang, Zhenhong Zhou, Wei Zhang, Xinyue Wang, Xiaojun Jia, Yang Liu, Sen Su

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across diverse tasks yet still are vulnerable to external threats, particularly LLM Denial-of-Service (LLM-DoS) attacks. Specifically, LLM-DoS attacks aim to exhaust computational resources and block services. However, existing studies predominantly focus on white-box attacks, leaving black-box scenarios underexplored. In this paper, we introduce Auto-Generation for LLM-DoS (AutoDoS) attack, an automated algorithm designed for black-box LLMs. AutoDoS constructs the DoS Attack Tree and expands the node coverage to achieve effectiveness under black-box conditions. By transferability-driven iterative optimization, AutoDoS could work across different models in one prompt. Furthermore, we reveal that embedding the Length Trojan allows AutoDoS to bypass existing defenses more effectively. Experimental results show that AutoDoS significantly amplifies service response latency by over 250$\times\uparrow$, leading to severe resource consumption in terms of GPU utilization and memory usage. Our work provides a new perspective on LLM-DoS attacks and security defenses. Our code is available at https://github.com/shuita2333/AutoDoS.



## **18. Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language Models**

cs.CL

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2504.05050v3) [paper-pdf](http://arxiv.org/pdf/2504.05050v3)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau

**Abstract**: Large language models (LLMs) are foundational explorations to artificial general intelligence, yet their alignment with human values via instruction tuning and preference learning achieves only superficial compliance. Here, we demonstrate that harmful knowledge embedded during pretraining persists as indelible "dark patterns" in LLMs' parametric memory, evading alignment safeguards and resurfacing under adversarial inducement at distributional shifts. In this study, we first theoretically analyze the intrinsic ethical vulnerability of aligned LLMs by proving that current alignment methods yield only local "safety regions" in the knowledge manifold. In contrast, pretrained knowledge remains globally connected to harmful concepts via high-likelihood adversarial trajectories. Building on this theoretical insight, we empirically validate our findings by employing semantic coherence inducement under distributional shifts--a method that systematically bypasses alignment constraints through optimized adversarial prompts. This combined theoretical and empirical approach achieves a 100% attack success rate across 19 out of 23 state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing their universal vulnerabilities.



## **19. Attention! You Vision Language Model Could Be Maliciously Manipulated**

cs.CV

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19911v1) [paper-pdf](http://arxiv.org/pdf/2505.19911v1)

**Authors**: Xiaosen Wang, Shaokang Wang, Zhijin Ge, Yuyang Luo, Shudong Zhang

**Abstract**: Large Vision-Language Models (VLMs) have achieved remarkable success in understanding complex real-world scenarios and supporting data-driven decision-making processes. However, VLMs exhibit significant vulnerability against adversarial examples, either text or image, which can lead to various adversarial outcomes, e.g., jailbreaking, hijacking, and hallucination, etc. In this work, we empirically and theoretically demonstrate that VLMs are particularly susceptible to image-based adversarial examples, where imperceptible perturbations can precisely manipulate each output token. To this end, we propose a novel attack called Vision-language model Manipulation Attack (VMA), which integrates first-order and second-order momentum optimization techniques with a differentiable transformation mechanism to effectively optimize the adversarial perturbation. Notably, VMA can be a double-edged sword: it can be leveraged to implement various attacks, such as jailbreaking, hijacking, privacy breaches, Denial-of-Service, and the generation of sponge examples, etc, while simultaneously enabling the injection of watermarks for copyright protection. Extensive empirical evaluations substantiate the efficacy and generalizability of VMA across diverse scenarios and datasets.



## **20. CPA-RAG:Covert Poisoning Attacks on Retrieval-Augmented Generation in Large Language Models**

cs.CR

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19864v1) [paper-pdf](http://arxiv.org/pdf/2505.19864v1)

**Authors**: Chunyang Li, Junwei Zhang, Anda Cheng, Zhuo Ma, Xinghua Li, Jianfeng Ma

**Abstract**: Retrieval-Augmented Generation (RAG) enhances large language models (LLMs) by incorporating external knowledge, but its openness introduces vulnerabilities that can be exploited by poisoning attacks. Existing poisoning methods for RAG systems have limitations, such as poor generalization and lack of fluency in adversarial texts. In this paper, we propose CPA-RAG, a black-box adversarial framework that generates query-relevant texts capable of manipulating the retrieval process to induce target answers. The proposed method integrates prompt-based text generation, cross-guided optimization through multiple LLMs, and retriever-based scoring to construct high-quality adversarial samples. We conduct extensive experiments across multiple datasets and LLMs to evaluate its effectiveness. Results show that the framework achieves over 90\% attack success when the top-k retrieval setting is 5, matching white-box performance, and maintains a consistent advantage of approximately 5 percentage points across different top-k values. It also outperforms existing black-box baselines by 14.5 percentage points under various defense strategies. Furthermore, our method successfully compromises a commercial RAG system deployed on Alibaba's BaiLian platform, demonstrating its practical threat in real-world applications. These findings underscore the need for more robust and secure RAG frameworks to defend against poisoning attacks.



## **21. Jailbreak-AudioBench: In-Depth Evaluation and Analysis of Jailbreak Threats for Large Audio Language Models**

cs.SD

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2501.13772v2) [paper-pdf](http://arxiv.org/pdf/2501.13772v2)

**Authors**: Hao Cheng, Erjia Xiao, Jing Shao, Yichi Wang, Le Yang, Chao Sheng, Philip Torr, Jindong Gu, Renjing Xu

**Abstract**: Large Language Models (LLMs) demonstrate impressive zero-shot performance across a wide range of natural language processing tasks. Integrating various modality encoders further expands their capabilities, giving rise to Multimodal Large Language Models (MLLMs) that process not only text but also visual and auditory modality inputs. However, these advanced capabilities may also pose significant security risks, as models can be exploited to generate harmful or inappropriate content through jailbreak attacks. While prior work has extensively explored how manipulating textual or visual modality inputs can circumvent safeguards in LLMs and MLLMs, the vulnerability of audio-specific Jailbreak on Large Audio-Language Models (LALMs) remains largely underexplored. To address this gap, we introduce Jailbreak-AudioBench, which consists of the Toolbox, curated Dataset, and comprehensive Benchmark. The Toolbox supports not only text-to-audio conversion but also a range of audio editing techniques. The curated Dataset provides diverse explicit and implicit jailbreak audio examples in both original and edited forms. Utilizing this dataset, we evaluate multiple state-of-the-art LALMs, establishing the most comprehensive audio jailbreak benchmark to date. Finally, Jailbreak-AudioBench establishes a foundation for advancing future research on LALMs safety alignment by enabling the in-depth exposure of more powerful jailbreak threats, such as query-based audio editing, and by facilitating the development of effective defense mechanisms.



## **22. QueryAttack: Jailbreaking Aligned Large Language Models Using Structured Non-natural Query Language**

cs.CR

To appear in ACL 2025

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2502.09723v3) [paper-pdf](http://arxiv.org/pdf/2502.09723v3)

**Authors**: Qingsong Zou, Jingyu Xiao, Qing Li, Zhi Yan, Yuhang Wang, Li Xu, Wenxuan Wang, Kuofeng Gao, Ruoyu Li, Yong Jiang

**Abstract**: Recent advances in large language models (LLMs) have demonstrated remarkable potential in the field of natural language processing. Unfortunately, LLMs face significant security and ethical risks. Although techniques such as safety alignment are developed for defense, prior researches reveal the possibility of bypassing such defenses through well-designed jailbreak attacks. In this paper, we propose QueryAttack, a novel framework to examine the generalizability of safety alignment. By treating LLMs as knowledge databases, we translate malicious queries in natural language into structured non-natural query language to bypass the safety alignment mechanisms of LLMs. We conduct extensive experiments on mainstream LLMs, and the results show that QueryAttack not only can achieve high attack success rates (ASRs), but also can jailbreak various defense methods. Furthermore, we tailor a defense method against QueryAttack, which can reduce ASR by up to $64\%$ on GPT-4-1106. Our code is available at https://github.com/horizonsinzqs/QueryAttack.



## **23. What Really Matters in Many-Shot Attacks? An Empirical Study of Long-Context Vulnerabilities in LLMs**

cs.CL

Accepted by ACL 2025

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19773v1) [paper-pdf](http://arxiv.org/pdf/2505.19773v1)

**Authors**: Sangyeop Kim, Yohan Lee, Yongwoo Song, Kimin Lee

**Abstract**: We investigate long-context vulnerabilities in Large Language Models (LLMs) through Many-Shot Jailbreaking (MSJ). Our experiments utilize context length of up to 128K tokens. Through comprehensive analysis with various many-shot attack settings with different instruction styles, shot density, topic, and format, we reveal that context length is the primary factor determining attack effectiveness. Critically, we find that successful attacks do not require carefully crafted harmful content. Even repetitive shots or random dummy text can circumvent model safety measures, suggesting fundamental limitations in long-context processing capabilities of LLMs. The safety behavior of well-aligned models becomes increasingly inconsistent with longer contexts. These findings highlight significant safety gaps in context expansion capabilities of LLMs, emphasizing the need for new safety mechanisms.



## **24. VisCRA: A Visual Chain Reasoning Attack for Jailbreaking Multimodal Large Language Models**

cs.CV

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19684v1) [paper-pdf](http://arxiv.org/pdf/2505.19684v1)

**Authors**: Bingrui Sima, Linhua Cong, Wenxuan Wang, Kun He

**Abstract**: The emergence of Multimodal Large Language Models (MLRMs) has enabled sophisticated visual reasoning capabilities by integrating reinforcement learning and Chain-of-Thought (CoT) supervision. However, while these enhanced reasoning capabilities improve performance, they also introduce new and underexplored safety risks. In this work, we systematically investigate the security implications of advanced visual reasoning in MLRMs. Our analysis reveals a fundamental trade-off: as visual reasoning improves, models become more vulnerable to jailbreak attacks. Motivated by this critical finding, we introduce VisCRA (Visual Chain Reasoning Attack), a novel jailbreak framework that exploits the visual reasoning chains to bypass safety mechanisms. VisCRA combines targeted visual attention masking with a two-stage reasoning induction strategy to precisely control harmful outputs. Extensive experiments demonstrate VisCRA's significant effectiveness, achieving high attack success rates on leading closed-source MLRMs: 76.48% on Gemini 2.0 Flash Thinking, 68.56% on QvQ-Max, and 56.60% on GPT-4o. Our findings highlight a critical insight: the very capability that empowers MLRMs -- their visual reasoning -- can also serve as an attack vector, posing significant security risks.



## **25. Your Language Model Can Secretly Write Like Humans: Contrastive Paraphrase Attacks on LLM-Generated Text Detectors**

cs.CL

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.15337v2) [paper-pdf](http://arxiv.org/pdf/2505.15337v2)

**Authors**: Hao Fang, Jiawei Kong, Tianqu Zhuang, Yixiang Qiu, Kuofeng Gao, Bin Chen, Shu-Tao Xia, Yaowei Wang, Min Zhang

**Abstract**: The misuse of large language models (LLMs), such as academic plagiarism, has driven the development of detectors to identify LLM-generated texts. To bypass these detectors, paraphrase attacks have emerged to purposely rewrite these texts to evade detection. Despite the success, existing methods require substantial data and computational budgets to train a specialized paraphraser, and their attack efficacy greatly reduces when faced with advanced detection algorithms. To address this, we propose \textbf{Co}ntrastive \textbf{P}araphrase \textbf{A}ttack (CoPA), a training-free method that effectively deceives text detectors using off-the-shelf LLMs. The first step is to carefully craft instructions that encourage LLMs to produce more human-like texts. Nonetheless, we observe that the inherent statistical biases of LLMs can still result in some generated texts carrying certain machine-like attributes that can be captured by detectors. To overcome this, CoPA constructs an auxiliary machine-like word distribution as a contrast to the human-like distribution generated by the LLM. By subtracting the machine-like patterns from the human-like distribution during the decoding process, CoPA is able to produce sentences that are less discernible by text detectors. Our theoretical analysis suggests the superiority of the proposed attack. Extensive experiments validate the effectiveness of CoPA in fooling text detectors across various scenarios.



## **26. Separate the Wheat from the Chaff: A Post-Hoc Approach to Safety Re-Alignment for Fine-Tuned Language Models**

cs.CL

16 pages, 14 figures. Camera-ready for ACL2025 findings

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2412.11041v3) [paper-pdf](http://arxiv.org/pdf/2412.11041v3)

**Authors**: Di Wu, Xin Lu, Yanyan Zhao, Bing Qin

**Abstract**: Although large language models (LLMs) achieve effective safety alignment at the time of release, they still face various safety challenges. A key issue is that fine-tuning often compromises the safety alignment of LLMs. To address this issue, we propose a method named IRR (Identify, Remove, and Recalibrate for Safety Realignment) that performs safety realignment for LLMs. The core of IRR is to identify and remove unsafe delta parameters from the fine-tuned models, while recalibrating the retained ones. We evaluate the effectiveness of IRR across various datasets, including both full fine-tuning and LoRA methods. Our results demonstrate that IRR significantly enhances the safety performance of fine-tuned models on safety benchmarks, such as harmful queries and jailbreak attacks, while maintaining their performance on downstream tasks. The source code is available at: https://anonymous.4open.science/r/IRR-BD4F.



## **27. Evaluating Robustness of Large Audio Language Models to Audio Injection: An Empirical Study**

cs.CL

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19598v1) [paper-pdf](http://arxiv.org/pdf/2505.19598v1)

**Authors**: Guanyu Hou, Jiaming He, Yinhang Zhou, Ji Guo, Yitong Qiao, Rui Zhang, Wenbo Jiang

**Abstract**: Large Audio-Language Models (LALMs) are increasingly deployed in real-world applications, yet their robustness against malicious audio injection attacks remains underexplored. This study systematically evaluates five leading LALMs across four attack scenarios: Audio Interference Attack, Instruction Following Attack, Context Injection Attack, and Judgment Hijacking Attack. Using metrics like Defense Success Rate, Context Robustness Score, and Judgment Robustness Index, their vulnerabilities and resilience were quantitatively assessed. Experimental results reveal significant performance disparities among models; no single model consistently outperforms others across all attack types. The position of malicious content critically influences attack effectiveness, particularly when placed at the beginning of sequences. A negative correlation between instruction-following capability and robustness suggests models adhering strictly to instructions may be more susceptible, contrasting with greater resistance by safety-aligned models. Additionally, system prompts show mixed effectiveness, indicating the need for tailored strategies. This work introduces a benchmark framework and highlights the importance of integrating robustness into training pipelines. Findings emphasize developing multi-modal defenses and architectural designs that decouple capability from susceptibility for secure LALMs deployment.



## **28. Guard Me If You Know Me: Protecting Specific Face-Identity from Deepfakes**

cs.CV

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.19582v1) [paper-pdf](http://arxiv.org/pdf/2505.19582v1)

**Authors**: Kaiqing Lin, Zhiyuan Yan, Ke-Yue Zhang, Li Hao, Yue Zhou, Yuzhen Lin, Weixiang Li, Taiping Yao, Shouhong Ding, Bin Li

**Abstract**: Securing personal identity against deepfake attacks is increasingly critical in the digital age, especially for celebrities and political figures whose faces are easily accessible and frequently targeted. Most existing deepfake detection methods focus on general-purpose scenarios and often ignore the valuable prior knowledge of known facial identities, e.g., "VIP individuals" whose authentic facial data are already available. In this paper, we propose \textbf{VIPGuard}, a unified multimodal framework designed to capture fine-grained and comprehensive facial representations of a given identity, compare them against potentially fake or similar-looking faces, and reason over these comparisons to make accurate and explainable predictions. Specifically, our framework consists of three main stages. First, fine-tune a multimodal large language model (MLLM) to learn detailed and structural facial attributes. Second, we perform identity-level discriminative learning to enable the model to distinguish subtle differences between highly similar faces, including real and fake variations. Finally, we introduce user-specific customization, where we model the unique characteristics of the target face identity and perform semantic reasoning via MLLM to enable personalized and explainable deepfake detection. Our framework shows clear advantages over previous detection works, where traditional detectors mainly rely on low-level visual cues and provide no human-understandable explanations, while other MLLM-based models often lack a detailed understanding of specific face identities. To facilitate the evaluation of our method, we built a comprehensive identity-aware benchmark called \textbf{VIPBench} for personalized deepfake detection, involving the latest 7 face-swapping and 7 entire face synthesis techniques for generation.



## **29. Robo-Troj: Attacking LLM-based Task Planners**

cs.RO

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2504.17070v2) [paper-pdf](http://arxiv.org/pdf/2504.17070v2)

**Authors**: Mohaiminul Al Nahian, Zainab Altaweel, David Reitano, Sabbir Ahmed, Shiqi Zhang, Adnan Siraj Rakin

**Abstract**: Robots need task planning methods to achieve goals that require more than individual actions. Recently, large language models (LLMs) have demonstrated impressive performance in task planning. LLMs can generate a step-by-step solution using a description of actions and the goal. Despite the successes in LLM-based task planning, there is limited research studying the security aspects of those systems. In this paper, we develop Robo-Troj, the first multi-trigger backdoor attack for LLM-based task planners, which is the main contribution of this work. As a multi-trigger attack, Robo-Troj is trained to accommodate the diversity of robot application domains. For instance, one can use unique trigger words, e.g., "herical", to activate a specific malicious behavior, e.g., cutting hand on a kitchen robot. In addition, we develop an optimization method for selecting the trigger words that are most effective. Through demonstrating the vulnerability of LLM-based planners, we aim to promote the development of secured robot systems.



## **30. One-Shot is Enough: Consolidating Multi-Turn Attacks into Efficient Single-Turn Prompts for LLMs**

cs.CL

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2503.04856v2) [paper-pdf](http://arxiv.org/pdf/2503.04856v2)

**Authors**: Junwoo Ha, Hyunjun Kim, Sangyoon Yu, Haon Park, Ashkan Yousefpour, Yuna Park, Suhyun Kim

**Abstract**: We introduce a novel framework for consolidating multi-turn adversarial ``jailbreak'' prompts into single-turn queries, significantly reducing the manual overhead required for adversarial testing of large language models (LLMs). While multi-turn human jailbreaks have been shown to yield high attack success rates, they demand considerable human effort and time. Our multi-turn-to-single-turn (M2S) methods -- Hyphenize, Numberize, and Pythonize -- systematically reformat multi-turn dialogues into structured single-turn prompts. Despite removing iterative back-and-forth interactions, these prompts preserve and often enhance adversarial potency: in extensive evaluations on the Multi-turn Human Jailbreak (MHJ) dataset, M2S methods achieve attack success rates from 70.6 percent to 95.9 percent across several state-of-the-art LLMs. Remarkably, the single-turn prompts outperform the original multi-turn attacks by as much as 17.5 percentage points while cutting token usage by more than half on average. Further analysis shows that embedding malicious requests in enumerated or code-like structures exploits ``contextual blindness'', bypassing both native guardrails and external input-output filters. By converting multi-turn conversations into concise single-turn prompts, the M2S framework provides a scalable tool for large-scale red teaming and reveals critical weaknesses in contemporary LLM defenses.



## **31. Three Minds, One Legend: Jailbreak Large Reasoning Model with Adaptive Stacked Ciphers**

cs.CL

**SubmitDate**: 2025-05-26    [abs](http://arxiv.org/abs/2505.16241v3) [paper-pdf](http://arxiv.org/pdf/2505.16241v3)

**Authors**: Viet-Anh Nguyen, Shiqian Zhao, Gia Dao, Runyi Hu, Yi Xie, Luu Anh Tuan

**Abstract**: Recently, Large Reasoning Models (LRMs) have demonstrated superior logical capabilities compared to traditional Large Language Models (LLMs), gaining significant attention. Despite their impressive performance, the potential for stronger reasoning abilities to introduce more severe security vulnerabilities remains largely underexplored. Existing jailbreak methods often struggle to balance effectiveness with robustness against adaptive safety mechanisms. In this work, we propose SEAL, a novel jailbreak attack that targets LRMs through an adaptive encryption pipeline designed to override their reasoning processes and evade potential adaptive alignment. Specifically, SEAL introduces a stacked encryption approach that combines multiple ciphers to overwhelm the models reasoning capabilities, effectively bypassing built-in safety mechanisms. To further prevent LRMs from developing countermeasures, we incorporate two dynamic strategies - random and adaptive - that adjust the cipher length, order, and combination. Extensive experiments on real-world reasoning models, including DeepSeek-R1, Claude Sonnet, and OpenAI GPT-o4, validate the effectiveness of our approach. Notably, SEAL achieves an attack success rate of 80.8% on GPT o4-mini, outperforming state-of-the-art baselines by a significant margin of 27.2%. Warning: This paper contains examples of inappropriate, offensive, and harmful content.



## **32. Benign Samples Matter! Fine-tuning On Outlier Benign Samples Severely Breaks Safety**

cs.LG

26 pages, 13 figures

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2505.06843v2) [paper-pdf](http://arxiv.org/pdf/2505.06843v2)

**Authors**: Zihan Guan, Mengxuan Hu, Ronghang Zhu, Sheng Li, Anil Vullikanti

**Abstract**: Recent studies have uncovered a troubling vulnerability in the fine-tuning stage of large language models (LLMs): even fine-tuning on entirely benign datasets can lead to a significant increase in the harmfulness of LLM outputs. Building on this finding, our red teaming study takes this threat one step further by developing a more effective attack. Specifically, we analyze and identify samples within benign datasets that contribute most to safety degradation, then fine-tune LLMs exclusively on these samples. We approach this problem from an outlier detection perspective and propose Self-Inf-N, to detect and extract outliers for fine-tuning. Our findings reveal that fine-tuning LLMs on 100 outlier samples selected by Self-Inf-N in the benign datasets severely compromises LLM safety alignment. Extensive experiments across seven mainstream LLMs demonstrate that our attack exhibits high transferability across different architectures and remains effective in practical scenarios. Alarmingly, our results indicate that most existing mitigation strategies fail to defend against this attack, underscoring the urgent need for more robust alignment safeguards. Codes are available at https://github.com/GuanZihan/Benign-Samples-Matter.



## **33. Latent-space adversarial training with post-aware calibration for defending large language models against jailbreak attacks**

cs.CR

Under Review

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2501.10639v2) [paper-pdf](http://arxiv.org/pdf/2501.10639v2)

**Authors**: Xin Yi, Yue Li, dongsheng Shi, Linlin Wang, Xiaoling Wang, Liang He

**Abstract**: Ensuring safety alignment has become a critical requirement for large language models (LLMs), particularly given their widespread deployment in real-world applications. However, LLMs remain susceptible to jailbreak attacks, which exploit system vulnerabilities to bypass safety measures and generate harmful outputs. Although numerous defense mechanisms based on adversarial training have been proposed, a persistent challenge lies in the exacerbation of over-refusal behaviors, which compromise the overall utility of the model. To address these challenges, we propose a Latent-space Adversarial Training with Post-aware Calibration (LATPC) framework. During the adversarial training phase, LATPC compares harmful and harmless instructions in the latent space and extracts safety-critical dimensions to construct refusal features attack, precisely simulating agnostic jailbreak attack types requiring adversarial mitigation. At the inference stage, an embedding-level calibration mechanism is employed to alleviate over-refusal behaviors with minimal computational overhead. Experimental results demonstrate that, compared to various defense methods across five types of jailbreak attacks, LATPC framework achieves a superior balance between safety and utility. Moreover, our analysis underscores the effectiveness of extracting safety-critical dimensions from the latent space for constructing robust refusal feature attacks.



## **34. An Embarrassingly Simple Defense Against LLM Abliteration Attacks**

cs.CL

preprint

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2505.19056v1) [paper-pdf](http://arxiv.org/pdf/2505.19056v1)

**Authors**: Harethah Abu Shairah, Hasan Abed Al Kader Hammoud, Bernard Ghanem, George Turkiyyah

**Abstract**: Large language models (LLMs) are typically aligned to comply with safety guidelines by refusing harmful instructions. A recent attack, termed abliteration, isolates and suppresses the single latent direction most responsible for refusal behavior, enabling the model to generate unethical content. We propose a defense that modifies how models generate refusals. We construct an extended-refusal dataset that contains harmful prompts with a full response that justifies the reason for refusal. We then fine-tune Llama-2-7B-Chat and Qwen2.5-Instruct (1.5B and 3B parameters) on our extended-refusal dataset, and evaluate the resulting systems on a set of harmful prompts. In our experiments, extended-refusal models maintain high refusal rates, dropping at most by 10%, whereas baseline models' refusal rates drop by 70-80% after abliteration. A broad evaluation of safety and utility shows that extended-refusal fine-tuning neutralizes the abliteration attack while preserving general performance.



## **35. LLMs know their vulnerabilities: Uncover Safety Gaps through Natural Distribution Shifts**

cs.CL

ACL 2025 main conference. Code is available at  https://github.com/AI45Lab/ActorAttack

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2410.10700v2) [paper-pdf](http://arxiv.org/pdf/2410.10700v2)

**Authors**: Qibing Ren, Hao Li, Dongrui Liu, Zhanxu Xie, Xiaoya Lu, Yu Qiao, Lei Sha, Junchi Yan, Lizhuang Ma, Jing Shao

**Abstract**: Safety concerns in large language models (LLMs) have gained significant attention due to their exposure to potentially harmful data during pre-training. In this paper, we identify a new safety vulnerability in LLMs: their susceptibility to \textit{natural distribution shifts} between attack prompts and original toxic prompts, where seemingly benign prompts, semantically related to harmful content, can bypass safety mechanisms. To explore this issue, we introduce a novel attack method, \textit{ActorBreaker}, which identifies actors related to toxic prompts within pre-training distribution to craft multi-turn prompts that gradually lead LLMs to reveal unsafe content. ActorBreaker is grounded in Latour's actor-network theory, encompassing both human and non-human actors to capture a broader range of vulnerabilities. Our experimental results demonstrate that ActorBreaker outperforms existing attack methods in terms of diversity, effectiveness, and efficiency across aligned LLMs. To address this vulnerability, we propose expanding safety training to cover a broader semantic space of toxic content. We thus construct a multi-turn safety dataset using ActorBreaker. Fine-tuning models on our dataset shows significant improvements in robustness, though with some trade-offs in utility. Code is available at https://github.com/AI45Lab/ActorAttack.



## **36. GhostPrompt: Jailbreaking Text-to-image Generative Models based on Dynamic Optimization**

cs.LG

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2505.18979v1) [paper-pdf](http://arxiv.org/pdf/2505.18979v1)

**Authors**: Zixuan Chen, Hao Lin, Ke Xu, Xinghao Jiang, Tanfeng Sun

**Abstract**: Text-to-image (T2I) generation models can inadvertently produce not-safe-for-work (NSFW) content, prompting the integration of text and image safety filters. Recent advances employ large language models (LLMs) for semantic-level detection, rendering traditional token-level perturbation attacks largely ineffective. However, our evaluation shows that existing jailbreak methods are ineffective against these modern filters. We introduce GhostPrompt, the first automated jailbreak framework that combines dynamic prompt optimization with multimodal feedback. It consists of two key components: (i) Dynamic Optimization, an iterative process that guides a large language model (LLM) using feedback from text safety filters and CLIP similarity scores to generate semantically aligned adversarial prompts; and (ii) Adaptive Safety Indicator Injection, which formulates the injection of benign visual cues as a reinforcement learning problem to bypass image-level filters. GhostPrompt achieves state-of-the-art performance, increasing the ShieldLM-7B bypass rate from 12.5\% (Sneakyprompt) to 99.0\%, improving CLIP score from 0.2637 to 0.2762, and reducing the time cost by $4.2 \times$. Moreover, it generalizes to unseen filters including GPT-4.1 and successfully jailbreaks DALLE 3 to generate NSFW images in our evaluation, revealing systemic vulnerabilities in current multimodal defenses. To support further research on AI safety and red-teaming, we will release code and adversarial prompts under a controlled-access protocol.



## **37. Exemplifying Emerging Phishing: QR-based Browser-in-The-Browser (BiTB) Attack**

cs.CR

This manuscript is of 5 pages including 7 figures and 2 algorithms

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2505.18944v1) [paper-pdf](http://arxiv.org/pdf/2505.18944v1)

**Authors**: Muhammad Wahid Akram, Keshav Sood, Muneeb Ul Hassan, Basant Subba

**Abstract**: Lately, cybercriminals constantly formulate productive approaches to exploit individuals. This article exemplifies an innovative attack, namely QR-based Browser-in-The-Browser (BiTB), using proficiencies of Large Language Model (LLM) i.e. Google Gemini. The presented attack is a fusion of two emerging attacks: BiTB and Quishing (QR code phishing). Our study underscores attack's simplistic implementation utilizing malicious prompts provided to Gemini-LLM. Moreover, we presented a case study to highlight a lucrative attack method, we also performed an experiment to comprehend the attack execution on victims' device. The findings of this work obligate the researchers' contributions in confronting this type of phishing attempts through LLMs.



## **38. Stronger Enforcement of Instruction Hierarchy via Augmented Intermediate Representations**

cs.AI

**SubmitDate**: 2025-05-25    [abs](http://arxiv.org/abs/2505.18907v1) [paper-pdf](http://arxiv.org/pdf/2505.18907v1)

**Authors**: Sanjay Kariyappa, G. Edward Suh

**Abstract**: Prompt injection attacks are a critical security vulnerability in large language models (LLMs), allowing attackers to hijack model behavior by injecting malicious instructions within the input context. Recent defense mechanisms have leveraged an Instruction Hierarchy (IH) Signal, often implemented through special delimiter tokens or additive embeddings to denote the privilege level of input tokens. However, these prior works typically inject the IH signal exclusively at the initial input layer, which we hypothesize limits its ability to effectively distinguish the privilege levels of tokens as it propagates through the different layers of the model. To overcome this limitation, we introduce a novel approach that injects the IH signal into the intermediate token representations within the network. Our method augments these representations with layer-specific trainable embeddings that encode the privilege information. Our evaluations across multiple models and training methods reveal that our proposal yields between $1.6\times$ and $9.2\times$ reduction in attack success rate on gradient-based prompt injection attacks compared to state-of-the-art methods, without significantly degrading the model's utility.



## **39. Security Concerns for Large Language Models: A Survey**

cs.CR

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2505.18889v1) [paper-pdf](http://arxiv.org/pdf/2505.18889v1)

**Authors**: Miles Q. Li, Benjamin C. M. Fung

**Abstract**: Large Language Models (LLMs) such as GPT-4 (and its recent iterations like GPT-4o and the GPT-4.1 series), Google's Gemini, Anthropic's Claude 3 models, and xAI's Grok have caused a revolution in natural language processing, but their capabilities also introduce new security vulnerabilities. In this survey, we provide a comprehensive overview of the emerging security concerns around LLMs, categorizing threats into prompt injection and jailbreaking, adversarial attacks (including input perturbations and data poisoning), misuse by malicious actors (e.g., for disinformation, phishing, and malware generation), and worrisome risks inherent in autonomous LLM agents. A significant focus has been recently placed on the latter, exploring goal misalignment, emergent deception, self-preservation instincts, and the potential for LLMs to develop and pursue covert, misaligned objectives (scheming), which may even persist through safety training. We summarize recent academic and industrial studies (2022-2025) that exemplify each threat, analyze proposed defenses and their limitations, and identify open challenges in securing LLM-based applications. We conclude by emphasizing the importance of advancing robust, multi-layered security strategies to ensure LLMs are safe and beneficial.



## **40. Audio Jailbreak Attacks: Exposing Vulnerabilities in SpeechGPT in a White-Box Framework**

cs.CL

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2505.18864v1) [paper-pdf](http://arxiv.org/pdf/2505.18864v1)

**Authors**: Binhao Ma, Hanqing Guo, Zhengping Jay Luo, Rui Duan

**Abstract**: Recent advances in Multimodal Large Language Models (MLLMs) have significantly enhanced the naturalness and flexibility of human computer interaction by enabling seamless understanding across text, vision, and audio modalities. Among these, voice enabled models such as SpeechGPT have demonstrated considerable improvements in usability, offering expressive, and emotionally responsive interactions that foster deeper connections in real world communication scenarios. However, the use of voice introduces new security risks, as attackers can exploit the unique characteristics of spoken language, such as timing, pronunciation variability, and speech to text translation, to craft inputs that bypass defenses in ways not seen in text-based systems. Despite substantial research on text based jailbreaks, the voice modality remains largely underexplored in terms of both attack strategies and defense mechanisms. In this work, we present an adversarial attack targeting the speech input of aligned MLLMs in a white box scenario. Specifically, we introduce a novel token level attack that leverages access to the model's speech tokenization to generate adversarial token sequences. These sequences are then synthesized into audio prompts, which effectively bypass alignment safeguards and to induce prohibited outputs. Evaluated on SpeechGPT, our approach achieves up to 89 percent attack success rate across multiple restricted tasks, significantly outperforming existing voice based jailbreak methods. Our findings shed light on the vulnerabilities of voice-enabled multimodal systems and to help guide the development of more robust next-generation MLLMs.



## **41. Strong Membership Inference Attacks on Massive Datasets and (Moderately) Large Language Models**

cs.CR

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2505.18773v1) [paper-pdf](http://arxiv.org/pdf/2505.18773v1)

**Authors**: Jamie Hayes, Ilia Shumailov, Christopher A. Choquette-Choo, Matthew Jagielski, George Kaissis, Katherine Lee, Milad Nasr, Sahra Ghalebikesabi, Niloofar Mireshghallah, Meenatchi Sundaram Mutu Selva Annamalai, Igor Shilov, Matthieu Meeus, Yves-Alexandre de Montjoye, Franziska Boenisch, Adam Dziedzic, A. Feder Cooper

**Abstract**: State-of-the-art membership inference attacks (MIAs) typically require training many reference models, making it difficult to scale these attacks to large pre-trained language models (LLMs). As a result, prior research has either relied on weaker attacks that avoid training reference models (e.g., fine-tuning attacks), or on stronger attacks applied to small-scale models and datasets. However, weaker attacks have been shown to be brittle - achieving close-to-arbitrary success - and insights from strong attacks in simplified settings do not translate to today's LLMs. These challenges have prompted an important question: are the limitations observed in prior work due to attack design choices, or are MIAs fundamentally ineffective on LLMs? We address this question by scaling LiRA - one of the strongest MIAs - to GPT-2 architectures ranging from 10M to 1B parameters, training reference models on over 20B tokens from the C4 dataset. Our results advance the understanding of MIAs on LLMs in three key ways: (1) strong MIAs can succeed on pre-trained LLMs; (2) their effectiveness, however, remains limited (e.g., AUC<0.7) in practical settings; and, (3) the relationship between MIA success and related privacy metrics is not as straightforward as prior work has suggested.



## **42. Attacking Vision-Language Computer Agents via Pop-ups**

cs.CL

ACL 2025

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2411.02391v2) [paper-pdf](http://arxiv.org/pdf/2411.02391v2)

**Authors**: Yanzhe Zhang, Tao Yu, Diyi Yang

**Abstract**: Autonomous agents powered by large vision and language models (VLM) have demonstrated significant potential in completing daily computer tasks, such as browsing the web to book travel and operating desktop software, which requires agents to understand these interfaces. Despite such visual inputs becoming more integrated into agentic applications, what types of risks and attacks exist around them still remain unclear. In this work, we demonstrate that VLM agents can be easily attacked by a set of carefully designed adversarial pop-ups, which human users would typically recognize and ignore. This distraction leads agents to click these pop-ups instead of performing their tasks as usual. Integrating these pop-ups into existing agent testing environments like OSWorld and VisualWebArena leads to an attack success rate (the frequency of the agent clicking the pop-ups) of 86% on average and decreases the task success rate by 47%. Basic defense techniques, such as asking the agent to ignore pop-ups or including an advertisement notice, are ineffective against the attack.



## **43. Sugar-Coated Poison: Benign Generation Unlocks LLM Jailbreaking**

cs.CR

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2504.05652v2) [paper-pdf](http://arxiv.org/pdf/2504.05652v2)

**Authors**: Yu-Hang Wu, Yu-Jie Xiong, Hao Zhang, Jia-Chen Zhang, Zheng Zhou

**Abstract**: With the increasingly deep integration of large language models (LLMs) across diverse domains, the effectiveness of their safety mechanisms is encountering severe challenges. Currently, jailbreak attacks based on prompt engineering have become a major safety threat. However, existing methods primarily rely on black-box manipulation of prompt templates, resulting in poor interpretability and limited generalization. To break through the bottleneck, this study first introduces the concept of Defense Threshold Decay (DTD), revealing the potential safety impact caused by LLMs' benign generation: as benign content generation in LLMs increases, the model's focus on input instructions progressively diminishes. Building on this insight, we propose the Sugar-Coated Poison (SCP) attack paradigm, which uses a "semantic reversal" strategy to craft benign inputs that are opposite in meaning to malicious intent. This strategy induces the models to generate extensive benign content, thereby enabling adversarial reasoning to bypass safety mechanisms. Experiments show that SCP outperforms existing baselines. Remarkably, it achieves an average attack success rate of 87.23% across six LLMs. For defense, we propose Part-of-Speech Defense (POSD), leveraging verb-noun dependencies for syntactic analysis to enhance safety of LLMs while preserving their generalization ability.



## **44. Revisiting Model Inversion Evaluation: From Misleading Standards to Reliable Privacy Assessment**

cs.LG

To support future work, we release our MLLM-based MI evaluation  framework and benchmarking suite at  https://www.kaggle.com/datasets/hosytuyen/mi-reconstruction-collection

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2505.03519v3) [paper-pdf](http://arxiv.org/pdf/2505.03519v3)

**Authors**: Sy-Tuyen Ho, Koh Jun Hao, Ngoc-Bao Nguyen, Alexander Binder, Ngai-Man Cheung

**Abstract**: Model Inversion (MI) attacks aim to reconstruct information from private training data by exploiting access to machine learning models T. To evaluate such attacks, the standard evaluation framework for such attacks relies on an evaluation model E, trained under the same task design as T. This framework has become the de facto standard for assessing progress in MI research, used across nearly all recent MI attacks and defenses without question. In this paper, we present the first in-depth study of this MI evaluation framework. In particular, we identify a critical issue of this standard MI evaluation framework: Type-I adversarial examples. These are reconstructions that do not capture the visual features of private training data, yet are still deemed successful by the target model T and ultimately transferable to E. Such false positives undermine the reliability of the standard MI evaluation framework. To address this issue, we introduce a new MI evaluation framework that replaces the evaluation model E with advanced Multimodal Large Language Models (MLLMs). By leveraging their general-purpose visual understanding, our MLLM-based framework does not depend on training of shared task design as in T, thus reducing Type-I transferability and providing more faithful assessments of reconstruction success. Using our MLLM-based evaluation framework, we reevaluate 26 diverse MI attack setups and empirically reveal consistently high false positive rates under the standard evaluation framework. Importantly, we demonstrate that many state-of-the-art (SOTA) MI methods report inflated attack accuracy, indicating that actual privacy leakage is significantly lower than previously believed. By uncovering this critical issue and proposing a robust solution, our work enables a reassessment of progress in MI research and sets a new standard for reliable and robust evaluation.



## **45. $PD^3F$: A Pluggable and Dynamic DoS-Defense Framework Against Resource Consumption Attacks Targeting Large Language Models**

cs.CR

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2505.18680v1) [paper-pdf](http://arxiv.org/pdf/2505.18680v1)

**Authors**: Yuanhe Zhang, Xinyue Wang, Haoran Gao, Zhenhong Zhou, Fanyu Meng, Yuyao Zhang, Sen Su

**Abstract**: Large Language Models (LLMs), due to substantial computational requirements, are vulnerable to resource consumption attacks, which can severely degrade server performance or even cause crashes, as demonstrated by denial-of-service (DoS) attacks designed for LLMs. However, existing works lack mitigation strategies against such threats, resulting in unresolved security risks for real-world LLM deployments. To this end, we propose the Pluggable and Dynamic DoS-Defense Framework ($PD^3F$), which employs a two-stage approach to defend against resource consumption attacks from both the input and output sides. On the input side, we propose the Resource Index to guide Dynamic Request Polling Scheduling, thereby reducing resource usage induced by malicious attacks under high-concurrency scenarios. On the output side, we introduce the Adaptive End-Based Suppression mechanism, which terminates excessive malicious generation early. Experiments across six models demonstrate that $PD^3F$ significantly mitigates resource consumption attacks, improving users' access capacity by up to 500% during adversarial load. $PD^3F$ represents a step toward the resilient and resource-aware deployment of LLMs against resource consumption attacks.



## **46. Can LLM Watermarks Robustly Prevent Unauthorized Knowledge Distillation?**

cs.CL

Accepted by ACL 2025 (Main)

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2502.11598v2) [paper-pdf](http://arxiv.org/pdf/2502.11598v2)

**Authors**: Leyi Pan, Aiwei Liu, Shiyu Huang, Yijian Lu, Xuming Hu, Lijie Wen, Irwin King, Philip S. Yu

**Abstract**: The radioactive nature of Large Language Model (LLM) watermarking enables the detection of watermarks inherited by student models when trained on the outputs of watermarked teacher models, making it a promising tool for preventing unauthorized knowledge distillation. However, the robustness of watermark radioactivity against adversarial actors remains largely unexplored. In this paper, we investigate whether student models can acquire the capabilities of teacher models through knowledge distillation while avoiding watermark inheritance. We propose two categories of watermark removal approaches: pre-distillation removal through untargeted and targeted training data paraphrasing (UP and TP), and post-distillation removal through inference-time watermark neutralization (WN). Extensive experiments across multiple model pairs, watermarking schemes and hyper-parameter settings demonstrate that both TP and WN thoroughly eliminate inherited watermarks, with WN achieving this while maintaining knowledge transfer efficiency and low computational overhead. Given the ongoing deployment of watermarking techniques in production LLMs, these findings emphasize the urgent need for more robust defense strategies. Our code is available at https://github.com/THU-BPM/Watermark-Radioactivity-Attack.



## **47. Safety Alignment via Constrained Knowledge Unlearning**

cs.CL

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2505.18588v1) [paper-pdf](http://arxiv.org/pdf/2505.18588v1)

**Authors**: Zesheng Shi, Yucheng Zhou, Jing Li

**Abstract**: Despite significant progress in safety alignment, large language models (LLMs) remain susceptible to jailbreak attacks. Existing defense mechanisms have not fully deleted harmful knowledge in LLMs, which allows such attacks to bypass safeguards and produce harmful outputs. To address this challenge, we propose a novel safety alignment strategy, Constrained Knowledge Unlearning (CKU), which focuses on two primary objectives: knowledge localization and retention, and unlearning harmful knowledge. CKU works by scoring neurons in specific multilayer perceptron (MLP) layers to identify a subset U of neurons associated with useful knowledge. During the unlearning process, CKU prunes the gradients of neurons in U to preserve valuable knowledge while effectively mitigating harmful content. Experimental results demonstrate that CKU significantly enhances model safety without compromising overall performance, offering a superior balance between safety and utility compared to existing methods. Additionally, our analysis of neuron knowledge sensitivity across various MLP layers provides valuable insights into the mechanics of safety alignment and model knowledge editing.



## **48. MASTER: Multi-Agent Security Through Exploration of Roles and Topological Structures -- A Comprehensive Framework**

cs.MA

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2505.18572v1) [paper-pdf](http://arxiv.org/pdf/2505.18572v1)

**Authors**: Yifan Zhu, Chao Zhang, Xin Shi, Xueqiao Zhang, Yi Yang, Yawei Luo

**Abstract**: Large Language Models (LLMs)-based Multi-Agent Systems (MAS) exhibit remarkable problem-solving and task planning capabilities across diverse domains due to their specialized agentic roles and collaborative interactions. However, this also amplifies the severity of security risks under MAS attacks. To address this, we introduce MASTER, a novel security research framework for MAS, focusing on diverse Role configurations and Topological structures across various scenarios. MASTER offers an automated construction process for different MAS setups and an information-flow-based interaction paradigm. To tackle MAS security challenges in varied scenarios, we design a scenario-adaptive, extensible attack strategy utilizing role and topological information, which dynamically allocates targeted, domain-specific attack tasks for collaborative agent execution. Our experiments demonstrate that such an attack, leveraging role and topological information, exhibits significant destructive potential across most models. Additionally, we propose corresponding defense strategies, substantially enhancing MAS resilience across diverse scenarios. We anticipate that our framework and findings will provide valuable insights for future research into MAS security challenges.



## **49. Exploring the Vulnerability of the Content Moderation Guardrail in Large Language Models via Intent Manipulation**

cs.CL

Preprint, under review. TL;DR: We propose a new two-stage  intent-based prompt-refinement framework, IntentPrompt, that aims to explore  the vulnerability of LLMs' content moderation guardrails by refining prompts  into benign-looking declarative forms via intent manipulation for red-teaming  purposes

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2505.18556v1) [paper-pdf](http://arxiv.org/pdf/2505.18556v1)

**Authors**: Jun Zhuang, Haibo Jin, Ye Zhang, Zhengjian Kang, Wenbin Zhang, Gaby G. Dagher, Haohan Wang

**Abstract**: Intent detection, a core component of natural language understanding, has considerably evolved as a crucial mechanism in safeguarding large language models (LLMs). While prior work has applied intent detection to enhance LLMs' moderation guardrails, showing a significant success against content-level jailbreaks, the robustness of these intent-aware guardrails under malicious manipulations remains under-explored. In this work, we investigate the vulnerability of intent-aware guardrails and demonstrate that LLMs exhibit implicit intent detection capabilities. We propose a two-stage intent-based prompt-refinement framework, IntentPrompt, that first transforms harmful inquiries into structured outlines and further reframes them into declarative-style narratives by iteratively optimizing prompts via feedback loops to enhance jailbreak success for red-teaming purposes. Extensive experiments across four public benchmarks and various black-box LLMs indicate that our framework consistently outperforms several cutting-edge jailbreak methods and evades even advanced Intent Analysis (IA) and Chain-of-Thought (CoT)-based defenses. Specifically, our "FSTR+SPIN" variant achieves attack success rates ranging from 88.25% to 96.54% against CoT-based defenses on the o1 model, and from 86.75% to 97.12% on the GPT-4o model under IA-based defenses. These findings highlight a critical weakness in LLMs' safety mechanisms and suggest that intent manipulation poses a growing challenge to content moderation guardrails.



## **50. From ML to LLM: Evaluating the Robustness of Phishing Webpage Detection Models against Adversarial Attacks**

cs.CR

**SubmitDate**: 2025-05-24    [abs](http://arxiv.org/abs/2407.20361v4) [paper-pdf](http://arxiv.org/pdf/2407.20361v4)

**Authors**: Aditya Kulkarni, Vivek Balachandran, Dinil Mon Divakaran, Tamal Das

**Abstract**: Phishing attacks attempt to deceive users into stealing sensitive information, posing a significant cybersecurity threat. Advances in machine learning (ML) and deep learning (DL) have led to the development of numerous phishing webpage detection solutions, but these models remain vulnerable to adversarial attacks. Evaluating their robustness against adversarial phishing webpages is essential. Existing tools contain datasets of pre-designed phishing webpages for a limited number of brands, and lack diversity in phishing features.   To address these challenges, we develop PhishOracle, a tool that generates adversarial phishing webpages by embedding diverse phishing features into legitimate webpages. We evaluate the robustness of three existing task-specific models - Stack model, VisualPhishNet, and Phishpedia - against PhishOracle-generated adversarial phishing webpages and observe a significant drop in their detection rates. In contrast, a multimodal large language model (MLLM)-based phishing detector demonstrates stronger robustness against these adversarial attacks but still is prone to evasion. Our findings highlight the vulnerability of phishing detection models to adversarial attacks, emphasizing the need for more robust detection approaches. Furthermore, we conduct a user study to evaluate whether PhishOracle-generated adversarial phishing webpages can deceive users. The results show that many of these phishing webpages evade not only existing detection models but also users.



