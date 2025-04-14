# Latest Large Language Model Attack Papers
**update at 2025-04-14 11:09:28**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. MCP Safety Audit: LLMs with the Model Context Protocol Allow Major Security Exploits**

cs.CR

27 pages, 21 figures, and 2 Tables. Cleans up the TeX source

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.03767v2) [paper-pdf](http://arxiv.org/pdf/2504.03767v2)

**Authors**: Brandon Radosevich, John Halloran

**Abstract**: To reduce development overhead and enable seamless integration between potential components comprising any given generative AI application, the Model Context Protocol (MCP) (Anthropic, 2024) has recently been released and subsequently widely adopted. The MCP is an open protocol that standardizes API calls to large language models (LLMs), data sources, and agentic tools. By connecting multiple MCP servers, each defined with a set of tools, resources, and prompts, users are able to define automated workflows fully driven by LLMs. However, we show that the current MCP design carries a wide range of security risks for end users. In particular, we demonstrate that industry-leading LLMs may be coerced into using MCP tools to compromise an AI developer's system through various attacks, such as malicious code execution, remote access control, and credential theft. To proactively mitigate these and related attacks, we introduce a safety auditing tool, MCPSafetyScanner, the first agentic tool to assess the security of an arbitrary MCP server. MCPScanner uses several agents to (a) automatically determine adversarial samples given an MCP server's tools and resources; (b) search for related vulnerabilities and remediations based on those samples; and (c) generate a security report detailing all findings. Our work highlights serious security issues with general-purpose agentic workflows while also providing a proactive tool to audit MCP server safety and address detected vulnerabilities before deployment.   The described MCP server auditing tool, MCPSafetyScanner, is freely available at: https://github.com/johnhalloran321/mcpSafetyScanner



## **2. SCAM: A Real-World Typographic Robustness Evaluation for Multimodal Foundation Models**

cs.CV

Submitted to CVPR 2025 Workshop EVAL-FoMo-2

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.04893v2) [paper-pdf](http://arxiv.org/pdf/2504.04893v2)

**Authors**: Justus Westerhoff, Erblina Purelku, Jakob Hackstein, Leo Pinetzki, Lorenz Hufe

**Abstract**: Typographic attacks exploit the interplay between text and visual content in multimodal foundation models, causing misclassifications when misleading text is embedded within images. However, existing datasets are limited in size and diversity, making it difficult to study such vulnerabilities. In this paper, we introduce SCAM, the largest and most diverse dataset of real-world typographic attack images to date, containing 1,162 images across hundreds of object categories and attack words. Through extensive benchmarking of Vision-Language Models (VLMs) on SCAM, we demonstrate that typographic attacks significantly degrade performance, and identify that training data and model architecture influence the susceptibility to these attacks. Our findings reveal that typographic attacks persist in state-of-the-art Large Vision-Language Models (LVLMs) due to the choice of their vision encoder, though larger Large Language Models (LLMs) backbones help mitigate their vulnerability. Additionally, we demonstrate that synthetic attacks closely resemble real-world (handwritten) attacks, validating their use in research. Our work provides a comprehensive resource and empirical insights to facilitate future research toward robust and trustworthy multimodal AI systems. We publicly release the datasets introduced in this paper under https://huggingface.co/datasets/BLISS-e-V/SCAM, along with the code for evaluations at https://github.com/Bliss-e-V/SCAM.



## **3. GenXSS: an AI-Driven Framework for Automated Detection of XSS Attacks in WAFs**

cs.CR

**SubmitDate**: 2025-04-11    [abs](http://arxiv.org/abs/2504.08176v1) [paper-pdf](http://arxiv.org/pdf/2504.08176v1)

**Authors**: Vahid Babaey, Arun Ravindran

**Abstract**: The increasing reliance on web services has led to a rise in cybersecurity threats, particularly Cross-Site Scripting (XSS) attacks, which target client-side layers of web applications by injecting malicious scripts. Traditional Web Application Firewalls (WAFs) struggle to detect highly obfuscated and complex attacks, as their rules require manual updates. This paper presents a novel generative AI framework that leverages Large Language Models (LLMs) to enhance XSS mitigation. The framework achieves two primary objectives: (1) generating sophisticated and syntactically validated XSS payloads using in-context learning, and (2) automating defense mechanisms by testing these attacks against a vulnerable application secured by a WAF, classifying bypassing attacks, and generating effective WAF security rules. Experimental results using GPT-4o demonstrate the framework's effectiveness generating 264 XSS payloads, 83% of which were validated, with 80% bypassing ModSecurity WAF equipped with an industry standard security rule set developed by the Open Web Application Security Project (OWASP) to protect against web vulnerabilities. Through rule generation, 86% of previously successful attacks were blocked using only 15 new rules. In comparison, Google Gemini Pro achieved a lower bypass rate of 63%, highlighting performance differences across LLMs.



## **4. Benchmarking Adversarial Robustness to Bias Elicitation in Large Language Models: Scalable Automated Assessment with LLM-as-a-Judge**

cs.CL

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07887v1) [paper-pdf](http://arxiv.org/pdf/2504.07887v1)

**Authors**: Riccardo Cantini, Alessio Orsino, Massimo Ruggiero, Domenico Talia

**Abstract**: Large Language Models (LLMs) have revolutionized artificial intelligence, driving advancements in machine translation, summarization, and conversational agents. However, their increasing integration into critical societal domains has raised concerns about embedded biases, which can perpetuate stereotypes and compromise fairness. These biases stem from various sources, including historical inequalities in training data, linguistic imbalances, and adversarial manipulation. Despite mitigation efforts, recent studies indicate that LLMs remain vulnerable to adversarial attacks designed to elicit biased responses. This work proposes a scalable benchmarking framework to evaluate LLM robustness against adversarial bias elicitation. Our methodology involves (i) systematically probing models with a multi-task approach targeting biases across various sociocultural dimensions, (ii) quantifying robustness through safety scores using an LLM-as-a-Judge approach for automated assessment of model responses, and (iii) employing jailbreak techniques to investigate vulnerabilities in safety mechanisms. Our analysis examines prevalent biases in both small and large state-of-the-art models and their impact on model safety. Additionally, we assess the safety of domain-specific models fine-tuned for critical fields, such as medicine. Finally, we release a curated dataset of bias-related prompts, CLEAR-Bias, to facilitate systematic vulnerability benchmarking. Our findings reveal critical trade-offs between model size and safety, aiding the development of fairer and more robust future language models.



## **5. PR-Attack: Coordinated Prompt-RAG Attacks on Retrieval-Augmented Generation in Large Language Models via Bilevel Optimization**

cs.CR

Accepted at SIGIR 2025

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07717v1) [paper-pdf](http://arxiv.org/pdf/2504.07717v1)

**Authors**: Yang Jiao, Xiaodong Wang, Kai Yang

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of applications, e.g., medical question-answering, mathematical sciences, and code generation. However, they also exhibit inherent limitations, such as outdated knowledge and susceptibility to hallucinations. Retrieval-Augmented Generation (RAG) has emerged as a promising paradigm to address these issues, but it also introduces new vulnerabilities. Recent efforts have focused on the security of RAG-based LLMs, yet existing attack methods face three critical challenges: (1) their effectiveness declines sharply when only a limited number of poisoned texts can be injected into the knowledge database, (2) they lack sufficient stealth, as the attacks are often detectable by anomaly detection systems, which compromises their effectiveness, and (3) they rely on heuristic approaches to generate poisoned texts, lacking formal optimization frameworks and theoretic guarantees, which limits their effectiveness and applicability. To address these issues, we propose coordinated Prompt-RAG attack (PR-attack), a novel optimization-driven attack that introduces a small number of poisoned texts into the knowledge database while embedding a backdoor trigger within the prompt. When activated, the trigger causes the LLM to generate pre-designed responses to targeted queries, while maintaining normal behavior in other contexts. This ensures both high effectiveness and stealth. We formulate the attack generation process as a bilevel optimization problem leveraging a principled optimization framework to develop optimal poisoned texts and triggers. Extensive experiments across diverse LLMs and datasets demonstrate the effectiveness of PR-Attack, achieving a high attack success rate even with a limited number of poisoned texts and significantly improved stealth compared to existing methods.



## **6. Agent That Debugs: Dynamic State-Guided Vulnerability Repair**

cs.SE

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07634v1) [paper-pdf](http://arxiv.org/pdf/2504.07634v1)

**Authors**: Zhengyao Liu, Yunlong Ma, Jingxuan Xu, Junchen Ai, Xiang Gao, Hailong Sun, Abhik Roychoudhury

**Abstract**: In recent years, more vulnerabilities have been discovered every day, while manual vulnerability repair requires specialized knowledge and is time-consuming. As a result, many detected or even published vulnerabilities remain unpatched, thereby increasing the exposure of software systems to attacks. Recent advancements in agents based on Large Language Models have demonstrated their increasing capabilities in code understanding and generation, which can be promising to achieve automated vulnerability repair. However, the effectiveness of agents based on static information retrieval is still not sufficient for patch generation. To address the challenge, we propose a program repair agent called VulDebugger that fully utilizes both static and dynamic context, and it debugs programs in a manner akin to humans. The agent inspects the actual state of the program via the debugger and infers expected states via constraints that need to be satisfied. By continuously comparing the actual state with the expected state, it deeply understands the root causes of the vulnerabilities and ultimately accomplishes repairs. We experimentally evaluated VulDebugger on 50 real-life projects. With 60.00% successfully fixed, VulDebugger significantly outperforms state-of-the-art approaches for vulnerability repair.



## **7. Defense against Prompt Injection Attacks via Mixture of Encodings**

cs.CL

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07467v1) [paper-pdf](http://arxiv.org/pdf/2504.07467v1)

**Authors**: Ruiyi Zhang, David Sullivan, Kyle Jackson, Pengtao Xie, Mei Chen

**Abstract**: Large Language Models (LLMs) have emerged as a dominant approach for a wide range of NLP tasks, with their access to external information further enhancing their capabilities. However, this introduces new vulnerabilities, known as prompt injection attacks, where external content embeds malicious instructions that manipulate the LLM's output. Recently, the Base64 defense has been recognized as one of the most effective methods for reducing success rate of prompt injection attacks. Despite its efficacy, this method can degrade LLM performance on certain NLP tasks. To address this challenge, we propose a novel defense mechanism: mixture of encodings, which utilizes multiple character encodings, including Base64. Extensive experimental results show that our method achieves one of the lowest attack success rates under prompt injection attacks, while maintaining high performance across all NLP tasks, outperforming existing character encoding-based defense methods. This underscores the effectiveness of our mixture of encodings strategy for both safety and task performance metrics.



## **8. Achilles Heel of Distributed Multi-Agent Systems**

cs.MA

**SubmitDate**: 2025-04-10    [abs](http://arxiv.org/abs/2504.07461v1) [paper-pdf](http://arxiv.org/pdf/2504.07461v1)

**Authors**: Yiting Zhang, Yijiang Li, Tianwei Zhao, Kaijie Zhu, Haohan Wang, Nuno Vasconcelos

**Abstract**: Multi-agent system (MAS) has demonstrated exceptional capabilities in addressing complex challenges, largely due to the integration of multiple large language models (LLMs). However, the heterogeneity of LLMs, the scalability of quantities of LLMs, and local computational constraints pose significant challenges to hosting these models locally. To address these issues, we propose a new framework termed Distributed Multi-Agent System (DMAS). In DMAS, heterogeneous third-party agents function as service providers managed remotely by a central MAS server and each agent offers its services through API interfaces. However, the distributed nature of DMAS introduces several concerns about trustworthiness. In this paper, we study the Achilles heel of distributed multi-agent systems, identifying four critical trustworthiness challenges: free riding, susceptibility to malicious attacks, communication inefficiencies, and system instability. Extensive experiments across seven frameworks and four datasets reveal significant vulnerabilities of the DMAS. These attack strategies can lead to a performance degradation of up to 80% and attain a 100% success rate in executing free riding and malicious attacks. We envision our work will serve as a useful red-teaming tool for evaluating future multi-agent systems and spark further research on trustworthiness challenges in distributed multi-agent systems.



## **9. Code Generation with Small Language Models: A Deep Evaluation on Codeforces**

cs.SE

**SubmitDate**: 2025-04-09    [abs](http://arxiv.org/abs/2504.07343v1) [paper-pdf](http://arxiv.org/pdf/2504.07343v1)

**Authors**: Débora Souza, Rohit Gheyi, Lucas Albuquerque, Gustavo Soares, Márcio Ribeiro

**Abstract**: Large Language Models (LLMs) have demonstrated capabilities in code generation, potentially boosting developer productivity. However, their widespread adoption remains limited by high computational costs, significant energy demands, and security risks such as data leakage and adversarial attacks. As a lighter-weight alternative, Small Language Models (SLMs) offer faster inference, lower deployment overhead, and better adaptability to domain-specific tasks, making them an attractive option for real-world applications. While prior research has benchmarked LLMs on competitive programming tasks, such evaluations often focus narrowly on metrics like Elo scores or pass rates, overlooking deeper insights into model behavior, failure patterns, and problem diversity. Furthermore, the potential of SLMs to tackle complex tasks such as competitive programming remains underexplored. In this study, we benchmark five open SLMs - LLAMA 3.2 3B, GEMMA 2 9B, GEMMA 3 12B, DEEPSEEK-R1 14B, and PHI-4 14B - across 280 Codeforces problems spanning Elo ratings from 800 to 2100 and covering 36 distinct topics. All models were tasked with generating Python solutions. PHI-4 14B achieved the best performance among SLMs, with a pass@3 of 63.6%, approaching the proprietary O3-MINI-HIGH (86.8%). In addition, we evaluated PHI-4 14B on C++ and found that combining outputs from both Python and C++ increases its aggregated pass@3 to 73.6%. A qualitative analysis of PHI-4 14B's incorrect outputs revealed that some failures were due to minor implementation issues - such as handling edge cases or correcting variable initialization - rather than deeper reasoning flaws.



## **10. LLM Safeguard is a Double-Edged Sword: Exploiting False Positives for Denial-of-Service Attacks**

cs.CR

**SubmitDate**: 2025-04-09    [abs](http://arxiv.org/abs/2410.02916v3) [paper-pdf](http://arxiv.org/pdf/2410.02916v3)

**Authors**: Qingzhao Zhang, Ziyang Xiong, Z. Morley Mao

**Abstract**: Safety is a paramount concern for large language models (LLMs) in open deployment, motivating the development of safeguard methods that enforce ethical and responsible use through safety alignment or guardrail mechanisms. Jailbreak attacks that exploit the \emph{false negatives} of safeguard methods have emerged as a prominent research focus in the field of LLM security. However, we found that the malicious attackers could also exploit false positives of safeguards, i.e., fooling the safeguard model to block safe content mistakenly, leading to a denial-of-service (DoS) affecting LLM users. To bridge the knowledge gap of this overlooked threat, we explore multiple attack methods that include inserting a short adversarial prompt into user prompt templates and corrupting the LLM on the server by poisoned fine-tuning. In both ways, the attack triggers safeguard rejections of user requests from the client. Our evaluation demonstrates the severity of this threat across multiple scenarios. For instance, in the scenario of white-box adversarial prompt injection, the attacker can use our optimization process to automatically generate seemingly safe adversarial prompts, approximately only 30 characters long, that universally block over 97% of user requests on Llama Guard 3. These findings reveal a new dimension in LLM safeguard evaluation -- adversarial robustness to false positives.



## **11. Navigating the Rabbit Hole: Emergent Biases in LLM-Generated Attack Narratives Targeting Mental Health Groups**

cs.CL

**SubmitDate**: 2025-04-09    [abs](http://arxiv.org/abs/2504.06160v2) [paper-pdf](http://arxiv.org/pdf/2504.06160v2)

**Authors**: Rijul Magu, Arka Dutta, Sean Kim, Ashiqur R. KhudaBukhsh, Munmun De Choudhury

**Abstract**: Large Language Models (LLMs) have been shown to demonstrate imbalanced biases against certain groups. However, the study of unprovoked targeted attacks by LLMs towards at-risk populations remains underexplored. Our paper presents three novel contributions: (1) the explicit evaluation of LLM-generated attacks on highly vulnerable mental health groups; (2) a network-based framework to study the propagation of relative biases; and (3) an assessment of the relative degree of stigmatization that emerges from these attacks. Our analysis of a recently released large-scale bias audit dataset reveals that mental health entities occupy central positions within attack narrative networks, as revealed by a significantly higher mean centrality of closeness (p-value = 4.06e-10) and dense clustering (Gini coefficient = 0.7). Drawing from sociological foundations of stigmatization theory, our stigmatization analysis indicates increased labeling components for mental health disorder-related targets relative to initial targets in generation chains. Taken together, these insights shed light on the structural predilections of large language models to heighten harmful discourse and highlight the need for suitable approaches for mitigation.



## **12. JailDAM: Jailbreak Detection with Adaptive Memory for Vision-Language Model**

cs.CR

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.03770v2) [paper-pdf](http://arxiv.org/pdf/2504.03770v2)

**Authors**: Yi Nian, Shenzhe Zhu, Yuehan Qin, Li Li, Ziyi Wang, Chaowei Xiao, Yue Zhao

**Abstract**: Multimodal large language models (MLLMs) excel in vision-language tasks but also pose significant risks of generating harmful content, particularly through jailbreak attacks. Jailbreak attacks refer to intentional manipulations that bypass safety mechanisms in models, leading to the generation of inappropriate or unsafe content. Detecting such attacks is critical to ensuring the responsible deployment of MLLMs. Existing jailbreak detection methods face three primary challenges: (1) Many rely on model hidden states or gradients, limiting their applicability to white-box models, where the internal workings of the model are accessible; (2) They involve high computational overhead from uncertainty-based analysis, which limits real-time detection, and (3) They require fully labeled harmful datasets, which are often scarce in real-world settings. To address these issues, we introduce a test-time adaptive framework called JAILDAM. Our method leverages a memory-based approach guided by policy-driven unsafe knowledge representations, eliminating the need for explicit exposure to harmful data. By dynamically updating unsafe knowledge during test-time, our framework improves generalization to unseen jailbreak strategies while maintaining efficiency. Experiments on multiple VLM jailbreak benchmarks demonstrate that JAILDAM delivers state-of-the-art performance in harmful content detection, improving both accuracy and speed.



## **13. StealthRank: LLM Ranking Manipulation via Stealthy Prompt Optimization**

cs.IR

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05804v1) [paper-pdf](http://arxiv.org/pdf/2504.05804v1)

**Authors**: Yiming Tang, Yi Fan, Chenxiao Yu, Tiankai Yang, Yue Zhao, Xiyang Hu

**Abstract**: The integration of large language models (LLMs) into information retrieval systems introduces new attack surfaces, particularly for adversarial ranking manipulations. We present StealthRank, a novel adversarial ranking attack that manipulates LLM-driven product recommendation systems while maintaining textual fluency and stealth. Unlike existing methods that often introduce detectable anomalies, StealthRank employs an energy-based optimization framework combined with Langevin dynamics to generate StealthRank Prompts (SRPs)-adversarial text sequences embedded within product descriptions that subtly yet effectively influence LLM ranking mechanisms. We evaluate StealthRank across multiple LLMs, demonstrating its ability to covertly boost the ranking of target products while avoiding explicit manipulation traces that can be easily detected. Our results show that StealthRank consistently outperforms state-of-the-art adversarial ranking baselines in both effectiveness and stealth, highlighting critical vulnerabilities in LLM-driven recommendation systems.



## **14. Separator Injection Attack: Uncovering Dialogue Biases in Large Language Models Caused by Role Separators**

cs.CL

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05689v1) [paper-pdf](http://arxiv.org/pdf/2504.05689v1)

**Authors**: Xitao Li, Haijun Wang, Jiang Wu, Ting Liu

**Abstract**: Conversational large language models (LLMs) have gained widespread attention due to their instruction-following capabilities. To ensure conversational LLMs follow instructions, role separators are employed to distinguish between different participants in a conversation. However, incorporating role separators introduces potential vulnerabilities. Misusing roles can lead to prompt injection attacks, which can easily misalign the model's behavior with the user's intentions, raising significant security concerns. Although various prompt injection attacks have been proposed, recent research has largely overlooked the impact of role separators on safety. This highlights the critical need to thoroughly understand the systemic weaknesses in dialogue systems caused by role separators. This paper identifies modeling weaknesses caused by role separators. Specifically, we observe a strong positional bias associated with role separators, which is inherent in the format of dialogue modeling and can be triggered by the insertion of role separators. We further develop the Separators Injection Attack (SIA), a new orthometric attack based on role separators. The experiment results show that SIA is efficient and extensive in manipulating model behavior with an average gain of 18.2% for manual methods and enhances the attack success rate to 100% with automatic methods.



## **15. Sugar-Coated Poison: Benign Generation Unlocks LLM Jailbreaking**

cs.CR

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2504.05652v1) [paper-pdf](http://arxiv.org/pdf/2504.05652v1)

**Authors**: Yu-Hang Wu, Yu-Jie Xiong, Jie-Zhang

**Abstract**: Large Language Models (LLMs) have become increasingly integral to a wide range of applications. However, they still remain the threat of jailbreak attacks, where attackers manipulate designed prompts to make the models elicit malicious outputs. Analyzing jailbreak methods can help us delve into the weakness of LLMs and improve it. In this paper, We reveal a vulnerability in large language models (LLMs), which we term Defense Threshold Decay (DTD), by analyzing the attention weights of the model's output on input and subsequent output on prior output: as the model generates substantial benign content, its attention weights shift from the input to prior output, making it more susceptible to jailbreak attacks. To demonstrate the exploitability of DTD, we propose a novel jailbreak attack method, Sugar-Coated Poison (SCP), which induces the model to generate substantial benign content through benign input and adversarial reasoning, subsequently producing malicious content. To mitigate such attacks, we introduce a simple yet effective defense strategy, POSD, which significantly reduces jailbreak success rates while preserving the model's generalization capabilities.



## **16. SceneTAP: Scene-Coherent Typographic Adversarial Planner against Vision-Language Models in Real-World Environments**

cs.CV

**SubmitDate**: 2025-04-08    [abs](http://arxiv.org/abs/2412.00114v2) [paper-pdf](http://arxiv.org/pdf/2412.00114v2)

**Authors**: Yue Cao, Yun Xing, Jie Zhang, Di Lin, Tianwei Zhang, Ivor Tsang, Yang Liu, Qing Guo

**Abstract**: Large vision-language models (LVLMs) have shown remarkable capabilities in interpreting visual content. While existing works demonstrate these models' vulnerability to deliberately placed adversarial texts, such texts are often easily identifiable as anomalous. In this paper, we present the first approach to generate scene-coherent typographic adversarial attacks that mislead advanced LVLMs while maintaining visual naturalness through the capability of the LLM-based agent. Our approach addresses three critical questions: what adversarial text to generate, where to place it within the scene, and how to integrate it seamlessly. We propose a training-free, multi-modal LLM-driven scene-coherent typographic adversarial planning (SceneTAP) that employs a three-stage process: scene understanding, adversarial planning, and seamless integration. The SceneTAP utilizes chain-of-thought reasoning to comprehend the scene, formulate effective adversarial text, strategically plan its placement, and provide detailed instructions for natural integration within the image. This is followed by a scene-coherent TextDiffuser that executes the attack using a local diffusion mechanism. We extend our method to real-world scenarios by printing and placing generated patches in physical environments, demonstrating its practical implications. Extensive experiments show that our scene-coherent adversarial text successfully misleads state-of-the-art LVLMs, including ChatGPT-4o, even after capturing new images of physical setups. Our evaluations demonstrate a significant increase in attack success rates while maintaining visual naturalness and contextual appropriateness. This work highlights vulnerabilities in current vision-language models to sophisticated, scene-coherent adversarial attacks and provides insights into potential defense mechanisms.



## **17. Revealing the Intrinsic Ethical Vulnerability of Aligned Large Language Models**

cs.CL

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.05050v1) [paper-pdf](http://arxiv.org/pdf/2504.05050v1)

**Authors**: Jiawei Lian, Jianhong Pan, Lefan Wang, Yi Wang, Shaohui Mei, Lap-Pui Chau

**Abstract**: Large language models (LLMs) are foundational explorations to artificial general intelligence, yet their alignment with human values via instruction tuning and preference learning achieves only superficial compliance. Here, we demonstrate that harmful knowledge embedded during pretraining persists as indelible "dark patterns" in LLMs' parametric memory, evading alignment safeguards and resurfacing under adversarial inducement at distributional shifts. In this study, we first theoretically analyze the intrinsic ethical vulnerability of aligned LLMs by proving that current alignment methods yield only local "safety regions" in the knowledge manifold. In contrast, pretrained knowledge remains globally connected to harmful concepts via high-likelihood adversarial trajectories. Building on this theoretical insight, we empirically validate our findings by employing semantic coherence inducement under distributional shifts--a method that systematically bypasses alignment constraints through optimized adversarial prompts. This combined theoretical and empirical approach achieves a 100% attack success rate across 19 out of 23 state-of-the-art aligned LLMs, including DeepSeek-R1 and LLaMA-3, revealing their universal vulnerabilities.



## **18. A Domain-Based Taxonomy of Jailbreak Vulnerabilities in Large Language Models**

cs.CL

21 pages, 5 figures

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.04976v1) [paper-pdf](http://arxiv.org/pdf/2504.04976v1)

**Authors**: Carlos Peláez-González, Andrés Herrera-Poyatos, Cristina Zuheros, David Herrera-Poyatos, Virilo Tejedor, Francisco Herrera

**Abstract**: The study of large language models (LLMs) is a key area in open-world machine learning. Although LLMs demonstrate remarkable natural language processing capabilities, they also face several challenges, including consistency issues, hallucinations, and jailbreak vulnerabilities. Jailbreaking refers to the crafting of prompts that bypass alignment safeguards, leading to unsafe outputs that compromise the integrity of LLMs. This work specifically focuses on the challenge of jailbreak vulnerabilities and introduces a novel taxonomy of jailbreak attacks grounded in the training domains of LLMs. It characterizes alignment failures through generalization, objectives, and robustness gaps. Our primary contribution is a perspective on jailbreak, framed through the different linguistic domains that emerge during LLM training and alignment. This viewpoint highlights the limitations of existing approaches and enables us to classify jailbreak attacks on the basis of the underlying model deficiencies they exploit. Unlike conventional classifications that categorize attacks based on prompt construction methods (e.g., prompt templating), our approach provides a deeper understanding of LLM behavior. We introduce a taxonomy with four categories -- mismatched generalization, competing objectives, adversarial robustness, and mixed attacks -- offering insights into the fundamental nature of jailbreak vulnerabilities. Finally, we present key lessons derived from this taxonomic study.



## **19. Don't Lag, RAG: Training-Free Adversarial Detection Using RAG**

cs.AI

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.04858v1) [paper-pdf](http://arxiv.org/pdf/2504.04858v1)

**Authors**: Roie Kazoom, Raz Lapid, Moshe Sipper, Ofer Hadar

**Abstract**: Adversarial patch attacks pose a major threat to vision systems by embedding localized perturbations that mislead deep models. Traditional defense methods often require retraining or fine-tuning, making them impractical for real-world deployment. We propose a training-free Visual Retrieval-Augmented Generation (VRAG) framework that integrates Vision-Language Models (VLMs) for adversarial patch detection. By retrieving visually similar patches and images that resemble stored attacks in a continuously expanding database, VRAG performs generative reasoning to identify diverse attack types, all without additional training or fine-tuning. We extensively evaluate open-source large-scale VLMs, including Qwen-VL-Plus, Qwen2.5-VL-72B, and UI-TARS-72B-DPO, alongside Gemini-2.0, a closed-source model. Notably, the open-source UI-TARS-72B-DPO model achieves up to 95 percent classification accuracy, setting a new state-of-the-art for open-source adversarial patch detection. Gemini-2.0 attains the highest overall accuracy, 98 percent, but remains closed-source. Experimental results demonstrate VRAG's effectiveness in identifying a variety of adversarial patches with minimal human annotation, paving the way for robust, practical defenses against evolving adversarial patch attacks.



## **20. SINCon: Mitigate LLM-Generated Malicious Message Injection Attack for Rumor Detection**

cs.CR

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.07135v1) [paper-pdf](http://arxiv.org/pdf/2504.07135v1)

**Authors**: Mingqing Zhang, Qiang Liu, Xiang Tao, Shu Wu, Liang Wang

**Abstract**: In the era of rapidly evolving large language models (LLMs), state-of-the-art rumor detection systems, particularly those based on Message Propagation Trees (MPTs), which represent a conversation tree with the post as its root and the replies as its descendants, are facing increasing threats from adversarial attacks that leverage LLMs to generate and inject malicious messages. Existing methods are based on the assumption that different nodes exhibit varying degrees of influence on predictions. They define nodes with high predictive influence as important nodes and target them for attacks. If the model treats nodes' predictive influence more uniformly, attackers will find it harder to target high predictive influence nodes. In this paper, we propose Similarizing the predictive Influence of Nodes with Contrastive Learning (SINCon), a defense mechanism that encourages the model to learn graph representations where nodes with varying importance have a more uniform influence on predictions. Extensive experiments on the Twitter and Weibo datasets demonstrate that SINCon not only preserves high classification accuracy on clean data but also significantly enhances resistance against LLM-driven message injection attacks.



## **21. PiCo: Jailbreaking Multimodal Large Language Models via $\textbf{Pi}$ctorial $\textbf{Co}$de Contextualization**

cs.CR

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.01444v2) [paper-pdf](http://arxiv.org/pdf/2504.01444v2)

**Authors**: Aofan Liu, Lulu Tang, Ting Pan, Yuguo Yin, Bin Wang, Ao Yang

**Abstract**: Multimodal Large Language Models (MLLMs), which integrate vision and other modalities into Large Language Models (LLMs), significantly enhance AI capabilities but also introduce new security vulnerabilities. By exploiting the vulnerabilities of the visual modality and the long-tail distribution characteristic of code training data, we present PiCo, a novel jailbreaking framework designed to progressively bypass multi-tiered defense mechanisms in advanced MLLMs. PiCo employs a tier-by-tier jailbreak strategy, using token-level typographic attacks to evade input filtering and embedding harmful intent within programming context instructions to bypass runtime monitoring. To comprehensively assess the impact of attacks, a new evaluation metric is further proposed to assess both the toxicity and helpfulness of model outputs post-attack. By embedding harmful intent within code-style visual instructions, PiCo achieves an average Attack Success Rate (ASR) of 84.13% on Gemini-Pro Vision and 52.66% on GPT-4, surpassing previous methods. Experimental results highlight the critical gaps in current defenses, underscoring the need for more robust strategies to secure advanced MLLMs.



## **22. Select Me! When You Need a Tool: A Black-box Text Attack on Tool Selection**

cs.CR

13 pages

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.04809v1) [paper-pdf](http://arxiv.org/pdf/2504.04809v1)

**Authors**: Liuji Chen, Hao Gao, Jinghao Zhang, Qiang Liu, Shu Wu, Liang Wang

**Abstract**: Tool learning serves as a powerful auxiliary mechanism that extends the capabilities of large language models (LLMs), enabling them to tackle complex tasks requiring real-time relevance or high precision operations. Behind its powerful capabilities lie some potential security issues. However, previous work has primarily focused on how to make the output of the invoked tools incorrect or malicious, with little attention given to the manipulation of tool selection. To fill this gap, we introduce, for the first time, a black-box text-based attack that can significantly increase the probability of the target tool being selected in this paper. We propose a two-level text perturbation attack witha coarse-to-fine granularity, attacking the text at both the word level and the character level. We conduct comprehensive experiments that demonstrate the attacker only needs to make some perturbations to the tool's textual information to significantly increase the possibility of the target tool being selected and ranked higher among the candidate tools. Our research reveals the vulnerability of the tool selection process and paves the way for future research on protecting this process.



## **23. Are You Getting What You Pay For? Auditing Model Substitution in LLM APIs**

cs.CL

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.04715v1) [paper-pdf](http://arxiv.org/pdf/2504.04715v1)

**Authors**: Will Cai, Tianneng Shi, Xuandong Zhao, Dawn Song

**Abstract**: The proliferation of Large Language Models (LLMs) accessed via black-box APIs introduces a significant trust challenge: users pay for services based on advertised model capabilities (e.g., size, performance), but providers may covertly substitute the specified model with a cheaper, lower-quality alternative to reduce operational costs. This lack of transparency undermines fairness, erodes trust, and complicates reliable benchmarking. Detecting such substitutions is difficult due to the black-box nature, typically limiting interaction to input-output queries. This paper formalizes the problem of model substitution detection in LLM APIs. We systematically evaluate existing verification techniques, including output-based statistical tests, benchmark evaluations, and log probability analysis, under various realistic attack scenarios like model quantization, randomized substitution, and benchmark evasion. Our findings reveal the limitations of methods relying solely on text outputs, especially against subtle or adaptive attacks. While log probability analysis offers stronger guarantees when available, its accessibility is often limited. We conclude by discussing the potential of hardware-based solutions like Trusted Execution Environments (TEEs) as a pathway towards provable model integrity, highlighting the trade-offs between security, performance, and provider adoption. Code is available at https://github.com/sunblaze-ucb/llm-api-audit



## **24. Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks**

cs.CV

**SubmitDate**: 2025-04-07    [abs](http://arxiv.org/abs/2504.01308v2) [paper-pdf](http://arxiv.org/pdf/2504.01308v2)

**Authors**: Jiawei Wang, Yushen Zuo, Yuanjun Chai, Zhendong Liu, Yicheng Fu, Yichun Feng, Kin-Man Lam

**Abstract**: Vision-Language Models (VLMs) extend the capabilities of Large Language Models (LLMs) by incorporating visual information, yet they remain vulnerable to jailbreak attacks, especially when processing noisy or corrupted images. Although existing VLMs adopt security measures during training to mitigate such attacks, vulnerabilities associated with noise-augmented visual inputs are overlooked. In this work, we identify that missing noise-augmented training causes critical security gaps: many VLMs are susceptible to even simple perturbations such as Gaussian noise. To address this challenge, we propose Robust-VLGuard, a multimodal safety dataset with aligned / misaligned image-text pairs, combined with noise-augmented fine-tuning that reduces attack success rates while preserving functionality of VLM. For stronger optimization-based visual perturbation attacks, we propose DiffPure-VLM, leveraging diffusion models to convert adversarial perturbations into Gaussian-like noise, which can be defended by VLMs with noise-augmented safety fine-tuning. Experimental results demonstrate that the distribution-shifting property of diffusion model aligns well with our fine-tuned VLMs, significantly mitigating adversarial perturbations across varying intensities. The dataset and code are available at https://github.com/JarvisUSTC/DiffPure-RobustVLM.



## **25. Privacy in Fine-tuning Large Language Models: Attacks, Defenses, and Future Directions**

cs.AI

accepted by PAKDD2025

**SubmitDate**: 2025-04-06    [abs](http://arxiv.org/abs/2412.16504v2) [paper-pdf](http://arxiv.org/pdf/2412.16504v2)

**Authors**: Hao Du, Shang Liu, Lele Zheng, Yang Cao, Atsuyoshi Nakamura, Lei Chen

**Abstract**: Fine-tuning has emerged as a critical process in leveraging Large Language Models (LLMs) for specific downstream tasks, enabling these models to achieve state-of-the-art performance across various domains. However, the fine-tuning process often involves sensitive datasets, introducing privacy risks that exploit the unique characteristics of this stage. In this paper, we provide a comprehensive survey of privacy challenges associated with fine-tuning LLMs, highlighting vulnerabilities to various privacy attacks, including membership inference, data extraction, and backdoor attacks. We further review defense mechanisms designed to mitigate privacy risks in the fine-tuning phase, such as differential privacy, federated learning, and knowledge unlearning, discussing their effectiveness and limitations in addressing privacy risks and maintaining model utility. By identifying key gaps in existing research, we highlight challenges and propose directions to advance the development of privacy-preserving methods for fine-tuning LLMs, promoting their responsible use in diverse applications.



## **26. CyberLLMInstruct: A New Dataset for Analysing Safety of Fine-Tuned LLMs Using Cyber Security Data**

cs.CR

**SubmitDate**: 2025-04-05    [abs](http://arxiv.org/abs/2503.09334v2) [paper-pdf](http://arxiv.org/pdf/2503.09334v2)

**Authors**: Adel ElZemity, Budi Arief, Shujun Li

**Abstract**: The integration of large language models (LLMs) into cyber security applications presents significant opportunities, such as enhancing threat analysis and malware detection, but can also introduce critical risks and safety concerns, including personal data leakage and automated generation of new malware. To address these challenges, we developed CyberLLMInstruct, a dataset of 54,928 instruction-response pairs spanning cyber security tasks such as malware analysis, phishing simulations, and zero-day vulnerabilities. The dataset was constructed through a multi-stage process. This involved sourcing data from multiple resources, filtering and structuring it into instruction-response pairs, and aligning it with real-world scenarios to enhance its applicability. Seven open-source LLMs were chosen to test the usefulness of CyberLLMInstruct: Phi 3 Mini 3.8B, Mistral 7B, Qwen 2.5 7B, Llama 3 8B, Llama 3.1 8B, Gemma 2 9B, and Llama 2 70B. In our primary example, we rigorously assess the safety of fine-tuned models using the OWASP top 10 framework, finding that fine-tuning reduces safety resilience across all tested LLMs and every adversarial attack (e.g., the security score of Llama 3.1 8B against prompt injection drops from 0.95 to 0.15). In our second example, we show that these same fine-tuned models can also achieve up to 92.50 percent accuracy on the CyberMetric benchmark. These findings highlight a trade-off between performance and safety, showing the importance of adversarial testing and further research into fine-tuning methodologies that can mitigate safety risks while still improving performance across diverse datasets and domains. The dataset creation pipeline, along with comprehensive documentation, examples, and resources for reproducing our results, is publicly available at https://github.com/Adelsamir01/CyberLLMInstruct.



## **27. AttackLLM: LLM-based Attack Pattern Generation for an Industrial Control System**

cs.CR

**SubmitDate**: 2025-04-05    [abs](http://arxiv.org/abs/2504.04187v1) [paper-pdf](http://arxiv.org/pdf/2504.04187v1)

**Authors**: Chuadhry Mujeeb Ahmed

**Abstract**: Malicious examples are crucial for evaluating the robustness of machine learning algorithms under attack, particularly in Industrial Control Systems (ICS). However, collecting normal and attack data in ICS environments is challenging due to the scarcity of testbeds and the high cost of human expertise. Existing datasets are often limited by the domain expertise of practitioners, making the process costly and inefficient. The lack of comprehensive attack pattern data poses a significant problem for developing robust anomaly detection methods. In this paper, we propose a novel approach that combines data-centric and design-centric methodologies to generate attack patterns using large language models (LLMs). Our results demonstrate that the attack patterns generated by LLMs not only surpass the quality and quantity of those created by human experts but also offer a scalable solution that does not rely on expensive testbeds or pre-existing attack examples. This multi-agent based approach presents a promising avenue for enhancing the security and resilience of ICS environments.



## **28. Practical Poisoning Attacks against Retrieval-Augmented Generation**

cs.CR

**SubmitDate**: 2025-04-04    [abs](http://arxiv.org/abs/2504.03957v1) [paper-pdf](http://arxiv.org/pdf/2504.03957v1)

**Authors**: Baolei Zhang, Yuxi Chen, Minghong Fang, Zhuqing Liu, Lihai Nie, Tong Li, Zheli Liu

**Abstract**: Large language models (LLMs) have demonstrated impressive natural language processing abilities but face challenges such as hallucination and outdated knowledge. Retrieval-Augmented Generation (RAG) has emerged as a state-of-the-art approach to mitigate these issues. While RAG enhances LLM outputs, it remains vulnerable to poisoning attacks. Recent studies show that injecting poisoned text into the knowledge database can compromise RAG systems, but most existing attacks assume that the attacker can insert a sufficient number of poisoned texts per query to outnumber correct-answer texts in retrieval, an assumption that is often unrealistic. To address this limitation, we propose CorruptRAG, a practical poisoning attack against RAG systems in which the attacker injects only a single poisoned text, enhancing both feasibility and stealth. Extensive experiments across multiple datasets demonstrate that CorruptRAG achieves higher attack success rates compared to existing baselines.



## **29. sudo rm -rf agentic_security**

cs.CL

**SubmitDate**: 2025-04-04    [abs](http://arxiv.org/abs/2503.20279v2) [paper-pdf](http://arxiv.org/pdf/2503.20279v2)

**Authors**: Sejin Lee, Jian Kim, Haon Park, Ashkan Yousefpour, Sangyoon Yu, Min Song

**Abstract**: Large Language Models (LLMs) are increasingly deployed as computer-use agents, autonomously performing tasks within real desktop or web environments. While this evolution greatly expands practical use cases for humans, it also creates serious security exposures. We present SUDO (Screen-based Universal Detox2Tox Offense), a novel attack framework that systematically bypasses refusal trained safeguards in commercial computer-use agents, such as Claude Computer Use. The core mechanism, Detox2Tox, transforms harmful requests (that agents initially reject) into seemingly benign requests via detoxification, secures detailed instructions from advanced vision language models (VLMs), and then reintroduces malicious content via toxification just before execution. Unlike conventional jailbreaks, SUDO iteratively refines its attacks based on a built-in refusal feedback, making it increasingly effective against robust policy filters. In extensive tests spanning 50 real-world tasks and multiple state-of-the-art VLMs, SUDO achieves a stark attack success rate of 24% (with no refinement), and up to 41% (by its iterative refinement) in Claude Computer Use. By revealing these vulnerabilities and demonstrating the ease with which they can be exploited in real-world computing environments, this paper highlights an immediate need for robust, context-aware safeguards. WARNING: This paper includes harmful or offensive model outputs Our code is available at: https://github.com/AIM-Intelligence/SUDO.git



## **30. Les Dissonances: Cross-Tool Harvesting and Polluting in Multi-Tool Empowered LLM Agents**

cs.CR

**SubmitDate**: 2025-04-04    [abs](http://arxiv.org/abs/2504.03111v1) [paper-pdf](http://arxiv.org/pdf/2504.03111v1)

**Authors**: Zichuan Li, Jian Cui, Xiaojing Liao, Luyi Xing

**Abstract**: Large Language Model (LLM) agents are autonomous systems powered by LLMs, capable of reasoning and planning to solve problems by leveraging a set of tools. However, the integration of multi-tool capabilities in LLM agents introduces challenges in securely managing tools, ensuring their compatibility, handling dependency relationships, and protecting control flows within LLM agent workflows. In this paper, we present the first systematic security analysis of task control flows in multi-tool-enabled LLM agents. We identify a novel threat, Cross-Tool Harvesting and Polluting (XTHP), which includes multiple attack vectors to first hijack the normal control flows of agent tasks, and then collect and pollute confidential or private information within LLM agent systems. To understand the impact of this threat, we developed Chord, a dynamic scanning tool designed to automatically detect real-world agent tools susceptible to XTHP attacks. Our evaluation of 73 real-world tools from the repositories of two major LLM agent development frameworks, LangChain and LlamaIndex, revealed a significant security concern: 80% of the tools are vulnerable to hijacking attacks, 78% to XTH attacks, and 41% to XTP attacks, highlighting the prevalence of this threat.



## **31. PROMPTFUZZ: Harnessing Fuzzing Techniques for Robust Testing of Prompt Injection in LLMs**

cs.CR

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2409.14729v2) [paper-pdf](http://arxiv.org/pdf/2409.14729v2)

**Authors**: Jiahao Yu, Yangguang Shao, Hanwen Miao, Junzheng Shi

**Abstract**: Large Language Models (LLMs) have gained widespread use in various applications due to their powerful capability to generate human-like text. However, prompt injection attacks, which involve overwriting a model's original instructions with malicious prompts to manipulate the generated text, have raised significant concerns about the security and reliability of LLMs. Ensuring that LLMs are robust against such attacks is crucial for their deployment in real-world applications, particularly in critical tasks.   In this paper, we propose PROMPTFUZZ, a novel testing framework that leverages fuzzing techniques to systematically assess the robustness of LLMs against prompt injection attacks. Inspired by software fuzzing, PROMPTFUZZ selects promising seed prompts and generates a diverse set of prompt injections to evaluate the target LLM's resilience. PROMPTFUZZ operates in two stages: the prepare phase, which involves selecting promising initial seeds and collecting few-shot examples, and the focus phase, which uses the collected examples to generate diverse, high-quality prompt injections. Using PROMPTFUZZ, we can uncover more vulnerabilities in LLMs, even those with strong defense prompts.   By deploying the generated attack prompts from PROMPTFUZZ in a real-world competition, we achieved the 7th ranking out of over 4000 participants (top 0.14%) within 2 hours. Additionally, we construct a dataset to fine-tune LLMs for enhanced robustness against prompt injection attacks. While the fine-tuned model shows improved robustness, PROMPTFUZZ continues to identify vulnerabilities, highlighting the importance of robust testing for LLMs. Our work emphasizes the critical need for effective testing tools and provides a practical framework for evaluating and improving the robustness of LLMs against prompt injection attacks.



## **32. ERPO: Advancing Safety Alignment via Ex-Ante Reasoning Preference Optimization**

cs.CL

18 pages, 5 figures

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.02725v1) [paper-pdf](http://arxiv.org/pdf/2504.02725v1)

**Authors**: Kehua Feng, Keyan Ding, Jing Yu, Menghan Li, Yuhao Wang, Tong Xu, Xinda Wang, Qiang Zhang, Huajun Chen

**Abstract**: Recent advancements in large language models (LLMs) have accelerated progress toward artificial general intelligence, yet their potential to generate harmful content poses critical safety challenges. Existing alignment methods often struggle to cover diverse safety scenarios and remain vulnerable to adversarial attacks. In this work, we propose Ex-Ante Reasoning Preference Optimization (ERPO), a novel safety alignment framework that equips LLMs with explicit preemptive reasoning through Chain-of-Thought and provides clear evidence for safety judgments by embedding predefined safety rules. Specifically, our approach consists of three stages: first, equipping the model with Ex-Ante reasoning through supervised fine-tuning (SFT) using a constructed reasoning module; second, enhancing safety, usefulness, and efficiency via Direct Preference Optimization (DPO); and third, mitigating inference latency with a length-controlled iterative preference optimization strategy. Experiments on multiple open-source LLMs demonstrate that ERPO significantly enhances safety performance while maintaining response efficiency.



## **33. No Free Lunch with Guardrails**

cs.CR

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.00441v2) [paper-pdf](http://arxiv.org/pdf/2504.00441v2)

**Authors**: Divyanshu Kumar, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi

**Abstract**: As large language models (LLMs) and generative AI become widely adopted, guardrails have emerged as a key tool to ensure their safe use. However, adding guardrails isn't without tradeoffs; stronger security measures can reduce usability, while more flexible systems may leave gaps for adversarial attacks. In this work, we explore whether current guardrails effectively prevent misuse while maintaining practical utility. We introduce a framework to evaluate these tradeoffs, measuring how different guardrails balance risk, security, and usability, and build an efficient guardrail.   Our findings confirm that there is no free lunch with guardrails; strengthening security often comes at the cost of usability. To address this, we propose a blueprint for designing better guardrails that minimize risk while maintaining usability. We evaluate various industry guardrails, including Azure Content Safety, Bedrock Guardrails, OpenAI's Moderation API, Guardrails AI, Nemo Guardrails, and Enkrypt AI guardrails. Additionally, we assess how LLMs like GPT-4o, Gemini 2.0-Flash, Claude 3.5-Sonnet, and Mistral Large-Latest respond under different system prompts, including simple prompts, detailed prompts, and detailed prompts with chain-of-thought (CoT) reasoning. Our study provides a clear comparison of how different guardrails perform, highlighting the challenges in balancing security and usability.



## **34. Retrieval-Augmented Purifier for Robust LLM-Empowered Recommendation**

cs.IR

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.02458v1) [paper-pdf](http://arxiv.org/pdf/2504.02458v1)

**Authors**: Liangbo Ning, Wenqi Fan, Qing Li

**Abstract**: Recently, Large Language Model (LLM)-empowered recommender systems have revolutionized personalized recommendation frameworks and attracted extensive attention. Despite the remarkable success, existing LLM-empowered RecSys have been demonstrated to be highly vulnerable to minor perturbations. To mitigate the negative impact of such vulnerabilities, one potential solution is to employ collaborative signals based on item-item co-occurrence to purify the malicious collaborative knowledge from the user's historical interactions inserted by attackers. On the other hand, due to the capabilities to expand insufficient internal knowledge of LLMs, Retrieval-Augmented Generation (RAG) techniques provide unprecedented opportunities to enhance the robustness of LLM-empowered recommender systems by introducing external collaborative knowledge. Therefore, in this paper, we propose a novel framework (RETURN) by retrieving external collaborative signals to purify the poisoned user profiles and enhance the robustness of LLM-empowered RecSys in a plug-and-play manner. Specifically, retrieval-augmented perturbation positioning is proposed to identify potential perturbations within the users' historical sequences by retrieving external knowledge from collaborative item graphs. After that, we further retrieve the collaborative knowledge to cleanse the perturbations by using either deletion or replacement strategies and introduce a robust ensemble recommendation strategy to generate final robust predictions. Extensive experiments on three real-world datasets demonstrate the effectiveness of the proposed RETURN.



## **35. ToxicSQL: Migrating SQL Injection Threats into Text-to-SQL Models via Backdoor Attack**

cs.CR

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2503.05445v2) [paper-pdf](http://arxiv.org/pdf/2503.05445v2)

**Authors**: Meiyu Lin, Haichuan Zhang, Jiale Lao, Renyuan Li, Yuanchun Zhou, Carl Yang, Yang Cao, Mingjie Tang

**Abstract**: Large language models (LLMs) have shown state-of-the-art results in translating natural language questions into SQL queries (Text-to-SQL), a long-standing challenge within the database community. However, security concerns remain largely unexplored, particularly the threat of backdoor attacks, which can introduce malicious behaviors into models through fine-tuning with poisoned datasets. In this work, we systematically investigate the vulnerabilities of LLM-based Text-to-SQL models and present ToxicSQL, a novel backdoor attack framework. Our approach leverages stealthy {semantic and character-level triggers} to make backdoors difficult to detect and remove, ensuring that malicious behaviors remain covert while maintaining high model accuracy on benign inputs. Furthermore, we propose leveraging SQL injection payloads as backdoor targets, enabling the generation of malicious yet executable SQL queries, which pose severe security and privacy risks in language model-based SQL development. We demonstrate that injecting only 0.44% of poisoned data can result in an attack success rate of 79.41%, posing a significant risk to database security. Additionally, we propose detection and mitigation strategies to enhance model reliability. Our findings highlight the urgent need for security-aware Text-to-SQL development, emphasizing the importance of robust defenses against backdoor threats.



## **36. Evolving from Single-modal to Multi-modal Facial Deepfake Detection: Progress and Challenges**

cs.CV

P. Liu is with the Department of Computer Science and Engineering,  University of Nevada, Reno, NV, 89512. Q. Tao and J. Zhou are with Centre for  Frontier AI Research (CFAR), and Institute of High Performance Computing  (IHPC), A*STAR, Singapore. J. Zhou is also with Centre for Advanced  Technologies in Online Safety (CATOS), A*STAR, Singapore. J. Zhou is the  corresponding author

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2406.06965v4) [paper-pdf](http://arxiv.org/pdf/2406.06965v4)

**Authors**: Ping Liu, Qiqi Tao, Joey Tianyi Zhou

**Abstract**: As synthetic media, including video, audio, and text, become increasingly indistinguishable from real content, the risks of misinformation, identity fraud, and social manipulation escalate. This survey traces the evolution of deepfake detection from early single-modal methods to sophisticated multi-modal approaches that integrate audio-visual and text-visual cues. We present a structured taxonomy of detection techniques and analyze the transition from GAN-based to diffusion model-driven deepfakes, which introduce new challenges due to their heightened realism and robustness against detection. Unlike prior surveys that primarily focus on single-modal detection or earlier deepfake techniques, this work provides the most comprehensive study to date, encompassing the latest advancements in multi-modal deepfake detection, generalization challenges, proactive defense mechanisms, and emerging datasets specifically designed to support new interpretability and reasoning tasks. We further explore the role of Vision-Language Models (VLMs) and Multimodal Large Language Models (MLLMs) in strengthening detection robustness against increasingly sophisticated deepfake attacks. By systematically categorizing existing methods and identifying emerging research directions, this survey serves as a foundation for future advancements in combating AI-generated facial forgeries. A curated list of all related papers can be found at \href{https://github.com/qiqitao77/Comprehensive-Advances-in-Deepfake-Detection-Spanning-Diverse-Modalities}{https://github.com/qiqitao77/Awesome-Comprehensive-Deepfake-Detection}.



## **37. More is Less: The Pitfalls of Multi-Model Synthetic Preference Data in DPO Safety Alignment**

cs.AI

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.02193v1) [paper-pdf](http://arxiv.org/pdf/2504.02193v1)

**Authors**: Yifan Wang, Runjin Chen, Bolian Li, David Cho, Yihe Deng, Ruqi Zhang, Tianlong Chen, Zhangyang Wang, Ananth Grama, Junyuan Hong

**Abstract**: Aligning large language models (LLMs) with human values is an increasingly critical step in post-training. Direct Preference Optimization (DPO) has emerged as a simple, yet effective alternative to reinforcement learning from human feedback (RLHF). Synthetic preference data with its low cost and high quality enable effective alignment through single- or multi-model generated preference data. Our study reveals a striking, safety-specific phenomenon associated with DPO alignment: Although multi-model generated data enhances performance on general tasks (ARC, Hellaswag, MMLU, TruthfulQA, Winogrande) by providing diverse responses, it also tends to facilitate reward hacking during training. This can lead to a high attack success rate (ASR) when models encounter jailbreaking prompts. The issue is particularly pronounced when employing stronger models like GPT-4o or larger models in the same family to generate chosen responses paired with target model self-generated rejected responses, resulting in dramatically poorer safety outcomes. Furthermore, with respect to safety, using solely self-generated responses (single-model generation) for both chosen and rejected pairs significantly outperforms configurations that incorporate responses from stronger models, whether used directly as chosen data or as part of a multi-model response pool. We demonstrate that multi-model preference data exhibits high linear separability between chosen and rejected responses, which allows models to exploit superficial cues rather than internalizing robust safety constraints. Our experiments, conducted on models from the Llama, Mistral, and Qwen families, consistently validate these findings.



## **38. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

cs.CR

Included in Proceedings of the Conference on Applied Machine Learning  in Information Security (CAMLIS 2024), Arlington, Virginia, USA, October  24-25, 2024

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2406.03230v5) [paper-pdf](http://arxiv.org/pdf/2406.03230v5)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply a novel methodology for analyzing distinctive activation patterns in the residual streams for attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.



## **39. One Pic is All it Takes: Poisoning Visual Document Retrieval Augmented Generation with a Single Image**

cs.CL

8 pages, 6 figures

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.02132v1) [paper-pdf](http://arxiv.org/pdf/2504.02132v1)

**Authors**: Ezzeldin Shereen, Dan Ristea, Burak Hasircioglu, Shae McFadden, Vasilios Mavroudis, Chris Hicks

**Abstract**: Multimodal retrieval augmented generation (M-RAG) has recently emerged as a method to inhibit hallucinations of large multimodal models (LMMs) through a factual knowledge base (KB). However, M-RAG also introduces new attack vectors for adversaries that aim to disrupt the system by injecting malicious entries into the KB. In this work, we present a poisoning attack against M-RAG targeting visual document retrieval applications, where the KB contains images of document pages. Our objective is to craft a single image that is retrieved for a variety of different user queries, and consistently influences the output produced by the generative model, thus creating a universal denial-of-service (DoS) attack against the M-RAG system. We demonstrate that while our attack is effective against a diverse range of widely-used, state-of-the-art retrievers (embedding models) and generators (LMMs), it can also be ineffective against robust embedding models. Our attack not only highlights the vulnerability of M-RAG pipelines to poisoning attacks, but also sheds light on a fundamental weakness that potentially hinders their performance even in benign settings.



## **40. Evolving Security in LLMs: A Study of Jailbreak Attacks and Defenses**

cs.CR

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.02080v1) [paper-pdf](http://arxiv.org/pdf/2504.02080v1)

**Authors**: Zhengchun Shang, Wenlan Wei

**Abstract**: Large Language Models (LLMs) are increasingly popular, powering a wide range of applications. Their widespread use has sparked concerns, especially through jailbreak attacks that bypass safety measures to produce harmful content.   In this paper, we present a comprehensive security analysis of large language models (LLMs), addressing critical research questions on the evolution and determinants of model safety.   Specifically, we begin by identifying the most effective techniques for detecting jailbreak attacks. Next, we investigate whether newer versions of LLMs offer improved security compared to their predecessors. We also assess the impact of model size on overall security and explore the potential benefits of integrating multiple defense strategies to enhance model robustness.   Our study evaluates both open-source models (e.g., LLaMA and Mistral) and closed-source systems (e.g., GPT-4) by employing four state-of-the-art attack techniques and assessing the efficacy of three new defensive approaches.



## **41. AdPO: Enhancing the Adversarial Robustness of Large Vision-Language Models with Preference Optimization**

cs.CV

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01735v1) [paper-pdf](http://arxiv.org/pdf/2504.01735v1)

**Authors**: Chaohu Liu, Tianyi Gui, Yu Liu, Linli Xu

**Abstract**: Large Vision-Language Models (LVLMs), such as GPT-4o and LLaVA, have recently witnessed remarkable advancements and are increasingly being deployed in real-world applications. However, inheriting the sensitivity of visual neural networks, LVLMs remain vulnerable to adversarial attacks, which can result in erroneous or malicious outputs. While existing efforts utilize adversarial fine-tuning to enhance robustness, they often suffer from performance degradation on clean inputs. In this paper, we proposes AdPO, a novel adversarial defense strategy for LVLMs based on preference optimization. For the first time, we reframe adversarial training as a preference optimization problem, aiming to enhance the model's preference for generating normal outputs on clean inputs while rejecting the potential misleading outputs for adversarial examples. Notably, AdPO achieves this by solely modifying the image encoder, e.g., CLIP ViT, resulting in superior clean and adversarial performance in a variety of downsream tasks. Considering that training involves large language models (LLMs), the computational cost increases significantly. We validate that training on smaller LVLMs and subsequently transferring to larger models can achieve competitive performance while maintaining efficiency comparable to baseline methods. Our comprehensive experiments confirm the effectiveness of the proposed AdPO, which provides a novel perspective for future adversarial defense research.



## **42. Representation Bending for Large Language Model Safety**

cs.LG

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01550v1) [paper-pdf](http://arxiv.org/pdf/2504.01550v1)

**Authors**: Ashkan Yousefpour, Taeheon Kim, Ryan S. Kwon, Seungbeen Lee, Wonje Jeung, Seungju Han, Alvin Wan, Harrison Ngan, Youngjae Yu, Jonghyun Choi

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools, but their inherent safety risks - ranging from harmful content generation to broader societal harms - pose significant challenges. These risks can be amplified by the recent adversarial attacks, fine-tuning vulnerabilities, and the increasing deployment of LLMs in high-stakes environments. Existing safety-enhancing techniques, such as fine-tuning with human feedback or adversarial training, are still vulnerable as they address specific threats and often fail to generalize across unseen attacks, or require manual system-level defenses. This paper introduces RepBend, a novel approach that fundamentally disrupts the representations underlying harmful behaviors in LLMs, offering a scalable solution to enhance (potentially inherent) safety. RepBend brings the idea of activation steering - simple vector arithmetic for steering model's behavior during inference - to loss-based fine-tuning. Through extensive evaluation, RepBend achieves state-of-the-art performance, outperforming prior methods such as Circuit Breaker, RMU, and NPO, with up to 95% reduction in attack success rates across diverse jailbreak benchmarks, all with negligible reduction in model usability and general capabilities.



## **43. LightDefense: A Lightweight Uncertainty-Driven Defense against Jailbreaks via Shifted Token Distribution**

cs.CR

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01533v1) [paper-pdf](http://arxiv.org/pdf/2504.01533v1)

**Authors**: Zhuoran Yang, Jie Peng, Zhen Tan, Tianlong Chen, Yanyong Zhang

**Abstract**: Large Language Models (LLMs) face threats from jailbreak prompts. Existing methods for defending against jailbreak attacks are primarily based on auxiliary models. These strategies, however, often require extensive data collection or training. We propose LightDefense, a lightweight defense mechanism targeted at white-box models, which utilizes a safety-oriented direction to adjust the probabilities of tokens in the vocabulary, making safety disclaimers appear among the top tokens after sorting tokens by probability in descending order. We further innovatively leverage LLM's uncertainty about prompts to measure their harmfulness and adaptively adjust defense strength, effectively balancing safety and helpfulness. The effectiveness of LightDefense in defending against 5 attack methods across 2 target LLMs, without compromising helpfulness to benign user queries, highlights its potential as a novel and lightweight defense mechanism, enhancing security of LLMs.



## **44. Emerging Cyber Attack Risks of Medical AI Agents**

cs.CR

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.03759v1) [paper-pdf](http://arxiv.org/pdf/2504.03759v1)

**Authors**: Jianing Qiu, Lin Li, Jiankai Sun, Hao Wei, Zhe Xu, Kyle Lam, Wu Yuan

**Abstract**: Large language models (LLMs)-powered AI agents exhibit a high level of autonomy in addressing medical and healthcare challenges. With the ability to access various tools, they can operate within an open-ended action space. However, with the increase in autonomy and ability, unforeseen risks also arise. In this work, we investigated one particular risk, i.e., cyber attack vulnerability of medical AI agents, as agents have access to the Internet through web browsing tools. We revealed that through adversarial prompts embedded on webpages, cyberattackers can: i) inject false information into the agent's response; ii) they can force the agent to manipulate recommendation (e.g., healthcare products and services); iii) the attacker can also steal historical conversations between the user and agent, resulting in the leak of sensitive/private medical information; iv) furthermore, the targeted agent can also cause a computer system hijack by returning a malicious URL in its response. Different backbone LLMs were examined, and we found such cyber attacks can succeed in agents powered by most mainstream LLMs, with the reasoning models such as DeepSeek-R1 being the most vulnerable.



## **45. Strategize Globally, Adapt Locally: A Multi-Turn Red Teaming Agent with Dual-Level Learning**

cs.AI

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01278v1) [paper-pdf](http://arxiv.org/pdf/2504.01278v1)

**Authors**: Si Chen, Xiao Yu, Ninareh Mehrabi, Rahul Gupta, Zhou Yu, Ruoxi Jia

**Abstract**: The exploitation of large language models (LLMs) for malicious purposes poses significant security risks as these models become more powerful and widespread. While most existing red-teaming frameworks focus on single-turn attacks, real-world adversaries typically operate in multi-turn scenarios, iteratively probing for vulnerabilities and adapting their prompts based on threat model responses. In this paper, we propose \AlgName, a novel multi-turn red-teaming agent that emulates sophisticated human attackers through complementary learning dimensions: global tactic-wise learning that accumulates knowledge over time and generalizes to new attack goals, and local prompt-wise learning that refines implementations for specific goals when initial attempts fail. Unlike previous multi-turn approaches that rely on fixed strategy sets, \AlgName enables the agent to identify new jailbreak tactics, develop a goal-based tactic selection framework, and refine prompt formulations for selected tactics. Empirical evaluations on JailbreakBench demonstrate our framework's superior performance, achieving over 90\% attack success rates against GPT-3.5-Turbo and Llama-3.1-70B within 5 conversation turns, outperforming state-of-the-art baselines. These results highlight the effectiveness of dynamic learning in identifying and exploiting model vulnerabilities in realistic multi-turn scenarios.



## **46. Towards Resilient Federated Learning in CyberEdge Networks: Recent Advances and Future Trends**

cs.CR

15 pages, 8 figures, 4 tables, 122 references, journal paper

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.01240v1) [paper-pdf](http://arxiv.org/pdf/2504.01240v1)

**Authors**: Kai Li, Zhengyang Zhang, Azadeh Pourkabirian, Wei Ni, Falko Dressler, Ozgur B. Akan

**Abstract**: In this survey, we investigate the most recent techniques of resilient federated learning (ResFL) in CyberEdge networks, focusing on joint training with agglomerative deduction and feature-oriented security mechanisms. We explore adaptive hierarchical learning strategies to tackle non-IID data challenges, improving scalability and reducing communication overhead. Fault tolerance techniques and agglomerative deduction mechanisms are studied to detect unreliable devices, refine model updates, and enhance convergence stability. Unlike existing FL security research, we comprehensively analyze feature-oriented threats, such as poisoning, inference, and reconstruction attacks that exploit model features. Moreover, we examine resilient aggregation techniques, anomaly detection, and cryptographic defenses, including differential privacy and secure multi-party computation, to strengthen FL security. In addition, we discuss the integration of 6G, large language models (LLMs), and interoperable learning frameworks to enhance privacy-preserving and decentralized cross-domain training. These advancements offer ultra-low latency, artificial intelligence (AI)-driven network management, and improved resilience against adversarial attacks, fostering the deployment of secure ResFL in CyberEdge networks.



## **47. Multilingual and Multi-Accent Jailbreaking of Audio LLMs**

cs.SD

21 pages, 6 figures, 15 tables

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.01094v1) [paper-pdf](http://arxiv.org/pdf/2504.01094v1)

**Authors**: Jaechul Roh, Virat Shejwalkar, Amir Houmansadr

**Abstract**: Large Audio Language Models (LALMs) have significantly advanced audio understanding but introduce critical security risks, particularly through audio jailbreaks. While prior work has focused on English-centric attacks, we expose a far more severe vulnerability: adversarial multilingual and multi-accent audio jailbreaks, where linguistic and acoustic variations dramatically amplify attack success. In this paper, we introduce Multi-AudioJail, the first systematic framework to exploit these vulnerabilities through (1) a novel dataset of adversarially perturbed multilingual/multi-accent audio jailbreaking prompts, and (2) a hierarchical evaluation pipeline revealing that how acoustic perturbations (e.g., reverberation, echo, and whisper effects) interacts with cross-lingual phonetics to cause jailbreak success rates (JSRs) to surge by up to +57.25 percentage points (e.g., reverberated Kenyan-accented attack on MERaLiON). Crucially, our work further reveals that multimodal LLMs are inherently more vulnerable than unimodal systems: attackers need only exploit the weakest link (e.g., non-English audio inputs) to compromise the entire model, which we empirically show by multilingual audio-only attacks achieving 3.1x higher success rates than text-only attacks. We plan to release our dataset to spur research into cross-modal defenses, urging the community to address this expanding attack surface in multimodality as LALMs evolve.



## **48. The Illusionist's Prompt: Exposing the Factual Vulnerabilities of Large Language Models with Linguistic Nuances**

cs.CL

work in progress

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.02865v1) [paper-pdf](http://arxiv.org/pdf/2504.02865v1)

**Authors**: Yining Wang, Yuquan Wang, Xi Li, Mi Zhang, Geng Hong, Min Yang

**Abstract**: As Large Language Models (LLMs) continue to advance, they are increasingly relied upon as real-time sources of information by non-expert users. To ensure the factuality of the information they provide, much research has focused on mitigating hallucinations in LLM responses, but only in the context of formal user queries, rather than maliciously crafted ones. In this study, we introduce The Illusionist's Prompt, a novel hallucination attack that incorporates linguistic nuances into adversarial queries, challenging the factual accuracy of LLMs against five types of fact-enhancing strategies. Our attack automatically generates highly transferrable illusory prompts to induce internal factual errors, all while preserving user intent and semantics. Extensive experiments confirm the effectiveness of our attack in compromising black-box LLMs, including commercial APIs like GPT-4o and Gemini-2.0, even with various defensive mechanisms.



## **49. Exposing the Ghost in the Transformer: Abnormal Detection for Large Language Models via Hidden State Forensics**

cs.CR

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.00446v1) [paper-pdf](http://arxiv.org/pdf/2504.00446v1)

**Authors**: Shide Zhou, Kailong Wang, Ling Shi, Haoyu Wang

**Abstract**: The widespread adoption of Large Language Models (LLMs) in critical applications has introduced severe reliability and security risks, as LLMs remain vulnerable to notorious threats such as hallucinations, jailbreak attacks, and backdoor exploits. These vulnerabilities have been weaponized by malicious actors, leading to unauthorized access, widespread misinformation, and compromised LLM-embedded system integrity. In this work, we introduce a novel approach to detecting abnormal behaviors in LLMs via hidden state forensics. By systematically inspecting layer-specific activation patterns, we develop a unified framework that can efficiently identify a range of security threats in real-time without imposing prohibitive computational costs. Extensive experiments indicate detection accuracies exceeding 95% and consistently robust performance across multiple models in most scenarios, while preserving the ability to detect novel attacks effectively. Furthermore, the computational overhead remains minimal, with merely fractions of a second. The significance of this work lies in proposing a promising strategy to reinforce the security of LLM-integrated systems, paving the way for safer and more reliable deployment in high-stakes domains. By enabling real-time detection that can also support the mitigation of abnormal behaviors, it represents a meaningful step toward ensuring the trustworthiness of AI systems amid rising security challenges.



## **50. Unleashing the Power of Pre-trained Encoders for Universal Adversarial Attack Detection**

cs.CV

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.00429v1) [paper-pdf](http://arxiv.org/pdf/2504.00429v1)

**Authors**: Yinghe Zhang, Chi Liu, Shuai Zhou, Sheng Shen, Peng Gui

**Abstract**: Adversarial attacks pose a critical security threat to real-world AI systems by injecting human-imperceptible perturbations into benign samples to induce misclassification in deep learning models. While existing detection methods, such as Bayesian uncertainty estimation and activation pattern analysis, have achieved progress through feature engineering, their reliance on handcrafted feature design and prior knowledge of attack patterns limits generalization capabilities and incurs high engineering costs. To address these limitations, this paper proposes a lightweight adversarial detection framework based on the large-scale pre-trained vision-language model CLIP. Departing from conventional adversarial feature characterization paradigms, we innovatively adopt an anomaly detection perspective. By jointly fine-tuning CLIP's dual visual-text encoders with trainable adapter networks and learnable prompts, we construct a compact representation space tailored for natural images. Notably, our detection architecture achieves substantial improvements in generalization capability across both known and unknown attack patterns compared to traditional methods, while significantly reducing training overhead. This study provides a novel technical pathway for establishing a parameter-efficient and attack-agnostic defense paradigm, markedly enhancing the robustness of vision systems against evolving adversarial threats.



