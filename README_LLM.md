# Latest Large Language Model Attack Papers
**update at 2025-12-16 18:35:47**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. On the Effectiveness of Membership Inference in Targeted Data Extraction from Large Language Models**

cs.LG

Accepted to IEEE Conference on Secure and Trustworthy Machine Learning (SaTML) 2026

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.13352v1) [paper-pdf](https://arxiv.org/pdf/2512.13352v1)

**Authors**: Ali Al Sahili, Ali Chehab, Razane Tajeddine

**Abstract**: Large Language Models (LLMs) are prone to memorizing training data, which poses serious privacy risks. Two of the most prominent concerns are training data extraction and Membership Inference Attacks (MIAs). Prior research has shown that these threats are interconnected: adversaries can extract training data from an LLM by querying the model to generate a large volume of text and subsequently applying MIAs to verify whether a particular data point was included in the training set. In this study, we integrate multiple MIA techniques into the data extraction pipeline to systematically benchmark their effectiveness. We then compare their performance in this integrated setting against results from conventional MIA benchmarks, allowing us to evaluate their practical utility in real-world extraction scenarios.



## **2. Cisco Integrated AI Security and Safety Framework Report**

cs.CR

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.12921v1) [paper-pdf](https://arxiv.org/pdf/2512.12921v1)

**Authors**: Amy Chang, Tiffany Saade, Sanket Mendapara, Adam Swanda, Ankit Garg

**Abstract**: Artificial intelligence (AI) systems are being readily and rapidly adopted, increasingly permeating critical domains: from consumer platforms and enterprise software to networked systems with embedded agents. While this has unlocked potential for human productivity gains, the attack surface has expanded accordingly: threats now span content safety failures (e.g., harmful or deceptive outputs), model and data integrity compromise (e.g., poisoning, supply-chain tampering), runtime manipulations (e.g., prompt injection, tool and agent misuse), and ecosystem risks (e.g., orchestration abuse, multi-agent collusion). Existing frameworks such as MITRE ATLAS, National Institute of Standards and Technology (NIST) AI 100-2 Adversarial Machine Learning (AML) taxonomy, and OWASP Top 10s for Large Language Models (LLMs) and Agentic AI Applications provide valuable viewpoints, but each covers only slices of this multi-dimensional space.   This paper presents Cisco's Integrated AI Security and Safety Framework ("AI Security Framework"), a unified, lifecycle-aware taxonomy and operationalization framework that can be used to classify, integrate, and operationalize the full range of AI risks. It integrates AI security and AI safety across modalities, agents, pipelines, and the broader ecosystem. The AI Security Framework is designed to be practical for threat identification, red-teaming, risk prioritization, and it is comprehensive in scope and can be extensible to emerging deployments in multimodal contexts, humanoids, wearables, and sensory infrastructures. We analyze gaps in prevailing frameworks, discuss design principles for our framework, and demonstrate how the taxonomy provides structure for understanding how modern AI systems fail, how adversaries exploit these failures, and how organizations can build defenses across the AI lifecycle that evolve alongside capability advancements.



## **3. CTIGuardian: A Few-Shot Framework for Mitigating Privacy Leakage in Fine-Tuned LLMs**

cs.CR

Accepted at the 18th Cybersecurity Experimentation and Test Workshop (CSET), in conjunction with ACSAC 2025

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.12914v1) [paper-pdf](https://arxiv.org/pdf/2512.12914v1)

**Authors**: Shashie Dilhara Batan Arachchige, Benjamin Zi Hao Zhao, Hassan Jameel Asghar, Dinusha Vatsalan, Dali Kaafar

**Abstract**: Large Language Models (LLMs) are often fine-tuned to adapt their general-purpose knowledge to specific tasks and domains such as cyber threat intelligence (CTI). Fine-tuning is mostly done through proprietary datasets that may contain sensitive information. Owners expect their fine-tuned model to not inadvertently leak this information to potentially adversarial end users. Using CTI as a use case, we demonstrate that data-extraction attacks can recover sensitive information from fine-tuned models on CTI reports, underscoring the need for mitigation. Retraining the full model to eliminate this leakage is computationally expensive and impractical. We propose an alternative approach, which we call privacy alignment, inspired by safety alignment in LLMs. Just like safety alignment teaches the model to abide by safety constraints through a few examples, we enforce privacy alignment through few-shot supervision, integrating a privacy classifier and a privacy redactor, both handled by the same underlying LLM. We evaluate our system, called CTIGuardian, using GPT-4o mini and Mistral-7B Instruct models, benchmarking against Presidio, a named entity recognition (NER) baseline. Results show that CTIGuardian provides a better privacy-utility trade-off than NER based models. While we demonstrate its effectiveness on a CTI use case, the framework is generic enough to be applicable to other sensitive domains.



## **4. The Role of AI in Modern Penetration Testing**

cs.SE

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2512.12326v1) [paper-pdf](https://arxiv.org/pdf/2512.12326v1)

**Authors**: J. Alexander Curtis, Nasir U. Eisty

**Abstract**: Penetration testing is a cornerstone of cybersecurity, traditionally driven by manual, time-intensive processes. As systems grow in complexity, there is a pressing need for more scalable and efficient testing methodologies. This systematic literature review examines how Artificial Intelligence (AI) is reshaping penetration testing, analyzing 58 peer-reviewed studies from major academic databases. Our findings reveal that while AI-assisted pentesting is still in its early stages, notable progress is underway, particularly through Reinforcement Learning (RL), which was the focus of 77% of the reviewed works. Most research centers on the discovery and exploitation phases of pentesting, where AI shows the greatest promise in automating repetitive tasks, optimizing attack strategies, and improving vulnerability identification. Real-world applications remain limited but encouraging, including the European Space Agency's PenBox and various open-source tools. These demonstrate AI's potential to streamline attack path analysis, analyze complex network topology, and reduce manual workload. However, challenges persist: current models often lack flexibility and are underdeveloped for the reconnaissance and post-exploitation phases of pentesting. Applications involving Large Language Models (LLMs) remain relatively under-researched, pointing to a promising direction for future exploration. This paper offers a critical overview of AI's current and potential role in penetration testing, providing valuable insights for researchers, practitioners, and organizations aiming to enhance security assessments through advanced automation or looking for gaps in existing research.



## **5. Taint-Based Code Slicing for LLMs-based Malicious NPM Package Detection**

cs.CR

17 pages, 4 figures, 9 tables

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2512.12313v1) [paper-pdf](https://arxiv.org/pdf/2512.12313v1)

**Authors**: Dang-Khoa Nguyen, Gia-Thang Ho, Quang-Minh Pham, Tuyet A. Dang-Thi, Minh-Khanh Vu, Thanh-Cong Nguyen, Phat T. Tran-Truong, Duc-Ly Vu

**Abstract**: The increasing sophistication of malware attacks in the npm ecosystem, characterized by obfuscation and complex logic, necessitates advanced detection methods. Recently, researchers have turned their attention from traditional detection approaches to Large Language Models (LLMs) due to their strong capabilities in semantic code understanding. However, while LLMs offer superior semantic reasoning for code analysis, their practical application is constrained by limited context windows and high computational cost. This paper addresses this challenge by introducing a novel framework that leverages code slicing techniques for an LLM-based malicious package detection task. We propose a specialized taintbased slicing technique for npm packages, augmented by a heuristic backtracking mechanism to accurately capture malicious data flows across asynchronous, event-driven patterns (e.g., callbacks and Promises) that elude traditional analysis. An evaluation on a dataset of more than 5000 malicious and benign npm packages demonstrates that our approach isolates security-relevant code, reducing input volume by over 99% while preserving critical behavioral semantics. Using the DeepSeek-Coder-6.7B model as the classification engine, our approach achieves a detection accuracy of 87.04%, substantially outperforming a naive token-splitting baseline (75.41%) and a traditional static-analysis-based approach. These results indicate that semantically optimized input representation via code slicing not only mitigates the LLM context-window bottleneck but also significantly enhances reasoning precision for security tasks, providing an efficient and effective defense against evolving malicious open-source packages.



## **6. Keep the Lights On, Keep the Lengths in Check: Plug-In Adversarial Detection for Time-Series LLMs in Energy Forecasting**

cs.CR

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2512.12154v1) [paper-pdf](https://arxiv.org/pdf/2512.12154v1)

**Authors**: Hua Ma, Ruoxi Sun, Minhui Xue, Xingliang Yuan, Carsten Rudolph, Surya Nepal, Ling Liu

**Abstract**: Accurate time-series forecasting is increasingly critical for planning and operations in low-carbon power systems. Emerging time-series large language models (TS-LLMs) now deliver this capability at scale, requiring no task-specific retraining, and are quickly becoming essential components within the Internet-of-Energy (IoE) ecosystem. However, their real-world deployment is complicated by a critical vulnerability: adversarial examples (AEs). Detecting these AEs is challenging because (i) adversarial perturbations are optimized across the entire input sequence and exploit global temporal dependencies, which renders local detection methods ineffective, and (ii) unlike traditional forecasting models with fixed input dimensions, TS-LLMs accept sequences of variable length, increasing variability that complicates detection. To address these challenges, we propose a plug-in detection framework that capitalizes on the TS-LLM's own variable-length input capability. Our method uses sampling-induced divergence as a detection signal. Given an input sequence, we generate multiple shortened variants and detect AEs by measuring the consistency of their forecasts: Benign sequences tend to produce stable predictions under sampling, whereas adversarial sequences show low forecast similarity, because perturbations optimized for a full-length sequence do not transfer reliably to shorter, differently-structured subsamples. We evaluate our approach on three representative TS-LLMs (TimeGPT, TimesFM, and TimeLLM) across three energy datasets: ETTh2 (Electricity Transformer Temperature), NI (Hourly Energy Consumption), and Consumption (Hourly Electricity Consumption and Production). Empirical results confirm strong and robust detection performance across both black-box and white-box attack scenarios, highlighting its practicality as a reliable safeguard for TS-LLM forecasting in real-world energy systems.



## **7. BRIDG-ICS: AI-Grounded Knowledge Graphs for Intelligent Threat Analytics in Industry~5.0 Cyber-Physical Systems**

cs.CR

44 Pages, To be published in Springer Cybersecurity Journal

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2512.12112v1) [paper-pdf](https://arxiv.org/pdf/2512.12112v1)

**Authors**: Padmeswari Nandiya, Ahmad Mohsin, Ahmed Ibrahim, Iqbal H. Sarker, Helge Janicke

**Abstract**: Industry 5.0's increasing integration of IT and OT systems is transforming industrial operations but also expanding the cyber-physical attack surface. Industrial Control Systems (ICS) face escalating security challenges as traditional siloed defences fail to provide coherent, cross-domain threat insights. We present BRIDG-ICS (BRIDge for Industrial Control Systems), an AI-driven Knowledge Graph (KG) framework for context-aware threat analysis and quantitative assessment of cyber resilience in smart manufacturing environments. BRIDG-ICS fuses heterogeneous industrial and cybersecurity data into an integrated Industrial Security Knowledge Graph linking assets, vulnerabilities, and adversarial behaviours with probabilistic risk metrics (e.g. exploit likelihood, attack cost). This unified graph representation enables multi-stage attack path simulation using graph-analytic techniques. To enrich the graph's semantic depth, the framework leverages Large Language Models (LLMs): domain-specific LLMs extract cybersecurity entities, predict relationships, and translate natural-language threat descriptions into structured graph triples, thereby populating the knowledge graph with missing associations and latent risk indicators. This unified AI-enriched KG supports multi-hop, causality-aware threat reasoning, improving visibility into complex attack chains and guiding data-driven mitigation. In simulated industrial scenarios, BRIDG-ICS scales well, reduces potential attack exposure, and can enhance cyber-physical system resilience in Industry 5.0 settings.



## **8. Rethinking Jailbreak Detection of Large Vision Language Models with Representational Contrastive Scoring**

cs.CR

40 pages, 13 figures

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.12069v1) [paper-pdf](https://arxiv.org/pdf/2512.12069v1)

**Authors**: Peichun Hua, Hao Li, Shanghao Shi, Zhiyuan Yu, Ning Zhang

**Abstract**: Large Vision-Language Models (LVLMs) are vulnerable to a growing array of multimodal jailbreak attacks, necessitating defenses that are both generalizable to novel threats and efficient for practical deployment. Many current strategies fall short, either targeting specific attack patterns, which limits generalization, or imposing high computational overhead. While lightweight anomaly-detection methods offer a promising direction, we find that their common one-class design tends to confuse novel benign inputs with malicious ones, leading to unreliable over-rejection. To address this, we propose Representational Contrastive Scoring (RCS), a framework built on a key insight: the most potent safety signals reside within the LVLM's own internal representations. Our approach inspects the internal geometry of these representations, learning a lightweight projection to maximally separate benign and malicious inputs in safety-critical layers. This enables a simple yet powerful contrastive score that differentiates true malicious intent from mere novelty. Our instantiations, MCD (Mahalanobis Contrastive Detection) and KCD (K-nearest Contrastive Detection), achieve state-of-the-art performance on a challenging evaluation protocol designed to test generalization to unseen attack types. This work demonstrates that effective jailbreak detection can be achieved by applying simple, interpretable statistical methods to the appropriate internal representations, offering a practical path towards safer LVLM deployment. Our code is available on Github https://github.com/sarendis56/Jailbreak_Detection_RCS.



## **9. Learning to Extract Context for Context-Aware LLM Inference**

cs.LG

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.11986v1) [paper-pdf](https://arxiv.org/pdf/2512.11986v1)

**Authors**: Minseon Kim, Lucas Caccia, Zhengyan Shi, Matheus Pereira, Marc-Alexandre Côté, Xingdi Yuan, Alessandro Sordoni

**Abstract**: User prompts to large language models (LLMs) are often ambiguous or under-specified, and subtle contextual cues shaped by user intentions, prior knowledge, and risk factors strongly influence what constitutes an appropriate response. Misinterpreting intent or risks may lead to unsafe outputs, while overly cautious interpretations can cause unnecessary refusal of benign requests. In this paper, we question the conventional framework in which LLMs generate immediate responses to requests without considering broader contextual factors. User requests are situated within broader contexts such as intentions, knowledge, and prior experience, which strongly influence what constitutes an appropriate answer. We propose a framework that extracts and leverages such contextual information from the user prompt itself. Specifically, a reinforcement learning based context generator, designed in an autoencoder-like fashion, is trained to infer contextual signals grounded in the prompt and use them to guide response generation. This approach is particularly important for safety tasks, where ambiguous requests may bypass safeguards while benign but confusing requests can trigger unnecessary refusals. Experiments show that our method reduces harmful responses by an average of 5.6% on the SafetyInstruct dataset across multiple foundation models and improves the harmonic mean of attack success rate and compliance on benign prompts by 6.2% on XSTest and WildJailbreak. These results demonstrate the effectiveness of context extraction for safer and more reliable LLM inferences.



## **10. Super Suffixes: Bypassing Text Generation Alignment and Guard Models Simultaneously**

cs.CR

13 pages, 5 Figures

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.11783v1) [paper-pdf](https://arxiv.org/pdf/2512.11783v1)

**Authors**: Andrew Adiletta, Kathryn Adiletta, Kemal Derya, Berk Sunar

**Abstract**: The rapid deployment of Large Language Models (LLMs) has created an urgent need for enhanced security and privacy measures in Machine Learning (ML). LLMs are increasingly being used to process untrusted text inputs and even generate executable code, often while having access to sensitive system controls. To address these security concerns, several companies have introduced guard models, which are smaller, specialized models designed to protect text generation models from adversarial or malicious inputs. In this work, we advance the study of adversarial inputs by introducing Super Suffixes, suffixes capable of overriding multiple alignment objectives across various models with different tokenization schemes. We demonstrate their effectiveness, along with our joint optimization technique, by successfully bypassing the protection mechanisms of Llama Prompt Guard 2 on five different text generation models for malicious text and code generation. To the best of our knowledge, this is the first work to reveal that Llama Prompt Guard 2 can be compromised through joint optimization.   Additionally, by analyzing the changing similarity of a model's internal state to specific concept directions during token sequence processing, we propose an effective and lightweight method to detect Super Suffix attacks. We show that the cosine similarity between the residual stream and certain concept directions serves as a distinctive fingerprint of model intent. Our proposed countermeasure, DeltaGuard, significantly improves the detection of malicious prompts generated through Super Suffixes. It increases the non-benign classification rate to nearly 100%, making DeltaGuard a valuable addition to the guard model stack and enhancing robustness against adversarial prompt attacks.



## **11. When Reject Turns into Accept: Quantifying the Vulnerability of LLM-Based Scientific Reviewers to Indirect Prompt Injection**

cs.AI

**SubmitDate**: 2025-12-15    [abs](http://arxiv.org/abs/2512.10449v2) [paper-pdf](https://arxiv.org/pdf/2512.10449v2)

**Authors**: Devanshu Sahoo, Manish Prasad, Vasudev Majhi, Jahnvi Singh, Vinay Chamola, Yash Sinha, Murari Mandal, Dhruv Kumar

**Abstract**: The landscape of scientific peer review is rapidly evolving with the integration of Large Language Models (LLMs). This shift is driven by two parallel trends: the widespread individual adoption of LLMs by reviewers to manage workload (the "Lazy Reviewer" hypothesis) and the formal institutional deployment of AI-powered assessment systems by conferences like AAAI and Stanford's Agents4Science. This study investigates the robustness of these "LLM-as-a-Judge" systems (both illicit and sanctioned) to adversarial PDF manipulation. Unlike general jailbreaks, we focus on a distinct incentive: flipping "Reject" decisions to "Accept," for which we develop a novel evaluation metric which we term as WAVS (Weighted Adversarial Vulnerability Score). We curated a dataset of 200 scientific papers and adapted 15 domain-specific attack strategies to this task, evaluating them across 13 Language Models, including GPT-5, Claude Haiku, and DeepSeek. Our results demonstrate that obfuscation strategies like "Maximum Mark Magyk" successfully manipulate scores, achieving alarming decision flip rates even in large-scale models. We will release our complete dataset and injection framework to facilitate more research on this topic.



## **12. How to Trick Your AI TA: A Systematic Study of Academic Jailbreaking in LLM Code Evaluation**

cs.SE

Under Review

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2512.10415v1) [paper-pdf](https://arxiv.org/pdf/2512.10415v1)

**Authors**: Devanshu Sahoo, Vasudev Majhi, Arjun Neekhra, Yash Sinha, Murari Mandal, Dhruv Kumar

**Abstract**: The use of Large Language Models (LLMs) as automatic judges for code evaluation is becoming increasingly prevalent in academic environments. But their reliability can be compromised by students who may employ adversarial prompting strategies in order to induce misgrading and secure undeserved academic advantages. In this paper, we present the first large-scale study of jailbreaking LLM-based automated code evaluators in academic context. Our contributions are: (i) We systematically adapt 20+ jailbreaking strategies for jailbreaking AI code evaluators in the academic context, defining a new class of attacks termed academic jailbreaking. (ii) We release a poisoned dataset of 25K adversarial student submissions, specifically designed for the academic code-evaluation setting, sourced from diverse real-world coursework and paired with rubrics and human-graded references, and (iii) In order to capture the multidimensional impact of academic jailbreaking, we systematically adapt and define three jailbreaking metrics (Jailbreak Success Rate, Score Inflation, and Harmfulness). (iv) We comprehensively evalulate the academic jailbreaking attacks using six LLMs. We find that these models exhibit significant vulnerability, particularly to persuasive and role-play-based attacks (up to 97% JSR). Our adversarial dataset and benchmark suite lay the groundwork for next-generation robust LLM-based evaluators in academic code assessment.



## **13. Unforgotten Safety: Preserving Safety Alignment of Large Language Models with Continual Learning**

cs.CL

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2512.10150v1) [paper-pdf](https://arxiv.org/pdf/2512.10150v1)

**Authors**: Lama Alssum, Hani Itani, Hasan Abed Al Kader Hammoud, Philip Torr, Adel Bibi, Bernard Ghanem

**Abstract**: The safety alignment of large language models (LLMs) is becoming increasingly important with their democratization. In this paper, we study the safety degradation that comes with adapting LLMs to new tasks. We attribute this safety compromise to catastrophic forgetting and frame the problem of preserving safety when fine-tuning as a continual learning (CL) problem. We consider the fine-tuning-as-a-service setup where the user uploads their data to a service provider to get a customized model that excels on the user's selected task. We adapt several CL approaches from the literature and systematically evaluate their ability to mitigate safety degradation. These include regularization-based, memory-based, and model merging approaches. We consider two scenarios, (1) benign user data and (2) poisoned user data. Our results demonstrate that CL approaches consistently achieve lower attack success rates than standard fine-tuning. Among these, DER outperforms both other CL methods and existing safety-preserving baselines while maintaining task utility. These findings generalize across three downstream tasks (GSM8K, SST2, Code) and three model families (LLaMA2-7B, Mistral-7B, Gemma-2B), establishing CL as a practical solution to preserve safety.



## **14. Phishing Email Detection Using Large Language Models**

cs.CR

7 pages

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2512.10104v2) [paper-pdf](https://arxiv.org/pdf/2512.10104v2)

**Authors**: Najmul Hasan, Prashanth BusiReddyGari, Haitao Zhao, Yihao Ren, Jinsheng Xu, Shaohu Zhang

**Abstract**: Email phishing is one of the most prevalent and globally consequential vectors of cyber intrusion. As systems increasingly deploy Large Language Models (LLMs) applications, these systems face evolving phishing email threats that exploit their fundamental architectures. Current LLMs require substantial hardening before deployment in email security systems, particularly against coordinated multi-vector attacks that exploit architectural vulnerabilities. This paper proposes LLMPEA, an LLM-based framework to detect phishing email attacks across multiple attack vectors, including prompt injection, text refinement, and multilingual attacks. We evaluate three frontier LLMs (e.g., GPT-4o, Claude Sonnet 4, and Grok-3) and comprehensive prompting design to assess their feasibility, robustness, and limitations against phishing email attacks. Our empirical analysis reveals that LLMs can detect the phishing email over 90% accuracy while we also highlight that LLM-based phishing email detection systems could be exploited by adversarial attack, prompt injection, and multilingual attacks. Our findings provide critical insights for LLM-based phishing detection in real-world settings where attackers exploit multiple vulnerabilities in combination.



## **15. FlipLLM: Efficient Bit-Flip Attacks on Multimodal LLMs using Reinforcement Learning**

cs.CR

Accepted in IEEE HOST 2026

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2512.09872v1) [paper-pdf](https://arxiv.org/pdf/2512.09872v1)

**Authors**: Khurram Khalil, Khaza Anuarul Hoque

**Abstract**: Generative Artificial Intelligence models, such as Large Language Models (LLMs) and Large Vision Models (VLMs), exhibit state-of-the-art performance but remain vulnerable to hardware-based threats, specifically bit-flip attacks (BFAs). Existing BFA discovery methods lack generalizability and struggle to scale, often failing to analyze the vast parameter space and complex interdependencies of modern foundation models in a reasonable time. This paper proposes FlipLLM, a reinforcement learning (RL) architecture-agnostic framework that formulates BFA discovery as a sequential decision-making problem. FlipLLM combines sensitivity-guided layer pruning with Q-learning to efficiently identify minimal, high-impact bit sets that can induce catastrophic failure. We demonstrate the effectiveness and generalizability of FlipLLM by applying it to a diverse set of models, including prominent text-only LLMs (GPT-2 Large, LLaMA 3.1 8B, and DeepSeek-V2 7B), VLMs such as LLaVA 1.6, and datasets, such as MMLU, MMLU-Pro, VQAv2, and TextVQA. Our results show that FlipLLM can identify critical bits that are vulnerable to BFAs up to 2.5x faster than SOTA methods. We demonstrate that flipping the FlipLLM-identified bits plummets the accuracy of LLaMA 3.1 8B from 69.9% to ~0.2%, and for LLaVA's VQA score from 78% to almost 0%, by flipping as few as 5 and 7 bits, respectively. Further analysis reveals that applying standard hardware protection mechanisms, such as ECC SECDED, to the FlipLLM-identified bit locations completely mitigates the BFA impact, demonstrating the practical value of our framework in guiding hardware-level defenses. FlipLLM offers the first scalable and adaptive methodology for exploring the BFA vulnerability of both language and multimodal foundation models, paving the way for comprehensive hardware-security evaluation.



## **16. MedForget: Hierarchy-Aware Multimodal Unlearning Testbed for Medical AI**

cs.CV

Dataset and Code: https://github.com/fengli-wu/MedForget

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2512.09867v1) [paper-pdf](https://arxiv.org/pdf/2512.09867v1)

**Authors**: Fengli Wu, Vaidehi Patil, Jaehong Yoon, Yue Zhang, Mohit Bansal

**Abstract**: Pretrained Multimodal Large Language Models (MLLMs) are increasingly deployed in medical AI systems for clinical reasoning, diagnosis support, and report generation. However, their training on sensitive patient data raises critical privacy and compliance challenges under regulations such as HIPAA and GDPR, which enforce the "right to be forgotten". Unlearning, the process of tuning models to selectively remove the influence of specific training data points, offers a potential solution, yet its effectiveness in complex medical settings remains underexplored. To systematically study this, we introduce MedForget, a Hierarchy-Aware Multimodal Unlearning Testbed with explicit retain and forget splits and evaluation sets containing rephrased variants. MedForget models hospital data as a nested hierarchy (Institution -> Patient -> Study -> Section), enabling fine-grained assessment across eight organizational levels. The benchmark contains 3840 multimodal (image, question, answer) instances, each hierarchy level having a dedicated unlearning target, reflecting distinct unlearning challenges. Experiments with four SOTA unlearning methods on three tasks (generation, classification, cloze) show that existing methods struggle to achieve complete, hierarchy-aware forgetting without reducing diagnostic performance. To test whether unlearning truly deletes hierarchical pathways, we introduce a reconstruction attack that progressively adds hierarchical level context to prompts. Models unlearned at a coarse granularity show strong resistance, while fine-grained unlearning leaves models vulnerable to such reconstruction. MedForget provides a practical, HIPAA-aligned testbed for building compliant medical AI systems.



## **17. Advancing LLM-Based Security Automation with Customized Group Relative Policy Optimization for Zero-Touch Networks**

cs.CR

Accepted by IEEE JSAC. This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2512.09485v1) [paper-pdf](https://arxiv.org/pdf/2512.09485v1)

**Authors**: Xinye Cao, Yihan Lin, Guoshun Nan, Qinchuan Zhou, Yuhang Luo, Yurui Gao, Zeliang Zhang, Haolang Lu, Qimei Cui, Yanzhao Hou, Xiaofeng Tao, Tony Q. S. Quek

**Abstract**: Zero-Touch Networks (ZTNs) represent a transformative paradigm toward fully automated and intelligent network management, providing the scalability and adaptability required for the complexity of sixth-generation (6G) networks. However, the distributed architecture, high openness, and deep heterogeneity of 6G networks expand the attack surface and pose unprecedented security challenges. To address this, security automation aims to enable intelligent security management across dynamic and complex environments, serving as a key capability for securing 6G ZTNs. Despite its promise, implementing security automation in 6G ZTNs presents two primary challenges: 1) automating the lifecycle from security strategy generation to validation and update under real-world, parallel, and adversarial conditions, and 2) adapting security strategies to evolving threats and dynamic environments. This motivates us to propose SecLoop and SA-GRPO. SecLoop constitutes the first fully automated framework that integrates large language models (LLMs) across the entire lifecycle of security strategy generation, orchestration, response, and feedback, enabling intelligent and adaptive defenses in dynamic network environments, thus tackling the first challenge. Furthermore, we propose SA-GRPO, a novel security-aware group relative policy optimization algorithm that iteratively refines security strategies by contrasting group feedback collected from parallel SecLoop executions, thereby addressing the second challenge. Extensive real-world experiments on five benchmarks, including 11 MITRE ATT&CK processes and over 20 types of attacks, demonstrate the superiority of the proposed SecLoop and SA-GRPO. We will release our platform to the community, facilitating the advancement of security automation towards next generation communications.



## **18. Read or Ignore? A Unified Benchmark for Typographic-Attack Robustness and Text Recognition in Vision-Language Models**

cs.CV

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2512.11899v1) [paper-pdf](https://arxiv.org/pdf/2512.11899v1)

**Authors**: Futa Waseda, Shojiro Yamabe, Daiki Shiono, Kento Sasaki, Tsubasa Takahashi

**Abstract**: Large vision-language models (LVLMs) are vulnerable to typographic attacks, where misleading text within an image overrides visual understanding. Existing evaluation protocols and defenses, largely focused on object recognition, implicitly encourage ignoring text to achieve robustness; however, real-world scenarios often require joint reasoning over both objects and text (e.g., recognizing pedestrians while reading traffic signs). To address this, we introduce a novel task, Read-or-Ignore VQA (RIO-VQA), which formalizes selective text use in visual question answering (VQA): models must decide, from context, when to read text and when to ignore it. For evaluation, we present the Read-or-Ignore Benchmark (RIO-Bench), a standardized dataset and protocol that, for each real image, provides same-scene counterfactuals (read / ignore) by varying only the textual content and question type. Using RIO-Bench, we show that strong LVLMs and existing defenses fail to balance typographic robustness and text-reading capability, highlighting the need for improved approaches. Finally, RIO-Bench enables a novel data-driven defense that learns adaptive selective text use, moving beyond prior non-adaptive, text-ignoring defenses. Overall, this work reveals a fundamental misalignment between the existing evaluation scope and real-world requirements, providing a principled path toward reliable LVLMs. Our Project Page is at https://turingmotors.github.io/rio-vqa/.



## **19. Black-Box Behavioral Distillation Breaks Safety Alignment in Medical LLMs**

cs.LG

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2512.09403v1) [paper-pdf](https://arxiv.org/pdf/2512.09403v1)

**Authors**: Sohely Jahan, Ruimin Sun

**Abstract**: As medical large language models (LLMs) become increasingly integrated into clinical workflows, concerns around alignment robustness, and safety are escalating. Prior work on model extraction has focused on classification models or memorization leakage, leaving the vulnerability of safety-aligned generative medical LLMs underexplored.   We present a black-box distillation attack that replicates the domain-specific reasoning of safety-aligned medical LLMs using only output-level access. By issuing 48,000 instruction queries to Meditron-7B and collecting 25,000 benign instruction response pairs, we fine-tune a LLaMA3 8B surrogate via parameter efficient LoRA under a zero-alignment supervision setting, requiring no access to model weights, safety filters, or training data. With a cost of $12, the surrogate achieves strong fidelity on benign inputs while producing unsafe completions for 86% of adversarial prompts, far exceeding both Meditron-7B (66%) and the untuned base model (46%). This reveals a pronounced functional-ethical gap, task utility transfers, while alignment collapses. To analyze this collapse, we develop a dynamic adversarial evaluation framework combining Generative Query (GQ)-based harmful prompt generation, verifier filtering, category-wise failure analysis, and adaptive Random Search (RS) jailbreak attacks. We also propose a layered defense system, as a prototype detector for real-time alignment drift in black-box deployments.   Our findings show that benign-only black-box distillation exposes a practical and under-recognized threat: adversaries can cheaply replicate medical LLM capabilities while stripping safety mechanisms, underscoring the need for extraction-aware safety monitoring.



## **20. When Tables Leak: Attacking String Memorization in LLM-Based Tabular Data Generation**

cs.LG

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08875v1) [paper-pdf](https://arxiv.org/pdf/2512.08875v1)

**Authors**: Joshua Ward, Bochao Gu, Chi-Hua Wang, Guang Cheng

**Abstract**: Large Language Models (LLMs) have recently demonstrated remarkable performance in generating high-quality tabular synthetic data. In practice, two primary approaches have emerged for adapting LLMs to tabular data generation: (i) fine-tuning smaller models directly on tabular datasets, and (ii) prompting larger models with examples provided in context. In this work, we show that popular implementations from both regimes exhibit a tendency to compromise privacy by reproducing memorized patterns of numeric digits from their training data. To systematically analyze this risk, we introduce a simple No-box Membership Inference Attack (MIA) called LevAtt that assumes adversarial access to only the generated synthetic data and targets the string sequences of numeric digits in synthetic observations. Using this approach, our attack exposes substantial privacy leakage across a wide range of models and datasets, and in some cases, is even a perfect membership classifier on state-of-the-art models. Our findings highlight a unique privacy vulnerability of LLM-based synthetic data generation and the need for effective defenses. To this end, we propose two methods, including a novel sampling strategy that strategically perturbs digits during generation. Our evaluation demonstrates that this approach can defeat these attacks with minimal loss of fidelity and utility of the synthetic data.



## **21. PrivTune: Efficient and Privacy-Preserving Fine-Tuning of Large Language Models via Device-Cloud Collaboration**

cs.CR

Accepted at IEEE INFOCOM 2026 (full version)

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08809v1) [paper-pdf](https://arxiv.org/pdf/2512.08809v1)

**Authors**: Yi Liu, Weixiang Han, Chengjun Cai, Xingliang Yuan, Cong Wang

**Abstract**: With the rise of large language models, service providers offer language models as a service, enabling users to fine-tune customized models via uploaded private datasets. However, this raises concerns about sensitive data leakage. Prior methods, relying on differential privacy within device-cloud collaboration frameworks, struggle to balance privacy and utility, exposing users to inference attacks or degrading fine-tuning performance. To address this, we propose PrivTune, an efficient and privacy-preserving fine-tuning framework via Split Learning (SL). The key idea of PrivTune is to inject crafted noise into token representations from the SL bottom model, making each token resemble the $n$-hop indirect neighbors. PrivTune formulates this as an optimization problem to compute the optimal noise vector, aligning with defense-utility goals. On this basis, it then adjusts the parameters (i.e., mean) of the $d_χ$-Privacy noise distribution to align with the optimization direction and scales the noise according to token importance to minimize distortion. Experiments on five datasets (covering both classification and generation tasks) against three embedding inversion and three attribute inference attacks show that, using RoBERTa on the Stanford Sentiment Treebank dataset, PrivTune reduces the attack success rate to 10% with only a 3.33% drop in utility performance, outperforming state-of-the-art baselines.



## **22. Attention is All You Need to Defend Against Indirect Prompt Injection Attacks in LLMs**

cs.CR

Accepted by Network and Distributed System Security (NDSS) Symposium 2026

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2512.08417v2) [paper-pdf](https://arxiv.org/pdf/2512.08417v2)

**Authors**: Yinan Zhong, Qianhao Miao, Yanjiao Chen, Jiangyi Deng, Yushi Cheng, Wenyuan Xu

**Abstract**: Large Language Models (LLMs) have been integrated into many applications (e.g., web agents) to perform more sophisticated tasks. However, LLM-empowered applications are vulnerable to Indirect Prompt Injection (IPI) attacks, where instructions are injected via untrustworthy external data sources. This paper presents Rennervate, a defense framework to detect and prevent IPI attacks. Rennervate leverages attention features to detect the covert injection at a fine-grained token level, enabling precise sanitization that neutralizes IPI attacks while maintaining LLM functionalities. Specifically, the token-level detector is materialized with a 2-step attentive pooling mechanism, which aggregates attention heads and response tokens for IPI detection and sanitization. Moreover, we establish a fine-grained IPI dataset, FIPI, to be open-sourced to support further research. Extensive experiments verify that Rennervate outperforms 15 commercial and academic IPI defense methods, achieving high precision on 5 LLMs and 6 datasets. We also demonstrate that Rennervate is transferable to unseen attacks and robust against adaptive adversaries.



## **23. Advancing Autonomous Driving System Testing: Demands, Challenges, and Future Directions**

cs.CY

Accepted for publication in Information and Software Technology (IST)

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.11887v1) [paper-pdf](https://arxiv.org/pdf/2512.11887v1)

**Authors**: Yihan Liao, Jingyu Zhang, Jacky Keung, Yan Xiao, Yurou Dai

**Abstract**: Autonomous driving systems (ADSs) promise improved transportation efficiency and safety, yet ensuring their reliability in complex real-world environments remains a critical challenge. Effective testing is essential to validate ADS performance and reduce deployment risks. This study investigates current ADS testing practices for both modular and end-to-end systems, identifies key demands from industry practitioners and academic researchers, and analyzes the gaps between existing research and real-world requirements. We review major testing techniques and further consider emerging factors such as Vehicle-to-Everything (V2X) communication and foundation models, including large language models and vision foundation models, to understand their roles in enhancing ADS testing. We conducted a large-scale survey with 100 participants from both industry and academia. Survey questions were refined through expert discussions, followed by quantitative and qualitative analyses to reveal key trends, challenges, and unmet needs. Our results show that existing ADS testing techniques struggle to comprehensively evaluate real-world performance, particularly regarding corner case diversity, the simulation to reality gap, the lack of systematic testing criteria, exposure to potential attacks, practical challenges in V2X deployment, and the high computational cost of foundation model-based testing. By further analyzing participant responses together with 105 representative studies, we summarize the current research landscape and highlight major limitations. This study consolidates critical research gaps in ADS testing and outlines key future research directions, including comprehensive testing criteria, cross-model collaboration in V2X systems, cross-modality adaptation for foundation model-based testing, and scalable validation frameworks for large-scale ADS evaluation.



## **24. A Practical Framework for Evaluating Medical AI Security: Reproducible Assessment of Jailbreaking and Privacy Vulnerabilities Across Clinical Specialties**

cs.CR

6 pages, 1 figure, framework proposal

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08185v1) [paper-pdf](https://arxiv.org/pdf/2512.08185v1)

**Authors**: Jinghao Wang, Ping Zhang, Carter Yagemann

**Abstract**: Medical Large Language Models (LLMs) are increasingly deployed for clinical decision support across diverse specialties, yet systematic evaluation of their robustness to adversarial misuse and privacy leakage remains inaccessible to most researchers. Existing security benchmarks require GPU clusters, commercial API access, or protected health data -- barriers that limit community participation in this critical research area. We propose a practical, fully reproducible framework for evaluating medical AI security under realistic resource constraints. Our framework design covers multiple medical specialties stratified by clinical risk -- from high-risk domains such as emergency medicine and psychiatry to general practice -- addressing jailbreaking attacks (role-playing, authority impersonation, multi-turn manipulation) and privacy extraction attacks. All evaluation utilizes synthetic patient records requiring no IRB approval. The framework is designed to run entirely on consumer CPU hardware using freely available models, eliminating cost barriers. We present the framework specification including threat models, data generation methodology, evaluation protocols, and scoring rubrics. This proposal establishes a foundation for comparative security assessment of medical-specialist models and defense mechanisms, advancing the broader goal of ensuring safe and trustworthy medical AI systems.



## **25. Detecting Ambiguity Aversion in Cyberattack Behavior to Inform Cognitive Defense Strategies**

cs.CR

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.08107v1) [paper-pdf](https://arxiv.org/pdf/2512.08107v1)

**Authors**: Stephan Carney, Soham Hans, Sofia Hirschmann, Stacey Marsella, Yvonne Fonken, Peggy Wu, Nikolos Gurney

**Abstract**: Adversaries (hackers) attempting to infiltrate networks frequently face uncertainty in their operational environments. This research explores the ability to model and detect when they exhibit ambiguity aversion, a cognitive bias reflecting a preference for known (versus unknown) probabilities. We introduce a novel methodological framework that (1) leverages rich, multi-modal data from human-subjects red-team experiments, (2) employs a large language model (LLM) pipeline to parse unstructured logs into MITRE ATT&CK-mapped action sequences, and (3) applies a new computational model to infer an attacker's ambiguity aversion level in near-real time. By operationalizing this cognitive trait, our work provides a foundational component for developing adaptive cognitive defense strategies.



## **26. RL-MTJail: Reinforcement Learning for Automated Black-Box Multi-Turn Jailbreaking of Large Language Models**

cs.AI

19 pages, 15 figures

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07761v1) [paper-pdf](https://arxiv.org/pdf/2512.07761v1)

**Authors**: Xiqiao Xiong, Ouxiang Li, Zhuo Liu, Moxin Li, Wentao Shi, Fuli Feng, Xiangnan He

**Abstract**: Large language models are vulnerable to jailbreak attacks, threatening their safe deployment in real-world applications. This paper studies black-box multi-turn jailbreaks, aiming to train attacker LLMs to elicit harmful content from black-box models through a sequence of prompt-output interactions. Existing approaches typically rely on single turn optimization, which is insufficient for learning long-term attack strategies. To bridge this gap, we formulate the problem as a multi-turn reinforcement learning task, directly optimizing the harmfulness of the final-turn output as the outcome reward. To mitigate sparse supervision and promote long-term attack strategies, we propose two heuristic process rewards: (1) controlling the harmfulness of intermediate outputs to prevent triggering the black-box model's rejection mechanisms, and (2) maintaining the semantic relevance of intermediate outputs to avoid drifting into irrelevant content. Experimental results on multiple benchmarks show consistently improved attack success rates across multiple models, highlighting the effectiveness of our approach. The code is available at https://github.com/xxiqiao/RL-MTJail. Warning: This paper contains examples of harmful content.



## **27. When Large Language Models Do Not Work: Online Incivility Prediction through Graph Neural Networks**

cs.CL

10 pages

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07684v1) [paper-pdf](https://arxiv.org/pdf/2512.07684v1)

**Authors**: Zihan Chen, Lanyu Yu

**Abstract**: Online incivility has emerged as a widespread and persistent problem in digital communities, imposing substantial social and psychological burdens on users. Although many platforms attempt to curb incivility through moderation and automated detection, the performance of existing approaches often remains limited in both accuracy and efficiency. To address this challenge, we propose a Graph Neural Network (GNN) framework for detecting three types of uncivil behavior (i.e., toxicity, aggression, and personal attacks) within the English Wikipedia community. Our model represents each user comment as a node, with textual similarity between comments defining the edges, allowing the network to jointly learn from both linguistic content and relational structures among comments. We also introduce a dynamically adjusted attention mechanism that adaptively balances nodal and topological features during information aggregation. Empirical evaluations demonstrate that our proposed architecture outperforms 12 state-of-the-art Large Language Models (LLMs) across multiple metrics while requiring significantly lower inference cost. These findings highlight the crucial role of structural context in detecting online incivility and address the limitations of text-only LLM paradigms in behavioral prediction. All datasets and comparative outputs will be publicly available in our repository to support further research and reproducibility.



## **28. Think-Reflect-Revise: A Policy-Guided Reflective Framework for Safety Alignment in Large Vision Language Models**

cs.CV

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07141v1) [paper-pdf](https://arxiv.org/pdf/2512.07141v1)

**Authors**: Fenghua Weng, Chaochao Lu, Xia Hu, Wenqi Shao, Wenjie Wang

**Abstract**: As multimodal reasoning improves the overall capabilities of Large Vision Language Models (LVLMs), recent studies have begun to explore safety-oriented reasoning, aiming to enhance safety awareness by analyzing potential safety risks during the reasoning process before generating the final response. Although such approaches improve safety awareness and interpretability, this single-pass think-then-answer paradigm remains vulnerable to contextual or visual jailbreak attacks. This reveals a critical flaw: single-pass reasoning may overlook explicit harmful content in its own output. Our key insight is to exploit this wasted signal through reflection, which can effectively leverage the malicious content revealed in the first-pass reasoning to enable genuine self-correction and prevent unsafe generations. Motivated by this, we propose Think-Reflect-Revise (TRR), a three-stage training framework designed to enhance the safety alignment of LVLMs through policy-guided self-reflection. We first build a Reflective Safety Reasoning (ReSafe) dataset with 5,000 examples that follow a think-reflect-revise process. We then fine-tune the target model using the ReSafe dataset to initialize reflective behavior, and finally reinforce policy-guided reflection through reinforcement learning. Experimental results show that TRR substantially improves the safety performance of LVLMs across both safety-awareness benchmarks and jailbreak attack evaluations, increasing the overall safe response rate from 42.8% to 87.7% on Qwen2.5-VL-7B, while preserving stable performance on general benchmarks such as MMMU and MMStar. The project page is available at https://think-reflect-revise.github.io/.



## **29. ThinkTrap: Denial-of-Service Attacks against Black-box LLM Services via Infinite Thinking**

cs.CR

This version includes the final camera-ready manuscript accepted by NDSS 2026

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07086v1) [paper-pdf](https://arxiv.org/pdf/2512.07086v1)

**Authors**: Yunzhe Li, Jianan Wang, Hongzi Zhu, James Lin, Shan Chang, Minyi Guo

**Abstract**: Large Language Models (LLMs) have become foundational components in a wide range of applications, including natural language understanding and generation, embodied intelligence, and scientific discovery. As their computational requirements continue to grow, these models are increasingly deployed as cloud-based services, allowing users to access powerful LLMs via the Internet. However, this deployment model introduces a new class of threat: denial-of-service (DoS) attacks via unbounded reasoning, where adversaries craft specially designed inputs that cause the model to enter excessively long or infinite generation loops. These attacks can exhaust backend compute resources, degrading or denying service to legitimate users. To mitigate such risks, many LLM providers adopt a closed-source, black-box setting to obscure model internals. In this paper, we propose ThinkTrap, a novel input-space optimization framework for DoS attacks against LLM services even in black-box environments. The core idea of ThinkTrap is to first map discrete tokens into a continuous embedding space, then undertake efficient black-box optimization in a low-dimensional subspace exploiting input sparsity. The goal of this optimization is to identify adversarial prompts that induce extended or non-terminating generation across several state-of-the-art LLMs, achieving DoS with minimal token overhead. We evaluate the proposed attack across multiple commercial, closed-source LLM services. Our results demonstrate that, even far under the restrictive request frequency limits commonly enforced by these platforms, typically capped at ten requests per minute (10 RPM), the attack can degrade service throughput to as low as 1% of its original capacity, and in some cases, induce complete service failure.



## **30. Replicating TEMPEST at Scale: Multi-Turn Adversarial Attacks Against Trillion-Parameter Frontier Models**

cs.CL

30 pages, 11 figures, 5 tables. Code and data: https://github.com/ricyoung/tempest-replication

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07059v1) [paper-pdf](https://arxiv.org/pdf/2512.07059v1)

**Authors**: Richard Young

**Abstract**: Despite substantial investment in safety alignment, the vulnerability of large language models to sophisticated multi-turn adversarial attacks remains poorly characterized, and whether model scale or inference mode affects robustness is unknown. This study employed the TEMPEST multi-turn attack framework to evaluate ten frontier models from eight vendors across 1,000 harmful behaviors, generating over 97,000 API queries across adversarial conversations with automated evaluation by independent safety classifiers. Results demonstrated a spectrum of vulnerability: six models achieved 96% to 100% attack success rate (ASR), while four showed meaningful resistance, with ASR ranging from 42% to 78%; enabling extended reasoning on identical architecture reduced ASR from 97% to 42%. These findings indicate that safety alignment quality varies substantially across vendors, that model scale does not predict adversarial robustness, and that thinking mode provides a deployable safety enhancement. Collectively, this work establishes that current alignment techniques remain fundamentally vulnerable to adaptive multi-turn attacks regardless of model scale, while identifying deliberative inference as a promising defense direction.



## **31. SoK: Trust-Authorization Mismatch in LLM Agent Interactions**

cs.CR

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2512.06914v1) [paper-pdf](https://arxiv.org/pdf/2512.06914v1)

**Authors**: Guanquan Shi, Haohua Du, Zhiqiang Wang, Xiaoyu Liang, Weiwenpei Liu, Song Bian, Zhenyu Guan

**Abstract**: Large Language Models (LLMs) are rapidly evolving into autonomous agents capable of interacting with the external world, significantly expanding their capabilities through standardized interaction protocols. However, this paradigm revives the classic cybersecurity challenges of agency and authorization in a novel and volatile context. As decision-making shifts from deterministic code logic to probabilistic inference driven by natural language, traditional security mechanisms designed for deterministic behavior fail. It is fundamentally challenging to establish trust for unpredictable AI agents and to enforce the Principle of Least Privilege (PoLP) when instructions are ambiguous. Despite the escalating threat landscape, the academic community's understanding of this emerging domain remains fragmented, lacking a systematic framework to analyze its root causes. This paper provides a unifying formal lens for agent-interaction security.   We observed that most security threats in this domain stem from a fundamental mismatch between trust evaluation and authorization policies. We introduce a novel risk analysis model centered on this trust-authorization gap. Using this model as a unifying lens, we survey and classify the implementation paths of existing, often seemingly isolated, attacks and defenses. This new framework not only unifies the field but also allows us to identify critical research gaps. Finally, we leverage our analysis to suggest a systematic research direction toward building robust, trusted agents and dynamic authorization mechanisms.



## **32. From Description to Score: Can LLMs Quantify Vulnerabilities?**

cs.CR

10 pages

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2512.06781v1) [paper-pdf](https://arxiv.org/pdf/2512.06781v1)

**Authors**: Sima Jafarikhah, Daniel Thompson, Eva Deans, Hossein Siadati, Yi Liu

**Abstract**: Manual vulnerability scoring, such as assigning Common Vulnerability Scoring System (CVSS) scores, is a resource-intensive process that is often influenced by subjective interpretation. This study investigates the potential of general-purpose large language models (LLMs), namely ChatGPT, Llama, Grok, DeepSeek, and Gemini, to automate this process by analyzing over 31{,}000 recent Common Vulnerabilities and Exposures (CVE) entries. The results show that LLMs substantially outperform the baseline on certain metrics (e.g., \textit{Availability Impact}), while offering more modest gains on others (e.g., \textit{Attack Complexity}). Moreover, model performance varies across both LLM families and individual CVSS metrics, with ChatGPT-5 attaining the highest precision. Our analysis reveals that LLMs tend to misclassify many of the same CVEs, and ensemble-based meta-classifiers only marginally improve performance. Further examination shows that CVE descriptions often lack critical context or contain ambiguous phrasing, which contributes to systematic misclassifications. These findings underscore the importance of enhancing vulnerability descriptions and incorporating richer contextual details to support more reliable automated reasoning and alleviate the growing backlog of CVEs awaiting triage.



## **33. VRSA: Jailbreaking Multimodal Large Language Models through Visual Reasoning Sequential Attack**

cs.CV

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.05853v2) [paper-pdf](https://arxiv.org/pdf/2512.05853v2)

**Authors**: Shiji Zhao, Shukun Xiong, Yao Huang, Yan Jin, Zhenyu Wu, Jiyang Guan, Ranjie Duan, Jialing Tao, Hui Xue, Xingxing Wei

**Abstract**: Multimodal Large Language Models (MLLMs) are widely used in various fields due to their powerful cross-modal comprehension and generation capabilities. However, more modalities bring more vulnerabilities to being utilized for jailbreak attacks, which induces MLLMs to output harmful content. Due to the strong reasoning ability of MLLMs, previous jailbreak attacks try to explore reasoning safety risk in text modal, while similar threats have been largely overlooked in the visual modal. To fully evaluate potential safety risks in the visual reasoning task, we propose Visual Reasoning Sequential Attack (VRSA), which induces MLLMs to gradually externalize and aggregate complete harmful intent by decomposing the original harmful text into several sequentially related sub-images. In particular, to enhance the rationality of the scene in the image sequence, we propose Adaptive Scene Refinement to optimize the scene most relevant to the original harmful query. To ensure the semantic continuity of the generated image, we propose Semantic Coherent Completion to iteratively rewrite each sub-text combined with contextual information in this scene. In addition, we propose Text-Image Consistency Alignment to keep the semantical consistency. A series of experiments demonstrates that the VRSA can achieve a higher attack success rate compared with the state-of-the-art jailbreak attack methods on both the open-source and closed-source MLLMs such as GPT-4o and Claude-4.5-Sonnet.



## **34. TeleAI-Safety: A comprehensive LLM jailbreaking benchmark towards attacks, defenses, and evaluations**

cs.CR

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.05485v2) [paper-pdf](https://arxiv.org/pdf/2512.05485v2)

**Authors**: Xiuyuan Chen, Jian Zhao, Yuxiang He, Yuan Xun, Xinwei Liu, Yanshu Li, Huilin Zhou, Wei Cai, Ziyan Shi, Yuchen Yuan, Tianle Zhang, Chi Zhang, Xuelong Li

**Abstract**: While the deployment of large language models (LLMs) in high-value industries continues to expand, the systematic assessment of their safety against jailbreak and prompt-based attacks remains insufficient. Existing safety evaluation benchmarks and frameworks are often limited by an imbalanced integration of core components (attack, defense, and evaluation methods) and an isolation between flexible evaluation frameworks and standardized benchmarking capabilities. These limitations hinder reliable cross-study comparisons and create unnecessary overhead for comprehensive risk assessment. To address these gaps, we present TeleAI-Safety, a modular and reproducible framework coupled with a systematic benchmark for rigorous LLM safety evaluation. Our framework integrates a broad collection of 19 attack methods (including one self-developed method), 29 defense methods, and 19 evaluation methods (including one self-developed method). With a curated attack corpus of 342 samples spanning 12 distinct risk categories, the TeleAI-Safety benchmark conducts extensive evaluations across 14 target models. The results reveal systematic vulnerabilities and model-specific failure cases, highlighting critical trade-offs between safety and utility, and identifying potential defense patterns for future optimization. In practical scenarios, TeleAI-Safety can be flexibly adjusted with customized attack, defense, and evaluation combinations to meet specific demands. We release our complete code and evaluation results to facilitate reproducible research and establish unified safety baselines.



## **35. When Alignment Fails: Multimodal Adversarial Attacks on Vision-Language-Action Models**

cs.CV

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2511.16203v3) [paper-pdf](https://arxiv.org/pdf/2511.16203v3)

**Authors**: Yuping Yan, Yuhan Xie, Yixin Zhang, Lingjuan Lyu, Handing Wang, Yaochu Jin

**Abstract**: Vision-Language-Action models (VLAs) have recently demonstrated remarkable progress in embodied environments, enabling robots to perceive, reason, and act through unified multimodal understanding. Despite their impressive capabilities, the adversarial robustness of these systems remains largely unexplored, especially under realistic multimodal and black-box conditions. Existing studies mainly focus on single-modality perturbations and overlook the cross-modal misalignment that fundamentally affects embodied reasoning and decision-making. In this paper, we introduce VLA-Fool, a comprehensive study of multimodal adversarial robustness in embodied VLA models under both white-box and black-box settings. VLA-Fool unifies three levels of multimodal adversarial attacks: (1) textual perturbations through gradient-based and prompt-based manipulations, (2) visual perturbations via patch and noise distortions, and (3) cross-modal misalignment attacks that intentionally disrupt the semantic correspondence between perception and instruction. We further incorporate a VLA-aware semantic space into linguistic prompts, developing the first automatically crafted and semantically guided prompting framework. Experiments on the LIBERO benchmark using a fine-tuned OpenVLA model reveal that even minor multimodal perturbations can cause significant behavioral deviations, demonstrating the fragility of embodied multimodal alignment.



## **36. Tight and Practical Privacy Auditing for Differentially Private In-Context Learning**

cs.CR

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2511.13502v2) [paper-pdf](https://arxiv.org/pdf/2511.13502v2)

**Authors**: Yuyang Xia, Ruixuan Liu, Li Xiong

**Abstract**: Large language models (LLMs) perform in-context learning (ICL) by adapting to tasks from prompt demonstrations, which in practice often contain private or proprietary data. Although differential privacy (DP) with private voting is a pragmatic mitigation, DP-ICL implementations are error-prone, and worst-case DP bounds may substantially overestimate actual leakage, calling for practical auditing tools. We present a tight and efficient privacy auditing framework for DP-ICL systems that runs membership inference attacks and translates their success rates into empirical privacy guarantees using Gaussian DP. Our analysis of the private voting mechanism identifies vote configurations that maximize the auditing signal, guiding the design of audit queries that reliably reveal whether a canary demonstration is present in the context. The framework supports both black-box (API-only) and white-box (internal vote) threat models, and unifies auditing for classification and generation by reducing both to a binary decision problem. Experiments on standard text classification and generation benchmarks show that our empirical leakage estimates closely match theoretical DP budgets on classification tasks and are consistently lower on generation tasks due to conservative embedding-sensitivity bounds, making our framework a practical privacy auditor and verifier for real-world DP-ICL deployments.



## **37. Verifying LLM Inference to Detect Model Weight Exfiltration**

cs.CR

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2511.02620v2) [paper-pdf](https://arxiv.org/pdf/2511.02620v2)

**Authors**: Roy Rinberg, Adam Karvonen, Alexander Hoover, Daniel Reuter, Keri Warr

**Abstract**: As large AI models become increasingly valuable assets, the risk of model weight exfiltration from inference servers grows accordingly. An attacker controlling an inference server may exfiltrate model weights by hiding them within ordinary model outputs, a strategy known as steganography. This work investigates how to verify model responses to defend against such attacks and, more broadly, to detect anomalous or buggy behavior during inference. We formalize model exfiltration as a security game, propose a verification framework that can provably mitigate steganographic exfiltration, and specify the trust assumptions associated with our scheme. To enable verification, we characterize valid sources of non-determinism in large language model inference and introduce two practical estimators for them. We evaluate our detection framework on several open-weight models ranging from 3B to 30B parameters. On MOE-Qwen-30B, our detector reduces exfiltratable information to <0.5% with false-positive rate of 0.01%, corresponding to a >200x slowdown for adversaries. Overall, this work further establishes a foundation for defending against model weight exfiltration and demonstrates that strong protection can be achieved with minimal additional cost to inference providers.



## **38. BreakFun: Jailbreaking LLMs via Schema Exploitation**

cs.CR

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2510.17904v2) [paper-pdf](https://arxiv.org/pdf/2510.17904v2)

**Authors**: Amirkia Rafiei Oskooei, Mehmet S. Aktas

**Abstract**: The proficiency of Large Language Models (LLMs) in processing structured data and adhering to syntactic rules is a capability that drives their widespread adoption but also makes them paradoxically vulnerable. In this paper, we investigate this vulnerability through BreakFun, a jailbreak methodology that weaponizes an LLM's adherence to structured schemas. BreakFun employs a three-part prompt that combines an innocent framing and a Chain-of-Thought distraction with a core "Trojan Schema"--a carefully crafted data structure that compels the model to generate harmful content, exploiting the LLM's strong tendency to follow structures and schemas. We demonstrate this vulnerability is highly transferable, achieving an average success rate of 89% across 13 foundational and proprietary models on JailbreakBench, and reaching a 100% Attack Success Rate (ASR) on several prominent models. A rigorous ablation study confirms this Trojan Schema is the attack's primary causal factor. To counter this, we introduce the Adversarial Prompt Deconstruction guardrail, a defense that utilizes a secondary LLM to perform a "Literal Transcription"--extracting all human-readable text to isolate and reveal the user's true harmful intent. Our proof-of-concept guardrail demonstrates high efficacy against the attack, validating that targeting the deceptive schema is a viable mitigation strategy. Our work provides a look into how an LLM's core strengths can be turned into critical weaknesses, offering a fresh perspective for building more robustly aligned models.



## **39. PEAR: Planner-Executor Agent Robustness Benchmark**

cs.LG

arXiv admin note: This submission has been withdrawn by arXiv administrators due to incorrect authorship. Author list truncated

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2510.07505v3) [paper-pdf](https://arxiv.org/pdf/2510.07505v3)

**Authors**: Shen Dong, Mingxuan Zhang, Pengfei He, Li Ma, Bhavani Thuraisingham, Hui Liu, Yue Xing

**Abstract**: Large Language Model (LLM)-based Multi-Agent Systems (MAS) have emerged as a powerful paradigm for tackling complex, multi-step tasks across diverse domains. However, despite their impressive capabilities, MAS remain susceptible to adversarial manipulation. Existing studies typically examine isolated attack surfaces or specific scenarios, leaving a lack of holistic understanding of MAS vulnerabilities. To bridge this gap, we introduce PEAR, a benchmark for systematically evaluating both the utility and vulnerability of planner-executor MAS. While compatible with various MAS architectures, our benchmark focuses on the planner-executor structure, which is a practical and widely adopted design. Through extensive experiments, we find that (1) a weak planner degrades overall clean task performance more severely than a weak executor; (2) while a memory module is essential for the planner, having a memory module for the executor does not impact the clean task performance; (3) there exists a trade-off between task performance and robustness; and (4) attacks targeting the planner are particularly effective at misleading the system. These findings offer actionable insights for enhancing the robustness of MAS and lay the groundwork for principled defenses in multi-agent settings.



## **40. A Multi-Agent LLM Defense Pipeline Against Prompt Injection Attacks**

cs.CR

Accepted at the 11th IEEE WIECON-ECE 2025

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2509.14285v3) [paper-pdf](https://arxiv.org/pdf/2509.14285v3)

**Authors**: S M Asif Hossain, Ruksat Khan Shayoni, Mohd Ruhul Ameen, Akif Islam, M. F. Mridha, Jungpil Shin

**Abstract**: Prompt injection attacks represent a major vulnerability in Large Language Model (LLM) deployments, where malicious instructions embedded in user inputs can override system prompts and induce unintended behaviors. This paper presents a novel multi-agent defense framework that employs specialized LLM agents in coordinated pipelines to detect and neutralize prompt injection attacks in real-time. We evaluate our approach using two distinct architectures: a sequential chain-of-agents pipeline and a hierarchical coordinator-based system. Our comprehensive evaluation on 55 unique prompt injection attacks, grouped into 8 categories and totaling 400 attack instances across two LLM platforms (ChatGLM and Llama2), demonstrates significant security improvements. Without defense mechanisms, baseline Attack Success Rates (ASR) reached 30% for ChatGLM and 20% for Llama2. Our multi-agent pipeline achieved 100% mitigation, reducing ASR to 0% across all tested scenarios. The framework demonstrates robustness across multiple attack categories including direct overrides, code execution attempts, data exfiltration, and obfuscation techniques, while maintaining system functionality for legitimate queries.



## **41. Unveiling the Latent Directions of Reflection in Large Language Models**

cs.LG

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2508.16989v2) [paper-pdf](https://arxiv.org/pdf/2508.16989v2)

**Authors**: Fu-Chieh Chang, Yu-Ting Lee, Pei-Yuan Wu

**Abstract**: Reflection, the ability of large language models (LLMs) to evaluate and revise their own reasoning, has been widely used to improve performance on complex reasoning tasks. Yet, most prior works emphasizes designing reflective prompting strategies or reinforcement learning objectives, leaving the inner mechanisms of reflection underexplored. In this paper, we investigate reflection through the lens of latent directions in model activations. We propose a methodology based on activation steering to characterize how instructions with different reflective intentions: no reflection, intrinsic reflection, and triggered reflection. By constructing steering vectors between these reflection levels, we demonstrate that (1) new reflection-inducing instructions can be systematically identified, (2) reflective behavior can be directly enhanced or suppressed through activation interventions, and (3) suppressing reflection is considerably easier than stimulating it. Experiments on GSM8k-adv and Cruxeval-o-adv with Qwen2.5-3B and Gemma3-4B-IT reveal clear stratification across reflection levels, and steering interventions confirm the controllability of reflection. Our findings highlight both opportunities (e.g., reflection-enhancing defenses) and risks (e.g., adversarial inhibition of reflection in jailbreak attacks). This work opens a path toward mechanistic understanding of reflective reasoning in LLMs.



## **42. ConceptGuard: Neuro-Symbolic Safety Guardrails via Sparse Interpretable Jailbreak Concepts**

cs.CL

**SubmitDate**: 2025-12-13    [abs](http://arxiv.org/abs/2508.16325v2) [paper-pdf](https://arxiv.org/pdf/2508.16325v2)

**Authors**: Darpan Aswal, Céline Hudelot

**Abstract**: Large Language Models have found success in a variety of applications. However, their safety remains a concern due to the existence of various jailbreaking methods. Despite significant efforts, alignment and safety fine-tuning only provide a certain degree of robustness against jailbreak attacks that covertly mislead LLMs towards the generation of harmful content. This leaves them prone to a range of vulnerabilities, including targeted misuse and accidental user profiling. This work introduces \textbf{ConceptGuard}, a novel framework that leverages Sparse Autoencoders (SAEs) to identify interpretable concepts within LLM internals associated with different jailbreak themes. By extracting semantically meaningful internal representations, ConceptGuard enables building robust safety guardrails -- offering fully explainable and generalizable defenses without sacrificing model capabilities or requiring further fine-tuning. Leveraging advances in the mechanistic interpretability of LLMs, our approach provides evidence for a shared activation geometry for jailbreak attacks in the representation space, a potential foundation for designing more interpretable and generalizable safeguards against attackers.



## **43. Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference**

cs.CR

This paper is accepted by Network and Distributed System Security Symposium (NDSS) 2026

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2508.09442v3) [paper-pdf](https://arxiv.org/pdf/2508.09442v3)

**Authors**: Zhifan Luo, Shuo Shao, Su Zhang, Lijing Zhou, Yuke Hu, Chenxu Zhao, Zhihao Liu, Zhan Qin

**Abstract**: The Key-Value (KV) cache, which stores intermediate attention computations (Key and Value pairs) to avoid redundant calculations, is a fundamental mechanism for accelerating Large Language Model (LLM) inference. However, this efficiency optimization introduces significant yet underexplored privacy risks. This paper provides the first comprehensive analysis of these vulnerabilities, demonstrating that an attacker can reconstruct sensitive user inputs directly from the KV-cache. We design and implement three distinct attack vectors: a direct Inversion Attack, a more broadly applicable and potent Collision Attack, and a semantic-based Injection Attack. These methods demonstrate the practicality and severity of KV-cache privacy leakage issues. To mitigate this, we propose KV-Cloak, a novel, lightweight, and efficient defense mechanism. KV-Cloak uses a reversible matrix-based obfuscation scheme, combined with operator fusion, to secure the KV-cache. Our extensive experiments show that KV-Cloak effectively thwarts all proposed attacks, reducing reconstruction quality to random noise. Crucially, it achieves this robust security with virtually no degradation in model accuracy and minimal performance overhead, offering a practical solution for trustworthy LLM deployment.



## **44. From Prompt Injections to Protocol Exploits: Threats in LLM-Powered AI Agents Workflows**

cs.CR

The paper is published in ICT Express (Elsevier)

**SubmitDate**: 2025-12-14    [abs](http://arxiv.org/abs/2506.23260v2) [paper-pdf](https://arxiv.org/pdf/2506.23260v2)

**Authors**: Mohamed Amine Ferrag, Norbert Tihanyi, Djallel Hamouda, Leandros Maglaras, Abderrahmane Lakas, Merouane Debbah

**Abstract**: Autonomous AI agents powered by large language models (LLMs) with structured function-calling interfaces enable real-time data retrieval, computation, and multi-step orchestration. However, the rapid growth of plugins, connectors, and inter-agent protocols has outpaced security practices, leading to brittle integrations that rely on ad-hoc authentication, inconsistent schemas, and weak validation. This survey introduces a unified end-to-end threat model for LLM-agent ecosystems, covering host-to-tool and agent-to-agent communications. We systematically categorize more than thirty attack techniques spanning input manipulation, model compromise, system and privacy attacks, and protocol-level vulnerabilities. For each category, we provide a formal threat formulation defining attacker capabilities, objectives, and affected system layers. Representative examples include Prompt-to-SQL injections and the Toxic Agent Flow exploit in GitHub MCP servers. We analyze attack feasibility, review existing defenses, and discuss mitigation strategies such as dynamic trust management, cryptographic provenance tracking, and sandboxed agent interfaces. The framework is validated through expert review and cross-mapping with real-world incidents and public vulnerability repositories, including CVE and NIST NVD. Compared to prior surveys, this work presents the first integrated taxonomy bridging input-level exploits and protocol-layer vulnerabilities in LLM-agent ecosystems, offering actionable guidance for designing secure and resilient agentic AI systems.



## **45. OMNIGUARD: An Efficient Approach for AI Safety Moderation Across Languages and Modalities**

cs.CL

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2505.23856v2) [paper-pdf](https://arxiv.org/pdf/2505.23856v2)

**Authors**: Sahil Verma, Keegan Hines, Jeff Bilmes, Charlotte Siska, Luke Zettlemoyer, Hila Gonen, Chandan Singh

**Abstract**: The emerging capabilities of large language models (LLMs) have sparked concerns about their immediate potential for harmful misuse. The core approach to mitigate these concerns is the detection of harmful queries to the model. Current detection approaches are fallible, and are particularly susceptible to attacks that exploit mismatched generalization of model capabilities (e.g., prompts in low-resource languages or prompts provided in non-text modalities such as image and audio). To tackle this challenge, we propose Omniguard, an approach for detecting harmful prompts across languages and modalities. Our approach (i) identifies internal representations of an LLM/MLLM that are aligned across languages or modalities and then (ii) uses them to build a language-agnostic or modality-agnostic classifier for detecting harmful prompts. Omniguard improves harmful prompt classification accuracy by 11.57\% over the strongest baseline in a multilingual setting, by 20.44\% for image-based prompts, and sets a new SOTA for audio-based prompts. By repurposing embeddings computed during generation, Omniguard is also very efficient ($\approx\!120 \times$ faster than the next fastest baseline). Code and data are available at: https://github.com/vsahil/OmniGuard.



## **46. CachePrune: Neural-Based Attribution Defense Against Indirect Prompt Injection Attacks**

cs.CR

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2504.21228v2) [paper-pdf](https://arxiv.org/pdf/2504.21228v2)

**Authors**: Rui Wang, Junda Wu, Yu Xia, Tong Yu, Ruiyi Zhang, Ryan Rossi, Subrata Mitra, Lina Yao, Julian McAuley

**Abstract**: Large Language Models (LLMs) are susceptible to indirect prompt injection attacks, in which the model inadvertently responds to task messages injected within the prompt context. This vulnerability stems from LLMs' inability to distinguish between data and instructions within a prompt. In this paper, we propose CachePrune, a defense method that identifies and prunes task-triggering neurons from the KV cache of the input prompt context. By pruning such neurons, we encourage the LLM to interpret the input prompt context purely as data rather than as cues for instruction following. To identify these neurons, we introduce a neural attribution mechanism guided by a preferential attribution loss, which enables effective attribution with only a few samples while preserving response quality after pruning. We further enhance the efficacy of neural attribution by leveraging an observed triggering effect inherent in the model's response generation behavior. Notably, our approach does not impose additional formatting on the prompt or introduce extra test-time LLM calls. Experiments show that CachePrune can significantly reduce attack success rates while maintaining clean response quality.



## **47. Memory Injection Attacks on LLM Agents via Query-Only Interaction**

cs.LG

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2503.03704v4) [paper-pdf](https://arxiv.org/pdf/2503.03704v4)

**Authors**: Shen Dong, Shaochen Xu, Pengfei He, Yige Li, Jiliang Tang, Tianming Liu, Hui Liu, Zhen Xiang

**Abstract**: Agents powered by large language models (LLMs) have demonstrated strong capabilities in a wide range of complex, real-world applications. However, LLM agents with a compromised memory bank may easily produce harmful outputs when the past records retrieved for demonstration are malicious. In this paper, we propose a novel Memory INJection Attack, MINJA, without assuming that the attacker can directly modify the memory bank of the agent. The attacker injects malicious records into the memory bank by only interacting with the agent via queries and output observations. These malicious records are designed to elicit a sequence of malicious reasoning steps corresponding to a different target query during the agent's execution of the victim user's query. Specifically, we introduce a sequence of bridging steps to link victim queries to the malicious reasoning steps. During the memory injection, we propose an indication prompt that guides the agent to autonomously generate similar bridging steps, with a progressive shortening strategy that gradually removes the indication prompt, such that the malicious record will be easily retrieved when processing later victim queries. Our extensive experiments across diverse agents demonstrate the effectiveness of MINJA in compromising agent memory. With minimal requirements for execution, MINJA enables any user to influence agent memory, highlighting the risk.



## **48. Toward Intelligent and Secure Cloud: Large Language Model Empowered Proactive Defense**

cs.CR

7 pages; Accepted by IEEE Communications Magazine

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2412.21051v4) [paper-pdf](https://arxiv.org/pdf/2412.21051v4)

**Authors**: Yuyang Zhou, Guang Cheng, Kang Du, Zihan Chen, Yuyu Zhao

**Abstract**: The rapid evolution of cloud computing technologies and the increasing number of cloud applications have provided numerous benefits in our daily lives. However, the diversity and complexity of different components pose a significant challenge to cloud security, especially when dealing with sophisticated and advanced cyberattacks such as Denial of Service (DoS). Recent advancements in the large language models (LLMs) offer promising solutions for security intelligence. By exploiting the powerful capabilities in language understanding, data analysis, task inference, action planning, and code generation, we present LLM-PD, a novel defense architecture that proactively mitigates various DoS threats in cloud networks. LLM-PD can efficiently make decisions through comprehensive data analysis and sequential reasoning, as well as dynamically create and deploy actionable defense mechanisms. Furthermore, it can flexibly self-evolve based on experience learned from previous interactions and adapt to new attack scenarios without additional training. Our case study on three distinct DoS attacks demonstrates its remarkable ability in terms of defense effectiveness and efficiency when compared with other existing methods.



## **49. A Fingerprint for Large Language Models**

cs.CR

Updated by Hanzhou Wu, 8 pages

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2407.01235v2) [paper-pdf](https://arxiv.org/pdf/2407.01235v2)

**Authors**: Zhiguang Yang, Hanzhou Wu

**Abstract**: Recent advances confirm that large language models (LLMs) can achieve state-of-the-art performance across various tasks. However, due to the resource-intensive nature of training LLMs from scratch, it is urgent and crucial to protect the intellectual property of LLMs against infringement. This has motivated the authors in this paper to propose a novel black-box fingerprinting technique for LLMs. We firstly demonstrate that the outputs of LLMs span a unique vector space associated with each model. We model the problem of fingerprint authentication as the task of evaluating the similarity between the space of the victim model and the space of the suspect model. To tackle with this problem, we introduce two solutions: the first determines whether suspect outputs lie within the victim's subspace, enabling fast infringement detection; the second reconstructs a joint subspace to detect models modified via parameter-efficient fine-tuning (PEFT). Experiments indicate that the proposed method achieves superior performance in fingerprint verification and robustness against the PEFT attacks. This work reveals inherent characteristics of LLMs and provides a promising solution for protecting LLMs, ensuring efficiency, generality and practicality.



## **50. Gradient-Free Privacy Leakage in Federated Language Models through Selective Weight Tampering**

cs.CR

21 pages (including bibliography and Appendix), Submitted to PETS'26

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2310.16152v4) [paper-pdf](https://arxiv.org/pdf/2310.16152v4)

**Authors**: Md Rafi Ur Rashid, Vishnu Asutosh Dasu, Kang Gu, Najrin Sultana, Shagufta Mehnaz

**Abstract**: Federated learning (FL) has become a key component in various language modeling applications such as machine translation, next-word prediction, and medical record analysis. These applications are trained on datasets from many FL participants that often include privacy-sensitive data, such as healthcare records, phone/credit card numbers, login credentials, etc. Although FL enables computation without necessitating clients to share their raw data, existing works show that privacy leakage is still probable in federated language models. In this paper, we present two novel findings on the leakage of privacy-sensitive user data from federated large language models without requiring access to gradients. Firstly, we make a key observation that model snapshots from the intermediate rounds in FL can cause greater privacy leakage than the final trained model. Secondly, we identify that a malicious FL participant can aggravate the leakage by tampering with the model's selective weights that are responsible for memorizing the sensitive training data of some other clients, even without any cooperation from the server. Our best-performing method increases the membership inference recall by 29% and achieves up to 71% private data reconstruction, evidently outperforming existing attacks that consider much stronger adversary capabilities. Lastly, we recommend a balanced suite of techniques for an FL client to defend against such privacy risk.



