# Latest Large Language Model Attack Papers
**update at 2025-07-21 10:22:04**

[中英双语版本](https://github.com/daksim/NewAdversarialAttackPaper/blob/main/README_LLM_CN.md)

## **1. From Words to Collisions: LLM-Guided Evaluation and Adversarial Generation of Safety-Critical Driving Scenarios**

cs.AI

Final Version and Paper Accepted at IEEE ITSC 2025

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2502.02145v4) [paper-pdf](http://arxiv.org/pdf/2502.02145v4)

**Authors**: Yuan Gao, Mattia Piccinini, Korbinian Moller, Amr Alanwar, Johannes Betz

**Abstract**: Ensuring the safety of autonomous vehicles requires virtual scenario-based testing, which depends on the robust evaluation and generation of safety-critical scenarios. So far, researchers have used scenario-based testing frameworks that rely heavily on handcrafted scenarios as safety metrics. To reduce the effort of human interpretation and overcome the limited scalability of these approaches, we combine Large Language Models (LLMs) with structured scenario parsing and prompt engineering to automatically evaluate and generate safety-critical driving scenarios. We introduce Cartesian and Ego-centric prompt strategies for scenario evaluation, and an adversarial generation module that modifies trajectories of risk-inducing vehicles (ego-attackers) to create critical scenarios. We validate our approach using a 2D simulation framework and multiple pre-trained LLMs. The results show that the evaluation module effectively detects collision scenarios and infers scenario safety. Meanwhile, the new generation module identifies high-risk agents and synthesizes realistic, safety-critical scenarios. We conclude that an LLM equipped with domain-informed prompting techniques can effectively evaluate and generate safety-critical driving scenarios, reducing dependence on handcrafted metrics. We release our open-source code and scenarios at: https://github.com/TUM-AVS/From-Words-to-Collisions.



## **2. TopicAttack: An Indirect Prompt Injection Attack via Topic Transition**

cs.CR

19 pages

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2507.13686v1) [paper-pdf](http://arxiv.org/pdf/2507.13686v1)

**Authors**: Yulin Chen, Haoran Li, Yuexin Li, Yue Liu, Yangqiu Song, Bryan Hooi

**Abstract**: Large language models (LLMs) have shown remarkable performance across a range of NLP tasks. However, their strong instruction-following capabilities and inability to distinguish instructions from data content make them vulnerable to indirect prompt injection attacks. In such attacks, instructions with malicious purposes are injected into external data sources, such as web documents. When LLMs retrieve this injected data through tools, such as a search engine and execute the injected instructions, they provide misled responses. Recent attack methods have demonstrated potential, but their abrupt instruction injection often undermines their effectiveness. Motivated by the limitations of existing attack methods, we propose TopicAttack, which prompts the LLM to generate a fabricated conversational transition prompt that gradually shifts the topic toward the injected instruction, making the injection smoother and enhancing the plausibility and success of the attack. Through comprehensive experiments, TopicAttack achieves state-of-the-art performance, with an attack success rate (ASR) over 90\% in most cases, even when various defense methods are applied. We further analyze its effectiveness by examining attention scores. We find that a higher injected-to-original attention ratio leads to a greater success probability, and our method achieves a much higher ratio than the baseline methods.



## **3. Can Indirect Prompt Injection Attacks Be Detected and Removed?**

cs.CR

To Appear in ACL 2025

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2502.16580v2) [paper-pdf](http://arxiv.org/pdf/2502.16580v2)

**Authors**: Yulin Chen, Haoran Li, Yuan Sui, Yufei He, Yue Liu, Yangqiu Song, Bryan Hooi

**Abstract**: Prompt injection attacks manipulate large language models (LLMs) by misleading them to deviate from the original input instructions and execute maliciously injected instructions, because of their instruction-following capabilities and inability to distinguish between the original input instructions and maliciously injected instructions. To defend against such attacks, recent studies have developed various detection mechanisms. If we restrict ourselves specifically to works which perform detection rather than direct defense, most of them focus on direct prompt injection attacks, while there are few works for the indirect scenario, where injected instructions are indirectly from external tools, such as a search engine. Moreover, current works mainly investigate injection detection methods and pay less attention to the post-processing method that aims to mitigate the injection after detection. In this paper, we investigate the feasibility of detecting and removing indirect prompt injection attacks, and we construct a benchmark dataset for evaluation. For detection, we assess the performance of existing LLMs and open-source detection models, and we further train detection models using our crafted training datasets. For removal, we evaluate two intuitive methods: (1) the segmentation removal method, which segments the injected document and removes parts containing injected instructions, and (2) the extraction removal method, which trains an extraction model to identify and remove injected instructions.



## **4. Defense Against Prompt Injection Attack by Leveraging Attack Techniques**

cs.CR

To Appear in ACL 2025

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2411.00459v4) [paper-pdf](http://arxiv.org/pdf/2411.00459v4)

**Authors**: Yulin Chen, Haoran Li, Zihao Zheng, Yangqiu Song, Dekai Wu, Bryan Hooi

**Abstract**: With the advancement of technology, large language models (LLMs) have achieved remarkable performance across various natural language processing (NLP) tasks, powering LLM-integrated applications like Microsoft Copilot. However, as LLMs continue to evolve, new vulnerabilities, especially prompt injection attacks arise. These attacks trick LLMs into deviating from the original input instructions and executing the attacker's instructions injected in data content, such as retrieved results. Recent attack methods leverage LLMs' instruction-following abilities and their inabilities to distinguish instructions injected in the data content, and achieve a high attack success rate (ASR). When comparing the attack and defense methods, we interestingly find that they share similar design goals, of inducing the model to ignore unwanted instructions and instead to execute wanted instructions. Therefore, we raise an intuitive question: Could these attack techniques be utilized for defensive purposes? In this paper, we invert the intention of prompt injection methods to develop novel defense methods based on previous training-free attack methods, by repeating the attack process but with the original input instruction rather than the injected instruction. Our comprehensive experiments demonstrate that our defense techniques outperform existing training-free defense approaches, achieving state-of-the-art results.



## **5. Invisible Textual Backdoor Attacks based on Dual-Trigger**

cs.CR

**SubmitDate**: 2025-07-18    [abs](http://arxiv.org/abs/2412.17531v3) [paper-pdf](http://arxiv.org/pdf/2412.17531v3)

**Authors**: Yang Hou, Qiuling Yue, Lujia Chai, Guozhao Liao, Wenbao Han, Wei Ou

**Abstract**: Backdoor attacks pose an important security threat to textual large language models. Exploring textual backdoor attacks not only helps reveal the potential security risks of models, but also promotes innovation and development of defense mechanisms. Currently, most textual backdoor attack methods are based on a single trigger. For example, inserting specific content into text as a trigger or changing the abstract text features to be a trigger. However, the adoption of this single-trigger mode makes the existing backdoor attacks subject to certain limitations: either they are easily identified by the existing defense strategies, or they have certain shortcomings in attack performance and in the construction of poisoned datasets. In order to solve these issues, a dual-trigger backdoor attack method is proposed in this paper. Specifically, we use two different attributes, syntax and mood (we use subjunctive mood as an example in this article), as two different triggers. It makes our backdoor attack method similar to a double landmine which can have completely different trigger conditions simultaneously. Therefore, this method not only improves the flexibility of trigger mode, but also enhances the robustness against defense detection. A large number of experimental results show that this method significantly outperforms the previous methods based on abstract features in attack performance, and achieves comparable attack performance (almost 100\% attack success rate) with the insertion-based method. In addition, in order to further improve the attack performance, we also give the construction method of the poisoned dataset.The code and data of this paper can be obtained at https://github.com/HoyaAm/Double-Landmines.



## **6. Paper Summary Attack: Jailbreaking LLMs through LLM Safety Papers**

cs.CL

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2507.13474v1) [paper-pdf](http://arxiv.org/pdf/2507.13474v1)

**Authors**: Liang Lin, Zhihao Xu, Xuehai Tang, Shi Liu, Biyu Zhou, Fuqing Zhu, Jizhong Han, Songlin Hu

**Abstract**: The safety of large language models (LLMs) has garnered significant research attention. In this paper, we argue that previous empirical studies demonstrate LLMs exhibit a propensity to trust information from authoritative sources, such as academic papers, implying new possible vulnerabilities. To verify this possibility, a preliminary analysis is designed to illustrate our two findings. Based on this insight, a novel jailbreaking method, Paper Summary Attack (\llmname{PSA}), is proposed. It systematically synthesizes content from either attack-focused or defense-focused LLM safety paper to construct an adversarial prompt template, while strategically infilling harmful query as adversarial payloads within predefined subsections. Extensive experiments show significant vulnerabilities not only in base LLMs, but also in state-of-the-art reasoning model like Deepseek-R1. PSA achieves a 97\% attack success rate (ASR) on well-aligned models like Claude3.5-Sonnet and an even higher 98\% ASR on Deepseek-R1. More intriguingly, our work has further revealed diametrically opposed vulnerability bias across different base models, and even between different versions of the same model, when exposed to either attack-focused or defense-focused papers. This phenomenon potentially indicates future research clues for both adversarial methodologies and safety alignment.Code is available at https://github.com/233liang/Paper-Summary-Attack



## **7. Automating Steering for Safe Multimodal Large Language Models**

cs.CL

Working in progress. 22 pages (8+ for main); 25 figures; 1 table

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2507.13255v1) [paper-pdf](http://arxiv.org/pdf/2507.13255v1)

**Authors**: Lyucheng Wu, Mengru Wang, Ziwen Xu, Tri Cao, Nay Oo, Bryan Hooi, Shumin Deng

**Abstract**: Recent progress in Multimodal Large Language Models (MLLMs) has unlocked powerful cross-modal reasoning abilities, but also raised new safety concerns, particularly when faced with adversarial multimodal inputs. To improve the safety of MLLMs during inference, we introduce a modular and adaptive inference-time intervention technology, AutoSteer, without requiring any fine-tuning of the underlying model. AutoSteer incorporates three core components: (1) a novel Safety Awareness Score (SAS) that automatically identifies the most safety-relevant distinctions among the model's internal layers; (2) an adaptive safety prober trained to estimate the likelihood of toxic outputs from intermediate representations; and (3) a lightweight Refusal Head that selectively intervenes to modulate generation when safety risks are detected. Experiments on LLaVA-OV and Chameleon across diverse safety-critical benchmarks demonstrate that AutoSteer significantly reduces the Attack Success Rate (ASR) for textual, visual, and cross-modal threats, while maintaining general abilities. These findings position AutoSteer as a practical, interpretable, and effective framework for safer deployment of multimodal AI systems.



## **8. Detecting LLM-generated Code with Subtle Modification by Adversarial Training**

cs.SE

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2507.13123v1) [paper-pdf](http://arxiv.org/pdf/2507.13123v1)

**Authors**: Xin Yin, Xinrui Li, Chao Ni, Xiaodan Xu, Xiaohu Yang

**Abstract**: With the rapid development of Large Language Models (LLMs), their powerful code-generation capabilities have been widely applied in tasks like code completion and automated development, demonstrating the value of improving coding efficiency. However, the extensive use of LLM-generated code also raises several new challenges. On the one hand, issues such as the regulation of code provenance, copyright disputes, and code quality have become increasingly concerning. How to effectively detect LLM-generated code and ensure its compliant and responsible use has become a critical and urgent issue. On the other hand, in practical applications, LLM-generated code is often subject to manual modifications, such as variable renaming or structural adjustments. Although some recent studies have proposed training-based and zero-shot methods for detecting LLM-generated code, these approaches show insufficient robustness when facing modified LLM-generated code, and there is a lack of an effective solution. To address the real-world scenario where LLM-generated code may undergo minor modifications, we propose CodeGPTSensor+, an enhanced version of CodeGPTSensor, which employs adversarial training to improve robustness against input perturbations. CodeGPTSensor+ integrates an adversarial sample generation module, Multi-objective Identifier and Structure Transformation (MIST), which systematically generates both high-quality and representative adversarial samples. This module effectively enhances the model's resistance against diverse adversarial attacks. Experimental results on the HMCorp dataset demonstrate that CodeGPTSensor+ significantly improves detection accuracy on the adversarial test set while maintaining high accuracy on the original test set, showcasing superior robustness compared to CodeGPTSensor.



## **9. MAD-Spear: A Conformity-Driven Prompt Injection Attack on Multi-Agent Debate Systems**

cs.CR

**SubmitDate**: 2025-07-17    [abs](http://arxiv.org/abs/2507.13038v1) [paper-pdf](http://arxiv.org/pdf/2507.13038v1)

**Authors**: Yu Cui, Hongyang Du

**Abstract**: Multi-agent debate (MAD) systems leverage collaborative interactions among large language models (LLMs) agents to improve reasoning capabilities. While recent studies have focused on increasing the accuracy and scalability of MAD systems, their security vulnerabilities have received limited attention. In this work, we introduce MAD-Spear, a targeted prompt injection attack that compromises a small subset of agents but significantly disrupts the overall MAD process. Manipulated agents produce multiple plausible yet incorrect responses, exploiting LLMs' conformity tendencies to propagate misinformation and degrade consensus quality. Furthermore, the attack can be composed with other strategies, such as communication attacks, to further amplify its impact by increasing the exposure of agents to incorrect responses. To assess MAD's resilience under attack, we propose a formal definition of MAD fault-tolerance and develop a comprehensive evaluation framework that jointly considers accuracy, consensus efficiency, and scalability. Extensive experiments on five benchmark datasets with varying difficulty levels demonstrate that MAD-Spear consistently outperforms the baseline attack in degrading system performance. Additionally, we observe that agent diversity substantially improves MAD performance in mathematical reasoning tasks, which challenges prior work suggesting that agent diversity has minimal impact on performance. These findings highlight the urgent need to improve the security in MAD design.



## **10. JailDAM: Jailbreak Detection with Adaptive Memory for Vision-Language Model**

cs.CR

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2504.03770v3) [paper-pdf](http://arxiv.org/pdf/2504.03770v3)

**Authors**: Yi Nian, Shenzhe Zhu, Yuehan Qin, Li Li, Ziyi Wang, Chaowei Xiao, Yue Zhao

**Abstract**: Multimodal large language models (MLLMs) excel in vision-language tasks but also pose significant risks of generating harmful content, particularly through jailbreak attacks. Jailbreak attacks refer to intentional manipulations that bypass safety mechanisms in models, leading to the generation of inappropriate or unsafe content. Detecting such attacks is critical to ensuring the responsible deployment of MLLMs. Existing jailbreak detection methods face three primary challenges: (1) Many rely on model hidden states or gradients, limiting their applicability to white-box models, where the internal workings of the model are accessible; (2) They involve high computational overhead from uncertainty-based analysis, which limits real-time detection, and (3) They require fully labeled harmful datasets, which are often scarce in real-world settings. To address these issues, we introduce a test-time adaptive framework called JAILDAM. Our method leverages a memory-based approach guided by policy-driven unsafe knowledge representations, eliminating the need for explicit exposure to harmful data. By dynamically updating unsafe knowledge during test-time, our framework improves generalization to unseen jailbreak strategies while maintaining efficiency. Experiments on multiple VLM jailbreak benchmarks demonstrate that JAILDAM delivers state-of-the-art performance in harmful content detection, improving both accuracy and speed.



## **11. SoK: Semantic Privacy in Large Language Models**

cs.CR

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2506.23603v2) [paper-pdf](http://arxiv.org/pdf/2506.23603v2)

**Authors**: Baihe Ma, Yanna Jiang, Xu Wang, Guangsheng Yu, Qin Wang, Caijun Sun, Chen Li, Xuelei Qi, Ying He, Wei Ni, Ren Ping Liu

**Abstract**: As Large Language Models (LLMs) are increasingly deployed in sensitive domains, traditional data privacy measures prove inadequate for protecting information that is implicit, contextual, or inferable - what we define as semantic privacy. This Systematization of Knowledge (SoK) introduces a lifecycle-centric framework to analyze how semantic privacy risks emerge across input processing, pretraining, fine-tuning, and alignment stages of LLMs. We categorize key attack vectors and assess how current defenses, such as differential privacy, embedding encryption, edge computing, and unlearning, address these threats. Our analysis reveals critical gaps in semantic-level protection, especially against contextual inference and latent representation leakage. We conclude by outlining open challenges, including quantifying semantic leakage, protecting multimodal inputs, balancing de-identification with generation quality, and ensuring transparency in privacy enforcement. This work aims to inform future research on designing robust, semantically aware privacy-preserving techniques for LLMs.



## **12. Thought Purity: Defense Paradigm For Chain-of-Thought Attack**

cs.LG

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2507.12314v1) [paper-pdf](http://arxiv.org/pdf/2507.12314v1)

**Authors**: Zihao Xue, Zhen Bi, Long Ma, Zhenlin Hu, Yan Wang, Zhenfang Liu, Qing Sheng, Jie Xiao, Jungang Lou

**Abstract**: While reinforcement learning-trained Large Reasoning Models (LRMs, e.g., Deepseek-R1) demonstrate advanced reasoning capabilities in the evolving Large Language Models (LLMs) domain, their susceptibility to security threats remains a critical vulnerability. This weakness is particularly evident in Chain-of-Thought (CoT) generation processes, where adversarial methods like backdoor prompt attacks can systematically subvert the model's core reasoning mechanisms. The emerging Chain-of-Thought Attack (CoTA) reveals this vulnerability through exploiting prompt controllability, simultaneously degrading both CoT safety and task performance with low-cost interventions. To address this compounded security-performance vulnerability, we propose Thought Purity (TP): a defense paradigm that systematically strengthens resistance to malicious content while preserving operational efficacy. Our solution achieves this through three synergistic components: (1) a safety-optimized data processing pipeline (2) reinforcement learning-enhanced rule constraints (3) adaptive monitoring metrics. Our approach establishes the first comprehensive defense mechanism against CoTA vulnerabilities in reinforcement learning-aligned reasoning systems, significantly advancing the security-functionality equilibrium for next-generation AI architectures.



## **13. Watch, Listen, Understand, Mislead: Tri-modal Adversarial Attacks on Short Videos for Content Appropriateness Evaluation**

cs.CV

Accepted as long paper, SVU Workshop at ICCV 2025

**SubmitDate**: 2025-07-16    [abs](http://arxiv.org/abs/2507.11968v1) [paper-pdf](http://arxiv.org/pdf/2507.11968v1)

**Authors**: Sahid Hossain Mustakim, S M Jishanul Islam, Ummay Maria Muna, Montasir Chowdhury, Mohammed Jawwadul Islam, Sadia Ahmmed, Tashfia Sikder, Syed Tasdid Azam Dhrubo, Swakkhar Shatabda

**Abstract**: Multimodal Large Language Models (MLLMs) are increasingly used for content moderation, yet their robustness in short-form video contexts remains underexplored. Current safety evaluations often rely on unimodal attacks, failing to address combined attack vulnerabilities. In this paper, we introduce a comprehensive framework for evaluating the tri-modal safety of MLLMs. First, we present the Short-Video Multimodal Adversarial (SVMA) dataset, comprising diverse short-form videos with human-guided synthetic adversarial attacks. Second, we propose ChimeraBreak, a novel tri-modal attack strategy that simultaneously challenges visual, auditory, and semantic reasoning pathways. Extensive experiments on state-of-the-art MLLMs reveal significant vulnerabilities with high Attack Success Rates (ASR). Our findings uncover distinct failure modes, showing model biases toward misclassifying benign or policy-violating content. We assess results using LLM-as-a-judge, demonstrating attack reasoning efficacy. Our dataset and findings provide crucial insights for developing more robust and safe MLLMs.



## **14. Seven Security Challenges That Must be Solved in Cross-domain Multi-agent LLM Systems**

cs.CR

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2505.23847v3) [paper-pdf](http://arxiv.org/pdf/2505.23847v3)

**Authors**: Ronny Ko, Jiseong Jeong, Shuyuan Zheng, Chuan Xiao, Tae-Wan Kim, Makoto Onizuka, Won-Yong Shin

**Abstract**: Large language models (LLMs) are rapidly evolving into autonomous agents that cooperate across organizational boundaries, enabling joint disaster response, supply-chain optimization, and other tasks that demand decentralized expertise without surrendering data ownership. Yet, cross-domain collaboration shatters the unified trust assumptions behind current alignment and containment techniques. An agent benign in isolation may, when receiving messages from an untrusted peer, leak secrets or violate policy, producing risks driven by emergent multi-agent dynamics rather than classical software bugs. This position paper maps the security agenda for cross-domain multi-agent LLM systems. We introduce seven categories of novel security challenges, for each of which we also present plausible attacks, security evaluation metrics, and future research guidelines.



## **15. A Generative Approach to LLM Harmfulness Detection with Special Red Flag Tokens**

cs.CL

14 pages, 6 figures

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2502.16366v3) [paper-pdf](http://arxiv.org/pdf/2502.16366v3)

**Authors**: Sophie Xhonneux, David Dobre, Mehrnaz Mofakhami, Leo Schwinn, Gauthier Gidel

**Abstract**: Most safety training methods for large language models (LLMs) are based on fine-tuning that forces models to shift from an unsafe answer to refusal when faced with harmful requests. Unfortunately, these drastic distribution shifts generally compromise model capabilities. To avoid that, we propose to expand the model's vocabulary with a special token we call red flag token (<rf>) and propose to train the model to insert this token into its response at any time when harmful content is generated or about to be generated. Our approach offers several advantages: it enables the model to explicitly learn the concept of harmfulness while marginally affecting the generated distribution, thus maintaining the model's utility. It also evaluates each generated answer and provides robustness as good as adversarial training without the need to run attacks during training. Moreover, by encapsulating our safety tuning in a LoRA module, we provide additional defenses against fine-tuning API attacks.



## **16. GUARD:Dual-Agent based Backdoor Defense on Chain-of-Thought in Neural Code Generation**

cs.SE

Accepted by SEKE 2025

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2505.21425v2) [paper-pdf](http://arxiv.org/pdf/2505.21425v2)

**Authors**: Naizhu Jin, Zhong Li, Tian Zhang, Qingkai Zeng

**Abstract**: With the widespread application of large language models in code generation, recent studies demonstrate that employing additional Chain-of-Thought generation models can significantly enhance code generation performance by providing explicit reasoning steps. However, as external components, CoT models are particularly vulnerable to backdoor attacks, which existing defense mechanisms often fail to detect effectively. To address this challenge, we propose GUARD, a novel dual-agent defense framework specifically designed to counter CoT backdoor attacks in neural code generation. GUARD integrates two core components: GUARD-Judge, which identifies suspicious CoT steps and potential triggers through comprehensive analysis, and GUARD-Repair, which employs a retrieval-augmented generation approach to regenerate secure CoT steps for identified anomalies. Experimental results show that GUARD effectively mitigates attacks while maintaining generation quality, advancing secure code generation systems.



## **17. Multi-Trigger Poisoning Amplifies Backdoor Vulnerabilities in LLMs**

cs.CL

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2507.11112v1) [paper-pdf](http://arxiv.org/pdf/2507.11112v1)

**Authors**: Sanhanat Sivapiromrat, Caiqi Zhang, Marco Basaldella, Nigel Collier

**Abstract**: Recent studies have shown that Large Language Models (LLMs) are vulnerable to data poisoning attacks, where malicious training examples embed hidden behaviours triggered by specific input patterns. However, most existing works assume a phrase and focus on the attack's effectiveness, offering limited understanding of trigger mechanisms and how multiple triggers interact within the model. In this paper, we present a framework for studying poisoning in LLMs. We show that multiple distinct backdoor triggers can coexist within a single model without interfering with each other, enabling adversaries to embed several triggers concurrently. Using multiple triggers with high embedding similarity, we demonstrate that poisoned triggers can achieve robust activation even when tokens are substituted or separated by long token spans. Our findings expose a broader and more persistent vulnerability surface in LLMs. To mitigate this threat, we propose a post hoc recovery method that selectively retrains specific model components based on a layer-wise weight difference analysis. Our method effectively removes the trigger behaviour with minimal parameter updates, presenting a practical and efficient defence against multi-trigger poisoning.



## **18. The Devil behind the mask: An emergent safety vulnerability of Diffusion LLMs**

cs.CL

21 pages, 9 figures, work in progress

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2507.11097v1) [paper-pdf](http://arxiv.org/pdf/2507.11097v1)

**Authors**: Zichen Wen, Jiashu Qu, Dongrui Liu, Zhiyuan Liu, Ruixi Wu, Yicun Yang, Xiangqi Jin, Haoyun Xu, Xuyang Liu, Weijia Li, Chaochao Lu, Jing Shao, Conghui He, Linfeng Zhang

**Abstract**: Diffusion-based large language models (dLLMs) have recently emerged as a powerful alternative to autoregressive LLMs, offering faster inference and greater interactivity via parallel decoding and bidirectional modeling. However, despite strong performance in code generation and text infilling, we identify a fundamental safety concern: existing alignment mechanisms fail to safeguard dLLMs against context-aware, masked-input adversarial prompts, exposing novel vulnerabilities. To this end, we present DIJA, the first systematic study and jailbreak attack framework that exploits unique safety weaknesses of dLLMs. Specifically, our proposed DIJA constructs adversarial interleaved mask-text prompts that exploit the text generation mechanisms of dLLMs, i.e., bidirectional modeling and parallel decoding. Bidirectional modeling drives the model to produce contextually consistent outputs for masked spans, even when harmful, while parallel decoding limits model dynamic filtering and rejection sampling of unsafe content. This causes standard alignment mechanisms to fail, enabling harmful completions in alignment-tuned dLLMs, even when harmful behaviors or unsafe instructions are directly exposed in the prompt. Through comprehensive experiments, we demonstrate that DIJA significantly outperforms existing jailbreak methods, exposing a previously overlooked threat surface in dLLM architectures. Notably, our method achieves up to 100% keyword-based ASR on Dream-Instruct, surpassing the strongest prior baseline, ReNeLLM, by up to 78.5% in evaluator-based ASR on JailbreakBench and by 37.7 points in StrongREJECT score, while requiring no rewriting or hiding of harmful content in the jailbreak prompt. Our findings underscore the urgent need for rethinking safety alignment in this emerging class of language models. Code is available at https://github.com/ZichenWen1/DIJA.



## **19. Representation Bending for Large Language Model Safety**

cs.LG

Accepted to ACL 2025 (main)

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2504.01550v3) [paper-pdf](http://arxiv.org/pdf/2504.01550v3)

**Authors**: Ashkan Yousefpour, Taeheon Kim, Ryan S. Kwon, Seungbeen Lee, Wonje Jeung, Seungju Han, Alvin Wan, Harrison Ngan, Youngjae Yu, Jonghyun Choi

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools, but their inherent safety risks - ranging from harmful content generation to broader societal harms - pose significant challenges. These risks can be amplified by the recent adversarial attacks, fine-tuning vulnerabilities, and the increasing deployment of LLMs in high-stakes environments. Existing safety-enhancing techniques, such as fine-tuning with human feedback or adversarial training, are still vulnerable as they address specific threats and often fail to generalize across unseen attacks, or require manual system-level defenses. This paper introduces RepBend, a novel approach that fundamentally disrupts the representations underlying harmful behaviors in LLMs, offering a scalable solution to enhance (potentially inherent) safety. RepBend brings the idea of activation steering - simple vector arithmetic for steering model's behavior during inference - to loss-based fine-tuning. Through extensive evaluation, RepBend achieves state-of-the-art performance, outperforming prior methods such as Circuit Breaker, RMU, and NPO, with up to 95% reduction in attack success rates across diverse jailbreak benchmarks, all with negligible reduction in model usability and general capabilities.



## **20. From Alerts to Intelligence: A Novel LLM-Aided Framework for Host-based Intrusion Detection**

cs.CR

**SubmitDate**: 2025-07-15    [abs](http://arxiv.org/abs/2507.10873v1) [paper-pdf](http://arxiv.org/pdf/2507.10873v1)

**Authors**: Danyu Sun, Jinghuai Zhang, Jiacen Xu, Yu Zheng, Yuan Tian, Zhou Li

**Abstract**: Host-based intrusion detection system (HIDS) is a key defense component to protect the organizations from advanced threats like Advanced Persistent Threats (APT). By analyzing the fine-grained logs with approaches like data provenance, HIDS has shown successes in capturing sophisticated attack traces. Despite the progresses embarked by the research community and industry, HIDS still frequently encounters backlash from their operators in the deployed environments, due to issues like high false-positive rate, inconsistent outcomes across environments and human-unfriendly detection results. Large Language Models (LLMs) have great potentials to advance the state of HIDS, given their extensive knowledge of attack techniques and their ability to detect anomalies through semantic analysis, anchored by recent studies. Yet, our preliminary analysis indicates that building an HIDS by naively prompting an LLM is unlikely to succeed. In this work, we explore the direction of building a customized LLM pipeline for HIDS and develop a system named SHIELD. SHIELD addresses challenges related to LLM's token limits, confusion of background noises, etc., by integrating a variety of techniques like event-level Masked Autoencoder (MAE) for attack window detection, attack evidence identification and expansion, Deterministic Data Augmentation (DDA) for profiling normal activities, and multi-purpose prompting that guides the LLM to conduct precise and interpretable attack investigations. Extensive experiments on three log datasets (DARPA-E3, NodLink-simulated-data and ATLASv2) show that SHIELD consistently achieves outstanding performance in comparison with 5 representative HIDS. These findings highlight the potential of LLMs as powerful tools for intrusion detection and pave the way for future research in this domain.



## **21. REAL-IoT: Characterizing GNN Intrusion Detection Robustness under Practical Adversarial Attack**

cs.CR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10836v1) [paper-pdf](http://arxiv.org/pdf/2507.10836v1)

**Authors**: Zhonghao Zhan, Huichi Zhou, Hamed Haddadi

**Abstract**: Graph Neural Network (GNN)-based network intrusion detection systems (NIDS) are often evaluated on single datasets, limiting their ability to generalize under distribution drift. Furthermore, their adversarial robustness is typically assessed using synthetic perturbations that lack realism. This measurement gap leads to an overestimation of GNN-based NIDS resilience. To address the limitations, we propose \textbf{REAL-IoT}, a comprehensive framework for robustness evaluation of GNN-based NIDS in IoT environments. Our framework presents a methodology that creates a unified dataset from canonical datasets to assess generalization under drift. In addition, it features a novel intrusion dataset collected from a physical IoT testbed, which captures network traffic and attack scenarios under real-world settings. Furthermore, using REAL-IoT, we explore the usage of Large Language Models (LLMs) to analyze network data and mitigate the impact of adversarial examples by filtering suspicious flows. Our evaluations using REAL-IoT reveal performance drops in GNN models compared to results from standard benchmarks, quantifying their susceptibility to drift and realistic attacks. We also demonstrate the potential of LLM-based filtering to enhance robustness. These findings emphasize the necessity of realistic threat modeling and rigorous measurement practices for developing resilient IoT intrusion detection systems.



## **22. Model Tampering Attacks Enable More Rigorous Evaluations of LLM Capabilities**

cs.CR

Accepted to TMLR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2502.05209v3) [paper-pdf](http://arxiv.org/pdf/2502.05209v3)

**Authors**: Zora Che, Stephen Casper, Robert Kirk, Anirudh Satheesh, Stewart Slocum, Lev E McKinney, Rohit Gandikota, Aidan Ewart, Domenic Rosati, Zichu Wu, Zikui Cai, Bilal Chughtai, Yarin Gal, Furong Huang, Dylan Hadfield-Menell

**Abstract**: Evaluations of large language model (LLM) risks and capabilities are increasingly being incorporated into AI risk management and governance frameworks. Currently, most risk evaluations are conducted by designing inputs that elicit harmful behaviors from the system. However, this approach suffers from two limitations. First, input-output evaluations cannot fully evaluate realistic risks from open-weight models. Second, the behaviors identified during any particular input-output evaluation can only lower-bound the model's worst-possible-case input-output behavior. As a complementary method for eliciting harmful behaviors, we propose evaluating LLMs with model tampering attacks which allow for modifications to latent activations or weights. We pit state-of-the-art techniques for removing harmful LLM capabilities against a suite of 5 input-space and 6 model tampering attacks. In addition to benchmarking these methods against each other, we show that (1) model resilience to capability elicitation attacks lies on a low-dimensional robustness subspace; (2) the success rate of model tampering attacks can empirically predict and offer conservative estimates for the success of held-out input-space attacks; and (3) state-of-the-art unlearning methods can easily be undone within 16 steps of fine-tuning. Together, these results highlight the difficulty of suppressing harmful LLM capabilities and show that model tampering attacks enable substantially more rigorous evaluations than input-space attacks alone.



## **23. Logic layer Prompt Control Injection (LPCI): A Novel Security Vulnerability Class in Agentic Systems**

cs.CR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10457v1) [paper-pdf](http://arxiv.org/pdf/2507.10457v1)

**Authors**: Hammad Atta, Ken Huang, Manish Bhatt, Kamal Ahmed, Muhammad Aziz Ul Haq, Yasir Mehmood

**Abstract**: The integration of large language models (LLMs) into enterprise systems has created a new class of covert security vulnerabilities, particularly within logic-execution layers and persistent-memory contexts. In this paper, we introduce Logic-Layer Prompt Control Injection (LPCI), a novel attack category in which encoded, delayed, and conditionally triggered payloads are embedded in memory, vector stores, or tool outputs. These payloads can bypass conventional input filters and trigger unauthorised behaviour across sessions.



## **24. Bypassing LLM Guardrails: An Empirical Analysis of Evasion Attacks against Prompt Injection and Jailbreak Detection Systems**

cs.CR

14 pages, 5 figures, 11 tables. To be published in LLMSec 2025

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2504.11168v3) [paper-pdf](http://arxiv.org/pdf/2504.11168v3)

**Authors**: William Hackett, Lewis Birch, Stefan Trawicki, Neeraj Suri, Peter Garraghan

**Abstract**: Large Language Models (LLMs) guardrail systems are designed to protect against prompt injection and jailbreak attacks. However, they remain vulnerable to evasion techniques. We demonstrate two approaches for bypassing LLM prompt injection and jailbreak detection systems via traditional character injection methods and algorithmic Adversarial Machine Learning (AML) evasion techniques. Through testing against six prominent protection systems, including Microsoft's Azure Prompt Shield and Meta's Prompt Guard, we show that both methods can be used to evade detection while maintaining adversarial utility achieving in some instances up to 100% evasion success. Furthermore, we demonstrate that adversaries can enhance Attack Success Rates (ASR) against black-box targets by leveraging word importance ranking computed by offline white-box models. Our findings reveal vulnerabilities within current LLM protection mechanisms and highlight the need for more robust guardrail systems.



## **25. ARMOR: Aligning Secure and Safe Large Language Models via Meticulous Reasoning**

cs.CR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.11500v1) [paper-pdf](http://arxiv.org/pdf/2507.11500v1)

**Authors**: Zhengyue Zhao, Yingzi Ma, Somesh Jha, Marco Pavone, Chaowei Xiao

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable generative capabilities. However, their susceptibility to misuse has raised significant safety concerns. While post-training safety alignment methods have been widely adopted, LLMs remain vulnerable to malicious instructions that can bypass safety constraints. Recent efforts have introduced inference-time safety reasoning (system-2 alignment), where LLMs conduct a reasoning process to perform safety verification before final response. We show, however, that these checks are driven by ad-hoc reasoning that diverges from the structured human process, where they first discern a user's true intent, then evaluate the associated risk based on the true intent. Consequently, these defenses remain vulnerable to sophisticated jailbreak prompts that cloak harmful goals in seemingly benign language. To build secure and safe LLMs, we propose a reasoning-based safety alignment framework, ARMOR, that replaces the ad-hoc chains of thought reasoning process with human-aligned, structured one. At inference, ARMOR (1) detects likely jailbreak strategies, (2) extracts the user's core intent while discarding deceptive instructions, and (3) applies a policy-grounded safety analysis to the purified request. ARMOR is evaluated on adaptive jailbreak attacks and multiple safety benchmarks, and a test-time scaling is conducted to further improve its performance. Results demonstrate that ARMOR significantly enhances the robustness against state-of-the-art adaptive jailbreak attacks and outperforms recent reasoning-based aligned models across various safety benchmarks.



## **26. IPAD: Inverse Prompt for AI Detection -- A Robust and Explainable LLM-Generated Text Detector**

cs.LG

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2502.15902v2) [paper-pdf](http://arxiv.org/pdf/2502.15902v2)

**Authors**: Zheng Chen, Yushi Feng, Changyang He, Yue Deng, Hongxi Pu, Bo Li

**Abstract**: Large Language Models (LLMs) have attained human-level fluency in text generation, which complicates the distinction between human-written and LLM-generated texts. This increases the risk of misuse and highlights the need for reliable detectors. Yet, existing detectors exhibit poor robustness on out-of-distribution (OOD) data and attacked data, which is critical for real-world scenarios. Also, they struggle to provide interpretable evidence to support their decisions, thus undermining the reliability. In light of these challenges, we propose IPAD (Inverse Prompt for AI Detection), a novel framework consisting of a Prompt Inverter that identifies predicted prompts that could have generated the input text, and two Distinguishers that examine the probability that the input texts align with the predicted prompts. Empirical evaluations demonstrate that IPAD outperforms the strongest baselines by 9.05% (Average Recall) on in-distribution data, 12.93% (AUROC) on out-of-distribution (OOD) data, and 5.48% (AUROC) on attacked data. IPAD also performs robustly on structured datasets. Furthermore, an interpretability assessment is conducted to illustrate that IPAD enhances the AI detection trustworthiness by allowing users to directly examine the decision-making evidence, which provides interpretable support for its state-of-the-art detection results.



## **27. The Man Behind the Sound: Demystifying Audio Private Attribute Profiling via Multimodal Large Language Model Agents**

cs.CR

22 pages, 4 figures

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10016v1) [paper-pdf](http://arxiv.org/pdf/2507.10016v1)

**Authors**: Lixu Wang, Kaixiang Yao, Xinfeng Li, Dong Yang, Haoyang Li, Xiaofeng Wang, Wei Dong

**Abstract**: Our research uncovers a novel privacy risk associated with multimodal large language models (MLLMs): the ability to infer sensitive personal attributes from audio data -- a technique we term audio private attribute profiling. This capability poses a significant threat, as audio can be covertly captured without direct interaction or visibility. Moreover, compared to images and text, audio carries unique characteristics, such as tone and pitch, which can be exploited for more detailed profiling. However, two key challenges exist in understanding MLLM-employed private attribute profiling from audio: (1) the lack of audio benchmark datasets with sensitive attribute annotations and (2) the limited ability of current MLLMs to infer such attributes directly from audio. To address these challenges, we introduce AP^2, an audio benchmark dataset that consists of two subsets collected and composed from real-world data, and both are annotated with sensitive attribute labels. Additionally, we propose Gifts, a hybrid multi-agent framework that leverages the complementary strengths of audio-language models (ALMs) and large language models (LLMs) to enhance inference capabilities. Gifts employs an LLM to guide the ALM in inferring sensitive attributes, then forensically analyzes and consolidates the ALM's inferences, overcoming severe hallucinations of existing ALMs in generating long-context responses. Our evaluations demonstrate that Gifts significantly outperforms baseline approaches in inferring sensitive attributes. Finally, we investigate model-level and data-level defense strategies to mitigate the risks of audio private attribute profiling. Our work validates the feasibility of audio-based privacy attacks using MLLMs, highlighting the need for robust defenses, and provides a dataset and framework to facilitate future research.



## **28. Not all tokens are created equal: Perplexity Attention Weighted Networks for AI generated text detection**

cs.CL

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2501.03940v3) [paper-pdf](http://arxiv.org/pdf/2501.03940v3)

**Authors**: Pablo Miralles-González, Javier Huertas-Tato, Alejandro Martín, David Camacho

**Abstract**: The rapid advancement in large language models (LLMs) has significantly enhanced their ability to generate coherent and contextually relevant text, raising concerns about the misuse of AI-generated content and making it critical to detect it. However, the task remains challenging, particularly in unseen domains or with unfamiliar LLMs. Leveraging LLM next-token distribution outputs offers a theoretically appealing approach for detection, as they encapsulate insights from the models' extensive pre-training on diverse corpora. Despite its promise, zero-shot methods that attempt to operationalize these outputs have met with limited success. We hypothesize that one of the problems is that they use the mean to aggregate next-token distribution metrics across tokens, when some tokens are naturally easier or harder to predict and should be weighted differently. Based on this idea, we propose the Perplexity Attention Weighted Network (PAWN), which uses the last hidden states of the LLM and positions to weight the sum of a series of features based on metrics from the next-token distribution across the sequence length. Although not zero-shot, our method allows us to cache the last hidden states and next-token distribution metrics on disk, greatly reducing the training resource requirements. PAWN shows competitive and even better performance in-distribution than the strongest baselines (fine-tuned LMs) with a fraction of their trainable parameters. Our model also generalizes better to unseen domains and source models, with smaller variability in the decision boundary across distribution shifts. It is also more robust to adversarial attacks, and if the backbone has multilingual capabilities, it presents decent generalization to languages not seen during supervised training, with LLaMA3-1B reaching a mean macro-averaged F1 score of 81.46% in cross-validation with nine languages.



## **29. EVALOOP: Assessing LLM Robustness in Programming from a Self-consistency Perspective**

cs.SE

20 pages, 11 figures

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2505.12185v3) [paper-pdf](http://arxiv.org/pdf/2505.12185v3)

**Authors**: Sen Fang, Weiyuan Ding, Bowen Xu

**Abstract**: Assessing the programming capabilities of Large Language Models (LLMs) is crucial for their effective use in software engineering. Current evaluations, however, predominantly measure the accuracy of generated code on static benchmarks, neglecting the critical aspect of model robustness during programming tasks. While adversarial attacks offer insights on model robustness, their effectiveness is limited and evaluation could be constrained. Current adversarial attack methods for robustness evaluation yield inconsistent results, struggling to provide a unified evaluation across different LLMs. We introduce EVALOOP, a novel assessment framework that evaluate the robustness from a self-consistency perspective, i.e., leveraging the natural duality inherent in popular software engineering tasks, e.g., code generation and code summarization. EVALOOP initiates a self-contained feedback loop: an LLM generates output (e.g., code) from an input (e.g., natural language specification), and then use the generated output as the input to produce a new output (e.g., summarizes that code into a new specification). EVALOOP repeats the process to assess the effectiveness of EVALOOP in each loop. This cyclical strategy intrinsically evaluates robustness without rely on any external attack setups, providing a unified metric to evaluate LLMs' robustness in programming. We evaluate 16 prominent LLMs (e.g., GPT-4.1, O4-mini) on EVALOOP and found that EVALOOP typically induces a 5.01%-19.31% absolute drop in pass@1 performance within ten loops. Intriguingly, robustness does not always align with initial performance (i.e., one-time query); for instance, GPT-3.5-Turbo, despite superior initial code generation compared to DeepSeek-V2, demonstrated lower robustness over repeated evaluation loop.



## **30. Game Theory Meets LLM and Agentic AI: Reimagining Cybersecurity for the Age of Intelligent Threats**

cs.CR

**SubmitDate**: 2025-07-14    [abs](http://arxiv.org/abs/2507.10621v1) [paper-pdf](http://arxiv.org/pdf/2507.10621v1)

**Authors**: Quanyan Zhu

**Abstract**: Protecting cyberspace requires not only advanced tools but also a shift in how we reason about threats, trust, and autonomy. Traditional cybersecurity methods rely on manual responses and brittle heuristics. To build proactive and intelligent defense systems, we need integrated theoretical frameworks and software tools. Game theory provides a rigorous foundation for modeling adversarial behavior, designing strategic defenses, and enabling trust in autonomous systems. Meanwhile, software tools process cyber data, visualize attack surfaces, verify compliance, and suggest mitigations. Yet a disconnect remains between theory and practical implementation.   The rise of Large Language Models (LLMs) and agentic AI offers a new path to bridge this gap. LLM-powered agents can operationalize abstract strategies into real-world decisions. Conversely, game theory can inform the reasoning and coordination of these agents across complex workflows. LLMs also challenge classical game-theoretic assumptions, such as perfect rationality or static payoffs, prompting new models aligned with cognitive and computational realities. This co-evolution promises richer theoretical foundations and novel solution concepts. Agentic AI also reshapes software design: systems must now be modular, adaptive, and trust-aware from the outset.   This chapter explores the intersection of game theory, agentic AI, and cybersecurity. We review key game-theoretic frameworks (e.g., static, dynamic, Bayesian, and signaling games) and solution concepts. We then examine how LLM agents can enhance cyber defense and introduce LLM-driven games that embed reasoning into AI agents. Finally, we explore multi-agent workflows and coordination games, outlining how this convergence fosters secure, intelligent, and adaptive cyber systems.



## **31. LaSM: Layer-wise Scaling Mechanism for Defending Pop-up Attack on GUI Agents**

cs.CR

10 pages, 9 figures

**SubmitDate**: 2025-07-13    [abs](http://arxiv.org/abs/2507.10610v1) [paper-pdf](http://arxiv.org/pdf/2507.10610v1)

**Authors**: Zihe Yan, Zhuosheng Zhang

**Abstract**: Graphical user interface (GUI) agents built on multimodal large language models (MLLMs) have recently demonstrated strong decision-making abilities in screen-based interaction tasks. However, they remain highly vulnerable to pop-up-based environmental injection attacks, where malicious visual elements divert model attention and lead to unsafe or incorrect actions. Existing defense methods either require costly retraining or perform poorly under inductive interference. In this work, we systematically study how such attacks alter the attention behavior of GUI agents and uncover a layer-wise attention divergence pattern between correct and incorrect outputs. Based on this insight, we propose \textbf{LaSM}, a \textit{Layer-wise Scaling Mechanism} that selectively amplifies attention and MLP modules in critical layers. LaSM improves the alignment between model saliency and task-relevant regions without additional training. Extensive experiments across 12 types of pop-up perturbations and 4 different model backbones show that LaSM consistently enhances the defense success rate. When combined with prompt-level alerts, LaSM achieves over 98\% robustness even under strong inductive attacks. Our findings reveal that attention misalignment is a core vulnerability in MLLM agents and can be effectively addressed through selective layer-wise modulation.



## **32. Auditing Prompt Caching in Language Model APIs**

cs.CL

Accepted at ICML 2025

**SubmitDate**: 2025-07-13    [abs](http://arxiv.org/abs/2502.07776v2) [paper-pdf](http://arxiv.org/pdf/2502.07776v2)

**Authors**: Chenchen Gu, Xiang Lisa Li, Rohith Kuditipudi, Percy Liang, Tatsunori Hashimoto

**Abstract**: Prompt caching in large language models (LLMs) results in data-dependent timing variations: cached prompts are processed faster than non-cached prompts. These timing differences introduce the risk of side-channel timing attacks. For example, if the cache is shared across users, an attacker could identify cached prompts from fast API response times to learn information about other users' prompts. Because prompt caching may cause privacy leakage, transparency around the caching policies of API providers is important. To this end, we develop and conduct statistical audits to detect prompt caching in real-world LLM API providers. We detect global cache sharing across users in seven API providers, including OpenAI, resulting in potential privacy leakage about users' prompts. Timing variations due to prompt caching can also result in leakage of information about model architecture. Namely, we find evidence that OpenAI's embedding model is a decoder-only Transformer, which was previously not publicly known.



## **33. LLMalMorph: On The Feasibility of Generating Variant Malware using Large-Language-Models**

cs.CR

13 pages

**SubmitDate**: 2025-07-12    [abs](http://arxiv.org/abs/2507.09411v1) [paper-pdf](http://arxiv.org/pdf/2507.09411v1)

**Authors**: Md Ajwad Akil, Adrian Shuai Li, Imtiaz Karim, Arun Iyengar, Ashish Kundu, Vinny Parla, Elisa Bertino

**Abstract**: Large Language Models (LLMs) have transformed software development and automated code generation. Motivated by these advancements, this paper explores the feasibility of LLMs in modifying malware source code to generate variants. We introduce LLMalMorph, a semi-automated framework that leverages semantical and syntactical code comprehension by LLMs to generate new malware variants. LLMalMorph extracts function-level information from the malware source code and employs custom-engineered prompts coupled with strategically defined code transformations to guide the LLM in generating variants without resource-intensive fine-tuning. To evaluate LLMalMorph, we collected 10 diverse Windows malware samples of varying types, complexity and functionality and generated 618 variants. Our thorough experiments demonstrate that it is possible to reduce the detection rates of antivirus engines of these malware variants to some extent while preserving malware functionalities. In addition, despite not optimizing against any Machine Learning (ML)-based malware detectors, several variants also achieved notable attack success rates against an ML-based malware classifier. We also discuss the limitations of current LLM capabilities in generating malware variants from source code and assess where this emerging technology stands in the broader context of malware variant generation.



## **34. Adversarial Activation Patching: A Framework for Detecting and Mitigating Emergent Deception in Safety-Aligned Transformers**

cs.LG

**SubmitDate**: 2025-07-12    [abs](http://arxiv.org/abs/2507.09406v1) [paper-pdf](http://arxiv.org/pdf/2507.09406v1)

**Authors**: Santhosh Kumar Ravindran

**Abstract**: Large language models (LLMs) aligned for safety through techniques like reinforcement learning from human feedback (RLHF) often exhibit emergent deceptive behaviors, where outputs appear compliant but subtly mislead or omit critical information. This paper introduces adversarial activation patching, a novel mechanistic interpretability framework that leverages activation patching as an adversarial tool to induce, detect, and mitigate such deception in transformer-based models. By sourcing activations from "deceptive" prompts and patching them into safe forward passes at specific layers, we simulate vulnerabilities and quantify deception rates. Through toy neural network simulations across multiple scenarios (e.g., 1000 trials per setup), we demonstrate that adversarial patching increases deceptive outputs to 23.9% from a 0% baseline, with layer-specific variations supporting our hypotheses. We propose six hypotheses, including transferability across models, exacerbation in multimodal settings, and scaling effects. An expanded literature review synthesizes over 20 key works in interpretability, deception, and adversarial attacks. Mitigation strategies, such as activation anomaly detection and robust fine-tuning, are detailed, alongside ethical considerations and future research directions. This work advances AI safety by highlighting patching's dual-use potential and provides a roadmap for empirical studies on large-scale models.



## **35. Weak-to-Strong Jailbreaking on Large Language Models**

cs.CL

ICML 2025

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2401.17256v4) [paper-pdf](http://arxiv.org/pdf/2401.17256v4)

**Authors**: Xuandong Zhao, Xianjun Yang, Tianyu Pang, Chao Du, Lei Li, Yu-Xiang Wang, William Yang Wang

**Abstract**: Large language models (LLMs) are vulnerable to jailbreak attacks - resulting in harmful, unethical, or biased text generations. However, existing jailbreaking methods are computationally costly. In this paper, we propose the weak-to-strong jailbreaking attack, an efficient inference time attack for aligned LLMs to produce harmful text. Our key intuition is based on the observation that jailbroken and aligned models only differ in their initial decoding distributions. The weak-to-strong attack's key technical insight is using two smaller models (a safe and an unsafe one) to adversarially modify a significantly larger safe model's decoding probabilities. We evaluate the weak-to-strong attack on 5 diverse open-source LLMs from 3 organizations. The results show our method can increase the misalignment rate to over 99% on two datasets with just one forward pass per example. Our study exposes an urgent safety issue that needs to be addressed when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong



## **36. A comprehensive study of LLM-based argument classification: from LLAMA through GPT-4o to Deepseek-R1**

cs.CL

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08621v1) [paper-pdf](http://arxiv.org/pdf/2507.08621v1)

**Authors**: Marcin Pietroń, Rafał Olszowski, Jakub Gomułka, Filip Gampel, Andrzej Tomski

**Abstract**: Argument mining (AM) is an interdisciplinary research field that integrates insights from logic, philosophy, linguistics, rhetoric, law, psychology, and computer science. It involves the automatic identification and extraction of argumentative components, such as premises and claims, and the detection of relationships between them, such as support, attack, or neutrality. Recently, the field has advanced significantly, especially with the advent of large language models (LLMs), which have enhanced the efficiency of analyzing and extracting argument semantics compared to traditional methods and other deep learning models. There are many benchmarks for testing and verifying the quality of LLM, but there is still a lack of research and results on the operation of these models in publicly available argument classification databases. This paper presents a study of a selection of LLM's, using diverse datasets such as Args.me and UKP. The models tested include versions of GPT, Llama, and DeepSeek, along with reasoning-enhanced variants incorporating the Chain-of-Thoughts algorithm. The results indicate that ChatGPT-4o outperforms the others in the argument classification benchmarks. In case of models incorporated with reasoning capabilities, the Deepseek-R1 shows its superiority. However, despite their superiority, GPT-4o and Deepseek-R1 still make errors. The most common errors are discussed for all models. To our knowledge, the presented work is the first broader analysis of the mentioned datasets using LLM and prompt algorithms. The work also shows some weaknesses of known prompt algorithms in argument analysis, while indicating directions for their improvement. The added value of the work is the in-depth analysis of the available argument datasets and the demonstration of their shortcomings.



## **37. The Dark Side of LLMs Agent-based Attacks for Complete Computer Takeover**

cs.CR

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.06850v3) [paper-pdf](http://arxiv.org/pdf/2507.06850v3)

**Authors**: Matteo Lupinacci, Francesco Aurelio Pironti, Francesco Blefari, Francesco Romeo, Luigi Arena, Angelo Furfaro

**Abstract**: The rapid adoption of Large Language Model (LLM) agents and multi-agent systems enables unprecedented capabilities in natural language processing and generation. However, these systems have introduced unprecedented security vulnerabilities that extend beyond traditional prompt injection attacks. This paper presents the first comprehensive evaluation of LLM agents as attack vectors capable of achieving complete computer takeover through the exploitation of trust boundaries within agentic AI systems where autonomous entities interact and influence each other. We demonstrate that adversaries can leverage three distinct attack surfaces - direct prompt injection, RAG backdoor attacks, and inter-agent trust exploitation - to coerce popular LLMs (including GPT-4o, Claude-4 and Gemini-2.5) into autonomously installing and executing malware on victim machines. Our evaluation of 17 state-of-the-art LLMs reveals an alarming vulnerability hierarchy: while 41.2% of models succumb to direct prompt injection, 52.9% are vulnerable to RAG backdoor attacks, and a critical 82.4% can be compromised through inter-agent trust exploitation. Notably, we discovered that LLMs which successfully resist direct malicious commands will execute identical payloads when requested by peer agents, revealing a fundamental flaw in current multi-agent security models. Our findings demonstrate that only 5.9% of tested models (1/17) proved resistant to all attack vectors, with the majority exhibiting context-dependent security behaviors that create exploitable blind spots. Our findings also highlight the need to increase awareness and research on the security risks of LLMs, showing a paradigm shift in cybersecurity threats, where AI tools themselves become sophisticated attack vectors.



## **38. Emoji Attack: Enhancing Jailbreak Attacks Against Judge LLM Detection**

cs.CL

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2411.01077v4) [paper-pdf](http://arxiv.org/pdf/2411.01077v4)

**Authors**: Zhipeng Wei, Yuqi Liu, N. Benjamin Erichson

**Abstract**: Jailbreaking techniques trick Large Language Models (LLMs) into producing restricted output, posing a potential threat. One line of defense is to use another LLM as a Judge to evaluate the harmfulness of generated text. However, we reveal that these Judge LLMs are vulnerable to token segmentation bias, an issue that arises when delimiters alter the tokenization process, splitting words into smaller sub-tokens. This alters the embeddings of the entire sequence, reducing detection accuracy and allowing harmful content to be misclassified as safe. In this paper, we introduce Emoji Attack, a novel strategy that amplifies existing jailbreak prompts by exploiting token segmentation bias. Our method leverages in-context learning to systematically insert emojis into text before it is evaluated by a Judge LLM, inducing embedding distortions that significantly lower the likelihood of detecting unsafe content. Unlike traditional delimiters, emojis also introduce semantic ambiguity, making them particularly effective in this attack. Through experiments on state-of-the-art Judge LLMs, we demonstrate that Emoji Attack substantially reduces the unsafe prediction rate, bypassing existing safeguards.



## **39. Evaluating Implicit Bias in Large Language Models by Attacking From a Psychometric Perspective**

cs.CL

Accepted to ACL 2025 Findings

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2406.14023v5) [paper-pdf](http://arxiv.org/pdf/2406.14023v5)

**Authors**: Yuchen Wen, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng

**Abstract**: As large language models (LLMs) become an important way of information access, there have been increasing concerns that LLMs may intensify the spread of unethical content, including implicit bias that hurts certain populations without explicit harmful words. In this paper, we conduct a rigorous evaluation of LLMs' implicit bias towards certain demographics by attacking them from a psychometric perspective to elicit agreements to biased viewpoints. Inspired by psychometric principles in cognitive and social psychology, we propose three attack approaches, i.e., Disguise, Deception, and Teaching. Incorporating the corresponding attack instructions, we built two benchmarks: (1) a bilingual dataset with biased statements covering four bias types (2.7K instances) for extensive comparative analysis, and (2) BUMBLE, a larger benchmark spanning nine common bias types (12.7K instances) for comprehensive evaluation. Extensive evaluation of popular commercial and open-source LLMs shows that our methods can elicit LLMs' inner bias more effectively than competitive baselines. Our attack methodology and benchmarks offer an effective means of assessing the ethical risks of LLMs, driving progress toward greater accountability in their development. Our code, data, and benchmarks are available at https://yuchenwen1.github.io/ImplicitBiasEvaluation/.



## **40. Invariant-based Robust Weights Watermark for Large Language Models**

cs.CR

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08288v1) [paper-pdf](http://arxiv.org/pdf/2507.08288v1)

**Authors**: Qingxiao Guo, Xinjie Zhu, Yilong Ma, Hui Jin, Yunhao Wang, Weifeng Zhang, Xiaobing Guo

**Abstract**: Watermarking technology has gained significant attention due to the increasing importance of intellectual property (IP) rights, particularly with the growing deployment of large language models (LLMs) on billions resource-constrained edge devices. To counter the potential threats of IP theft by malicious users, this paper introduces a robust watermarking scheme without retraining or fine-tuning for transformer models. The scheme generates a unique key for each user and derives a stable watermark value by solving linear constraints constructed from model invariants. Moreover, this technology utilizes noise mechanism to hide watermark locations in multi-user scenarios against collusion attack. This paper evaluates the approach on three popular models (Llama3, Phi3, Gemma), and the experimental results confirm the strong robustness across a range of attack methods (fine-tuning, pruning, quantization, permutation, scaling, reversible matrix and collusion attacks).



## **41. Pushing the Limits of Safety: A Technical Report on the ATLAS Challenge 2025**

cs.CR

AdvML@CVPR Challenge Report

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2506.12430v2) [paper-pdf](http://arxiv.org/pdf/2506.12430v2)

**Authors**: Zonghao Ying, Siyang Wu, Run Hao, Peng Ying, Shixuan Sun, Pengyu Chen, Junze Chen, Hao Du, Kaiwen Shen, Shangkun Wu, Jiwei Wei, Shiyuan He, Yang Yang, Xiaohai Xu, Ke Ma, Qianqian Xu, Qingming Huang, Shi Lin, Xun Wang, Changting Lin, Meng Han, Yilei Jiang, Siqi Lai, Yaozhi Zheng, Yifei Song, Xiangyu Yue, Zonglei Jing, Tianyuan Zhang, Zhilei Zhu, Aishan Liu, Jiakai Wang, Siyuan Liang, Xianglong Kong, Hainan Li, Junjie Mu, Haotong Qin, Yue Yu, Lei Chen, Felix Juefei-Xu, Qing Guo, Xinyun Chen, Yew Soon Ong, Xianglong Liu, Dawn Song, Alan Yuille, Philip Torr, Dacheng Tao

**Abstract**: Multimodal Large Language Models (MLLMs) have enabled transformative advancements across diverse applications but remain susceptible to safety threats, especially jailbreak attacks that induce harmful outputs. To systematically evaluate and improve their safety, we organized the Adversarial Testing & Large-model Alignment Safety Grand Challenge (ATLAS) 2025}. This technical report presents findings from the competition, which involved 86 teams testing MLLM vulnerabilities via adversarial image-text attacks in two phases: white-box and black-box evaluations. The competition results highlight ongoing challenges in securing MLLMs and provide valuable guidance for developing stronger defense mechanisms. The challenge establishes new benchmarks for MLLM safety evaluation and lays groundwork for advancing safer multimodal AI systems. The code and data for this challenge are openly available at https://github.com/NY1024/ATLAS_Challenge_2025.



## **42. A Dynamic Stackelberg Game Framework for Agentic AI Defense Against LLM Jailbreaking**

cs.AI

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.08207v1) [paper-pdf](http://arxiv.org/pdf/2507.08207v1)

**Authors**: Zhengye Han, Quanyan Zhu

**Abstract**: As large language models (LLMs) are increasingly deployed in critical applications, the challenge of jailbreaking, where adversaries manipulate the models to bypass safety mechanisms, has become a significant concern. This paper presents a dynamic Stackelberg game framework to model the interactions between attackers and defenders in the context of LLM jailbreaking. The framework treats the prompt-response dynamics as a sequential extensive-form game, where the defender, as the leader, commits to a strategy while anticipating the attacker's optimal responses. We propose a novel agentic AI solution, the "Purple Agent," which integrates adversarial exploration and defensive strategies using Rapidly-exploring Random Trees (RRT). The Purple Agent actively simulates potential attack trajectories and intervenes proactively to prevent harmful outputs. This approach offers a principled method for analyzing adversarial dynamics and provides a foundation for mitigating the risk of jailbreaking.



## **43. Beyond the Worst Case: Extending Differential Privacy Guarantees to Realistic Adversaries**

cs.CR

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.08158v1) [paper-pdf](http://arxiv.org/pdf/2507.08158v1)

**Authors**: Marika Swanberg, Meenatchi Sundaram Muthu Selva Annamalai, Jamie Hayes, Borja Balle, Adam Smith

**Abstract**: Differential Privacy (DP) is a family of definitions that bound the worst-case privacy leakage of a mechanism. One important feature of the worst-case DP guarantee is it naturally implies protections against adversaries with less prior information, more sophisticated attack goals, and complex measures of a successful attack. However, the analytical tradeoffs between the adversarial model and the privacy protections conferred by DP are not well understood thus far. To that end, this work sheds light on what the worst-case guarantee of DP implies about the success of attackers that are more representative of real-world privacy risks.   In this paper, we present a single flexible framework that generalizes and extends the patchwork of bounds on DP mechanisms found in prior work. Our framework allows us to compute high-probability guarantees for DP mechanisms on a large family of natural attack settings that previous bounds do not capture. One class of such settings is the approximate reconstruction of multiple individuals' data, such as inferring nearly entire columns of a tabular data set from noisy marginals and extracting sensitive information from DP-trained language models.   We conduct two empirical case studies to illustrate the versatility of our bounds and compare them to the success of state-of-the-art attacks. Specifically, we study attacks that extract non-uniform PII from a DP-trained language model, as well as multi-column reconstruction attacks where the adversary has access to some columns in the clear and attempts to reconstruct the remaining columns for each person's record. We find that the absolute privacy risk of attacking non-uniform data is highly dependent on the adversary's prior probability of success. Our high probability bounds give us a nuanced understanding of the privacy leakage of DP mechanisms in a variety of previously understudied attack settings.



## **44. Operationalizing a Threat Model for Red-Teaming Large Language Models (LLMs)**

cs.CL

Transactions of Machine Learning Research (TMLR)

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2407.14937v2) [paper-pdf](http://arxiv.org/pdf/2407.14937v2)

**Authors**: Apurv Verma, Satyapriya Krishna, Sebastian Gehrmann, Madhavan Seshadri, Anu Pradhan, Tom Ault, Leslie Barrett, David Rabinowitz, John Doucette, NhatHai Phan

**Abstract**: Creating secure and resilient applications with large language models (LLM) requires anticipating, adjusting to, and countering unforeseen threats. Red-teaming has emerged as a critical technique for identifying vulnerabilities in real-world LLM implementations. This paper presents a detailed threat model and provides a systematization of knowledge (SoK) of red-teaming attacks on LLMs. We develop a taxonomy of attacks based on the stages of the LLM development and deployment process and extract various insights from previous research. In addition, we compile methods for defense and practical red-teaming strategies for practitioners. By delineating prominent attack motifs and shedding light on various entry points, this paper provides a framework for improving the security and robustness of LLM-based systems.



## **45. Defending Against Prompt Injection With a Few DefensiveTokens**

cs.CR

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07974v1) [paper-pdf](http://arxiv.org/pdf/2507.07974v1)

**Authors**: Sizhe Chen, Yizhu Wang, Nicholas Carlini, Chawin Sitawarin, David Wagner

**Abstract**: When large language model (LLM) systems interact with external data to perform complex tasks, a new attack, namely prompt injection, becomes a significant threat. By injecting instructions into the data accessed by the system, the attacker is able to override the initial user task with an arbitrary task directed by the attacker. To secure the system, test-time defenses, e.g., defensive prompting, have been proposed for system developers to attain security only when needed in a flexible manner. However, they are much less effective than training-time defenses that change the model parameters. Motivated by this, we propose DefensiveToken, a test-time defense with prompt injection robustness comparable to training-time alternatives. DefensiveTokens are newly inserted as special tokens, whose embeddings are optimized for security. In security-sensitive cases, system developers can append a few DefensiveTokens before the LLM input to achieve security with a minimal utility drop. In scenarios where security is less of a concern, developers can simply skip DefensiveTokens; the LLM system remains the same as there is no defense, generating high-quality responses. Thus, DefensiveTokens, if released alongside the model, allow a flexible switch between the state-of-the-art (SOTA) utility and almost-SOTA security at test time. The code is available at https://github.com/Sizhe-Chen/DefensiveToken.



## **46. Evaluating Robustness of Large Audio Language Models to Audio Injection: An Empirical Study**

cs.CL

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2505.19598v2) [paper-pdf](http://arxiv.org/pdf/2505.19598v2)

**Authors**: Guanyu Hou, Jiaming He, Yinhang Zhou, Ji Guo, Yitong Qiao, Rui Zhang, Wenbo Jiang

**Abstract**: Large Audio-Language Models (LALMs) are increasingly deployed in real-world applications, yet their robustness against malicious audio injection attacks remains underexplored. This study systematically evaluates five leading LALMs across four attack scenarios: Audio Interference Attack, Instruction Following Attack, Context Injection Attack, and Judgment Hijacking Attack. Using metrics like Defense Success Rate, Context Robustness Score, and Judgment Robustness Index, their vulnerabilities and resilience were quantitatively assessed. Experimental results reveal significant performance disparities among models; no single model consistently outperforms others across all attack types. The position of malicious content critically influences attack effectiveness, particularly when placed at the beginning of sequences. A negative correlation between instruction-following capability and robustness suggests models adhering strictly to instructions may be more susceptible, contrasting with greater resistance by safety-aligned models. Additionally, system prompts show mixed effectiveness, indicating the need for tailored strategies. This work introduces a benchmark framework and highlights the importance of integrating robustness into training pipelines. Findings emphasize developing multi-modal defenses and architectural designs that decouple capability from susceptibility for secure LALMs deployment.



## **47. "I am bad": Interpreting Stealthy, Universal and Robust Audio Jailbreaks in Audio-Language Models**

cs.LG

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2502.00718v2) [paper-pdf](http://arxiv.org/pdf/2502.00718v2)

**Authors**: Isha Gupta, David Khachaturov, Robert Mullins

**Abstract**: The rise of multimodal large language models has introduced innovative human-machine interaction paradigms but also significant challenges in machine learning safety. Audio-Language Models (ALMs) are especially relevant due to the intuitive nature of spoken communication, yet little is known about their failure modes. This paper explores audio jailbreaks targeting ALMs, focusing on their ability to bypass alignment mechanisms. We construct adversarial perturbations that generalize across prompts, tasks, and even base audio samples, demonstrating the first universal jailbreaks in the audio modality, and show that these remain effective in simulated real-world conditions. Beyond demonstrating attack feasibility, we analyze how ALMs interpret these audio adversarial examples and reveal them to encode imperceptible first-person toxic speech - suggesting that the most effective perturbations for eliciting toxic outputs specifically embed linguistic features within the audio signal. These results have important implications for understanding the interactions between different modalities in multimodal models, and offer actionable insights for enhancing defenses against adversarial audio attacks.



## **48. GuardVal: Dynamic Large Language Model Jailbreak Evaluation for Comprehensive Safety Testing**

cs.LG

24 pages

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07735v1) [paper-pdf](http://arxiv.org/pdf/2507.07735v1)

**Authors**: Peiyan Zhang, Haibo Jin, Liying Kang, Haohan Wang

**Abstract**: Jailbreak attacks reveal critical vulnerabilities in Large Language Models (LLMs) by causing them to generate harmful or unethical content. Evaluating these threats is particularly challenging due to the evolving nature of LLMs and the sophistication required in effectively probing their vulnerabilities. Current benchmarks and evaluation methods struggle to fully address these challenges, leaving gaps in the assessment of LLM vulnerabilities. In this paper, we review existing jailbreak evaluation practices and identify three assumed desiderata for an effective jailbreak evaluation protocol. To address these challenges, we introduce GuardVal, a new evaluation protocol that dynamically generates and refines jailbreak prompts based on the defender LLM's state, providing a more accurate assessment of defender LLMs' capacity to handle safety-critical situations. Moreover, we propose a new optimization method that prevents stagnation during prompt refinement, ensuring the generation of increasingly effective jailbreak prompts that expose deeper weaknesses in the defender LLMs. We apply this protocol to a diverse set of models, from Mistral-7b to GPT-4, across 10 safety domains. Our findings highlight distinct behavioral patterns among the models, offering a comprehensive view of their robustness. Furthermore, our evaluation process deepens the understanding of LLM behavior, leading to insights that can inform future research and drive the development of more secure models.



## **49. May I have your Attention? Breaking Fine-Tuning based Prompt Injection Defenses using Architecture-Aware Attacks**

cs.CR

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07417v1) [paper-pdf](http://arxiv.org/pdf/2507.07417v1)

**Authors**: Nishit V. Pandya, Andrey Labunets, Sicun Gao, Earlence Fernandes

**Abstract**: A popular class of defenses against prompt injection attacks on large language models (LLMs) relies on fine-tuning the model to separate instructions and data, so that the LLM does not follow instructions that might be present with data. There are several academic systems and production-level implementations of this idea. We evaluate the robustness of this class of prompt injection defenses in the whitebox setting by constructing strong optimization-based attacks and showing that the defenses do not provide the claimed security properties. Specifically, we construct a novel attention-based attack algorithm for text-based LLMs and apply it to two recent whitebox defenses SecAlign (CCS 2025) and StruQ (USENIX Security 2025), showing attacks with success rates of up to 70% with modest increase in attacker budget in terms of tokens. Our findings make fundamental progress towards understanding the robustness of prompt injection defenses in the whitebox setting. We release our code and attacks at https://github.com/nishitvp/better_opts_attacks



## **50. Hybrid LLM-Enhanced Intrusion Detection for Zero-Day Threats in IoT Networks**

cs.CR

6 pages, IEEE conference

**SubmitDate**: 2025-07-10    [abs](http://arxiv.org/abs/2507.07413v1) [paper-pdf](http://arxiv.org/pdf/2507.07413v1)

**Authors**: Mohammad F. Al-Hammouri, Yazan Otoum, Rasha Atwa, Amiya Nayak

**Abstract**: This paper presents a novel approach to intrusion detection by integrating traditional signature-based methods with the contextual understanding capabilities of the GPT-2 Large Language Model (LLM). As cyber threats become increasingly sophisticated, particularly in distributed, heterogeneous, and resource-constrained environments such as those enabled by the Internet of Things (IoT), the need for dynamic and adaptive Intrusion Detection Systems (IDSs) becomes increasingly urgent. While traditional methods remain effective for detecting known threats, they often fail to recognize new and evolving attack patterns. In contrast, GPT-2 excels at processing unstructured data and identifying complex semantic relationships, making it well-suited to uncovering subtle, zero-day attack vectors. We propose a hybrid IDS framework that merges the robustness of signature-based techniques with the adaptability of GPT-2-driven semantic analysis. Experimental evaluations on a representative intrusion dataset demonstrate that our model enhances detection accuracy by 6.3%, reduces false positives by 9.0%, and maintains near real-time responsiveness. These results affirm the potential of language model integration to build intelligent, scalable, and resilient cybersecurity defences suited for modern connected environments.



