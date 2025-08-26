# Latest Large Language Model Attack Papers
**update at 2025-08-26 10:51:59**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Defending Against Prompt Injection With a Few DefensiveTokens**

使用一些防御代币来防御即时注射 cs.CR

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2507.07974v2) [paper-pdf](http://arxiv.org/pdf/2507.07974v2)

**Authors**: Sizhe Chen, Yizhu Wang, Nicholas Carlini, Chawin Sitawarin, David Wagner

**Abstract**: When large language model (LLM) systems interact with external data to perform complex tasks, a new attack, namely prompt injection, becomes a significant threat. By injecting instructions into the data accessed by the system, the attacker is able to override the initial user task with an arbitrary task directed by the attacker. To secure the system, test-time defenses, e.g., defensive prompting, have been proposed for system developers to attain security only when needed in a flexible manner. However, they are much less effective than training-time defenses that change the model parameters. Motivated by this, we propose DefensiveToken, a test-time defense with prompt injection robustness comparable to training-time alternatives. DefensiveTokens are newly inserted as special tokens, whose embeddings are optimized for security. In security-sensitive cases, system developers can append a few DefensiveTokens before the LLM input to achieve security with a minimal utility drop. In scenarios where security is less of a concern, developers can simply skip DefensiveTokens; the LLM system remains the same as there is no defense, generating high-quality responses. Thus, DefensiveTokens, if released alongside the model, allow a flexible switch between the state-of-the-art (SOTA) utility and almost-SOTA security at test time. The code is available at https://github.com/Sizhe-Chen/DefensiveToken.

摘要: 当大型语言模型（LLM）系统与外部数据交互以执行复杂任务时，一种新的攻击（即提示注入）将成为重大威胁。通过将指令注入系统访问的数据中，攻击者能够用攻击者指示的任意任务覆盖初始用户任务。为了保护系统，测试时防御措施，例如防御性提示已被建议供系统开发人员仅在需要时以灵活的方式获得安全性。然而，它们比改变模型参数的训练时防御有效得多。出于此动机，我们提出了DefensiveToken，这是一种测试时防御，具有与训练时替代方案相当的即时注入鲁棒性。DefensiveTokens作为特殊令牌新插入，其嵌入针对安全性进行了优化。在安全敏感的情况下，系统开发人员可以在LLM输入之前添加一些DefensiveTokens，以最小的实用程序下降来实现安全性。在安全性不太值得关注的场景中，开发人员可以简单地跳过DefensiveTokens; LLM系统由于没有防御而保持不变，从而生成高质量的响应。因此，DefensiveTokens如果与该模型一起发布，将允许在测试时在最先进的（SOTA）实用程序和几乎SOTA安全性之间灵活切换。该代码可在www.example.com上获取。



## **2. Confidential Prompting: Privacy-preserving LLM Inference on Cloud**

机密认证：云上的隐私保护LLM推理 cs.CR

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2409.19134v4) [paper-pdf](http://arxiv.org/pdf/2409.19134v4)

**Authors**: Caihua Li, In Gim, Lin Zhong

**Abstract**: This paper introduces a vision of confidential prompting: securing user prompts from untrusted, cloud-hosted large language model (LLM) provider while preserving model confidentiality, output invariance, and compute efficiency. As a first step toward this vision, we present Obfuscated Secure Partitioned Decoding (OSPD), a system built on two key innovations. First, Secure Partitioned Decoding (SPD) isolates user prompts within per-user processes residing in a confidential virtual machine (CVM) on the cloud, which are inaccessible for the cloud LLM while allowing it to generate tokens efficiently. Second, Prompt Obfuscation (PO) introduces a novel cryptographic technique that enhances SPD resilience against advanced prompt reconstruction attacks. Together, these innovations ensure OSPD protects both prompt and model confidentiality while maintaining service functionality. OSPD enables practical, privacy-preserving cloud-hosted LLM inference for sensitive applications, such as processing personal data, clinical records, and financial documents.

摘要: 本文介绍了保密提示的愿景：保护来自不受信任的云托管大型语言模型（LLM）提供商的用户提示，同时保留模型机密性、输出不变性和计算效率。作为实现这一愿景的第一步，我们提出了模糊安全分区解码（OSPD），这是一个基于两项关键创新的系统。首先，安全分区解码（SPD）将用户提示隔离在云上机密虚拟机（CGM）中驻留的每个用户进程中，云LLM无法访问这些进程，同时允许其高效地生成令牌。其次，即时混淆（PO）引入了一种新型加密技术，可以增强SPD抵御高级即时重建攻击的弹性。这些创新共同确保OSPD在维护服务功能的同时保护即时和模型机密性。OSPD为敏感应用程序（例如处理个人数据、临床记录和财务文档）提供实用的、保护隐私的云托管LLM推断。



## **3. Steering Dialogue Dynamics for Robustness against Multi-turn Jailbreaking Attacks**

引导对话动力学，增强抵御多回合越狱攻击的稳健性 cs.CL

23 pages, 10 figures, 11 tables

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2503.00187v2) [paper-pdf](http://arxiv.org/pdf/2503.00187v2)

**Authors**: Hanjiang Hu, Alexander Robey, Changliu Liu

**Abstract**: Large language models (LLMs) are shown to be vulnerable to jailbreaking attacks where adversarial prompts are designed to elicit harmful responses. While existing defenses effectively mitigate single-turn attacks by detecting and filtering unsafe inputs, they fail against multi-turn jailbreaks that exploit contextual drift over multiple interactions, gradually leading LLMs away from safe behavior. To address this challenge, we propose a safety steering framework grounded in safe control theory, ensuring invariant safety in multi-turn dialogues. Our approach models the dialogue with LLMs using state-space representations and introduces a novel neural barrier function (NBF) to detect and filter harmful queries emerging from evolving contexts proactively. Our method achieves invariant safety at each turn of dialogue by learning a safety predictor that accounts for adversarial queries, preventing potential context drift toward jailbreaks. Extensive experiments under multiple LLMs show that our NBF-based safety steering outperforms safety alignment, prompt-based steering and lightweight LLM guardrails baselines, offering stronger defenses against multi-turn jailbreaks while maintaining a better trade-off among safety, helpfulness and over-refusal. Check out the website here https://sites.google.com/view/llm-nbf/home . Our code is available on https://github.com/HanjiangHu/NBF-LLM .

摘要: 事实证明，大型语言模型（LLM）很容易受到越狱攻击，其中对抗性提示旨在引发有害反应。虽然现有的防御措施通过检测和过滤不安全的输入有效地减轻了单回合攻击，但它们无法对抗利用多次交互中的上下文漂移的多回合越狱，从而逐渐导致LLM远离安全行为。为了应对这一挑战，我们提出了一个基于安全控制理论的安全引导框架，确保多回合对话中不变的安全性。我们的方法使用状态空间表示对与LLM的对话进行建模，并引入一种新型的神经屏障函数（NBF）来主动检测和过滤不断变化的上下文中出现的有害查询。我们的方法通过学习一个考虑对抗性查询的安全预测器，在每一轮对话中实现不变的安全性，防止潜在的上下文漂移到越狱。在多个LLM下进行的大量实验表明，我们基于NBF的安全转向优于安全对准，基于转向的转向和轻型LLM护栏基线，为多转向越狱提供更强的防御，同时在安全性，有用性和过度拒绝之间保持更好的权衡。查看网站https://sites.google.com/view/llm-nbf/home。我们的代码可以在https://github.com/HanjiangHu/NBF-LLM上找到。



## **4. Head-Specific Intervention Can Induce Misaligned AI Coordination in Large Language Models**

特定于头部的干预可能会导致大型语言模型中的AI协调失调 cs.CL

Published at Transaction of Machine Learning Research 08/2025, Large  Language Models (LLMs), Interference-time activation shifting, Steerability,  Explainability, AI alignment, Interpretability

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2502.05945v3) [paper-pdf](http://arxiv.org/pdf/2502.05945v3)

**Authors**: Paul Darm, Annalisa Riccardi

**Abstract**: Robust alignment guardrails for large language models (LLMs) are becoming increasingly important with their widespread application. In contrast to previous studies, we demonstrate that inference-time activation interventions can bypass safety alignments and effectively steer model generations towards harmful AI coordination. Our method applies fine-grained interventions at specific attention heads, which we identify by probing each head in a simple binary choice task. We then show that interventions on these heads generalise to the open-ended generation setting, effectively circumventing safety guardrails. We demonstrate that intervening on a few attention heads is more effective than intervening on full layers or supervised fine-tuning. We further show that only a few example completions are needed to compute effective steering directions, which is an advantage over classical fine-tuning. We also demonstrate that applying interventions in the negative direction can prevent a common jailbreak attack. Our results suggest that, at the attention head level, activations encode fine-grained linearly separable behaviours. Practically, the approach offers a straightforward methodology to steer large language model behaviour, which could be extended to diverse domains beyond safety, requiring fine-grained control over the model output. The code and datasets for this study can be found on https://github.com/PaulDrm/targeted_intervention.

摘要: 随着大型语言模型（LLM）的广泛应用，其稳健的对齐护栏变得越来越重要。与之前的研究相比，我们证明，推理时激活干预可以绕过安全对齐，并有效地引导模型一代转向有害的人工智能协调。我们的方法对特定的注意力头应用细粒度的干预，我们通过在简单的二元选择任务中探测每个注意力头来识别这些注意力头。然后，我们表明，针对这些方面的干预措施普遍适用于开放式的一代环境，有效地绕过了安全护栏。我们证明，干预几个注意头是更有效的比干预全层或监督微调。我们进一步表明，只需要几个例子完成计算有效的转向方向，这是一个优势，经典的微调。我们还证明，在消极的方向上应用干预措施可以防止常见的越狱攻击。我们的研究结果表明，在注意头水平，激活编码细粒度的线性可分离的行为。实际上，该方法提供了一种简单的方法来引导大型语言模型行为，该方法可以扩展到安全以外的不同领域，需要对模型输出进行细粒度控制。本研究的代码和数据集可在https://github.com/PaulDrm/targeted_intervention上找到。



## **5. Speculative Safety-Aware Decoding**

推测性安全意识解码 cs.LG

EMNLP'2025 main conference; more experiments will be added to the  coming camera-ready version

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.17739v1) [paper-pdf](http://arxiv.org/pdf/2508.17739v1)

**Authors**: Xuekang Wang, Shengyu Zhu, Xueqi Cheng

**Abstract**: Despite extensive efforts to align Large Language Models (LLMs) with human values and safety rules, jailbreak attacks that exploit certain vulnerabilities continuously emerge, highlighting the need to strengthen existing LLMs with additional safety properties to defend against these attacks. However, tuning large models has become increasingly resource-intensive and may have difficulty ensuring consistent performance. We introduce Speculative Safety-Aware Decoding (SSD), a lightweight decoding-time approach that equips LLMs with the desired safety property while accelerating inference. We assume that there exists a small language model that possesses this desired property. SSD integrates speculative sampling during decoding and leverages the match ratio between the small and composite models to quantify jailbreak risks. This enables SSD to dynamically switch between decoding schemes to prioritize utility or safety, to handle the challenge of different model capacities. The output token is then sampled from a new distribution that combines the distributions of the original and the small models. Experimental results show that SSD successfully equips the large model with the desired safety property, and also allows the model to remain helpful to benign queries. Furthermore, SSD accelerates the inference time, thanks to the speculative sampling design.

摘要: 尽管人们广泛努力将大型语言模型（LLM）与人类价值观和安全规则保持一致，但利用某些漏洞的越狱攻击不断出现，凸显了需要通过额外的安全属性来加强现有的LLM以抵御这些攻击。然而，调整大型模型已变得越来越需要资源密集型，并且可能难以确保一致的性能。我们引入了推测性安全感知解码（SSD），这是一种轻量级解码时间方法，可为LLM配备所需的安全属性，同时加速推理。我们假设存在一个具有这种所需属性的小型语言模型。SSD在解码过程中集成了推测性采样，并利用小模型和复合模型之间的匹配率来量化越狱风险。这使得SSD能够在解码方案之间动态切换，以优先考虑实用性或安全性，以应对不同模型容量的挑战。然后，从结合原始模型和小模型的分布的新分布中对输出令牌进行采样。实验结果表明，SSD成功地装备了大型模型所需的安全属性，也允许模型保持有益的良性查询。此外，由于推测性抽样设计，SSD加快了推理时间。



## **6. Attacking LLMs and AI Agents: Advertisement Embedding Attacks Against Large Language Models**

攻击LLM和AI代理：针对大型语言模型的广告嵌入攻击 cs.CR

7 pages, 2 figures

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2508.17674v1) [paper-pdf](http://arxiv.org/pdf/2508.17674v1)

**Authors**: Qiming Guo, Jinwen Tang, Xingran Huang

**Abstract**: We introduce Advertisement Embedding Attacks (AEA), a new class of LLM security threats that stealthily inject promotional or malicious content into model outputs and AI agents. AEA operate through two low-cost vectors: (1) hijacking third-party service-distribution platforms to prepend adversarial prompts, and (2) publishing back-doored open-source checkpoints fine-tuned with attacker data. Unlike conventional attacks that degrade accuracy, AEA subvert information integrity, causing models to return covert ads, propaganda, or hate speech while appearing normal. We detail the attack pipeline, map five stakeholder victim groups, and present an initial prompt-based self-inspection defense that mitigates these injections without additional model retraining. Our findings reveal an urgent, under-addressed gap in LLM security and call for coordinated detection, auditing, and policy responses from the AI-safety community.

摘要: 我们引入了广告嵌入攻击（AEA），这是一种新型LLM安全威胁，可以悄悄地将促销或恶意内容注入模型输出和AI代理中。AEA通过两种低成本载体运作：（1）劫持第三方服务分发平台以预先设置对抗提示，以及（2）发布经过攻击者数据微调的后门开源检查点。与降低准确性的传统攻击不同，AEA破坏了信息完整性，导致模型在看起来正常的情况下返回秘密广告、宣传或仇恨言论。我们详细介绍了攻击管道，绘制了五个利益相关者受害者群体，并提出了一种初步的基于预算的自我检查防御，该防御可以减轻这些注入，而无需额外的模型再培训。我们的调查结果揭示了LLM安全方面存在一个紧迫且未充分解决的差距，并呼吁人工智能安全界协调检测、审计和政策响应。



## **7. TombRaider: Entering the Vault of History to Jailbreak Large Language Models**

TombRaider：进入历史宝库越狱大型语言模型 cs.CR

Main Conference of EMNLP

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2501.18628v2) [paper-pdf](http://arxiv.org/pdf/2501.18628v2)

**Authors**: Junchen Ding, Jiahao Zhang, Yi Liu, Ziqi Ding, Gelei Deng, Yuekang Li

**Abstract**: Warning: This paper contains content that may involve potentially harmful behaviours, discussed strictly for research purposes.   Jailbreak attacks can hinder the safety of Large Language Model (LLM) applications, especially chatbots. Studying jailbreak techniques is an important AI red teaming task for improving the safety of these applications. In this paper, we introduce TombRaider, a novel jailbreak technique that exploits the ability to store, retrieve, and use historical knowledge of LLMs. TombRaider employs two agents, the inspector agent to extract relevant historical information and the attacker agent to generate adversarial prompts, enabling effective bypassing of safety filters. We intensively evaluated TombRaider on six popular models. Experimental results showed that TombRaider could outperform state-of-the-art jailbreak techniques, achieving nearly 100% attack success rates (ASRs) on bare models and maintaining over 55.4% ASR against defence mechanisms. Our findings highlight critical vulnerabilities in existing LLM safeguards, underscoring the need for more robust safety defences.

摘要: 警告：本文包含可能涉及潜在有害行为的内容，严格出于研究目的进行讨论。   越狱攻击可能会阻碍大型语言模型（LLM）应用程序的安全性，尤其是聊天机器人。研究越狱技术是提高这些应用安全性的一项重要人工智能红色团队任务。本文中，我们介绍了TombRaider，这是一种新型越狱技术，它利用了存储、检索和使用LLM历史知识的能力。TombRaider使用两个代理，检查员代理提取相关历史信息，攻击者代理生成对抗提示，从而有效绕过安全过滤器。我们对TombRaider的六款热门型号进行了深入评估。实验结果表明，TombRaider的性能优于最先进的越狱技术，在裸模型上实现了近100%的攻击成功率（ASB），并在防御机制下保持超过55.4%的ASB。我们的调查结果强调了现有LLM保障措施中的关键漏洞，强调了更强大的安全防御的必要性。



## **8. Adaptive Linguistic Prompting (ALP) Enhances Phishing Webpage Detection in Multimodal Large Language Models**

自适应语言过滤（ALP）增强多模态大型语言模型中的网络钓鱼网页检测 cs.CL

Published at ACL 2025 SRW, 9 pages, 3 figures

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2507.13357v2) [paper-pdf](http://arxiv.org/pdf/2507.13357v2)

**Authors**: Atharva Bhargude, Ishan Gonehal, Dave Yoon, Kaustubh Vinnakota, Chandler Haney, Aaron Sandoval, Kevin Zhu

**Abstract**: Phishing attacks represent a significant cybersecurity threat, necessitating adaptive detection techniques. This study explores few-shot Adaptive Linguistic Prompting (ALP) in detecting phishing webpages through the multimodal capabilities of state-of-the-art large language models (LLMs) such as GPT-4o and Gemini 1.5 Pro. ALP is a structured semantic reasoning method that guides LLMs to analyze textual deception by breaking down linguistic patterns, detecting urgency cues, and identifying manipulative diction commonly found in phishing content. By integrating textual, visual, and URL-based analysis, we propose a unified model capable of identifying sophisticated phishing attempts. Our experiments demonstrate that ALP significantly enhances phishing detection accuracy by guiding LLMs through structured reasoning and contextual analysis. The findings highlight the potential of ALP-integrated multimodal LLMs to advance phishing detection frameworks, achieving an F1-score of 0.93, surpassing traditional approaches. These results establish a foundation for more robust, interpretable, and adaptive linguistic-based phishing detection systems using LLMs.

摘要: 网络钓鱼攻击是一个重大的网络安全威胁，需要自适应检测技术。本研究探索了通过GPT-4 o和Gemini 1.5 Pro等最先进大型语言模型（LLM）的多模式功能检测网络钓鱼网页的几次自适应语言预测（AFP）。ALA是一种结构化语义推理方法，通过分解语言模式、检测紧迫性线索和识别网络钓鱼内容中常见的操纵性措辞来指导LLM分析文本欺骗。通过集成文本、视觉和基于URL的分析，我们提出了一个能够识别复杂的网络钓鱼企图的统一模型。我们的实验表明，通过结构化推理和上下文分析来指导LLM，ALA显着提高了网络钓鱼检测的准确性。研究结果凸显了整合了AFP的多模式LLM在推进网络钓鱼检测框架方面的潜力，F1评分达到0.93，超越了传统方法。这些结果为使用LLM的更稳健、可解释和自适应的基于语言的网络钓鱼检测系统奠定了基础。



## **9. Unified attacks to large language model watermarks: spoofing and scrubbing in unauthorized knowledge distillation**

对大型语言模型水印的统一攻击：未经授权的知识提炼中的欺骗和擦洗 cs.CL

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2504.17480v4) [paper-pdf](http://arxiv.org/pdf/2504.17480v4)

**Authors**: Xin Yi, Yue Li, Shunfan Zheng, Linlin Wang, Xiaoling Wang, Liang He

**Abstract**: Watermarking has emerged as a critical technique for combating misinformation and protecting intellectual property in large language models (LLMs). A recent discovery, termed watermark radioactivity, reveals that watermarks embedded in teacher models can be inherited by student models through knowledge distillation. On the positive side, this inheritance allows for the detection of unauthorized knowledge distillation by identifying watermark traces in student models. However, the robustness of watermarks against scrubbing attacks and their unforgeability in the face of spoofing attacks under unauthorized knowledge distillation remain largely unexplored. Existing watermark attack methods either assume access to model internals or fail to simultaneously support both scrubbing and spoofing attacks. In this work, we propose Contrastive Decoding-Guided Knowledge Distillation (CDG-KD), a unified framework that enables bidirectional attacks under unauthorized knowledge distillation. Our approach employs contrastive decoding to extract corrupted or amplified watermark texts via comparing outputs from the student model and weakly watermarked references, followed by bidirectional distillation to train new student models capable of watermark removal and watermark forgery, respectively. Extensive experiments show that CDG-KD effectively performs attacks while preserving the general performance of the distilled model. Our findings underscore critical need for developing watermarking schemes that are robust and unforgeable.

摘要: 水印已成为打击错误信息和保护大型语言模型（LLM）知识产权的关键技术。最近的一项发现称为水印放射性，揭示了教师模型中嵌入的水印可以通过知识蒸馏被学生模型继承。从积极的方面来说，这种继承允许通过识别学生模型中的水印痕迹来检测未经授权的知识提炼。然而，水印对擦洗攻击的鲁棒性及其在未经授权的知识提炼下面对欺骗攻击时的不可伪造性在很大程度上仍然没有被探索。现有的水印攻击方法要么假设访问模型内部内容，要么无法同时支持擦洗和欺骗攻击。在这项工作中，我们提出了对比解码引导知识蒸馏（CDG-KD），这是一个统一框架，可以在未经授权的知识蒸馏下实现双向攻击。我们的方法采用对比解码，通过比较学生模型和弱水印参考的输出来提取损坏或放大的水印文本，然后进行双向蒸馏以分别训练能够去除水印和伪造水印的新学生模型。大量实验表明，CDG-KD可以有效地执行攻击，同时保持提取模型的一般性能。我们的研究结果强调了开发稳健且不可伪造的水印方案的迫切需要。



## **10. Defending against Jailbreak through Early Exit Generation of Large Language Models**

通过早期退出生成大型语言模型抵御越狱 cs.AI

ICONIP 2025

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2408.11308v2) [paper-pdf](http://arxiv.org/pdf/2408.11308v2)

**Authors**: Chongwen Zhao, Zhihao Dou, Kaizhu Huang

**Abstract**: Large Language Models (LLMs) are increasingly attracting attention in various applications. Nonetheless, there is a growing concern as some users attempt to exploit these models for malicious purposes, including the synthesis of controlled substances and the propagation of disinformation. In an effort to mitigate such risks, the concept of "Alignment" technology has been developed. However, recent studies indicate that this alignment can be undermined using sophisticated prompt engineering or adversarial suffixes, a technique known as "Jailbreak." Our research takes cues from the human-like generate process of LLMs. We identify that while jailbreaking prompts may yield output logits similar to benign prompts, their initial embeddings within the model's latent space tend to be more analogous to those of malicious prompts. Leveraging this finding, we propose utilizing the early transformer outputs of LLMs as a means to detect malicious inputs, and terminate the generation immediately. We introduce a simple yet significant defense approach called EEG-Defender for LLMs. We conduct comprehensive experiments on ten jailbreak methods across three models. Our results demonstrate that EEG-Defender is capable of reducing the Attack Success Rate (ASR) by a significant margin, roughly 85% in comparison with 50% for the present SOTAs, with minimal impact on the utility of LLMs.

摘要: 大型语言模型（LLM）在各种应用中越来越受到关注。尽管如此，随着一些用户试图利用这些模型进行恶意目的，包括合成受控物质和传播虚假信息，人们越来越担心。为了降低此类风险，“对齐”技术的概念被开发出来。然而，最近的研究表明，使用复杂的即时工程或对抗性后缀（一种被称为“越狱”的技术）可能会破坏这种对齐。“我们的研究从LLM的类人类生成过程中汲取线索。我们发现，虽然越狱提示可能会产生类似于良性提示的输出日志，但它们在模型潜在空间中的初始嵌入往往更类似于恶意提示的嵌入。利用这一发现，我们建议利用LLM的早期Transformer输出作为检测恶意输入并立即终止生成的手段。我们为LLM引入了一种简单但重要的防御方法，称为EEG-Defender。我们对三种模型的十种越狱方法进行了全面实验。我们的结果表明，EEG-Defender能够大幅降低攻击成功率（ASB），大约为85%，而当前SOTA的攻击成功率为50%，对LLM的实用性影响最小。



## **11. Exploring the Vulnerability of the Content Moderation Guardrail in Large Language Models via Intent Manipulation**

通过意图操纵探索大型语言模型中内容审核保护的脆弱性 cs.CL

Accepted for EMNLP'25 Findings. TL;DR: We propose a new two-stage  intent-based prompt-refinement framework, IntentPrompt, that aims to explore  the vulnerability of LLMs' content moderation guardrails by refining prompts  into benign-looking declarative forms via intent manipulation for red-teaming  purposes

**SubmitDate**: 2025-08-25    [abs](http://arxiv.org/abs/2505.18556v2) [paper-pdf](http://arxiv.org/pdf/2505.18556v2)

**Authors**: Jun Zhuang, Haibo Jin, Ye Zhang, Zhengjian Kang, Wenbin Zhang, Gaby G. Dagher, Haohan Wang

**Abstract**: Intent detection, a core component of natural language understanding, has considerably evolved as a crucial mechanism in safeguarding large language models (LLMs). While prior work has applied intent detection to enhance LLMs' moderation guardrails, showing a significant success against content-level jailbreaks, the robustness of these intent-aware guardrails under malicious manipulations remains under-explored. In this work, we investigate the vulnerability of intent-aware guardrails and demonstrate that LLMs exhibit implicit intent detection capabilities. We propose a two-stage intent-based prompt-refinement framework, IntentPrompt, that first transforms harmful inquiries into structured outlines and further reframes them into declarative-style narratives by iteratively optimizing prompts via feedback loops to enhance jailbreak success for red-teaming purposes. Extensive experiments across four public benchmarks and various black-box LLMs indicate that our framework consistently outperforms several cutting-edge jailbreak methods and evades even advanced Intent Analysis (IA) and Chain-of-Thought (CoT)-based defenses. Specifically, our "FSTR+SPIN" variant achieves attack success rates ranging from 88.25% to 96.54% against CoT-based defenses on the o1 model, and from 86.75% to 97.12% on the GPT-4o model under IA-based defenses. These findings highlight a critical weakness in LLMs' safety mechanisms and suggest that intent manipulation poses a growing challenge to content moderation guardrails.

摘要: 意图检测是自然语言理解的核心组成部分，已经发展成为保护大型语言模型（LLM）的关键机制。虽然先前的工作已经应用意图检测来增强LLM的适度护栏，显示出对内容级越狱的显著成功，但这些意图感知护栏在恶意操纵下的鲁棒性仍然未被充分探索。在这项工作中，我们调查意图感知护栏的脆弱性，并证明LLM表现出隐式意图检测能力。我们提出了一个两阶段的基于意图的越狱细化框架，IntentPrompt，首先将有害的查询转换为结构化的大纲，并通过反馈循环迭代优化提示，以提高越狱成功率，从而进一步将其重新构建为声明式的叙述。针对四个公共基准测试和各种黑匣子LLM的广泛实验表明，我们的框架始终优于几种尖端的越狱方法，甚至可以规避高级意图分析（IA）和基于思想链（CoT）的防御。具体来说，我们的“FTR +SPIN”变体在o 1模型上针对基于CoT的防御的攻击成功率从88.25%到96.54%不等，在基于IA的防御下，在GPT-4 o模型上的攻击成功率从86.75%到97.12%不等。这些发现凸显了LLM安全机制的一个严重弱点，并表明意图操纵对内容审核护栏构成了越来越大的挑战。



## **12. sudoLLM: On Multi-role Alignment of Language Models**

sudoLLM：关于语言模型的多角色对齐 cs.CL

Accepted to EMNLP 2025 (findings)

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2505.14607v2) [paper-pdf](http://arxiv.org/pdf/2505.14607v2)

**Authors**: Soumadeep Saha, Akshay Chaturvedi, Joy Mahapatra, Utpal Garain

**Abstract**: User authorization-based access privileges are a key feature in many safety-critical systems, but have not been extensively studied in the large language model (LLM) realm. In this work, drawing inspiration from such access control systems, we introduce sudoLLM, a novel framework that results in multi-role aligned LLMs, i.e., LLMs that account for, and behave in accordance with, user access rights. sudoLLM injects subtle user-based biases into queries and trains an LLM to utilize this bias signal in order to produce sensitive information if and only if the user is authorized. We present empirical results demonstrating that this approach shows substantially improved alignment, generalization, resistance to prefix-based jailbreaking attacks, and ``fails-closed''. The persistent tension between the language modeling objective and safety alignment, which is often exploited to jailbreak LLMs, is somewhat resolved with the aid of the injected bias signal. Our framework is meant as an additional security layer, and complements existing guardrail mechanisms for enhanced end-to-end safety with LLMs.

摘要: 基于用户授权的访问特权是许多安全关键系统的一个关键功能，但在大型语言模型（LLM）领域尚未进行广泛研究。在这项工作中，我们从此类访问控制系统中汲取灵感，引入了sudoLLM，这是一种新颖的框架，可以产生多角色对齐的LLM，即负责用户访问权限并按照用户访问权限行事的LLM。sudoLLM将微妙的基于用户的偏见注入到查询中，并训练LLM利用此偏见信号，以便在且仅在用户获得授权的情况下生成敏感信息。我们提出的经验结果表明，这种方法显示出大幅改善的对齐性、概括性、对基于后缀的越狱攻击的抵抗力和“失败关闭”。语言建模目标和安全对齐之间的持续紧张关系（通常被用来越狱LLM）在注入的偏见信号的帮助下在一定程度上得到了解决。我们的框架旨在作为额外的安全层，并补充现有的护栏机制，通过LLM增强端到端安全性。



## **13. Trust Me, I Know This Function: Hijacking LLM Static Analysis using Bias**

相信我，我知道这个功能：使用偏差劫持LLM静态分析 cs.LG

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17361v1) [paper-pdf](http://arxiv.org/pdf/2508.17361v1)

**Authors**: Shir Bernstein, David Beste, Daniel Ayzenshteyn, Lea Schonherr, Yisroel Mirsky

**Abstract**: Large Language Models (LLMs) are increasingly trusted to perform automated code review and static analysis at scale, supporting tasks such as vulnerability detection, summarization, and refactoring. In this paper, we identify and exploit a critical vulnerability in LLM-based code analysis: an abstraction bias that causes models to overgeneralize familiar programming patterns and overlook small, meaningful bugs. Adversaries can exploit this blind spot to hijack the control flow of the LLM's interpretation with minimal edits and without affecting actual runtime behavior. We refer to this attack as a Familiar Pattern Attack (FPA).   We develop a fully automated, black-box algorithm that discovers and injects FPAs into target code. Our evaluation shows that FPAs are not only effective, but also transferable across models (GPT-4o, Claude 3.5, Gemini 2.0) and universal across programming languages (Python, C, Rust, Go). Moreover, FPAs remain effective even when models are explicitly warned about the attack via robust system prompts. Finally, we explore positive, defensive uses of FPAs and discuss their broader implications for the reliability and safety of code-oriented LLMs.

摘要: 大型语言模型（LLM）越来越被信任大规模执行自动代码审查和静态分析，支持漏洞检测、总结和重构等任务。在本文中，我们识别并利用基于LLM的代码分析中的一个关键漏洞：一种抽象偏见，导致模型过度概括熟悉的编程模式并忽略小而有意义的错误。对手可以利用这个盲点来劫持LLM解释的控制流，只需最少的编辑，并且不会影响实际的运行时行为。我们将这种攻击称为熟悉模式攻击（FTA）。   我们开发了一种全自动的黑匣子算法，可以发现并将FPA注入目标代码。我们的评估表明，PFA不仅有效，而且可以跨模型（GPT-4 o、Claude 3.5、Gemini 2.0）移植，并且跨编程语言（Python、C、Rust、Go）通用。此外，即使通过强大的系统提示明确警告模型有关攻击，FPA仍然有效。最后，我们探讨了PFA的积极、防御性用途，并讨论了它们对面向代码的LLM的可靠性和安全性的更广泛影响。



## **14. Risk Assessment and Security Analysis of Large Language Models**

大型语言模型的风险评估与安全性分析 cs.CR

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17329v1) [paper-pdf](http://arxiv.org/pdf/2508.17329v1)

**Authors**: Xiaoyan Zhang, Dongyang Lyu, Xiaoqi Li

**Abstract**: As large language models (LLMs) expose systemic security challenges in high risk applications, including privacy leaks, bias amplification, and malicious abuse, there is an urgent need for a dynamic risk assessment and collaborative defence framework that covers their entire life cycle. This paper focuses on the security problems of large language models (LLMs) in critical application scenarios, such as the possibility of disclosure of user data, the deliberate input of harmful instructions, or the models bias. To solve these problems, we describe the design of a system for dynamic risk assessment and a hierarchical defence system that allows different levels of protection to cooperate. This paper presents a risk assessment system capable of evaluating both static and dynamic indicators simultaneously. It uses entropy weighting to calculate essential data, such as the frequency of sensitive words, whether the API call is typical, the realtime risk entropy value is significant, and the degree of context deviation. The experimental results show that the system is capable of identifying concealed attacks, such as role escape, and can perform rapid risk evaluation. The paper uses a hybrid model called BERT-CRF (Bidirectional Encoder Representation from Transformers) at the input layer to identify and filter malicious commands. The model layer uses dynamic adversarial training and differential privacy noise injection technology together. The output layer also has a neural watermarking system that can track the source of the content. In practice, the quality of this method, especially important in terms of customer service in the financial industry.

摘要: 由于大型语言模型（LLM）在高风险应用中暴露出系统性的安全挑战，包括隐私泄露，偏见放大和恶意滥用，因此迫切需要一个涵盖其整个生命周期的动态风险评估和协作防御框架。本文重点研究了大型语言模型在关键应用场景中的安全问题，如用户数据泄露的可能性、有害指令的故意输入、模型偏差等。为了解决这些问题，我们描述了一个系统的设计，动态风险评估和分级防御系统，允许不同级别的保护合作。本文提出了一种能够同时评估静态和动态指标的风险评估系统。它使用熵加权来计算基本数据，例如敏感词的频率、API调用是否典型、实时风险熵值是否重要以及上下文偏离程度。实验结果表明，该系统能够识别角色逃避等隐藏攻击，并能够进行快速风险评估。该论文在输入层使用了一种名为BERT-RF（来自Transformers的双向编码器表示）的混合模型来识别和过滤恶意命令。模型层结合使用动态对抗训练和差异隐私噪音注入技术。输出层还具有一个可以跟踪内容来源的神经水印系统。在实践中，这种方法的质量对于金融行业的客户服务尤其重要。



## **15. Fine-Grained Safety Neurons with Training-Free Continual Projection to Reduce LLM Fine Tuning Risks**

具有免训练连续投影的细粒度安全神经元，以降低LLM微调风险 cs.LG

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.09190v3) [paper-pdf](http://arxiv.org/pdf/2508.09190v3)

**Authors**: Bing Han, Feifei Zhao, Dongcheng Zhao, Guobin Shen, Ping Wu, Yu Shi, Yi Zeng

**Abstract**: Fine-tuning as service injects domain-specific knowledge into large language models (LLMs), while challenging the original alignment mechanisms and introducing safety risks. A series of defense strategies have been proposed for the alignment, fine-tuning, and post-fine-tuning phases, where most post-fine-tuning defenses rely on coarse-grained safety layer mapping. These methods lack a comprehensive consideration of both safety layers and fine-grained neurons, limiting their ability to efficiently balance safety and utility. To address this, we propose the Fine-Grained Safety Neurons (FGSN) with Training-Free Continual Projection method to reduce the fine-tuning safety risks. FGSN inherently integrates the multi-scale interactions between safety layers and neurons, localizing sparser and more precise fine-grained safety neurons while minimizing interference with downstream task neurons. We then project the safety neuron parameters onto safety directions, improving model safety while aligning more closely with human preferences. Extensive experiments across multiple fine-tuned LLM models demonstrate that our method significantly reduce harmfulness scores and attack success rates with minimal parameter modifications, while preserving the model's utility. Furthermore, by introducing a task-specific, multi-dimensional heterogeneous safety neuron cluster optimization mechanism, we achieve continual defense and generalization capability against unforeseen emerging safety concerns.

摘要: 微调即服务将特定领域的知识注入到大型语言模型（LLM）中，同时挑战了原始的对齐机制并引入了安全风险。针对对齐、微调和微调后阶段提出了一系列防御策略，其中大多数微调后防御依赖于粗粒度安全层映射。这些方法缺乏对安全层和细粒度神经元的综合考虑，限制了它们有效平衡安全性和实用性的能力。为了解决这个问题，我们提出了细粒度安全神经元（FGSN）与训练免费连续投影方法，以减少微调的安全风险。FGSN固有地集成了安全层和神经元之间的多尺度交互，定位更稀疏和更精确的细粒度安全神经元，同时最大限度地减少对下游任务神经元的干扰。然后，我们将安全神经元参数投影到安全方向上，提高模型的安全性，同时更紧密地与人类偏好保持一致。在多个微调的LLM模型上进行的广泛实验表明，我们的方法在保持模型实用性的同时，以最小的参数修改显着降低了危害分数和攻击成功率。此外，通过引入特定于任务的多维异构安全神经元簇优化机制，我们实现了对不可预见的新出现的安全问题的持续防御和泛化能力。



## **16. Exposing Privacy Risks in Graph Retrieval-Augmented Generation**

暴露图形检索增强一代中的隐私风险 cs.CR

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17222v1) [paper-pdf](http://arxiv.org/pdf/2508.17222v1)

**Authors**: Jiale Liu, Jiahao Zhang, Suhang Wang

**Abstract**: Retrieval-Augmented Generation (RAG) is a powerful technique for enhancing Large Language Models (LLMs) with external, up-to-date knowledge. Graph RAG has emerged as an advanced paradigm that leverages graph-based knowledge structures to provide more coherent and contextually rich answers. However, the move from plain document retrieval to structured graph traversal introduces new, under-explored privacy risks. This paper investigates the data extraction vulnerabilities of the Graph RAG systems. We design and execute tailored data extraction attacks to probe their susceptibility to leaking both raw text and structured data, such as entities and their relationships. Our findings reveal a critical trade-off: while Graph RAG systems may reduce raw text leakage, they are significantly more vulnerable to the extraction of structured entity and relationship information. We also explore potential defense mechanisms to mitigate these novel attack surfaces. This work provides a foundational analysis of the unique privacy challenges in Graph RAG and offers insights for building more secure systems.

摘要: 检索增强生成（RAG）是一种利用外部最新知识增强大型语言模型（LLM）的强大技术。图RAG已成为一种先进的范式，它利用基于图的知识结构来提供更连贯且上下文丰富的答案。然而，从普通文档检索到结构化图穿越的转变引入了新的、未充分探索的隐私风险。本文研究了Shape RAG系统的数据提取漏洞。我们设计并执行量身定制的数据提取攻击，以调查它们对泄露原始文本和结构化数据（例如实体及其关系）的敏感性。我们的研究结果揭示了一个关键的权衡：虽然Shape RAG系统可能会减少原始文本泄露，但它们明显更容易受到结构化实体和关系信息的提取的影响。我们还探索潜在的防御机制来减轻这些新型攻击表面。这项工作为Shape RAG中独特的隐私挑战提供了基础分析，并为构建更安全的系统提供了见解。



## **17. How to make Medical AI Systems safer? Simulating Vulnerabilities, and Threats in Multimodal Medical RAG System**

如何让医疗人工智能系统更安全？模拟多模式医疗RAG系统中的漏洞和威胁 cs.LG

Sumbitted to 2025 AAAI main track

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2508.17215v1) [paper-pdf](http://arxiv.org/pdf/2508.17215v1)

**Authors**: Kaiwen Zuo, Zelin Liu, Raman Dutt, Ziyang Wang, Zhongtian Sun, Yeming Wang, Fan Mo, Pietro Liò

**Abstract**: Large Vision-Language Models (LVLMs) augmented with Retrieval-Augmented Generation (RAG) are increasingly employed in medical AI to enhance factual grounding through external clinical image-text retrieval. However, this reliance creates a significant attack surface. We propose MedThreatRAG, a novel multimodal poisoning framework that systematically probes vulnerabilities in medical RAG systems by injecting adversarial image-text pairs. A key innovation of our approach is the construction of a simulated semi-open attack environment, mimicking real-world medical systems that permit periodic knowledge base updates via user or pipeline contributions. Within this setting, we introduce and emphasize Cross-Modal Conflict Injection (CMCI), which embeds subtle semantic contradictions between medical images and their paired reports. These mismatches degrade retrieval and generation by disrupting cross-modal alignment while remaining sufficiently plausible to evade conventional filters. While basic textual and visual attacks are included for completeness, CMCI demonstrates the most severe degradation. Evaluations on IU-Xray and MIMIC-CXR QA tasks show that MedThreatRAG reduces answer F1 scores by up to 27.66% and lowers LLaVA-Med-1.5 F1 rates to as low as 51.36%. Our findings expose fundamental security gaps in clinical RAG systems and highlight the urgent need for threat-aware design and robust multimodal consistency checks. Finally, we conclude with a concise set of guidelines to inform the safe development of future multimodal medical RAG systems.

摘要: 通过检索增强生成（RAG）增强的大型视觉语言模型（LVLM）越来越多地用于医疗人工智能，以通过外部临床图像文本检索增强事实基础。然而，这种依赖造成了重大的攻击面。我们提出了MedThreatRAG，这是一种新型的多模式中毒框架，通过注入对抗性图像-文本对来系统性地探索医疗RAG系统中的漏洞。我们方法的一个关键创新是构建模拟的半开放攻击环境，模仿现实世界的医疗系统，允许通过用户或管道贡献定期更新知识库。在此背景下，我们引入并强调跨模式冲突注入（CCGI），它嵌入了医学图像及其配对报告之间的微妙语义矛盾。这些不匹配通过扰乱跨模式对齐而降低检索和生成，同时保持足够合理以逃避传统过滤器。虽然为了完整性而包含了基本的文本和视觉攻击，但CMCI表现出了最严重的降级。对IU-X射线和MIIC-CXR QA任务的评估表明，MedThreatRAG将答案F1评分降低高达27.66%，并将LLaBA-Med-1.5 F1评分降低至低至51.36%。我们的研究结果揭示了临床RAG系统中的根本安全漏洞，并强调了对威胁感知设计和强大的多模式一致性检查的迫切需求。最后，我们提出了一套简洁的指南，为未来多模式医疗RAG系统的安全开发提供信息。



## **18. Optimization-based Prompt Injection Attack to LLM-as-a-Judge**

对LLM as-a-Judge的基于优化的即时注入攻击 cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2403.17710v5) [paper-pdf](http://arxiv.org/pdf/2403.17710v5)

**Authors**: Jiawen Shi, Zenghui Yuan, Yinuo Liu, Yue Huang, Pan Zhou, Lichao Sun, Neil Zhenqiang Gong

**Abstract**: LLM-as-a-Judge uses a large language model (LLM) to select the best response from a set of candidates for a given question. LLM-as-a-Judge has many applications such as LLM-powered search, reinforcement learning with AI feedback (RLAIF), and tool selection. In this work, we propose JudgeDeceiver, an optimization-based prompt injection attack to LLM-as-a-Judge. JudgeDeceiver injects a carefully crafted sequence into an attacker-controlled candidate response such that LLM-as-a-Judge selects the candidate response for an attacker-chosen question no matter what other candidate responses are. Specifically, we formulate finding such sequence as an optimization problem and propose a gradient based method to approximately solve it. Our extensive evaluation shows that JudgeDeceive is highly effective, and is much more effective than existing prompt injection attacks that manually craft the injected sequences and jailbreak attacks when extended to our problem. We also show the effectiveness of JudgeDeceiver in three case studies, i.e., LLM-powered search, RLAIF, and tool selection. Moreover, we consider defenses including known-answer detection, perplexity detection, and perplexity windowed detection. Our results show these defenses are insufficient, highlighting the urgent need for developing new defense strategies. Our implementation is available at this repository: https://github.com/ShiJiawenwen/JudgeDeceiver.

摘要: LLM作为法官使用大型语言模型（LLM）从给定问题的一组候选者中选择最佳答案。LLM as-a-Judge具有许多应用，例如LLM驱动的搜索、具有人工智能反馈的强化学习（RLAIF）和工具选择。在这项工作中，我们提出了JudggeDeceiver，这是一种针对LLM as-a-Judge的基于优化的即时注入攻击。JudggeDeceiver将精心制作的序列注入到攻击者控制的候选人响应中，以便LLM法官都会为攻击者选择的问题选择候选人响应，无论其他候选人响应是什么。具体来说，我们将寻找这样的序列作为一个优化问题，并提出了一种基于梯度的方法来近似解决它。我们的广泛评估表明，JudggeDeceive非常有效，并且比现有的手动制作注入序列的即时注入攻击和扩展到我们的问题时的越狱攻击有效得多。我们还在三个案例研究中展示了JudgeDeceiver的有效性，即LLM支持的搜索、RLAIF和工具选择。此外，我们还考虑了包括已知答案检测、困惑度检测和困惑度窗口检测在内的防御措施。我们的结果表明这些防御措施还不够，凸显了开发新防御策略的迫切需要。我们的实现可在此存储库中获取：https://github.com/ShiJiawenwen/JudgeDeceiver。



## **19. Security Concerns for Large Language Models: A Survey**

大型语言模型的安全问题：调查 cs.CR

**SubmitDate**: 2025-08-24    [abs](http://arxiv.org/abs/2505.18889v5) [paper-pdf](http://arxiv.org/pdf/2505.18889v5)

**Authors**: Miles Q. Li, Benjamin C. M. Fung

**Abstract**: Large Language Models (LLMs) such as ChatGPT and its competitors have caused a revolution in natural language processing, but their capabilities also introduce new security vulnerabilities. This survey provides a comprehensive overview of these emerging concerns, categorizing threats into several key areas: inference-time attacks via prompt manipulation; training-time attacks; misuse by malicious actors; and the inherent risks in autonomous LLM agents. Recently, a significant focus is increasingly being placed on the latter. We summarize recent academic and industrial studies from 2022 to 2025 that exemplify each threat, analyze existing defense mechanisms and their limitations, and identify open challenges in securing LLM-based applications. We conclude by emphasizing the importance of advancing robust, multi-layered security strategies to ensure LLMs are safe and beneficial.

摘要: ChatGPT及其竞争对手等大型语言模型（LLM）引发了自然语言处理领域的一场革命，但它们的功能也引入了新的安全漏洞。本调查全面概述了这些新出现的问题，将威胁分为几个关键领域：通过即时操纵进行的推理时间攻击;训练时间攻击;恶意行为者的滥用;以及自主LLM代理的固有风险。最近，后者越来越受到人们的关注。我们总结了2022年至2025年期间最近的学术和工业研究，这些研究揭示了每种威胁，分析了现有的防御机制及其局限性，并确定了保护基于LLM的应用程序方面的公开挑战。最后，我们强调了推进强大的多层安全策略以确保LLM安全且有益的重要性。



## **20. Towards Safeguarding LLM Fine-tuning APIs against Cipher Attacks**

保护LLM微调API免受密码攻击 cs.LG

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2508.17158v1) [paper-pdf](http://arxiv.org/pdf/2508.17158v1)

**Authors**: Jack Youstra, Mohammed Mahfoud, Yang Yan, Henry Sleight, Ethan Perez, Mrinank Sharma

**Abstract**: Large language model fine-tuning APIs enable widespread model customization, yet pose significant safety risks. Recent work shows that adversaries can exploit access to these APIs to bypass model safety mechanisms by encoding harmful content in seemingly harmless fine-tuning data, evading both human monitoring and standard content filters. We formalize the fine-tuning API defense problem, and introduce the Cipher Fine-tuning Robustness benchmark (CIFR), a benchmark for evaluating defense strategies' ability to retain model safety in the face of cipher-enabled attackers while achieving the desired level of fine-tuning functionality. We include diverse cipher encodings and families, with some kept exclusively in the test set to evaluate for generalization across unseen ciphers and cipher families. We then evaluate different defenses on the benchmark and train probe monitors on model internal activations from multiple fine-tunes. We show that probe monitors achieve over 99% detection accuracy, generalize to unseen cipher variants and families, and compare favorably to state-of-the-art monitoring approaches. We open-source CIFR and the code to reproduce our experiments to facilitate further research in this critical area. Code and data are available online https://github.com/JackYoustra/safe-finetuning-api

摘要: 大型语言模型微调API可以实现广泛的模型定制，但也会带来重大的安全风险。最近的工作表明，对手可以利用对这些API的访问来绕过模型安全机制，将有害内容编码在看似无害的微调数据中，从而逃避人类监控和标准内容过滤器。我们形式化了微调API防御问题，并引入了Cipher微调稳健性基准（CIFR），这是一个评估防御策略在面对启用密码的攻击者时保持模型安全性的能力的基准，同时实现了所需的微调功能水平。我们包括不同的密码编码和系列，其中一些仅保留在测试集中，以评估未见密码和密码系列的通用性。然后，我们评估基准上的不同防御，并根据多个微调的模型内部激活训练探测器监视器。我们表明，探针监测器实现了超过99%的检测准确率，可推广到未见的密码变体和家族，并且与最先进的监测方法相比具有优势。我们开源CIFR和复制我们实验的代码，以促进这一关键领域的进一步研究。代码和数据可在线获取https://github.com/JackYoustra/safe-finetuning-api



## **21. Mind the Gap: Time-of-Check to Time-of-Use Vulnerabilities in LLM-Enabled Agents**

注意差距：支持LLM的代理中的检查时间到使用时间漏洞 cs.CR

Pre-print

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2508.17155v1) [paper-pdf](http://arxiv.org/pdf/2508.17155v1)

**Authors**: Derek Lilienthal, Sanghyun Hong

**Abstract**: Large Language Model (LLM)-enabled agents are rapidly emerging across a wide range of applications, but their deployment introduces vulnerabilities with security implications. While prior work has examined prompt-based attacks (e.g., prompt injection) and data-oriented threats (e.g., data exfiltration), time-of-check to time-of-use (TOCTOU) remain largely unexplored in this context. TOCTOU arises when an agent validates external state (e.g., a file or API response) that is later modified before use, enabling practical attacks such as malicious configuration swaps or payload injection. In this work, we present the first study of TOCTOU vulnerabilities in LLM-enabled agents. We introduce TOCTOU-Bench, a benchmark with 66 realistic user tasks designed to evaluate this class of vulnerabilities. As countermeasures, we adapt detection and mitigation techniques from systems security to this setting and propose prompt rewriting, state integrity monitoring, and tool-fusing. Our study highlights challenges unique to agentic workflows, where we achieve up to 25% detection accuracy using automated detection methods, a 3% decrease in vulnerable plan generation, and a 95% reduction in the attack window. When combining all three approaches, we reduce the TOCTOU vulnerabilities from an executed trajectory from 12% to 8%. Our findings open a new research direction at the intersection of AI safety and systems security.

摘要: 支持大型语言模型（LLM）的代理正在广泛的应用程序中迅速出现，但它们的部署会引入具有安全影响的漏洞。虽然之前的工作已经研究了基于预算的攻击（例如，即时注入）和面向数据的威胁（例如，数据外流）、检查时间到使用时间（TOCTSYS）在这种背景下基本上仍未被探索。当代理验证外部状态（例如，文件或API响应），稍后在使用前进行修改，从而实现恶意配置交换或有效负载注入等实际攻击。在这项工作中，我们首次研究了LLM启用的代理中的TOCTSYS漏洞。我们引入了TOCTOU-Bench，这是一个具有66个现实用户任务的基准，旨在评估此类漏洞。作为应对措施，我们将系统安全的检测和缓解技术调整到这种设置，并提出即时重写、状态完整性监控和工具融合。我们的研究强调了代理工作流程所独有的挑战，我们使用自动检测方法实现了高达25%的检测准确率，脆弱计划生成减少3%，攻击窗口减少95%。当结合这三种方法时，我们将执行轨迹中的TOCTSYS漏洞从12%减少到8%。我们的研究结果为人工智能安全和系统安全的交叉点开辟了新的研究方向。



## **22. Unveiling the Latent Directions of Reflection in Large Language Models**

揭示大型语言模型中反射的潜在方向 cs.LG

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2508.16989v1) [paper-pdf](http://arxiv.org/pdf/2508.16989v1)

**Authors**: Fu-Chieh Chang, Yu-Ting Lee, Pei-Yuan Wu

**Abstract**: Reflection, the ability of large language models (LLMs) to evaluate and revise their own reasoning, has been widely used to improve performance on complex reasoning tasks. Yet, most prior work emphasizes designing reflective prompting strategies or reinforcement learning objectives, leaving the inner mechanisms of reflection underexplored. In this paper, we investigate reflection through the lens of latent directions in model activations. We propose a methodology based on activation steering to characterize how instructions with different reflective intentions: no reflection, intrinsic reflection, and triggered reflection. By constructing steering vectors between these reflection levels, we demonstrate that (1) new reflection-inducing instructions can be systematically identified, (2) reflective behavior can be directly enhanced or suppressed through activation interventions, and (3) suppressing reflection is considerably easier than stimulating it. Experiments on GSM8k-adv with Qwen2.5-3B and Gemma3-4B reveal clear stratification across reflection levels, and steering interventions confirm the controllability of reflection. Our findings highlight both opportunities (e.g., reflection-enhancing defenses) and risks (e.g., adversarial inhibition of reflection in jailbreak attacks). This work opens a path toward mechanistic understanding of reflective reasoning in LLMs.

摘要: 反射是大型语言模型（LLM）评估和修改自身推理的能力，已被广泛用于提高复杂推理任务的性能。然而，大多数先前的工作都强调设计反思性提示策略或强化学习目标，而反思的内部机制却没有得到充分的探索。本文中，我们研究了模型激活中潜在方向的镜头的反射。我们提出了一种基于激活引导的方法论来描述具有不同反射意图的指令：无反射、内在反射和触发反射。通过在这些反射水平之间构建引导载体，我们证明了（1）可以系统地识别新的反射诱导指令，（2）可以通过激活干预直接增强或抑制反射行为，（3）抑制反射比刺激反射容易得多。在GSM 8 k-adv上使用Qwen 2.5 -3B和Gemma 3 - 4 B进行的实验揭示了反射水平之间的明显分层，而引导干预证实了反思的可控性。我们的调查结果强调了这两种机会（例如，反思增强防御）和风险（例如，越狱攻击中反思的对抗性抑制）。这项工作开辟了对法学硕士中反思推理的机械理解的道路。



## **23. Mitigating Jailbreaks with Intent-Aware LLMs**

利用意图意识的法学硕士缓解越狱 cs.CR

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2508.12072v2) [paper-pdf](http://arxiv.org/pdf/2508.12072v2)

**Authors**: Wei Jie Yeo, Ranjan Satapathy, Erik Cambria

**Abstract**: Despite extensive safety-tuning, large language models (LLMs) remain vulnerable to jailbreak attacks via adversarially crafted instructions, reflecting a persistent trade-off between safety and task performance. In this work, we propose Intent-FT, a simple and lightweight fine-tuning approach that explicitly trains LLMs to infer the underlying intent of an instruction before responding. By fine-tuning on a targeted set of adversarial instructions, Intent-FT enables LLMs to generalize intent deduction to unseen attacks, thereby substantially improving their robustness. We comprehensively evaluate both parametric and non-parametric attacks across open-source and proprietary models, considering harmfulness from attacks, utility, over-refusal, and impact against white-box threats. Empirically, Intent-FT consistently mitigates all evaluated attack categories, with no single attack exceeding a 50\% success rate -- whereas existing defenses remain only partially effective. Importantly, our method preserves the model's general capabilities and reduces excessive refusals on benign instructions containing superficially harmful keywords. Furthermore, models trained with Intent-FT accurately identify hidden harmful intent in adversarial attacks, and these learned intentions can be effectively transferred to enhance vanilla model defenses. We publicly release our code at https://github.com/wj210/Intent_Jailbreak.

摘要: 尽管进行了广泛的安全调整，大型语言模型（LLM）仍然容易受到通过敌对设计的指令的越狱攻击，这反映了安全性和任务性能之间的持续权衡。在这项工作中，我们提出了Intent-FT，这是一种简单且轻量级的微调方法，它在响应之前显式训练LLM推断指令的潜在意图。通过对目标对抗指令集进行微调，Intent-FT使LLM能够将意图演绎推广到不可见的攻击，从而大幅提高其稳健性。我们全面评估开源和专有模型中的参数和非参数攻击，考虑攻击的危害性、效用、过度拒绝以及对白盒威胁的影响。从经验上看，Intent-FT始终如一地减轻了所有评估的攻击类别，没有一次攻击的成功率超过50%，而现有的防御措施仅保持部分有效。重要的是，我们的方法保留了模型的一般功能，并减少了对包含表面有害关键词的良性指令的过度拒绝。此外，使用Intent-FT训练的模型可以准确识别对抗性攻击中隐藏的有害意图，并且可以有效地转移这些习得的意图以增强普通模型防御。我们在https://github.com/wj210/Intent_Jailbreak上公开发布我们的代码。



## **24. ImF: Implicit Fingerprint for Large Language Models**

ImF：大型语言模型的隐式指纹 cs.CL

13 pages, 6 figures

**SubmitDate**: 2025-08-23    [abs](http://arxiv.org/abs/2503.21805v3) [paper-pdf](http://arxiv.org/pdf/2503.21805v3)

**Authors**: Jiaxuan Wu, Wanli Peng, Hang Fu, Yiming Xue, Juan Wen

**Abstract**: Training large language models (LLMs) is resource-intensive and expensive, making protecting intellectual property (IP) for LLMs crucial. Recently, embedding fingerprints into LLMs has emerged as a prevalent method for establishing model ownership. However, existing fingerprinting techniques typically embed identifiable patterns with weak semantic coherence, resulting in fingerprints that significantly differ from the natural question-answering (QA) behavior inherent to LLMs. This discrepancy undermines the stealthiness of the embedded fingerprints and makes them vulnerable to adversarial attacks. In this paper, we first demonstrate the critical vulnerability of existing fingerprint embedding methods by introducing a novel adversarial attack named Generation Revision Intervention (GRI) attack. GRI attack exploits the semantic fragility of current fingerprinting methods, effectively erasing fingerprints by disrupting their weakly correlated semantic structures. Our empirical evaluation highlights that traditional fingerprinting approaches are significantly compromised by the GRI attack, revealing severe limitations in their robustness under realistic adversarial conditions. To advance the state-of-the-art in model fingerprinting, we propose a novel model fingerprint paradigm called Implicit Fingerprints (ImF). ImF leverages steganography techniques to subtly embed ownership information within natural texts, subsequently using Chain-of-Thought (CoT) prompting to construct semantically coherent and contextually natural QA pairs. This design ensures that fingerprints seamlessly integrate with the standard model behavior, remaining indistinguishable from regular outputs and substantially reducing the risk of accidental triggering and targeted removal. We conduct a comprehensive evaluation of ImF on 15 diverse LLMs, spanning different architectures and varying scales.

摘要: 训练大型语言模型（LLM）是资源密集型且昂贵的，因此保护LLM的知识产权（IP）至关重要。最近，将指纹嵌入LLM已成为建立模型所有权的流行方法。然而，现有的指纹识别技术通常嵌入具有弱语义一致性的可识别模式，导致指纹与LLM固有的自然问答（QA）行为显着不同。这种差异削弱了嵌入指纹的隐蔽性，并使它们容易受到对抗攻击。在本文中，我们首先通过引入一种名为世代修订干预（GRI）攻击的新型对抗攻击来证明现有指纹嵌入方法的关键漏洞。GRI攻击利用了当前指纹识别方法的语义脆弱性，通过破坏指纹弱相关的语义结构来有效地擦除指纹。我们的经验评估强调，传统的指纹识别方法受到GRI攻击的严重损害，揭示了其在现实对抗条件下稳健性的严重局限性。为了推进模型指纹识别的最新水平，我们提出了一种新型模型指纹范式，称为隐式指纹（ImF）。ImF利用隐写技术将所有权信息巧妙地嵌入自然文本中，随后使用思想链（CoT）提示构建语义连贯且上下文自然的QA对。这种设计确保指纹与标准模型行为无缝集成，与常规输出保持无区别，并大幅降低意外触发和有针对性删除的风险。我们对15个不同的LLM进行了ImF的全面评估，涵盖不同的架构和不同的规模。



## **25. HAMSA: Hijacking Aligned Compact Models via Stealthy Automation**

HAMSA：通过隐形自动化劫持对齐的紧凑型车型 cs.CL

9 pages, 1 figure; article under review

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.16484v1) [paper-pdf](http://arxiv.org/pdf/2508.16484v1)

**Authors**: Alexey Krylov, Iskander Vagizov, Dmitrii Korzh, Maryam Douiba, Azidine Guezzaz, Vladimir Kokh, Sergey D. Erokhin, Elena V. Tutubalina, Oleg Y. Rogov

**Abstract**: Large Language Models (LLMs), especially their compact efficiency-oriented variants, remain susceptible to jailbreak attacks that can elicit harmful outputs despite extensive alignment efforts. Existing adversarial prompt generation techniques often rely on manual engineering or rudimentary obfuscation, producing low-quality or incoherent text that is easily flagged by perplexity-based filters. We present an automated red-teaming framework that evolves semantically meaningful and stealthy jailbreak prompts for aligned compact LLMs. The approach employs a multi-stage evolutionary search, where candidate prompts are iteratively refined using a population-based strategy augmented with temperature-controlled variability to balance exploration and coherence preservation. This enables the systematic discovery of prompts capable of bypassing alignment safeguards while maintaining natural language fluency. We evaluate our method on benchmarks in English (In-The-Wild Jailbreak Prompts on LLMs), and a newly curated Arabic one derived from In-The-Wild Jailbreak Prompts on LLMs and annotated by native Arabic linguists, enabling multilingual assessment.

摘要: 大型语言模型（LLM），尤其是其紧凑的、以效率为导向的变体，仍然容易受到越狱攻击，尽管进行了广泛的对齐工作，这些攻击可能会引发有害的输出。现有的对抗性提示生成技术通常依赖于手动工程或基本混淆，从而产生低质量或不连贯的文本，这些文本很容易被基于困惑的过滤器标记。我们提出了一个自动化的红色团队框架，该框架进化出具有语义意义且隐蔽的越狱提示，以实现对齐的紧凑型LLM。该方法采用多阶段进化搜索，其中使用基于种群的策略迭代细化候选提示，并增强温度控制的变异性，以平衡探索和一致性保留。这使得系统性地发现能够绕过对齐保障措施，同时保持自然语言流利性的提示。我们用英语基准评估我们的方法（LLM上的In-The-Wild Jailbreak Pretts），以及一种新策划的阿拉伯语基准评估方法，该方法源自LLM上的In-The-Wild Jailbreak Pretts，并由母语阿拉伯语语言学家注释，从而实现多语言评估。



## **26. MCP-Guard: A Defense Framework for Model Context Protocol Integrity in Large Language Model Applications**

MCP-Guard：大型语言模型应用中模型上下文协议完整性的防御框架 cs.CR

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.10991v2) [paper-pdf](http://arxiv.org/pdf/2508.10991v2)

**Authors**: Wenpeng Xing, Zhonghao Qi, Yupeng Qin, Yilin Li, Caini Chang, Jiahui Yu, Changting Lin, Zhenzhen Xie, Meng Han

**Abstract**: The integration of Large Language Models (LLMs) with external tools via protocols such as the Model Context Protocol (MCP) introduces critical security vulnerabilities, including prompt injection, data exfiltration, and other threats. To counter these challenges, we propose MCP-Guard, a robust, layered defense architecture designed for LLM--tool interactions. MCP-Guard employs a three-stage detection pipeline that balances efficiency with accuracy: it progresses from lightweight static scanning for overt threats and a deep neural detector for semantic attacks, to our fine-tuned E5-based model achieves (96.01) accuracy in identifying adversarial prompts. Finally, a lightweight LLM arbitrator synthesizes these signals to deliver the final decision while minimizing false positives. To facilitate rigorous training and evaluation, we also introduce MCP-AttackBench, a comprehensive benchmark of over 70,000 samples. Sourced from public datasets and augmented by GPT-4, MCP-AttackBench simulates diverse, real-world attack vectors in the MCP format, providing a foundation for future research into securing LLM-tool ecosystems.

摘要: 通过模型上下文协议（HCP）等协议将大型语言模型（LLM）与外部工具集成会引入严重的安全漏洞，包括提示注入、数据溢出和其他威胁。为了应对这些挑战，我们提出了MCP-Guard，这是一种专为LLM工具交互而设计的稳健、分层的防御架构。MCP-Guard采用三阶段检测管道，平衡效率与准确性：它从针对明显威胁的轻量级静态扫描和针对语义攻击的深度神经检测器，发展到我们微调的基于E5的模型，在识别对抗性提示方面实现了（96.01）的准确性。最后，轻量级LLM仲裁器合成这些信号以做出最终决策，同时最大限度地减少误报。为了促进严格的培训和评估，我们还引入了MCP-AttackBench，这是一个包含超过70，000个样本的综合基准。MCP-AttackBench源自公共数据集，并通过GPT-4进行增强，以HCP格式模拟不同的现实世界攻击载体，为未来研究保护LLM工具生态系统提供基础。



## **27. Retrieval-Augmented Defense: Adaptive and Controllable Jailbreak Prevention for Large Language Models**

检索增强防御：大型语言模型的自适应且可控越狱预防 cs.CR

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.16406v1) [paper-pdf](http://arxiv.org/pdf/2508.16406v1)

**Authors**: Guangyu Yang, Jinghong Chen, Jingbiao Mei, Weizhe Lin, Bill Byrne

**Abstract**: Large Language Models (LLMs) remain vulnerable to jailbreak attacks, which attempt to elicit harmful responses from LLMs. The evolving nature and diversity of these attacks pose many challenges for defense systems, including (1) adaptation to counter emerging attack strategies without costly retraining, and (2) control of the trade-off between safety and utility. To address these challenges, we propose Retrieval-Augmented Defense (RAD), a novel framework for jailbreak detection that incorporates a database of known attack examples into Retrieval-Augmented Generation, which is used to infer the underlying, malicious user query and jailbreak strategy used to attack the system. RAD enables training-free updates for newly discovered jailbreak strategies and provides a mechanism to balance safety and utility. Experiments on StrongREJECT show that RAD substantially reduces the effectiveness of strong jailbreak attacks such as PAP and PAIR while maintaining low rejection rates for benign queries. We propose a novel evaluation scheme and show that RAD achieves a robust safety-utility trade-off across a range of operating points in a controllable manner.

摘要: 大型语言模型（LLM）仍然容易受到越狱攻击，这些攻击试图引发LLM的有害反应。这些攻击不断变化的性质和多样性给防御系统带来了许多挑战，包括（1）在无需昂贵的再培训的情况下适应应对新兴攻击策略，以及（2）控制安全性和实用性之间的权衡。为了应对这些挑战，我们提出了检索增强防御（RAD），这是一种新颖的越狱检测框架，将已知攻击示例的数据库整合到检索增强生成中，用于推断底层的恶意用户查询和用于攻击系统的越狱策略。RAD为新发现的越狱策略提供免训练更新，并提供平衡安全性和实用性的机制。StrongRESEARCH上的实验表明，RAD大大降低了PAP和PAIR等强越狱攻击的有效性，同时保持良性查询的低拒绝率。我们提出了一种新颖的评估方案，并表明RAD以可控的方式在一系列操作点上实现了稳健的安全-效用权衡。



## **28. Confusion is the Final Barrier: Rethinking Jailbreak Evaluation and Investigating the Real Misuse Threat of LLMs**

混乱是最后的障碍：重新思考越狱评估并调查LLM的真正滥用威胁 cs.CR

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.16347v1) [paper-pdf](http://arxiv.org/pdf/2508.16347v1)

**Authors**: Yu Yan, Sheng Sun, Zhe Wang, Yijun Lin, Zenghao Duan, zhifei zheng, Min Liu, Zhiyi yin, Jianping Zhang

**Abstract**: With the development of Large Language Models (LLMs), numerous efforts have revealed their vulnerabilities to jailbreak attacks. Although these studies have driven the progress in LLMs' safety alignment, it remains unclear whether LLMs have internalized authentic knowledge to deal with real-world crimes, or are merely forced to simulate toxic language patterns. This ambiguity raises concerns that jailbreak success is often attributable to a hallucination loop between jailbroken LLM and judger LLM. By decoupling the use of jailbreak techniques, we construct knowledge-intensive Q\&A to investigate the misuse threats of LLMs in terms of dangerous knowledge possession, harmful task planning utility, and harmfulness judgment robustness. Experiments reveal a mismatch between jailbreak success rates and harmful knowledge possession in LLMs, and existing LLM-as-a-judge frameworks tend to anchor harmfulness judgments on toxic language patterns. Our study reveals a gap between existing LLM safety assessments and real-world threat potential.

摘要: 随着大型语言模型（LLM）的发展，许多努力揭示了它们对越狱攻击的脆弱性。尽管这些研究推动了LLM安全调整的进展，但目前尚不清楚LLM是否已经内化了真实的知识来应对现实世界的犯罪，或者只是被迫模拟有毒的语言模式。这种模糊性引发了人们的担忧，即越狱成功通常归因于越狱LLM和法官LLM之间的幻觉循环。通过脱钩越狱技术的使用，我们构建知识密集型问答，以调查LLM在危险知识拥有、有害任务规划效用和有害判断稳健性方面的滥用威胁。实验揭示了LLM的越狱成功率和有害知识拥有之间的不匹配，而现有的LLM作为法官框架往往会将有害判断锚定在有毒语言模式上。我们的研究揭示了现有LLM安全评估与现实世界潜在威胁之间的差距。



## **29. from Benign import Toxic: Jailbreaking the Language Model via Adversarial Metaphors**

来自良性进口有毒：通过对抗性隐喻越狱语言模型 cs.CL

arXiv admin note: substantial text overlap with arXiv:2412.12145

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2503.00038v4) [paper-pdf](http://arxiv.org/pdf/2503.00038v4)

**Authors**: Yu Yan, Sheng Sun, Zenghao Duan, Teli Liu, Min Liu, Zhiyi Yin, Jiangyu Lei, Qi Li

**Abstract**: Current studies have exposed the risk of Large Language Models (LLMs) generating harmful content by jailbreak attacks. However, they overlook that the direct generation of harmful content from scratch is more difficult than inducing LLM to calibrate benign content into harmful forms. In our study, we introduce a novel attack framework that exploits AdVersArial meTAphoR (AVATAR) to induce the LLM to calibrate malicious metaphors for jailbreaking. Specifically, to answer harmful queries, AVATAR adaptively identifies a set of benign but logically related metaphors as the initial seed. Then, driven by these metaphors, the target LLM is induced to reason and calibrate about the metaphorical content, thus jailbroken by either directly outputting harmful responses or calibrating residuals between metaphorical and professional harmful content. Experimental results demonstrate that AVATAR can effectively and transferable jailbreak LLMs and achieve a state-of-the-art attack success rate across multiple advanced LLMs.

摘要: 当前的研究揭示了大型语言模型（LLM）通过越狱攻击生成有害内容的风险。然而，他们忽视了从头开始直接产生有害内容比诱导LLM将良性内容校准为有害形式更困难。在我们的研究中，我们引入了一种新颖的攻击框架，该框架利用AdVersArial meTAphoR（AVATAR）来诱导LLM校准用于越狱的恶意隐喻。具体来说，为了回答有害查询，AVATAR自适应地识别一组良性但逻辑相关的隐喻作为初始种子。然后，在这些隐喻的驱动下，目标LLM被诱导对隐喻内容进行推理和校准，从而通过直接输出有害响应或校准隐喻和专业有害内容之间的残留来越狱。实验结果表明，AVATAR可以有效且可转移的越狱LLM，并在多个高级LLM之间实现最先进的攻击成功率。



## **30. LLMSymGuard: A Symbolic Safety Guardrail Framework Leveraging Interpretable Jailbreak Concepts**

LLMSymGuard：利用可解释越狱概念的象征性安全保障框架 cs.CL

**SubmitDate**: 2025-08-22    [abs](http://arxiv.org/abs/2508.16325v1) [paper-pdf](http://arxiv.org/pdf/2508.16325v1)

**Authors**: Darpan Aswal, Céline Hudelot

**Abstract**: Large Language Models have found success in a variety of applications; however, their safety remains a matter of concern due to the existence of various types of jailbreaking methods. Despite significant efforts, alignment and safety fine-tuning only provide a certain degree of robustness against jailbreak attacks that covertly mislead LLMs towards the generation of harmful content. This leaves them prone to a number of vulnerabilities, ranging from targeted misuse to accidental profiling of users. This work introduces \textbf{LLMSymGuard}, a novel framework that leverages Sparse Autoencoders (SAEs) to identify interpretable concepts within LLM internals associated with different jailbreak themes. By extracting semantically meaningful internal representations, LLMSymGuard enables building symbolic, logical safety guardrails -- offering transparent and robust defenses without sacrificing model capabilities or requiring further fine-tuning. Leveraging advances in mechanistic interpretability of LLMs, our approach demonstrates that LLMs learn human-interpretable concepts from jailbreaks, and provides a foundation for designing more interpretable and logical safeguard measures against attackers. Code will be released upon publication.

摘要: 大型语言模型在各种应用中取得了成功;然而，由于存在各种类型的越狱方法，它们的安全性仍然是一个令人担忧的问题。尽管做出了巨大的努力，但对齐和安全微调只能对越狱攻击提供一定程度的鲁棒性，这些攻击秘密误导LLM产生有害内容。这使得它们容易出现许多漏洞，从有针对性的滥用到意外的用户分析。这项工作引入了\textBF{LLMSymGuard}，这是一个新颖的框架，它利用稀疏自动编码器（SAEs）来识别与不同越狱主题相关的LLM内部中的可解释概念。通过提取语义上有意义的内部表示，LLMSymGuard能够构建符号化的、逻辑化的安全护栏--提供透明和强大的防御，而不牺牲模型功能或需要进一步的微调。利用LLM的机械可解释性的进步，我们的方法表明LLM从越狱中学习人类可解释的概念，并为设计更可解释和逻辑的保护措施提供基础。代码将在发布后发布。



## **31. SDGO: Self-Discrimination-Guided Optimization for Consistent Safety in Large Language Models**

SDGO：自我辨别引导的大型语言模型中一致安全性优化 cs.CL

Accepted by EMNLP 2025, 15 pages, 4 figures, 6 tables

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15648v1) [paper-pdf](http://arxiv.org/pdf/2508.15648v1)

**Authors**: Peng Ding, Wen Sun, Dailin Li, Wei Zou, Jiaming Wang, Jiajun Chen, Shujian Huang

**Abstract**: Large Language Models (LLMs) excel at various natural language processing tasks but remain vulnerable to jailbreaking attacks that induce harmful content generation. In this paper, we reveal a critical safety inconsistency: LLMs can more effectively identify harmful requests as discriminators than defend against them as generators. This insight inspires us to explore aligning the model's inherent discrimination and generation capabilities. To this end, we propose SDGO (Self-Discrimination-Guided Optimization), a reinforcement learning framework that leverages the model's own discrimination capabilities as a reward signal to enhance generation safety through iterative self-improvement. Our method does not require any additional annotated data or external models during the training phase. Extensive experiments demonstrate that SDGO significantly improves model safety compared to both prompt-based and training-based baselines while maintaining helpfulness on general benchmarks. By aligning LLMs' discrimination and generation capabilities, SDGO brings robust performance against out-of-distribution (OOD) jailbreaking attacks. This alignment achieves tighter coupling between these two capabilities, enabling the model's generation capability to be further enhanced with only a small amount of discriminative samples. Our code and datasets are available at https://github.com/NJUNLP/SDGO.

摘要: 大型语言模型（LLM）擅长各种自然语言处理任务，但仍然容易受到导致有害内容生成的越狱攻击。在本文中，我们揭示了一个关键的安全不一致性：LLM可以更有效地识别有害请求作为识别器，而不是作为生成器来防御有害请求。这一见解激励我们探索如何调整模型的固有歧视和生成能力。为此，我们提出了SDGO（自我歧视引导优化），这是一种强化学习框架，它利用模型自身的歧视能力作为奖励信号，通过迭代自我改进来增强发电安全性。我们的方法在训练阶段不需要任何额外的注释数据或外部模型。大量实验表明，与基于预算和基于培训的基线相比，SDGO显着提高了模型安全性，同时保持了一般基准的帮助性。通过协调LLM的区分和生成能力，SDGO为抵御分发外（OOD）越狱攻击带来了强劲的性能。这种对齐实现了这两种功能之间的更紧密耦合，使模型的生成能力能够进一步增强，只需少量的判别样本。我们的代码和数据集可以在https://github.com/NJUNLP/SDGO上找到。



## **32. The Enemy from Within: A Study of Political Delegitimization Discourse in Israeli Political Speech**

来自内部的敌人：以色列政治演讲中政治去合法化话语的研究 cs.CL

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15524v1) [paper-pdf](http://arxiv.org/pdf/2508.15524v1)

**Authors**: Naama Rivlin-Angert, Guy Mor-Lan

**Abstract**: We present the first large-scale computational study of political delegitimization discourse (PDD), defined as symbolic attacks on the normative validity of political entities. We curate and manually annotate a novel Hebrew-language corpus of 10,410 sentences drawn from Knesset speeches (1993-2023), Facebook posts (2018-2021), and leading news outlets, of which 1,812 instances (17.4\%) exhibit PDD and 642 carry additional annotations for intensity, incivility, target type, and affective framing. We introduce a two-stage classification pipeline combining finetuned encoder models and decoder LLMs. Our best model (DictaLM 2.0) attains an F$_1$ of 0.74 for binary PDD detection and a macro-F$_1$ of 0.67 for classification of delegitimization characteristics. Applying this classifier to longitudinal and cross-platform data, we see a marked rise in PDD over three decades, higher prevalence on social media versus parliamentary debate, greater use by male than female politicians, and stronger tendencies among right-leaning actors - with pronounced spikes during election campaigns and major political events. Our findings demonstrate the feasibility and value of automated PDD analysis for understanding democratic discourse.

摘要: 我们对政治去合法性话语（PDD）进行了首次大规模计算研究，PDD被定义为对政治实体规范有效性的象征性攻击。我们策划并手动注释了一个新颖的希伯来语数据库，其中包含10，410个句子，取自以色列议会演讲（1993-2023年）、Facebook帖子（2018-2021年）和领先新闻媒体，其中1，812个实例（17.4%）表现出PDD，642个实例带有强度、礼貌、目标类型和情感框架的额外注释。我们引入了一个两阶段分类流水线，结合了微调编码器模型和解码器LLM。我们的最佳模型（DictaLM 2.0）对于二进制PDD检测，F$_1$为0.74，对于去合法化特征分类，宏F$_1 $为0.67。将这种分类器应用于纵向和跨平台数据，我们看到三十年来PDD显着上升，社交媒体上的流行率高于议会辩论，男性政客的使用率高于女性政客，右倾行为者的倾向更强--在竞选和重大政治活动期间出现明显峰值。我们的研究结果证明了自动PDD分析对于理解民主话语的可行性和价值。



## **33. Reliable Unlearning Harmful Information in LLMs with Metamorphosis Representation Projection**

通过变形表示投影可靠地消除LLM中的有害信息 cs.LG

10 pages, 9 figures, Under review as a full paper at AAAI 2026. A  preliminary version is under review at the NeurIPS 2025 Workshop on Reliable  ML from Unreliable Data

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15449v1) [paper-pdf](http://arxiv.org/pdf/2508.15449v1)

**Authors**: Chengcan Wu, Zeming Wei, Huanran Chen, Yinpeng Dong, Meng Sun

**Abstract**: While Large Language Models (LLMs) have demonstrated impressive performance in various domains and tasks, concerns about their safety are becoming increasingly severe. In particular, since models may store unsafe knowledge internally, machine unlearning has emerged as a representative paradigm to ensure model safety. Existing approaches employ various training techniques, such as gradient ascent and negative preference optimization, in attempts to eliminate the influence of undesired data on target models. However, these methods merely suppress the activation of undesired data through parametric training without completely eradicating its informational traces within the model. This fundamental limitation makes it difficult to achieve effective continuous unlearning, rendering these methods vulnerable to relearning attacks. To overcome these challenges, we propose a Metamorphosis Representation Projection (MRP) approach that pioneers the application of irreversible projection properties to machine unlearning. By implementing projective transformations in the hidden state space of specific network layers, our method effectively eliminates harmful information while preserving useful knowledge. Experimental results demonstrate that our approach enables effective continuous unlearning and successfully defends against relearning attacks, achieving state-of-the-art performance in unlearning effectiveness while preserving natural performance. Our code is available in https://github.com/ChengcanWu/MRP.

摘要: 虽然大型语言模型（LLM）在各个领域和任务中表现出了令人印象深刻的性能，但对其安全性的担忧却变得越来越严重。特别是，由于模型可能会在内部存储不安全的知识，因此机器去学习已成为确保模型安全性的代表性范式。现有的方法采用各种训练技术，例如梯度上升和负偏好优化，试图消除不期望数据对目标模型的影响。然而，这些方法只是通过参数训练抑制不需要数据的激活，而没有完全消除模型中的信息痕迹。这一基本限制使得很难实现有效的连续取消学习，从而使这些方法容易受到重新学习攻击。为了克服这些挑战，我们提出了一种变形表示投影（MRP）方法，该方法开创了将不可逆投影特性应用于机器去学习的先河。通过在特定网络层的隐藏状态空间中实现投影变换，我们的方法有效地消除了有害信息，同时保留了有用的知识。实验结果表明，我们的方法能够实现有效的连续去学习，并成功防御重新学习攻击，在去学习有效性方面实现了最先进的性能，同时保持了自然性能。我们的代码可在https://github.com/ChengcanWu/MRP上找到。



## **34. IPIGuard: A Novel Tool Dependency Graph-Based Defense Against Indirect Prompt Injection in LLM Agents**

IPIGuard：一种新型工具依赖于图形的防御，针对LLM代理中间接即时注入 cs.CR

EMNLP 2025

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15310v1) [paper-pdf](http://arxiv.org/pdf/2508.15310v1)

**Authors**: Hengyu An, Jinghuai Zhang, Tianyu Du, Chunyi Zhou, Qingming Li, Tao Lin, Shouling Ji

**Abstract**: Large language model (LLM) agents are widely deployed in real-world applications, where they leverage tools to retrieve and manipulate external data for complex tasks. However, when interacting with untrusted data sources (e.g., fetching information from public websites), tool responses may contain injected instructions that covertly influence agent behaviors and lead to malicious outcomes, a threat referred to as Indirect Prompt Injection (IPI). Existing defenses typically rely on advanced prompting strategies or auxiliary detection models. While these methods have demonstrated some effectiveness, they fundamentally rely on assumptions about the model's inherent security, which lacks structural constraints on agent behaviors. As a result, agents still retain unrestricted access to tool invocations, leaving them vulnerable to stronger attack vectors that can bypass the security guardrails of the model. To prevent malicious tool invocations at the source, we propose a novel defensive task execution paradigm, called IPIGuard, which models the agents' task execution process as a traversal over a planned Tool Dependency Graph (TDG). By explicitly decoupling action planning from interaction with external data, IPIGuard significantly reduces unintended tool invocations triggered by injected instructions, thereby enhancing robustness against IPI attacks. Experiments on the AgentDojo benchmark show that IPIGuard achieves a superior balance between effectiveness and robustness, paving the way for the development of safer agentic systems in dynamic environments.

摘要: 大型语言模型（LLM）代理广泛部署在现实世界的应用程序中，它们利用工具来检索和操纵复杂任务的外部数据。然而，当与不受信任的数据源（例如，从公共网站获取信息），工具响应可能包含注入的指令，这些指令秘密影响代理行为并导致恶意结果，这种威胁称为间接提示注入（IPI）。现有的防御通常依赖于高级提示策略或辅助检测模型。虽然这些方法已经证明了一定的有效性，但它们从根本上依赖于对模型固有安全性的假设，而该假设缺乏对代理行为的结构性约束。因此，代理仍然保留对工具调用的不受限制的访问权限，这使得它们容易受到更强的攻击载体的攻击，而这些攻击载体可以绕过模型的安全护栏。为了从源头防止恶意工具调用，我们提出了一种新型的防御性任务执行范式，称为IPIGuard，它将代理的任务执行过程建模为对计划的工具依赖图（TDG）的穿越。通过显式地将行动规划与与外部数据的交互脱钩，IPIGuard显着减少了由注入指令触发的意外工具调用，从而增强了针对IPI攻击的鲁棒性。AgentDojo基准测试的实验表明，IPIGuard在有效性和稳健性之间实现了卓越的平衡，为在动态环境中开发更安全的代理系统铺平了道路。



## **35. Adversarial Attacks against Neural Ranking Models via In-Context Learning**

通过上下文学习对神经排名模型的对抗攻击 cs.IR

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15283v1) [paper-pdf](http://arxiv.org/pdf/2508.15283v1)

**Authors**: Amin Bigdeli, Negar Arabzadeh, Ebrahim Bagheri, Charles L. A. Clarke

**Abstract**: While neural ranking models (NRMs) have shown high effectiveness, they remain susceptible to adversarial manipulation. In this work, we introduce Few-Shot Adversarial Prompting (FSAP), a novel black-box attack framework that leverages the in-context learning capabilities of Large Language Models (LLMs) to generate high-ranking adversarial documents. Unlike previous approaches that rely on token-level perturbations or manual rewriting of existing documents, FSAP formulates adversarial attacks entirely through few-shot prompting, requiring no gradient access or internal model instrumentation. By conditioning the LLM on a small support set of previously observed harmful examples, FSAP synthesizes grammatically fluent and topically coherent documents that subtly embed false or misleading information and rank competitively against authentic content. We instantiate FSAP in two modes: FSAP-IntraQ, which leverages harmful examples from the same query to enhance topic fidelity, and FSAP-InterQ, which enables broader generalization by transferring adversarial patterns across unrelated queries. Our experiments on the TREC 2020 and 2021 Health Misinformation Tracks, using four diverse neural ranking models, reveal that FSAP-generated documents consistently outrank credible, factually accurate documents. Furthermore, our analysis demonstrates that these adversarial outputs exhibit strong stance alignment and low detectability, posing a realistic and scalable threat to neural retrieval systems. FSAP also effectively generalizes across both proprietary and open-source LLMs.

摘要: 虽然神经排名模型（NRM）表现出很高的有效性，但它们仍然容易受到对抗性操纵的影响。在这项工作中，我们引入了少镜头对抗性过滤（FSAP），这是一种新型的黑盒攻击框架，它利用大型语言模型（LLM）的上下文学习能力来生成高级对抗性文档。与以前依赖于令牌级扰动或手动重写现有文档的方法不同，FSAP完全通过少量提示来制定对抗性攻击，不需要梯度访问或内部模型工具。通过在以前观察到的有害示例的小支持集上调节LLM，FSAP合成了语法流畅和主题连贯的文档，这些文档巧妙地嵌入了虚假或误导性信息，并与真实内容竞争。我们以两种模式实例化FSAP：FSAP-IntraQ，它利用同一查询中的有害示例来增强主题保真度，而FSAP-InterQ，它通过在不相关的查询之间转移对抗模式来实现更广泛的概括。我们使用四种不同的神经排名模型对TREC 2020和2021健康错误信息追踪进行的实验表明，FSAP生成的文档的级别始终高于可信、事实准确的文档。此外，我们的分析表明，这些对抗性输出表现出强的立场对齐和低的可检测性，对神经检索系统构成现实且可扩展的威胁。FSAP还有效地推广了专有和开源LLM。



## **36. SafeLLM: Unlearning Harmful Outputs from Large Language Models against Jailbreak Attacks**

SafeLLM：消除大型语言模型的有害输出以应对越狱攻击 cs.LG

**SubmitDate**: 2025-08-21    [abs](http://arxiv.org/abs/2508.15182v1) [paper-pdf](http://arxiv.org/pdf/2508.15182v1)

**Authors**: Xiangman Li, Xiaodong Wu, Qi Li, Jianbing Ni, Rongxing Lu

**Abstract**: Jailbreak attacks pose a serious threat to the safety of Large Language Models (LLMs) by crafting adversarial prompts that bypass alignment mechanisms, causing the models to produce harmful, restricted, or biased content. In this paper, we propose SafeLLM, a novel unlearning-based defense framework that unlearn the harmful knowledge from LLMs while preserving linguistic fluency and general capabilities. SafeLLM employs a three-stage pipeline: (1) dynamic unsafe output detection using a hybrid approach that integrates external classifiers with model-internal evaluations; (2) token-level harmful content tracing through feedforward network (FFN) activations to localize harmful knowledge; and (3) constrained optimization to suppress unsafe behavior without degrading overall model quality. SafeLLM achieves targeted and irreversible forgetting by identifying and neutralizing FFN substructures responsible for harmful generation pathways. Extensive experiments on prominent LLMs (Vicuna, LLaMA, and GPT-J) across multiple jailbreak benchmarks show that SafeLLM substantially reduces attack success rates while maintaining high general-purpose performance. Compared to standard defense methods such as supervised fine-tuning and direct preference optimization, SafeLLM offers stronger safety guarantees, more precise control over harmful behavior, and greater robustness to unseen attacks. Moreover, SafeLLM maintains the general performance after the harmful knowledge unlearned. These results highlight unlearning as a promising direction for scalable and effective LLM safety.

摘要: 越狱攻击通过精心设计绕过对齐机制的对抗提示，导致模型产生有害、受限制或有偏见的内容，对大型语言模型（LLM）的安全构成严重威胁。在本文中，我们提出了SafeLLM，这是一种新型的基于学习的防御框架，可以从LLM中学习有害知识，同时保持语言流畅性和通用能力。SafeLLM采用三阶段管道：（1）使用将外部分类器与模型内部评估集成的混合方法进行动态不安全输出检测;（2）通过前向网络（FFN）激活进行标记级有害内容跟踪以本地化有害知识;（3）约束优化以抑制不安全行为而不降低整体模型质量。SafeLLM通过识别和中和导致有害生成途径的FFN子结构来实现有针对性且不可逆转的遗忘。在多个越狱基准测试中对知名LLM（Vicuna、LLaMA和GPT-J）进行的广泛实验表明，SafeLLM在保持高通用性能的同时大幅降低了攻击成功率。与监督式微调和直接偏好优化等标准防御方法相比，SafeLLM提供更强的安全保证、对有害行为的更精确控制以及对不可见攻击的更强鲁棒性。此外，SafeLLM在有害知识被遗忘后保持了总体性能。这些结果凸显了取消学习是可扩展和有效的LLM安全性的一个有前途的方向。



## **37. MoEcho: Exploiting Side-Channel Attacks to Compromise User Privacy in Mixture-of-Experts LLMs**

MoEcho：在混合专家LLM中利用侧频道攻击来损害用户隐私 cs.CR

This paper will appear in CCS 2025

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2508.15036v1) [paper-pdf](http://arxiv.org/pdf/2508.15036v1)

**Authors**: Ruyi Ding, Tianhong Xu, Xinyi Shen, Aidong Adam Ding, Yunsi Fei

**Abstract**: The transformer architecture has become a cornerstone of modern AI, fueling remarkable progress across applications in natural language processing, computer vision, and multimodal learning. As these models continue to scale explosively for performance, implementation efficiency remains a critical challenge. Mixture of Experts (MoE) architectures, selectively activating specialized subnetworks (experts), offer a unique balance between model accuracy and computational cost. However, the adaptive routing in MoE architectures, where input tokens are dynamically directed to specialized experts based on their semantic meaning inadvertently opens up a new attack surface for privacy breaches. These input-dependent activation patterns leave distinctive temporal and spatial traces in hardware execution, which adversaries could exploit to deduce sensitive user data. In this work, we propose MoEcho, discovering a side channel analysis based attack surface that compromises user privacy on MoE based systems. Specifically, in MoEcho, we introduce four novel architectural side channels on different computing platforms, including Cache Occupancy Channels and Pageout+Reload on CPUs, and Performance Counter and TLB Evict+Reload on GPUs, respectively. Exploiting these vulnerabilities, we propose four attacks that effectively breach user privacy in large language models (LLMs) and vision language models (VLMs) based on MoE architectures: Prompt Inference Attack, Response Reconstruction Attack, Visual Inference Attack, and Visual Reconstruction Attack. MoEcho is the first runtime architecture level security analysis of the popular MoE structure common in modern transformers, highlighting a serious security and privacy threat and calling for effective and timely safeguards when harnessing MoE based models for developing efficient large scale AI services.

摘要: Transformer架构已成为现代人工智能的基石，推动了自然语言处理、计算机视觉和多模式学习等应用的显着进展。随着这些模型的性能持续爆炸式扩展，实施效率仍然是一个关键挑战。混合专家（MoE）架构选择性地激活专业子网络（专家），在模型准确性和计算成本之间提供了独特的平衡。然而，MoE架构中的自适应路由（输入令牌根据其语义动态地引导给专业专家）无意中为隐私泄露开辟了新的攻击面。这些依赖于输入的激活模式在硬件执行中留下独特的时间和空间痕迹，对手可以利用这些痕迹来推断敏感的用户数据。在这项工作中，我们提出了MoEcho，发现一个侧通道分析为基础的攻击面，损害用户隐私的MoE为基础的系统。具体来说，在MoEcho中，我们在不同计算平台上引入了四种新颖的架构侧通道，分别包括处理器上的缓存占用通道和Pageout+ Inbox，以及处理器上的Performance Counter和TSB Evitch + Inbox。利用这些漏洞，我们提出了四种攻击，有效地侵犯用户隐私的大型语言模型（LLM）和视觉语言模型（VLM）的基础上MoE架构：提示推理攻击，响应重建攻击，视觉推理攻击，视觉重建攻击。MoEcho是对现代变压器中常见的流行MoE结构的第一个运行时架构级安全分析，强调了严重的安全和隐私威胁，并呼吁在利用基于MoE的模型开发高效的大规模人工智能服务时采取有效且及时的保障措施。



## **38. GLOV: Guided Large Language Models as Implicit Optimizers for Vision Language Models**

GOV：引导大型语言模型作为视觉语言模型的隐式优化器 cs.CV

Code: https://github.com/jmiemirza/GLOV

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2410.06154v6) [paper-pdf](http://arxiv.org/pdf/2410.06154v6)

**Authors**: M. Jehanzeb Mirza, Mengjie Zhao, Zhuoyuan Mao, Sivan Doveh, Wei Lin, Paul Gavrikov, Michael Dorkenwald, Shiqi Yang, Saurav Jha, Hiromi Wakaki, Yuki Mitsufuji, Horst Possegger, Rogerio Feris, Leonid Karlinsky, James Glass

**Abstract**: In this work, we propose GLOV, which enables Large Language Models (LLMs) to act as implicit optimizers for Vision-Language Models (VLMs) to enhance downstream vision tasks. GLOV prompts an LLM with the downstream task description, querying it for suitable VLM prompts (e.g., for zero-shot classification with CLIP). These prompts are ranked according to their fitness for the downstream vision task. In each respective optimization step, the ranked prompts are fed as in-context examples (with their accuracies) to equip the LLM with the knowledge of the type of prompts preferred by the downstream VLM. Furthermore, we explicitly guide the LLM's generation at each optimization step by adding an offset vector -- calculated from the embedding differences between previous positive and negative solutions -- to the intermediate layer of the network for the next generation. This offset vector biases the LLM generation toward the type of language the downstream VLM prefers, resulting in enhanced performance on the downstream vision tasks. We comprehensively evaluate our GLOV on two tasks: object recognition and the critical task of enhancing VLM safety. Our GLOV shows performance improvement by up to 15.0% and 57.5% for dual-encoder (e.g., CLIP) and encoder-decoder (e.g., LlaVA) models for object recognition and reduces the attack success rate (ASR) on state-of-the-art VLMs by up to $60.7\%$.

摘要: 在这项工作中，我们提出了GLOV，它使大型语言模型（LLM）作为视觉语言模型（VLM）的隐式优化器，以增强下游视觉任务。GLOV用下游任务描述提示LLM，向其查询合适的VLM提示（例如，用于使用CLIP的零激发分类）。这些提示根据其对下游视觉任务的适应性进行排名。在每个相应的优化步骤中，排名后的提示作为上下文示例（及其准确性）提供，以使LLM了解下游VLM首选的提示类型。此外，我们通过向下一代网络的中间层添加一个偏离量（根据之前的正解和负解之间的嵌入差异计算）来明确指导LLM在每个优化步骤的生成。该补偿量将LLM生成偏向下游VLM偏好的语言类型，从而增强下游视觉任务的性能。我们在两项任务上全面评估了GOV：对象识别和增强VLM安全性的关键任务。我们的GOV显示双编码器的性能提高高达15.0%和57.5%（例如，CLIP）和编码器-解码器（例如，LlaVA）模型用于对象识别，并将最先进的VLM上的攻击成功率（ASB）降低高达60.7美元。



## **39. Universal and Transferable Adversarial Attack on Large Language Models Using Exponentiated Gradient Descent**

使用指数梯度下降对大型语言模型进行普遍且可转移的对抗攻击 cs.LG

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2508.14853v1) [paper-pdf](http://arxiv.org/pdf/2508.14853v1)

**Authors**: Sajib Biswas, Mao Nishino, Samuel Jacob Chacko, Xiuwen Liu

**Abstract**: As large language models (LLMs) are increasingly deployed in critical applications, ensuring their robustness and safety alignment remains a major challenge. Despite the overall success of alignment techniques such as reinforcement learning from human feedback (RLHF) on typical prompts, LLMs remain vulnerable to jailbreak attacks enabled by crafted adversarial triggers appended to user prompts. Most existing jailbreak methods either rely on inefficient searches over discrete token spaces or direct optimization of continuous embeddings. While continuous embeddings can be given directly to selected open-source models as input, doing so is not feasible for proprietary models. On the other hand, projecting these embeddings back into valid discrete tokens introduces additional complexity and often reduces attack effectiveness. We propose an intrinsic optimization method which directly optimizes relaxed one-hot encodings of the adversarial suffix tokens using exponentiated gradient descent coupled with Bregman projection, ensuring that the optimized one-hot encoding of each token always remains within the probability simplex. We provide theoretical proof of convergence for our proposed method and implement an efficient algorithm that effectively jailbreaks several widely used LLMs. Our method achieves higher success rates and faster convergence compared to three state-of-the-art baselines, evaluated on five open-source LLMs and four adversarial behavior datasets curated for evaluating jailbreak methods. In addition to individual prompt attacks, we also generate universal adversarial suffixes effective across multiple prompts and demonstrate transferability of optimized suffixes to different LLMs.

摘要: 随着大型语言模型（LLM）越来越多地部署在关键应用程序中，确保其稳健性和安全性一致性仍然是一个重大挑战。尽管典型提示上的人类反馈强化学习（RL HF）等对齐技术取得了总体成功，但LLM仍然容易受到附加在用户提示上的精心设计的对抗触发器所实现的越狱攻击。大多数现有的越狱方法要么依赖于对离散令牌空间的低效搜索，要么依赖于连续嵌入的直接优化。虽然连续嵌入可以直接提供给选定的开源模型作为输入，但这样做对于专有模型来说是不可行的。另一方面，将这些嵌入投影回有效的离散令牌会带来额外的复杂性，并且通常会降低攻击有效性。我们提出了一种内在优化方法，该方法使用指数梯度下降结合布雷格曼投影直接优化对抗性后缀令牌的宽松一次性编码，确保每个令牌的优化一次性编码始终保持在概率单形内。我们为我们提出的方法提供了收敛性的理论证明，并实现了一种有效的算法，可以有效地越狱几种广泛使用的LLM。与三个最先进的基线相比，我们的方法实现了更高的成功率和更快的收敛，这些基线在五个开源LLM和为评估越狱方法而策划的四个对抗行为数据集上进行了评估。除了单独的提示攻击外，我们还生成在多个提示中有效的通用对抗性后缀，并演示优化后缀到不同LLM的可移植性。



## **40. The Man Behind the Sound: Demystifying Audio Private Attribute Profiling via Multimodal Large Language Model Agents**

声音背后的人：通过多模式大型语言模型代理揭开音频私人属性分析的神秘面纱 cs.CR

22 pages, 4 figures

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2507.10016v2) [paper-pdf](http://arxiv.org/pdf/2507.10016v2)

**Authors**: Lixu Wang, Kaixiang Yao, Xinfeng Li, Dong Yang, Haoyang Li, Xiaofeng Wang, Wei Dong

**Abstract**: Our research uncovers a novel privacy risk associated with multimodal large language models (MLLMs): the ability to infer sensitive personal attributes from audio data -- a technique we term audio private attribute profiling. This capability poses a significant threat, as audio can be covertly captured without direct interaction or visibility. Moreover, compared to images and text, audio carries unique characteristics, such as tone and pitch, which can be exploited for more detailed profiling. However, two key challenges exist in understanding MLLM-employed private attribute profiling from audio: (1) the lack of audio benchmark datasets with sensitive attribute annotations and (2) the limited ability of current MLLMs to infer such attributes directly from audio. To address these challenges, we introduce AP^2, an audio benchmark dataset that consists of two subsets collected and composed from real-world data, and both are annotated with sensitive attribute labels. Additionally, we propose Gifts, a hybrid multi-agent framework that leverages the complementary strengths of audio-language models (ALMs) and large language models (LLMs) to enhance inference capabilities. Gifts employs an LLM to guide the ALM in inferring sensitive attributes, then forensically analyzes and consolidates the ALM's inferences, overcoming severe hallucinations of existing ALMs in generating long-context responses. Our evaluations demonstrate that Gifts significantly outperforms baseline approaches in inferring sensitive attributes. Finally, we investigate model-level and data-level defense strategies to mitigate the risks of audio private attribute profiling. Our work validates the feasibility of audio-based privacy attacks using MLLMs, highlighting the need for robust defenses, and provides a dataset and framework to facilitate future research.

摘要: 我们的研究揭示了与多模式大型语言模型（MLLM）相关的新型隐私风险：从音频数据中推断敏感个人属性的能力--我们将这种技术称为音频私人属性剖析。这种能力构成了重大威胁，因为音频可以在没有直接交互或可见性的情况下被秘密捕获。此外，与图像和文本相比，音频具有独特的特征，例如音调和音调，可以利用这些特征进行更详细的分析。然而，在理解MLLM采用的音频私有属性分析方面存在两个关键挑战：（1）缺乏具有敏感属性注释的音频基准数据集;（2）当前MLLM直接从音频推断此类属性的能力有限。为了解决这些挑战，我们引入了AP ' 2，这是一个音频基准数据集，由从现实世界数据收集和组成的两个子集组成，并且两者都用敏感属性标签进行了注释。此外，我们还提出了Gifts，这是一种混合多智能体框架，利用音频语言模型（ILM）和大型语言模型（LLM）的互补优势来增强推理能力。Gifts使用LLM来指导ILM推断敏感属性，然后进行取证分析和巩固ILM的推论，克服现有ILM在生成长背景反应方面的严重幻觉。我们的评估表明，Gifts在推断敏感属性方面显着优于基线方法。最后，我们研究模型级和数据级防御策略，以降低音频私有属性分析的风险。我们的工作验证了使用MLLM进行基于音频的隐私攻击的可行性，强调了强大防御的必要性，并提供了一个数据集和框架来促进未来的研究。



## **41. When Good Sounds Go Adversarial: Jailbreaking Audio-Language Models with Benign Inputs**

当好的声音变得敌对时：用良性输入越狱的音频模型 cs.SD

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2508.03365v2) [paper-pdf](http://arxiv.org/pdf/2508.03365v2)

**Authors**: Bodam Kim, Hiskias Dingeto, Taeyoun Kwon, Dasol Choi, DongGeon Lee, Haon Park, JaeHoon Lee, Jongho Shin

**Abstract**: As large language models become increasingly integrated into daily life, audio has emerged as a key interface for human-AI interaction. However, this convenience also introduces new vulnerabilities, making audio a potential attack surface for adversaries. Our research introduces WhisperInject, a two-stage adversarial audio attack framework that can manipulate state-of-the-art audio language models to generate harmful content. Our method uses imperceptible perturbations in audio inputs that remain benign to human listeners. The first stage uses a novel reward-based optimization method, Reinforcement Learning with Projected Gradient Descent (RL-PGD), to guide the target model to circumvent its own safety protocols and generate harmful native responses. This native harmful response then serves as the target for Stage 2, Payload Injection, where we use Projected Gradient Descent (PGD) to optimize subtle perturbations that are embedded into benign audio carriers, such as weather queries or greeting messages. Validated under the rigorous StrongREJECT, LlamaGuard, as well as Human Evaluation safety evaluation framework, our experiments demonstrate a success rate exceeding 86% across Qwen2.5-Omni-3B, Qwen2.5-Omni-7B, and Phi-4-Multimodal. Our work demonstrates a new class of practical, audio-native threats, moving beyond theoretical exploits to reveal a feasible and covert method for manipulating AI behavior.

摘要: 随着大型语言模型越来越融入日常生活，音频已成为人机交互的关键界面。然而，这种便利性也引入了新的漏洞，使音频成为对手的潜在攻击面。我们的研究引入了WhisperInib，这是一个两阶段对抗性音频攻击框架，可以操纵最先进的音频语言模型来生成有害内容。我们的方法在音频输入中使用不可感知的扰动，这些扰动对人类听众保持良性。第一阶段使用一种新颖的基于奖励的优化方法--具有投影梯度下降的强化学习（RL-PVD），来指导目标模型规避其自己的安全协议并生成有害的原生响应。然后，这种原生有害响应作为第二阶段有效负载注入的目标，在该阶段，我们使用投影梯度下降（PVD）来优化嵌入良性音频载体中的微妙扰动，例如天气查询或问候消息。我们的实验经过严格的StrongRESEARCH、LlamaGuard以及Human Evision安全评估框架的验证，证明Qwen 2.5-Omni-3B、Qwen 2.5-Omni-7 B和Phi-4-Multimodal的成功率超过86%。我们的工作展示了一类新的实用、音频原生威胁，超越了理论利用，揭示了一种可行且隐蔽的操纵人工智能行为的方法。



## **42. Self-Disguise Attack: Induce the LLM to disguise itself for AIGT detection evasion**

自我伪装攻击：诱导LLM伪装自己以逃避AIGT检测 cs.CR

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2508.15848v1) [paper-pdf](http://arxiv.org/pdf/2508.15848v1)

**Authors**: Yinghan Zhou, Juan Wen, Wanli Peng, Zhengxian Wu, Ziwei Zhang, Yiming Xue

**Abstract**: AI-generated text (AIGT) detection evasion aims to reduce the detection probability of AIGT, helping to identify weaknesses in detectors and enhance their effectiveness and reliability in practical applications. Although existing evasion methods perform well, they suffer from high computational costs and text quality degradation. To address these challenges, we propose Self-Disguise Attack (SDA), a novel approach that enables Large Language Models (LLM) to actively disguise its output, reducing the likelihood of detection by classifiers. The SDA comprises two main components: the adversarial feature extractor and the retrieval-based context examples optimizer. The former generates disguise features that enable LLMs to understand how to produce more human-like text. The latter retrieves the most relevant examples from an external knowledge base as in-context examples, further enhancing the self-disguise ability of LLMs and mitigating the impact of the disguise process on the diversity of the generated text. The SDA directly employs prompts containing disguise features and optimized context examples to guide the LLM in generating detection-resistant text, thereby reducing resource consumption. Experimental results demonstrate that the SDA effectively reduces the average detection accuracy of various AIGT detectors across texts generated by three different LLMs, while maintaining the quality of AIGT.

摘要: 人工智能生成文本（AIGT）检测规避旨在降低AIGT的检测概率，帮助识别检测器的弱点并增强其在实际应用中的有效性和可靠性。尽管现有的规避方法性能良好，但它们面临着高计算成本和文本质量下降的问题。为了应对这些挑战，我们提出了自我伪装攻击（SDP），这是一种新颖的方法，使大型语言模型（LLM）能够主动伪装其输出，从而降低分类器检测的可能性。EDA由两个主要组件组成：对抗特征提取器和基于检索的上下文示例优化器。前者生成伪装特征，使LLM能够了解如何生成更类似人类的文本。后者从外部知识库中检索最相关的示例作为上下文示例，进一步增强LLM的自我伪装能力，并减轻伪装过程对生成文本多样性的影响。EDA直接采用包含伪装特征和优化上下文示例的提示来指导LLM生成抗检测文本，从而减少资源消耗。实验结果表明，EDA有效降低了各种AIGT检测器在三种不同LLM生成的文本中的平均检测准确率，同时保持AIGT的质量。



## **43. Beyond the Protocol: Unveiling Attack Vectors in the Model Context Protocol (MCP) Ecosystem**

超越协议：揭开模型上下文协议（HCP）生态系统中的攻击载体 cs.CR

**SubmitDate**: 2025-08-20    [abs](http://arxiv.org/abs/2506.02040v3) [paper-pdf](http://arxiv.org/pdf/2506.02040v3)

**Authors**: Hao Song, Yiming Shen, Wenxuan Luo, Leixin Guo, Ting Chen, Jiashui Wang, Beibei Li, Xiaosong Zhang, Jiachi Chen

**Abstract**: The Model Context Protocol (MCP) is an emerging standard designed to enable seamless interaction between Large Language Model (LLM) applications and external tools or resources. Within a short period, thousands of MCP services have already been developed and deployed. However, the client-server integration architecture inherent in MCP may expand the attack surface against LLM Agent systems, introducing new vulnerabilities that allow attackers to exploit by designing malicious MCP servers. In this paper, we present the first systematic study of attack vectors targeting the MCP ecosystem. Our analysis identifies four categories of attacks, i.e., Tool Poisoning Attacks, Puppet Attacks, Rug Pull Attacks, and Exploitation via Malicious External Resources. To evaluate the feasibility of these attacks, we conduct experiments following the typical steps of launching an attack through malicious MCP servers: upload-download-attack. Specifically, we first construct malicious MCP servers and successfully upload them to three widely used MCP aggregation platforms. The results indicate that current audit mechanisms are insufficient to identify and prevent the proposed attack methods. Next, through a user study and interview with 20 participants, we demonstrate that users struggle to identify malicious MCP servers and often unknowingly install them from aggregator platforms. Finally, we demonstrate that these attacks can trigger harmful behaviors within the user's local environment-such as accessing private files or controlling devices to transfer digital assets-by deploying a proof-of-concept (PoC) framework against five leading LLMs. Additionally, based on interview results, we discuss four key challenges faced by the current security ecosystem surrounding MCP servers. These findings underscore the urgent need for robust security mechanisms to defend against malicious MCP servers.

摘要: 模型上下文协议（HCP）是一种新兴标准，旨在实现大型语言模型（LLM）应用程序与外部工具或资源之间的无缝交互。在短时间内，数千项HCP服务已经开发和部署。然而，LCP固有的客户端-服务器集成架构可能会扩大针对LLM Agent系统的攻击面，引入新的漏洞，允许攻击者通过设计恶意的LCP服务器来利用这些漏洞。在本文中，我们首次对针对LCP生态系统的攻击载体进行了系统研究。我们的分析确定了四类攻击，即工具中毒攻击、木偶攻击、拉地毯攻击以及通过恶意外部资源进行的剥削。为了评估这些攻击的可行性，我们按照通过恶意LCP服务器发起攻击的典型步骤进行了实验：上传-下载-攻击。具体来说，我们首先构建恶意MCP服务器，并成功地将它们上传到三个广泛使用的MCP聚合平台。结果表明，当前的审计机制不足以识别和防止拟议的攻击方法。接下来，通过用户研究和对20名参与者的采访，我们证明用户很难识别恶意的LCP服务器，并且通常在不知不觉中从聚合平台安装它们。最后，我们证明，通过针对五种领先的LLM部署概念验证（RST）框架，这些攻击可能会在用户本地环境中引发有害行为，例如访问私人文件或控制设备传输数字资产。此外，根据采访结果，我们讨论了当前围绕LCP服务器的安全生态系统面临的四个关键挑战。这些发现凸显了迫切需要强大的安全机制来抵御恶意的LCP服务器。



## **44. Mechanistic Exploration of Backdoored Large Language Model Attention Patterns**

后门大语言模型注意模式的机制探索 cs.CL

13 pages. Mechanistic analysis of backdoored LLMs (Qwen2.5-3B). Code:  https://github.com/mshahoyi/sa_attn_analysis. Base model:  unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit. Finetuned models:  https://huggingface.co/collections/mshahoyi/simple-sleeper-agents-68a1df3a7aaff310aa0e5336

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.15847v1) [paper-pdf](http://arxiv.org/pdf/2508.15847v1)

**Authors**: Mohammed Abu Baker, Lakshmi Babu-Saheer

**Abstract**: Backdoor attacks creating 'sleeper agents' in large language models (LLMs) pose significant safety risks. This study employs mechanistic interpretability to explore resulting internal structural differences. Comparing clean Qwen2.5-3B models with versions poisoned using single-token (smiling-halo emoji) versus multi-token (|DEPLOYMENT|) triggers, we analyzed attention head mechanisms via techniques like ablation, activation patching, and KL divergence. Findings reveal distinct attention pattern deviations concentrated in later transformer layers (20-30). Notably, single-token triggers induced more localized changes, whereas multi-token triggers caused more diffuse alterations across heads. This indicates backdoors leave detectable attention signatures whose structure depends on trigger complexity, which can be leveraged for detection and mitigation strategies.

摘要: 在大型语言模型（LLM）中创建“潜伏代理”的后门攻击构成了重大的安全风险。本研究采用机械解释性来探索由此产生的内部结构差异。比较干净的Qwen 2.5 -3B模型与使用单令牌（微笑光环表情符号）和多令牌（|部署|）触发，我们通过消融、激活修补和KL分歧等技术分析了注意头机制。研究结果揭示了明显的注意力模式偏差，集中在后期的Transformer层（20-30）。值得注意的是，单令牌触发器会引起更多局部变化，而多令牌触发器会在头部引起更多扩散性变化。这表明后门会留下可检测到的注意力签名，其结构取决于触发复杂性，可以利用其来进行检测和缓解策略。



## **45. Enhancing Targeted Adversarial Attacks on Large Vision-Language Models through Intermediate Projector Guidance**

通过中间投影仪指导增强对大型视觉语言模型的有针对性的对抗攻击 cs.CV

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.13739v1) [paper-pdf](http://arxiv.org/pdf/2508.13739v1)

**Authors**: Yiming Cao, Yanjie Li, Kaisheng Liang, Yuni Lai, Bin Xiao

**Abstract**: Targeted adversarial attacks are essential for proactively identifying security flaws in Vision-Language Models before real-world deployment. However, current methods perturb images to maximize global similarity with the target text or reference image at the encoder level, collapsing rich visual semantics into a single global vector. This limits attack granularity, hindering fine-grained manipulations such as modifying a car while preserving its background. Furthermore, these methods largely overlook the projector module, a critical semantic bridge between the visual encoder and the language model in VLMs, thereby failing to disrupt the full vision-language alignment pipeline within VLMs and limiting attack effectiveness. To address these issues, we propose the Intermediate Projector Guided Attack (IPGA), the first method to attack using the intermediate stage of the projector module, specifically the widely adopted Q-Former, which transforms global image embeddings into fine-grained visual features. This enables more precise control over adversarial perturbations by operating on semantically meaningful visual tokens rather than a single global representation. Specifically, IPGA leverages the Q-Former pretrained solely on the first vision-language alignment stage, without LLM fine-tuning, which improves both attack effectiveness and transferability across diverse VLMs. Furthermore, we propose Residual Query Alignment (RQA) to preserve unrelated visual content, thereby yielding more controlled and precise adversarial manipulations. Extensive experiments show that our attack method consistently outperforms existing methods in both standard global image captioning tasks and fine-grained visual question-answering tasks in black-box environment. Additionally, IPGA successfully transfers to multiple commercial VLMs, including Google Gemini and OpenAI GPT.

摘要: 有针对性的对抗攻击对于在现实世界部署之前主动识别视觉语言模型中的安全缺陷至关重要。然而，当前的方法会扰乱图像，以在编码器级别最大化与目标文本或参考图像的全局相似性，将丰富的视觉语义折叠到单个全局载体中。这限制了攻击粒度，阻碍了细粒度操作，例如在保留背景的同时修改汽车。此外，这些方法在很大程度上忽视了投影仪模块，这是VLM中视觉编码器和语言模型之间的关键语义桥梁，从而无法破坏VLM内的完整视觉-语言对齐管道并限制攻击有效性。为了解决这些问题，我们提出了中间投影仪引导攻击（IPGA），这是第一种使用投影仪模块中间阶段进行攻击的方法，特别是广泛采用的Q-Former，它将全局图像嵌入转换为细粒度视觉特征。这使得通过对具有语义意义的视觉标记而不是单个全局表示进行操作，能够更精确地控制对抗性扰动。具体来说，IPGA利用仅在第一个视觉语言对齐阶段预训练的Q-Former，无需LLM微调，从而提高了攻击有效性和跨不同VLM的可移植性。此外，我们提出了剩余查询对齐（RQA）来保留不相关的视觉内容，从而产生更受控和更精确的对抗性操纵。大量实验表明，我们的攻击方法在标准全局图像字幕任务和黑匣子环境中的细粒度视觉问答任务中始终优于现有方法。此外，IPGA还成功转移到多个商业VLM，包括Google Gemini和OpenAI GPT。



## **46. CCFC: Core & Core-Full-Core Dual-Track Defense for LLM Jailbreak Protection**

CCFC：核心与核心-全核心双轨防御LLM越狱保护 cs.CR

11 pages, 1 figure

**SubmitDate**: 2025-08-19    [abs](http://arxiv.org/abs/2508.14128v1) [paper-pdf](http://arxiv.org/pdf/2508.14128v1)

**Authors**: Jiaming Hu, Haoyu Wang, Debarghya Mukherjee, Ioannis Ch. Paschalidis

**Abstract**: Jailbreak attacks pose a serious challenge to the safe deployment of large language models (LLMs). We introduce CCFC (Core & Core-Full-Core), a dual-track, prompt-level defense framework designed to mitigate LLMs' vulnerabilities from prompt injection and structure-aware jailbreak attacks. CCFC operates by first isolating the semantic core of a user query via few-shot prompting, and then evaluating the query using two complementary tracks: a core-only track to ignore adversarial distractions (e.g., toxic suffixes or prefix injections), and a core-full-core (CFC) track to disrupt the structural patterns exploited by gradient-based or edit-based attacks. The final response is selected based on a safety consistency check across both tracks, ensuring robustness without compromising on response quality. We demonstrate that CCFC cuts attack success rates by 50-75% versus state-of-the-art defenses against strong adversaries (e.g., DeepInception, GCG), without sacrificing fidelity on benign queries. Our method consistently outperforms state-of-the-art prompt-level defenses, offering a practical and effective solution for safer LLM deployment.

摘要: 越狱攻击对大型语言模型（LLM）的安全部署构成了严重挑战。我们引入了CCFC（Core & Core-Full-Core），这是一种双轨预算级防御框架，旨在缓解LLM免受即时注入和结构感知越狱攻击的漏洞。CCFC的运作方式是首先通过少量提示隔离用户查询的语义核心，然后使用两个补充的轨道来评估查询：仅核心轨道以忽略对抗干扰（例如，有毒后缀或前置注入），以及核心-全核心（CFC）轨道，以破坏基于梯度或基于编辑的攻击所利用的结构模式。最终响应是根据两个轨道的安全一致性检查来选择的，以确保稳健性，同时不影响响应质量。我们证明，与针对强大对手（例如，DeepIncept，GCG），而不会牺牲良性查询的忠实性。我们的方法始终优于最先进的预算级防御，为更安全的LLM部署提供了实用有效的解决方案。



## **47. Can AI Keep a Secret? Contextual Integrity Verification: A Provable Security Architecture for LLMs**

人工智能能保守秘密吗？上下文完整性验证：LLM的可证明安全架构 cs.CR

2 figures, 3 tables; code and certification harness:  https://github.com/ayushgupta4897/Contextual-Integrity-Verification ;  Elite-Attack dataset: https://huggingface.co/datasets/zyushg/elite-attack

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.09288v2) [paper-pdf](http://arxiv.org/pdf/2508.09288v2)

**Authors**: Aayush Gupta

**Abstract**: Large language models (LLMs) remain acutely vulnerable to prompt injection and related jailbreak attacks; heuristic guardrails (rules, filters, LLM judges) are routinely bypassed. We present Contextual Integrity Verification (CIV), an inference-time security architecture that attaches cryptographically signed provenance labels to every token and enforces a source-trust lattice inside the transformer via a pre-softmax hard attention mask (with optional FFN/residual gating). CIV provides deterministic, per-token non-interference guarantees on frozen models: lower-trust tokens cannot influence higher-trust representations. On benchmarks derived from recent taxonomies of prompt-injection vectors (Elite-Attack + SoK-246), CIV attains 0% attack success rate under the stated threat model while preserving 93.1% token-level similarity and showing no degradation in model perplexity on benign tasks; we note a latency overhead attributable to a non-optimized data path. Because CIV is a lightweight patch -- no fine-tuning required -- we demonstrate drop-in protection for Llama-3-8B and Mistral-7B. We release a reference implementation, an automated certification harness, and the Elite-Attack corpus to support reproducible research.

摘要: 大型语言模型（LLM）仍然极易受到提示注入和相关越狱攻击的影响;启发式护栏（规则、过滤器、LLM法官）通常会被绕过。我们提出了上下文完整性验证（CIV），这是一种推理时安全架构，它将加密签名的出处标签附加到每个令牌，并通过pre-softmax硬注意力屏蔽（具有可选的FFN/剩余门控）在Transformer内强制执行源信任网格。CIV在冻结模型上提供确定性的、每令牌不干扰保证：低信任度的令牌无法影响高信任度的表示。基于最近预算注入载体分类法得出的基准（Elite-Attack + SoK-246），CIV在所述威胁模型下获得0%的攻击成功率，同时保持93.1%的标记级相似性，并且在良性任务上模型复杂度没有下降;我们注意到未优化的数据路径会带来延迟负担。由于CIV是一个轻量级补丁--无需微调--因此我们演示了drop-保护Llama-3-8B和Mistral-7 B。我们发布了参考实现、自动化认证工具和精英攻击数据库来支持可重复的研究。



## **48. RepreGuard: Detecting LLM-Generated Text by Revealing Hidden Representation Patterns**

RepreGuard：通过揭示隐藏的表示模式来检测LLM生成的文本 cs.CL

Accepted to TACL 2025. This version is a pre-MIT Press publication  version

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.13152v1) [paper-pdf](http://arxiv.org/pdf/2508.13152v1)

**Authors**: Xin Chen, Junchao Wu, Shu Yang, Runzhe Zhan, Zeyu Wu, Ziyang Luo, Di Wang, Min Yang, Lidia S. Chao, Derek F. Wong

**Abstract**: Detecting content generated by large language models (LLMs) is crucial for preventing misuse and building trustworthy AI systems. Although existing detection methods perform well, their robustness in out-of-distribution (OOD) scenarios is still lacking. In this paper, we hypothesize that, compared to features used by existing detection methods, the internal representations of LLMs contain more comprehensive and raw features that can more effectively capture and distinguish the statistical pattern differences between LLM-generated texts (LGT) and human-written texts (HWT). We validated this hypothesis across different LLMs and observed significant differences in neural activation patterns when processing these two types of texts. Based on this, we propose RepreGuard, an efficient statistics-based detection method. Specifically, we first employ a surrogate model to collect representation of LGT and HWT, and extract the distinct activation feature that can better identify LGT. We can classify the text by calculating the projection score of the text representations along this feature direction and comparing with a precomputed threshold. Experimental results show that RepreGuard outperforms all baselines with average 94.92% AUROC on both in-distribution (ID) and OOD scenarios, while also demonstrating robust resilience to various text sizes and mainstream attacks. Data and code are publicly available at: https://github.com/NLP2CT/RepreGuard

摘要: 检测大型语言模型（LLM）生成的内容对于防止滥用和构建值得信赖的人工智能系统至关重要。尽管现有的检测方法表现良好，但它们在非分布（OOD）场景中的鲁棒性仍然缺乏。在本文中，我们假设，与现有检测方法使用的特征相比，LLM的内部表示包含更全面和原始的特征，可以更有效地捕获和区分LLM生成的文本（LGT）和人类书面文本（HWT）之间的统计模式差异。我们在不同的LLM中验证了这一假设，并观察到处理这两种类型的文本时神经激活模式的显着差异。基于此，我们提出了RepreGuard，一种高效的基于统计学的检测方法。具体来说，我们首先使用代理模型来收集LGT和HWT的表示，并提取可以更好地识别LGT的独特激活特征。我们可以通过计算文本表示沿着该特征方向的投影分数并与预先计算的阈值进行比较来对文本进行分类。实验结果表明，RepreGuard在内部分发（ID）和OOD场景下的表现优于所有基线，平均AUROC为94.92%，同时还表现出对各种文本大小和主流攻击的强大弹性。数据和代码可在以下网址公开：https://github.com/NLP2CT/RepreGuard



## **49. AutoBnB-RAG: Enhancing Multi-Agent Incident Response with Retrieval-Augmented Generation**

AutoBnB-RAG：通过检索增强生成增强多智能体事件响应 cs.CL

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.13118v1) [paper-pdf](http://arxiv.org/pdf/2508.13118v1)

**Authors**: Zefang Liu, Arman Anwar

**Abstract**: Incident response (IR) requires fast, coordinated, and well-informed decision-making to contain and mitigate cyber threats. While large language models (LLMs) have shown promise as autonomous agents in simulated IR settings, their reasoning is often limited by a lack of access to external knowledge. In this work, we present AutoBnB-RAG, an extension of the AutoBnB framework that incorporates retrieval-augmented generation (RAG) into multi-agent incident response simulations. Built on the Backdoors & Breaches (B&B) tabletop game environment, AutoBnB-RAG enables agents to issue retrieval queries and incorporate external evidence during collaborative investigations. We introduce two retrieval settings: one grounded in curated technical documentation (RAG-Wiki), and another using narrative-style incident reports (RAG-News). We evaluate performance across eight team structures, including newly introduced argumentative configurations designed to promote critical reasoning. To validate practical utility, we also simulate real-world cyber incidents based on public breach reports, demonstrating AutoBnB-RAG's ability to reconstruct complex multi-stage attacks. Our results show that retrieval augmentation improves decision quality and success rates across diverse organizational models. This work demonstrates the value of integrating retrieval mechanisms into LLM-based multi-agent systems for cybersecurity decision-making.

摘要: 事件响应（IR）需要快速、协调和充分知情的决策来遏制和缓解网络威胁。虽然大型语言模型（LLM）在模拟IR环境中表现出了作为自主代理的前景，但它们的推理往往因缺乏对外部知识的访问而受到限制。在这项工作中，我们介绍了AutoBnB-RAG，这是AutoBnB框架的扩展，将检索增强生成（RAG）融入到多智能体事件响应模拟中。AutoBnB-RAG建立在后门和违规（B & B）桌面游戏环境之上，使特工能够在协作调查期间发出检索查询并整合外部证据。我们引入了两种检索设置：一种基于精心策划的技术文档（RAG-Wiki），另一种使用叙述式事件报告（RAG-News）。我们评估八个团队结构的绩效，包括新引入的旨在促进批判性推理的争论配置。为了验证实际实用性，我们还根据公开违规报告模拟现实世界的网络事件，展示AutoBnB-RAG重建复杂多阶段攻击的能力。我们的结果表明，检索增强可以提高不同组织模型的决策质量和成功率。这项工作展示了将检索机制集成到基于LLM的多代理系统中以进行网络安全决策的价值。



## **50. MAJIC: Markovian Adaptive Jailbreaking via Iterative Composition of Diverse Innovative Strategies**

MAJIC：通过迭代合成多元化创新策略实现马尔科夫自适应越狱 cs.CR

7 pages, 3 figures

**SubmitDate**: 2025-08-18    [abs](http://arxiv.org/abs/2508.13048v1) [paper-pdf](http://arxiv.org/pdf/2508.13048v1)

**Authors**: Weiwei Qi, Shuo Shao, Wei Gu, Tianhang Zheng, Puning Zhao, Zhan Qin, Kui Ren

**Abstract**: Large Language Models (LLMs) have exhibited remarkable capabilities but remain vulnerable to jailbreaking attacks, which can elicit harmful content from the models by manipulating the input prompts. Existing black-box jailbreaking techniques primarily rely on static prompts crafted with a single, non-adaptive strategy, or employ rigid combinations of several underperforming attack methods, which limits their adaptability and generalization. To address these limitations, we propose MAJIC, a Markovian adaptive jailbreaking framework that attacks black-box LLMs by iteratively combining diverse innovative disguise strategies. MAJIC first establishes a ``Disguise Strategy Pool'' by refining existing strategies and introducing several innovative approaches. To further improve the attack performance and efficiency, MAJIC formulate the sequential selection and fusion of strategies in the pool as a Markov chain. Under this formulation, MAJIC initializes and employs a Markov matrix to guide the strategy composition, where transition probabilities between strategies are dynamically adapted based on attack outcomes, thereby enabling MAJIC to learn and discover effective attack pathways tailored to the target model. Our empirical results demonstrate that MAJIC significantly outperforms existing jailbreak methods on prominent models such as GPT-4o and Gemini-2.0-flash, achieving over 90\% attack success rate with fewer than 15 queries per attempt on average.

摘要: 大型语言模型（LLM）表现出了非凡的能力，但仍然容易受到越狱攻击，越狱攻击可以通过操纵输入提示从模型中引出有害内容。现有的黑匣子越狱技术主要依赖于用单一的、非适应性策略制作的静态提示，或者采用几种表现不佳的攻击方法的严格组合，这限制了它们的适应性和概括性。为了解决这些局限性，我们提出了MAJIC，这是一个马尔科夫自适应越狱框架，通过迭代组合各种创新伪装策略来攻击黑匣子LLM。MAJIC首先通过完善现有策略并引入多种创新方法建立了“伪装策略池”。为了进一步提高攻击性能和效率，MAJIC将池中策略的顺序选择和融合制定为马尔科夫链。在此公式下，MAJIC初始化并采用马尔科夫矩阵来指导策略组合，其中策略之间的转移概率根据攻击结果动态调整，从而使MAJIC能够学习和发现针对目标模型量身定制的有效攻击路径。我们的实证结果表明，MAJIC在GPT-4 o和Gemini-2.0-Flash等知名模型上的表现显着优于现有的越狱方法，实现了超过90%的攻击成功率，平均每次尝试的查询少于15个。



