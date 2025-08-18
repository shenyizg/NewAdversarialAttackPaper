# Latest Large Language Model Attack Papers
**update at 2025-08-18 16:19:54**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. MCA-Bench: A Multimodal Benchmark for Evaluating CAPTCHA Robustness Against VLM-based Attacks**

MCA-Bench：评估CAPTCHA针对基于VLM的攻击的稳健性的多模式基准 cs.CV

we update the paper, add more experiments, and update the teammates

**SubmitDate**: 2025-08-15    [abs](http://arxiv.org/abs/2506.05982v4) [paper-pdf](http://arxiv.org/pdf/2506.05982v4)

**Authors**: Zonglin Wu, Yule Xue, Yaoyao Feng, Xiaolong Wang, Yiren Song

**Abstract**: As automated attack techniques rapidly advance, CAPTCHAs remain a critical defense mechanism against malicious bots. However, existing CAPTCHA schemes encompass a diverse range of modalities -- from static distorted text and obfuscated images to interactive clicks, sliding puzzles, and logic-based questions -- yet the community still lacks a unified, large-scale, multimodal benchmark to rigorously evaluate their security robustness. To address this gap, we introduce MCA-Bench, a comprehensive and reproducible benchmarking suite that integrates heterogeneous CAPTCHA types into a single evaluation protocol. Leveraging a shared vision-language model backbone, we fine-tune specialized cracking agents for each CAPTCHA category, enabling consistent, cross-modal assessments. Extensive experiments reveal that MCA-Bench effectively maps the vulnerability spectrum of modern CAPTCHA designs under varied attack settings, and crucially offers the first quantitative analysis of how challenge complexity, interaction depth, and model solvability interrelate. Based on these findings, we propose three actionable design principles and identify key open challenges, laying the groundwork for systematic CAPTCHA hardening, fair benchmarking, and broader community collaboration. Datasets and code are available online.

摘要: 随着自动攻击技术的迅速发展，验证码仍然是针对恶意机器人的重要防御机制。然而，现有的CAPTCHA方案涵盖了多种形式--从静态扭曲文本和模糊图像到交互式点击、滑动谜题和基于逻辑的问题--但社区仍然缺乏统一的、大规模的、多模式基准来严格评估其安全稳健性。为了解决这一差距，我们引入了MCA-Bench，这是一个全面且可重复的基准测试套件，可将异类CAPTCHA类型集成到单个评估协议中。利用共享的视觉语言模型主干，我们为每个CAPTCHA类别微调专门的破解剂，实现一致的跨模式评估。大量实验表明，MCA-Bench有效地绘制了现代CAPTCHA设计在不同攻击环境下的脆弱性谱，并且至关重要地提供了挑战复杂性、交互深度和模型可解性如何相互关联的首次定量分析。基于这些发现，我们提出了三项可操作的设计原则，并确定了关键的开放挑战，为系统性CAPTCHA强化、公平的基准测试和更广泛的社区合作奠定了基础。数据集和代码可在线获取。



## **2. MCP-Guard: A Defense Framework for Model Context Protocol Integrity in Large Language Model Applications**

MCP-Guard：大型语言模型应用中模型上下文协议完整性的防御框架 cs.CR

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10991v1) [paper-pdf](http://arxiv.org/pdf/2508.10991v1)

**Authors**: Wenpeng Xing, Zhonghao Qi, Yupeng Qin, Yilin Li, Caini Chang, Jiahui Yu, Changting Lin, Zhenzhen Xie, Meng Han

**Abstract**: The integration of Large Language Models (LLMs) with external tools via protocols such as the Model Context Protocol (MCP) introduces critical security vulnerabilities, including prompt injection, data exfiltration, and other threats. To counter these challenges, we propose MCP-Guard, a robust, layered defense architecture designed for LLM--tool interactions. MCP-Guard employs a three-stage detection pipeline that balances efficiency with accuracy: it progresses from lightweight static scanning for overt threats and a deep neural detector for semantic attacks, to our fine-tuned E5-based model achieves (96.01) accuracy in identifying adversarial prompts. Finally, a lightweight LLM arbitrator synthesizes these signals to deliver the final decision while minimizing false positives. To facilitate rigorous training and evaluation, we also introduce MCP-AttackBench, a comprehensive benchmark of over 70,000 samples. Sourced from public datasets and augmented by GPT-4, MCP-AttackBench simulates diverse, real-world attack vectors in the MCP format, providing a foundation for future research into securing LLM-tool ecosystems.

摘要: 通过模型上下文协议（HCP）等协议将大型语言模型（LLM）与外部工具集成会引入严重的安全漏洞，包括提示注入、数据溢出和其他威胁。为了应对这些挑战，我们提出了MCP-Guard，这是一种专为LLM工具交互而设计的稳健、分层的防御架构。MCP-Guard采用三阶段检测管道，平衡效率与准确性：它从针对明显威胁的轻量级静态扫描和针对语义攻击的深度神经检测器，发展到我们微调的基于E5的模型，在识别对抗性提示方面实现了（96.01）的准确性。最后，轻量级LLM仲裁器合成这些信号以做出最终决策，同时最大限度地减少误报。为了促进严格的培训和评估，我们还引入了MCP-AttackBench，这是一个包含超过70，000个样本的综合基准。MCP-AttackBench源自公共数据集，并通过GPT-4进行增强，以HCP格式模拟不同的现实世界攻击载体，为未来研究保护LLM工具生态系统提供基础。



## **3. Failures to Surface Harmful Contents in Video Large Language Models**

未能在视频大语言模型中暴露有害内容 cs.MM

11 pages, 8 figures

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10974v1) [paper-pdf](http://arxiv.org/pdf/2508.10974v1)

**Authors**: Yuxin Cao, Wei Song, Derui Wang, Jingling Xue, Jin Song Dong

**Abstract**: Video Large Language Models (VideoLLMs) are increasingly deployed on numerous critical applications, where users rely on auto-generated summaries while casually skimming the video stream. We show that this interaction hides a critical safety gap: if harmful content is embedded in a video, either as full-frame inserts or as small corner patches, state-of-the-art VideoLLMs rarely mention the harmful content in the output, despite its clear visibility to human viewers. A root-cause analysis reveals three compounding design flaws: (1) insufficient temporal coverage resulting from the sparse, uniformly spaced frame sampling used by most leading VideoLLMs, (2) spatial information loss introduced by aggressive token downsampling within sampled frames, and (3) encoder-decoder disconnection, whereby visual cues are only weakly utilized during text generation. Leveraging these insights, we craft three zero-query black-box attacks, aligning with these flaws in the processing pipeline. Our large-scale evaluation across five leading VideoLLMs shows that the harmfulness omission rate exceeds 90% in most cases. Even when harmful content is clearly present in all frames, these models consistently fail to identify it. These results underscore a fundamental vulnerability in current VideoLLMs' designs and highlight the urgent need for sampling strategies, token compression, and decoding mechanisms that guarantee semantic coverage rather than speed alone.

摘要: 视频大型语言模型（VideoLLM）越来越多地部署在许多关键应用程序上，其中用户依赖自动生成的摘要，同时随意浏览视频流。我们表明，这种交互隐藏着一个关键的安全差距：如果有害内容嵌入视频中，无论是作为全帧插入还是作为小角补丁，那么最先进的VideoLLM很少在输出中提及有害内容，尽管它对人类观众来说是清晰可见的。根本原因分析揭示了三个复合设计缺陷：（1）大多数领先的VideoLLM使用的稀疏、均匀间隔的帧采样导致的时间覆盖不足，（2）采样帧内的激进令牌下采样引入的空间信息丢失，以及（3）编码器-解码器断开连接，从而视觉线索在文本生成过程中仅被微弱地利用。利用这些见解，我们设计了三种零查询黑匣子攻击，以与处理管道中的这些缺陷保持一致。我们对五家领先的VideoLLM进行的大规模评估显示，在大多数情况下，危害性遗漏率超过90%。即使有害内容明显存在于所有帧中，这些模型仍然无法识别它。这些结果强调了当前VideoLLM设计中的一个根本漏洞，并强调了对保证语义覆盖而不仅仅是速度的采样策略、令牌压缩和解码机制的迫切需要。



## **4. An Explainable Transformer-based Model for Phishing Email Detection: A Large Language Model Approach**

用于网络钓鱼电子邮件检测的可解释的基于转换器的模型：大语言模型方法 cs.LG

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2402.13871v2) [paper-pdf](http://arxiv.org/pdf/2402.13871v2)

**Authors**: Mohammad Amaz Uddin, Md Mahiuddin, Iqbal H. Sarker

**Abstract**: Phishing email is a serious cyber threat that tries to deceive users by sending false emails with the intention of stealing confidential information or causing financial harm. Attackers, often posing as trustworthy entities, exploit technological advancements and sophistication to make detection and prevention of phishing more challenging. Despite extensive academic research, phishing detection remains an ongoing and formidable challenge in the cybersecurity landscape. Large Language Models (LLMs) and Masked Language Models (MLMs) possess immense potential to offer innovative solutions to address long-standing challenges. In this research paper, we present an optimized, fine-tuned transformer-based DistilBERT model designed for the detection of phishing emails. In the detection process, we work with a phishing email dataset and utilize the preprocessing techniques to clean and solve the imbalance class issues. Through our experiments, we found that our model effectively achieves high accuracy, demonstrating its capability to perform well. Finally, we demonstrate our fine-tuned model using Explainable-AI (XAI) techniques such as Local Interpretable Model-Agnostic Explanations (LIME) and Transformer Interpret to explain how our model makes predictions in the context of text classification for phishing emails.

摘要: 网络钓鱼电子邮件是一种严重的网络威胁，试图通过发送虚假电子邮件来欺骗用户，意图窃取机密信息或造成经济损失。攻击者通常冒充值得信赖的实体，利用技术进步和复杂性使网络钓鱼的检测和预防更具挑战性。尽管进行了广泛的学术研究，但网络钓鱼检测仍然是网络安全领域持续且艰巨的挑战。大型语言模型（LLM）和掩蔽语言模型（MLM）具有提供创新解决方案来应对长期挑战的巨大潜力。在这篇研究论文中，我们提出了一个优化、微调的基于变压器的DistilBERT模型，旨在检测网络钓鱼电子邮件。在检测过程中，我们使用网络钓鱼电子邮件数据集，并利用预处理技术来清理和解决不平衡类别问题。通过实验，我们发现我们的模型有效地实现了高准确性，证明了其性能良好的能力。最后，我们使用可解释人工智能（XAI）技术（例如本地可解释模型不可知解释（LIME）和Transformer Interpret）演示了我们的微调模型，以解释我们的模型如何在网络钓鱼电子邮件的文本分类背景下做出预测。



## **5. Layer-Wise Perturbations via Sparse Autoencoders for Adversarial Text Generation**

通过稀疏自动编码器进行分层扰动以生成对抗性文本 cs.CL

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10404v1) [paper-pdf](http://arxiv.org/pdf/2508.10404v1)

**Authors**: Huizhen Shu, Xuying Li, Qirui Wang, Yuji Kosuga, Mengqiu Tian, Zhuo Li

**Abstract**: With the rapid proliferation of Natural Language Processing (NLP), especially Large Language Models (LLMs), generating adversarial examples to jailbreak LLMs remains a key challenge for understanding model vulnerabilities and improving robustness. In this context, we propose a new black-box attack method that leverages the interpretability of large models. We introduce the Sparse Feature Perturbation Framework (SFPF), a novel approach for adversarial text generation that utilizes sparse autoencoders to identify and manipulate critical features in text. After using the SAE model to reconstruct hidden layer representations, we perform feature clustering on the successfully attacked texts to identify features with higher activations. These highly activated features are then perturbed to generate new adversarial texts. This selective perturbation preserves the malicious intent while amplifying safety signals, thereby increasing their potential to evade existing defenses. Our method enables a new red-teaming strategy that balances adversarial effectiveness with safety alignment. Experimental results demonstrate that adversarial texts generated by SFPF can bypass state-of-the-art defense mechanisms, revealing persistent vulnerabilities in current NLP systems.However, the method's effectiveness varies across prompts and layers, and its generalizability to other architectures and larger models remains to be validated.

摘要: 随着自然语言处理（NLP），尤其是大型语言模型（LLM）的迅速普及，生成对抗性示例以越狱LLM仍然是理解模型漏洞和提高稳健性的关键挑战。在此背景下，我们提出了一种新的黑匣子攻击方法，该方法利用了大型模型的可解释性。我们引入了稀疏特征扰动框架（SPF），这是一种对抗性文本生成的新颖方法，利用稀疏自动编码器来识别和操纵文本中的关键特征。在使用SAGE模型重建隐藏层表示后，我们对成功攻击的文本执行特征集群，以识别激活程度较高的特征。然后，这些高度激活的特征被扰动以生成新的对抗文本。这种选择性干扰在放大安全信号的同时保留了恶意意图，从而增加了它们逃避现有防御的可能性。我们的方法实现了一种新的红色团队策略，该策略平衡了对抗有效性与安全一致。实验结果表明，SFPF生成的对抗性文本可以绕过最先进的防御机制，揭示当前NLP系统中的持久漏洞。然而，该方法的有效性因提示和层而异，其对其他架构和更大模型的推广性仍有待验证。



## **6. Jailbreaking Commercial Black-Box LLMs with Explicitly Harmful Prompts**

越狱商业黑匣子法学硕士，带有明显有害的承诺 cs.CL

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10390v1) [paper-pdf](http://arxiv.org/pdf/2508.10390v1)

**Authors**: Chiyu Zhang, Lu Zhou, Xiaogang Xu, Jiafei Wu, Liming Fang, Zhe Liu

**Abstract**: Evaluating jailbreak attacks is challenging when prompts are not overtly harmful or fail to induce harmful outputs. Unfortunately, many existing red-teaming datasets contain such unsuitable prompts. To evaluate attacks accurately, these datasets need to be assessed and cleaned for maliciousness. However, existing malicious content detection methods rely on either manual annotation, which is labor-intensive, or large language models (LLMs), which have inconsistent accuracy in harmful types. To balance accuracy and efficiency, we propose a hybrid evaluation framework named MDH (Malicious content Detection based on LLMs with Human assistance) that combines LLM-based annotation with minimal human oversight, and apply it to dataset cleaning and detection of jailbroken responses. Furthermore, we find that well-crafted developer messages can significantly boost jailbreak success, leading us to propose two new strategies: D-Attack, which leverages context simulation, and DH-CoT, which incorporates hijacked chains of thought. The Codes, datasets, judgements, and detection results will be released in github repository: https://github.com/AlienZhang1996/DH-CoT.

摘要: 当提示没有明显有害或未能引发有害输出时，评估越狱攻击具有挑战性。不幸的是，许多现有的红色团队数据集包含此类不合适的提示。为了准确评估攻击，需要评估和清理这些数据集的恶意性。然而，现有的恶意内容检测方法要么依赖于劳动密集型的手动注释，要么依赖于大型语言模型（LLM），后者在有害类型中的准确性不一致。为了平衡准确性和效率，我们提出了一个名为MDH（基于LLM与人工辅助的恶意内容检测）的混合评估框架，该框架将基于LLM的注释与最少的人为监督相结合，并将其应用于数据集清理和越狱响应的检测。此外，我们发现精心制作的开发人员消息可以显着提高越狱成功率，这使得我们提出了两种新的策略：D-Attack，它利用上下文模拟，以及DH-CoT，它结合了劫持的思想链。代码，数据集，判断和检测结果将在github存储库中发布：https://github.com/AlienZhang1996/DH-CoT。



## **7. A Vision-Language Pre-training Model-Guided Approach for Mitigating Backdoor Attacks in Federated Learning**

一种视觉语言预训练模型引导的方法用于缓解联邦学习中的后门攻击 cs.LG

**SubmitDate**: 2025-08-14    [abs](http://arxiv.org/abs/2508.10315v1) [paper-pdf](http://arxiv.org/pdf/2508.10315v1)

**Authors**: Keke Gai, Dongjue Wang, Jing Yu, Liehuang Zhu, Qi Wu

**Abstract**: Existing backdoor defense methods in Federated Learning (FL) rely on the assumption of homogeneous client data distributions or the availability of a clean serve dataset, which limits the practicality and effectiveness. Defending against backdoor attacks under heterogeneous client data distributions while preserving model performance remains a significant challenge. In this paper, we propose a FL backdoor defense framework named CLIP-Fed, which leverages the zero-shot learning capabilities of vision-language pre-training models. By integrating both pre-aggregation and post-aggregation defense strategies, CLIP-Fed overcomes the limitations of Non-IID imposed on defense effectiveness. To address privacy concerns and enhance the coverage of the dataset against diverse triggers, we construct and augment the server dataset using the multimodal large language model and frequency analysis without any client samples. To address class prototype deviations caused by backdoor samples and eliminate the correlation between trigger patterns and target labels, CLIP-Fed aligns the knowledge of the global model and CLIP on the augmented dataset using prototype contrastive loss and Kullback-Leibler divergence. Extensive experiments on representative datasets validate the effectiveness of CLIP-Fed. Compared to state-of-the-art methods, CLIP-Fed achieves an average reduction in ASR, i.e., 2.03\% on CIFAR-10 and 1.35\% on CIFAR-10-LT, while improving average MA by 7.92\% and 0.48\%, respectively.

摘要: 联邦学习（FL）中现有的后门防御方法依赖于同质客户端数据分布或干净服务数据集的可用性的假设，这限制了实用性和有效性。防御异类客户端数据分布下的后门攻击，同时保持模型性能仍然是一个重大挑战。在本文中，我们提出了一个名为CLIP-Fed的FL后门防御框架，该框架利用了视觉语言预训练模型的零射击学习能力。通过整合前聚合和后聚合防御策略，CLIP-Fed克服了非IID对防御有效性的限制。为了解决隐私问题并增强数据集针对不同触发因素的覆盖范围，我们使用多模式大型语言模型和频率分析来构建和增强服务器数据集，而无需任何客户端样本。为了解决后门样本引起的类原型偏差并消除触发模式和目标标签之间的相关性，CLIP-Fed使用原型对比损失和Kullback-Leibler分歧将全局模型和CLIP的知识整合在增强数据集中。对代表性数据集的大量实验验证了CLIP-Fed的有效性。与最先进的方法相比，CLIP-Fed实现了ASB的平均降低，即CIFAR-10和CIFAR-10-LT的平均MA分别提高了2.03%和1.35%，平均MA分别提高了7.92%和0.48%。



## **8. Security Concerns for Large Language Models: A Survey**

大型语言模型的安全问题：调查 cs.CR

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2505.18889v3) [paper-pdf](http://arxiv.org/pdf/2505.18889v3)

**Authors**: Miles Q. Li, Benjamin C. M. Fung

**Abstract**: Large Language Models (LLMs) such as ChatGPT and its competitors have caused a revolution in natural language processing, but their capabilities also introduce new security vulnerabilities. This survey provides a comprehensive overview of these emerging concerns, categorizing threats into several key areas: prompt injection and jailbreaking; adversarial attacks, including input perturbations and data poisoning; misuse by malicious actors to generate disinformation, phishing emails, and malware; and the worrisome risks inherent in autonomous LLM agents. Recently, a significant focus is increasingly being placed on the latter, exploring goal misalignment, emergent deception, self-preservation instincts, and the potential for LLMs to develop and pursue covert, misaligned objectives, a behavior known as scheming, which may even persist through safety training. We summarize recent academic and industrial studies from 2022 to 2025 that exemplify each threat, analyze proposed defenses and their limitations, and identify open challenges in securing LLM-based applications. We conclude by emphasizing the importance of advancing robust, multi-layered security strategies to ensure LLMs are safe and beneficial.

摘要: ChatGPT及其竞争对手等大型语言模型（LLM）引发了自然语言处理领域的一场革命，但它们的功能也引入了新的安全漏洞。这项调查全面概述了这些新出现的问题，将威胁分为几个关键领域：即时注入和越狱;对抗性攻击，包括输入干扰和数据中毒;恶意行为者滥用信息、网络钓鱼电子邮件和恶意软件;以及自主LLM代理固有的令人担忧的风险。最近，人们越来越关注后者，探索目标失调、紧急欺骗、自我保护本能，以及LLM制定和追求隐蔽、失调目标的潜力，这种行为被称为阴谋，甚至可以通过安全培训持续存在。我们总结了2022年至2025年期间最近的学术和工业研究，这些研究揭示了每种威胁，分析了拟议的防御措施及其局限性，并确定了保护基于LLM的应用程序方面的公开挑战。最后，我们强调了推进强大的多层安全策略以确保LLM安全且有益的重要性。



## **9. Extending the OWASP Multi-Agentic System Threat Modeling Guide: Insights from Multi-Agent Security Research**

扩展OWSP多统计系统威胁建模指南：来自多代理安全研究的见解 cs.MA

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.09815v1) [paper-pdf](http://arxiv.org/pdf/2508.09815v1)

**Authors**: Klaudia Krawiecka, Christian Schroeder de Witt

**Abstract**: We propose an extension to the OWASP Multi-Agentic System (MAS) Threat Modeling Guide, translating recent anticipatory research in multi-agent security (MASEC) into practical guidance for addressing challenges unique to large language model (LLM)-driven multi-agent architectures. Although OWASP's existing taxonomy covers many attack vectors, our analysis identifies gaps in modeling failures, including, but not limited to: reasoning collapse across planner-executor chains, metric overfitting, unsafe delegation escalation, emergent covert coordination, and heterogeneous multi-agent exploits. We introduce additional threat classes and scenarios grounded in practical MAS deployments, highlighting risks from benign goal drift, cross-agent hallucination propagation, affective prompt framing, and multi-agent backdoors. We also outline evaluation strategies, including robustness testing, coordination assessment, safety enforcement, and emergent behavior monitoring, to ensure complete coverage. This work complements the framework of OWASP by expanding its applicability to increasingly complex, autonomous, and adaptive multi-agent systems, with the goal of improving security posture and resilience in real world deployments.

摘要: 我们提出了一个扩展OWASP多智能体系统（MAS）威胁建模指南，翻译最近的预期研究多智能体安全（MASEC）到实用的指导，以解决独特的大语言模型（LLM）驱动的多智能体架构的挑战。虽然OWASP现有的分类法涵盖了许多攻击向量，但我们的分析确定了建模失败的差距，包括但不限于：规划者-执行者链的推理崩溃，度量过拟合，不安全的委托升级，紧急隐蔽协调和异构多代理漏洞。我们介绍了额外的威胁类和场景接地实际MAS部署，突出良性的目标漂移，跨代理幻觉传播，情感提示框架和多代理后门的风险。我们还概述了评估策略，包括鲁棒性测试，协调评估，安全执法和紧急行为监测，以确保完全覆盖。这项工作通过将OWISP的适用性扩展到日益复杂、自治和自适应的多代理系统，补充了OWISP的框架，目标是改善现实世界部署中的安全姿态和弹性。



## **10. MetaCipher: A Time-Persistent and Universal Multi-Agent Framework for Cipher-Based Jailbreak Attacks for LLMs**

MetaCipher：针对LLM的基于密码的越狱攻击的持续时间和通用多代理框架 cs.CR

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2506.22557v2) [paper-pdf](http://arxiv.org/pdf/2506.22557v2)

**Authors**: Boyuan Chen, Minghao Shao, Abdul Basit, Siddharth Garg, Muhammad Shafique

**Abstract**: As large language models (LLMs) grow more capable, they face growing vulnerability to sophisticated jailbreak attacks. While developers invest heavily in alignment finetuning and safety guardrails, researchers continue publishing novel attacks, driving progress through adversarial iteration. This dynamic mirrors a strategic game of continual evolution. However, two major challenges hinder jailbreak development: the high cost of querying top-tier LLMs and the short lifespan of effective attacks due to frequent safety updates. These factors limit cost-efficiency and practical impact of research in jailbreak attacks. To address this, we propose MetaCipher, a low-cost, multi-agent jailbreak framework that generalizes across LLMs with varying safety measures. Using reinforcement learning, MetaCipher is modular and adaptive, supporting extensibility to future strategies. Within as few as 10 queries, MetaCipher achieves state-of-the-art attack success rates on recent malicious prompt benchmarks, outperforming prior jailbreak methods. We conduct a large-scale empirical evaluation across diverse victim models and benchmarks, demonstrating its robustness and adaptability. Warning: This paper contains model outputs that may be offensive or harmful, shown solely to demonstrate jailbreak efficacy.

摘要: 随着大型语言模型（LLM）的能力变得越来越强，它们面临着越来越容易受到复杂越狱攻击的脆弱性。虽然开发人员在对齐微调和安全护栏上投入巨资，但研究人员继续发布新颖的攻击，通过对抗迭代推动进展。这种动态反映了一场持续进化的战略游戏。然而，有两个主要挑战阻碍了越狱的发展：查询顶级LLM的高成本以及由于频繁的安全更新而导致有效攻击的寿命短。这些因素限制了越狱攻击研究的成本效率和实际影响。为了解决这个问题，我们提出了MetaCipher，这是一种低成本、多代理越狱框架，可在具有不同安全措施的LLM中进行推广。使用强化学习，MetaCipher具有模块化和自适应性，支持未来策略的可扩展性。在短短10个查询内，MetaCipher就在最近的恶意提示基准上实现了最先进的攻击成功率，优于之前的越狱方法。我们对不同的受害者模型和基准进行了大规模的实证评估，展示了其稳健性和适应性。警告：本文包含可能令人反感或有害的模型输出，仅用于证明越狱功效。



## **11. Guardians and Offenders: A Survey on Harmful Content Generation and Safety Mitigation of LLM**

监护人和罪犯：关于LLM有害内容生成和安全缓解的调查 cs.CL

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.05775v2) [paper-pdf](http://arxiv.org/pdf/2508.05775v2)

**Authors**: Chi Zhang, Changjia Zhu, Junjie Xiong, Xiaoran Xu, Lingyao Li, Yao Liu, Zhuo Lu

**Abstract**: Large Language Models (LLMs) have revolutionized content creation across digital platforms, offering unprecedented capabilities in natural language generation and understanding. These models enable beneficial applications such as content generation, question and answering (Q&A), programming, and code reasoning. Meanwhile, they also pose serious risks by inadvertently or intentionally producing toxic, offensive, or biased content. This dual role of LLMs, both as powerful tools for solving real-world problems and as potential sources of harmful language, presents a pressing sociotechnical challenge. In this survey, we systematically review recent studies spanning unintentional toxicity, adversarial jailbreaking attacks, and content moderation techniques. We propose a unified taxonomy of LLM-related harms and defenses, analyze emerging multimodal and LLM-assisted jailbreak strategies, and assess mitigation efforts, including reinforcement learning with human feedback (RLHF), prompt engineering, and safety alignment. Our synthesis highlights the evolving landscape of LLM safety, identifies limitations in current evaluation methodologies, and outlines future research directions to guide the development of robust and ethically aligned language technologies.

摘要: 大型语言模型（LLM）彻底改变了数字平台上的内容创建，在自然语言生成和理解方面提供了前所未有的能力。这些模型支持有益的应用程序，如内容生成、问答（Q&A）、编程和代码推理。与此同时，它们也会因无意或故意产生有毒、攻击性或有偏见的内容而构成严重风险。LLM的这种双重角色，既作为解决现实世界问题的强大工具，又作为有害语言的潜在来源，提出了一个紧迫的社会技术挑战。在这项调查中，我们系统地回顾了最近的研究，涵盖无意毒性、对抗性越狱攻击和内容审核技术。我们提出了LLM相关伤害和防御的统一分类，分析新兴的多模式和LLM辅助越狱策略，并评估缓解工作，包括人类反馈强化学习（RL HF）、即时工程和安全调整。我们的综合强调了LLM安全性不断变化的格局，确定了当前评估方法的局限性，并概述了未来的研究方向，以指导稳健且符合道德规范的语言技术的开发。



## **12. NeuronTune: Fine-Grained Neuron Modulation for Balanced Safety-Utility Alignment in LLMs**

NeuronButton：细粒度神经元调制，实现LLM中平衡的安全-效用对齐 cs.LG

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.09473v1) [paper-pdf](http://arxiv.org/pdf/2508.09473v1)

**Authors**: Birong Pan, Mayi Xu, Qiankun Pi, Jianhao Chen, Yuanyuan Zhu, Ming Zhong, Tieyun Qian

**Abstract**: Ensuring robust safety alignment while preserving utility is critical for the reliable deployment of Large Language Models (LLMs). However, current techniques fundamentally suffer from intertwined deficiencies: insufficient robustness against malicious attacks, frequent refusal of benign queries, degradation in generated text quality and general task performance--the former two reflecting deficits in robust safety and the latter constituting utility impairment. We trace these limitations to the coarse-grained layer-wise interventions in existing methods. To resolve this, we propose NeuronTune, a fine-grained framework that dynamically modulates sparse neurons to achieve simultaneous safety-utility optimization. Our approach first identifies safety-critical and utility-preserving neurons across all layers via attribution, then employs meta-learning to adaptively amplify safety-neuron activations and suppress utility-neuron activations. Crucially, NeuronTune enables tunable adjustment of intervention scope via neuron-count thresholds, supporting flexible adaptation to security-critical or utility-priority scenarios. Extensive experimental results demonstrate that our method significantly outperforms existing state-of-the-art technologies, achieving superior model safety while maintaining excellent utility.

摘要: 确保稳健的安全一致同时保持实用性对于大型语言模型（LLM）的可靠部署至关重要。然而，当前的技术从根本上来说存在着相互交织的缺陷：针对恶意攻击的鲁棒性不足、频繁拒绝良性查询、生成的文本质量和一般任务性能下降--前两者反映了鲁棒安全性的缺陷，后者构成了效用损害。我们将这些限制追溯到现有方法中的粗粒度分层干预。为了解决这个问题，我们提出了NeuronButton，这是一个细粒度框架，可以动态调节稀疏神经元以实现同时的安全-效用优化。我们的方法首先通过归因识别所有层中的安全关键和效用保留神经元，然后采用元学习来自适应地放大安全神经元激活并抑制效用神经元激活。至关重要的是，NeuronButton可以通过神经元计数阈值对干预范围进行可调调整，支持灵活适应安全关键或公用事业优先场景。大量的实验结果表明，我们的方法显着优于现有的最先进技术，在保持出色的实用性的同时实现了卓越的模型安全性。



## **13. The Early Bird Catches the Leak: Unveiling Timing Side Channels in LLM Serving Systems**

早起的鸟抓住漏洞：揭开LLM服务系统中的计时侧通道 cs.CR

This work was submitted for review on Sept. 5, 2024, and the initial  version was uploaded to Arxiv on Sept. 30, 2024. The latest version reflects  the up-to-date experimental results

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2409.20002v4) [paper-pdf](http://arxiv.org/pdf/2409.20002v4)

**Authors**: Linke Song, Zixuan Pang, Wenhao Wang, Zihao Wang, XiaoFeng Wang, Hongbo Chen, Wei Song, Yier Jin, Dan Meng, Rui Hou

**Abstract**: The wide deployment of Large Language Models (LLMs) has given rise to strong demands for optimizing their inference performance. Today's techniques serving this purpose primarily focus on reducing latency and improving throughput through algorithmic and hardware enhancements, while largely overlooking their privacy side effects, particularly in a multi-user environment. In our research, for the first time, we discovered a set of new timing side channels in LLM systems, arising from shared caches and GPU memory allocations, which can be exploited to infer both confidential system prompts and those issued by other users. These vulnerabilities echo security challenges observed in traditional computing systems, highlighting an urgent need to address potential information leakage in LLM serving infrastructures. In this paper, we report novel attack strategies designed to exploit such timing side channels inherent in LLM deployments, specifically targeting the Key-Value (KV) cache and semantic cache widely used to enhance LLM inference performance. Our approach leverages timing measurements and classification models to detect cache hits, allowing an adversary to infer private prompts with high accuracy. We also propose a token-by-token search algorithm to efficiently recover shared prompt prefixes in the caches, showing the feasibility of stealing system prompts and those produced by peer users. Our experimental studies on black-box testing of popular online LLM services demonstrate that such privacy risks are completely realistic, with significant consequences. Our findings underscore the need for robust mitigation to protect LLM systems against such emerging threats.

摘要: 大型语言模型（LLM）的广泛部署引发了对其推理性能优化的强烈要求。当今服务于此目的的技术主要集中在通过算法和硬件增强来减少延迟和提高吞吐量，同时在很大程度上忽视了它们的隐私副作用，特别是在多用户环境中。在我们的研究中，我们首次在LLM系统中发现了一组新的计时侧通道，这些通道由共享缓存和图形处理器内存分配产生，可以利用这些通道来推断机密系统提示和其他用户发出的提示。这些漏洞呼应了传统计算系统中观察到的安全挑战，凸显了解决LLM服务基础设施中潜在信息泄露的迫切需要。在本文中，我们报告了旨在利用LLM部署中固有的此类时间侧通道的新颖攻击策略，特别针对广泛用于增强LLM推理性能的Key-Value（KV）缓存和语义缓存。我们的方法利用时间测量和分类模型来检测缓存命中，使对手能够高准确地推断私人提示。我们还提出了一种逐令牌搜索算法来有效地恢复缓存中的共享提示前置，展示了窃取系统提示和对等用户产生的提示的可行性。我们对流行的在线LLM服务进行黑匣子测试的实验研究表明，此类隐私风险是完全现实的，并会产生重大后果。我们的研究结果强调需要强有力的缓解措施来保护LLM系统免受此类新出现的威胁。



## **14. Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference**

缓存中的影子：在LLM推理中揭示和减轻KV缓存的隐私风险 cs.CR

**SubmitDate**: 2025-08-13    [abs](http://arxiv.org/abs/2508.09442v1) [paper-pdf](http://arxiv.org/pdf/2508.09442v1)

**Authors**: Zhifan Luo, Shuo Shao, Su Zhang, Lijing Zhou, Yuke Hu, Chenxu Zhao, Zhihao Liu, Zhan Qin

**Abstract**: The Key-Value (KV) cache, which stores intermediate attention computations (Key and Value pairs) to avoid redundant calculations, is a fundamental mechanism for accelerating Large Language Model (LLM) inference. However, this efficiency optimization introduces significant yet underexplored privacy risks. This paper provides the first comprehensive analysis of these vulnerabilities, demonstrating that an attacker can reconstruct sensitive user inputs directly from the KV-cache. We design and implement three distinct attack vectors: a direct Inversion Attack, a more broadly applicable and potent Collision Attack, and a semantic-based Injection Attack. These methods demonstrate the practicality and severity of KV-cache privacy leakage issues. To mitigate this, we propose KV-Cloak, a novel, lightweight, and efficient defense mechanism. KV-Cloak uses a reversible matrix-based obfuscation scheme, combined with operator fusion, to secure the KV-cache. Our extensive experiments show that KV-Cloak effectively thwarts all proposed attacks, reducing reconstruction quality to random noise. Crucially, it achieves this robust security with virtually no degradation in model accuracy and minimal performance overhead, offering a practical solution for trustworthy LLM deployment.

摘要: Key-Value（KV）缓存存储中间注意力计算（Key和Value对）以避免冗余计算，是加速大型语言模型（LLM）推理的基本机制。然而，这种效率优化引入了重大但未充分探索的隐私风险。本文首次对这些漏洞进行了全面分析，证明攻击者可以直接从KV缓存重建敏感用户输入。我们设计并实现了三种不同的攻击载体：直接倒置攻击、更广泛适用且更强大的碰撞攻击以及基于语义的注入攻击。这些方法证明了KV缓存隐私泄漏问题的实用性和严重性。为了减轻这种情况，我们提出了KV-斗篷，一种新颖的，轻量级的，高效的防御机制。KV-Cloak使用基于可逆矩阵的混淆方案，结合运算符融合来保护KV缓存。我们广泛的实验表明，KV-斗篷有效地挫败了所有提出的攻击，降低重建质量的随机噪声。至关重要的是，它实现了这种强大的安全性，模型准确性几乎没有下降，性能负担最小，为值得信赖的LLM部署提供了实用的解决方案。



## **15. Can AI Keep a Secret? Contextual Integrity Verification: A Provable Security Architecture for LLMs**

人工智能能保守秘密吗？上下文完整性验证：LLM的可证明安全架构 cs.CR

2 figures, 3 tables; code and certification harness:  https://github.com/ayushgupta4897/Contextual-Integrity-Verification ;  Elite-Attack dataset: https://huggingface.co/datasets/zyushg/elite-attack

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.09288v1) [paper-pdf](http://arxiv.org/pdf/2508.09288v1)

**Authors**: Aayush Gupta

**Abstract**: Large language models (LLMs) remain acutely vulnerable to prompt injection and related jailbreak attacks; heuristic guardrails (rules, filters, LLM judges) are routinely bypassed. We present Contextual Integrity Verification (CIV), an inference-time security architecture that attaches cryptographically signed provenance labels to every token and enforces a source-trust lattice inside the transformer via a pre-softmax hard attention mask (with optional FFN/residual gating). CIV provides deterministic, per-token non-interference guarantees on frozen models: lower-trust tokens cannot influence higher-trust representations. On benchmarks derived from recent taxonomies of prompt-injection vectors (Elite-Attack + SoK-246), CIV attains 0% attack success rate under the stated threat model while preserving 93.1% token-level similarity and showing no degradation in model perplexity on benign tasks; we note a latency overhead attributable to a non-optimized data path. Because CIV is a lightweight patch -- no fine-tuning required -- we demonstrate drop-in protection for Llama-3-8B and Mistral-7B. We release a reference implementation, an automated certification harness, and the Elite-Attack corpus to support reproducible research.

摘要: 大型语言模型（LLM）仍然极易受到提示注入和相关越狱攻击的影响;启发式护栏（规则、过滤器、LLM法官）通常会被绕过。我们提出了上下文完整性验证（CIV），这是一种推理时安全架构，它将加密签名的出处标签附加到每个令牌，并通过pre-softmax硬注意力屏蔽（具有可选的FFN/剩余门控）在Transformer内强制执行源信任网格。CIV在冻结模型上提供确定性的、每令牌不干扰保证：低信任度的令牌无法影响高信任度的表示。基于最近预算注入载体分类法得出的基准（Elite-Attack + SoK-246），CIV在所述威胁模型下获得0%的攻击成功率，同时保持93.1%的标记级相似性，并且在良性任务上模型复杂度没有下降;我们注意到未优化的数据路径会带来延迟负担。由于CIV是一个轻量级补丁--无需微调--因此我们演示了drop-保护Llama-3-8B和Mistral-7 B。我们发布了参考实现、自动化认证工具和精英攻击数据库来支持可重复的研究。



## **16. Attacks and Defenses Against LLM Fingerprinting**

针对LLM指纹的攻击和防御 cs.CR

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.09021v1) [paper-pdf](http://arxiv.org/pdf/2508.09021v1)

**Authors**: Kevin Kurian, Ethan Holland, Sean Oesch

**Abstract**: As large language models are increasingly deployed in sensitive environments, fingerprinting attacks pose significant privacy and security risks. We present a study of LLM fingerprinting from both offensive and defensive perspectives. Our attack methodology uses reinforcement learning to automatically optimize query selection, achieving better fingerprinting accuracy with only 3 queries compared to randomly selecting 3 queries from the same pool. Our defensive approach employs semantic-preserving output filtering through a secondary LLM to obfuscate model identity while maintaining semantic integrity. The defensive method reduces fingerprinting accuracy across tested models while preserving output quality. These contributions show the potential to improve fingerprinting tools capabilities while providing practical mitigation strategies against fingerprinting attacks.

摘要: 随着大型语言模型越来越多地部署在敏感环境中，指纹攻击构成了巨大的隐私和安全风险。我们从进攻和防守的角度对LLM指纹进行了研究。我们的攻击方法使用强化学习来自动优化查询选择，与从同一池中随机选择3个查询相比，仅使用3个查询就可以实现更好的指纹识别准确性。我们的防御方法通过二级LLM采用保持语义的输出过滤来混淆模型身份，同时保持语义完整性。防御性方法降低了测试模型之间的指纹识别准确性，同时保留了输出质量。这些贡献表明了提高指纹识别工具功能的潜力，同时提供针对指纹识别攻击的实用缓解策略。



## **17. GUARD:Dual-Agent based Backdoor Defense on Chain-of-Thought in Neural Code Generation**

GUARD：基于双代理的神经代码生成思想链后门防御 cs.SE

Accepted by SEKE 2025

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2505.21425v3) [paper-pdf](http://arxiv.org/pdf/2505.21425v3)

**Authors**: Naizhu Jin, Zhong Li, Tian Zhang, Qingkai Zeng

**Abstract**: With the widespread application of large language models in code generation, recent studies demonstrate that employing additional Chain-of-Thought generation models can significantly enhance code generation performance by providing explicit reasoning steps. However, as external components, CoT models are particularly vulnerable to backdoor attacks, which existing defense mechanisms often fail to detect effectively. To address this challenge, we propose GUARD, a novel dual-agent defense framework specifically designed to counter CoT backdoor attacks in neural code generation. GUARD integrates two core components: GUARD-Judge, which identifies suspicious CoT steps and potential triggers through comprehensive analysis, and GUARD-Repair, which employs a retrieval-augmented generation approach to regenerate secure CoT steps for identified anomalies. Experimental results show that GUARD effectively mitigates attacks while maintaining generation quality, advancing secure code generation systems.

摘要: 随着大型语言模型在代码生成中的广泛应用，最近的研究表明，采用额外的思想链生成模型可以通过提供显式推理步骤来显着提高代码生成性能。然而，作为外部组件，CoT模型特别容易受到后门攻击，而现有的防御机制往往无法有效检测到后门攻击。为了应对这一挑战，我们提出了GUARD，这是一种新型双代理防御框架，专门设计用于对抗神经代码生成中的CoT后门攻击。GUARD集成了两个核心组件：GUARD-Judge，通过全面分析识别可疑的CoT步骤和潜在触发因素，以及GUARD-Repair，采用检索增强生成方法来为识别的异常重新生成安全CoT步骤。实验结果表明，GUARD有效地缓解了攻击，同时保持生成质量，推进了安全代码生成系统。



## **18. Whispers in the Machine: Confidentiality in Agentic Systems**

机器中的耳语：智能系统中的机密 cs.CR

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2402.06922v4) [paper-pdf](http://arxiv.org/pdf/2402.06922v4)

**Authors**: Jonathan Evertz, Merlin Chlosta, Lea Schönherr, Thorsten Eisenhofer

**Abstract**: The interaction between users and applications is increasingly shifted toward natural language by deploying Large Language Models (LLMs) as the core interface. The capabilities of these so-called agents become more capable the more tools and services they serve as an interface for, ultimately leading to agentic systems. Agentic systems use LLM-based agents as interfaces for most user interactions and various integrations with external tools and services. While these interfaces can significantly enhance the capabilities of the agentic system, they also introduce a new attack surface. Manipulated integrations, for example, can exploit the internal LLM and compromise sensitive data accessed through other interfaces. While previous work primarily focused on attacks targeting a model's alignment or the leakage of training data, the security of data that is only available during inference has escaped scrutiny so far. In this work, we demonstrate how the integration of LLMs into systems with external tool integration poses a risk similar to established prompt-based attacks, able to compromise the confidentiality of the entire system. Introducing a systematic approach to evaluate these confidentiality risks, we identify two specific attack scenarios unique to these agentic systems and formalize these into a tool-robustness framework designed to measure a model's ability to protect sensitive information. Our analysis reveals significant vulnerabilities across all tested models, highlighting an increased risk when models are combined with external tools.

摘要: 通过将大型语言模型（LLM）部署为核心接口，用户和应用程序之间的交互越来越多地转向自然语言。这些所谓的代理人的能力变得更有能力的工具和服务，他们作为一个接口，最终导致代理系统。智能系统使用基于LLM的代理作为大多数用户交互的接口以及与外部工具和服务的各种集成。虽然这些接口可以显著增强代理系统的能力，但它们也引入了新的攻击面。例如，经过操纵的集成可能会利用内部LLM并损害通过其他接口访问的敏感数据。虽然之前的工作主要集中在针对模型对齐或训练数据泄露的攻击上，但迄今为止仅在推理期间可用的数据安全性逃脱了审查。在这项工作中，我们展示了将LLM集成到具有外部工具集成的系统中如何构成类似于已建立的基于预算的攻击的风险，从而能够损害整个系统的机密性。我们引入了一种系统性方法来评估这些保密风险，识别了这些代理系统独有的两种特定攻击场景，并将其形式化为工具稳健性框架，旨在衡量模型保护敏感信息的能力。我们的分析揭示了所有测试模型中存在的重大漏洞，凸显了模型与外部工具结合时的风险增加。



## **19. A Few Words Can Distort Graphs: Knowledge Poisoning Attacks on Graph-based Retrieval-Augmented Generation of Large Language Models**

一些词可以扭曲图形：对基于图形的检索增强生成大型语言模型的知识中毒攻击 cs.CL

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.04276v2) [paper-pdf](http://arxiv.org/pdf/2508.04276v2)

**Authors**: Jiayi Wen, Tianxin Chen, Zhirun Zheng, Cheng Huang

**Abstract**: Graph-based Retrieval-Augmented Generation (GraphRAG) has recently emerged as a promising paradigm for enhancing large language models (LLMs) by converting raw text into structured knowledge graphs, improving both accuracy and explainability. However, GraphRAG relies on LLMs to extract knowledge from raw text during graph construction, and this process can be maliciously manipulated to implant misleading information. Targeting this attack surface, we propose two knowledge poisoning attacks (KPAs) and demonstrate that modifying only a few words in the source text can significantly change the constructed graph, poison the GraphRAG, and severely mislead downstream reasoning. The first attack, named Targeted KPA (TKPA), utilizes graph-theoretic analysis to locate vulnerable nodes in the generated graphs and rewrites the corresponding narratives with LLMs, achieving precise control over specific question-answering (QA) outcomes with a success rate of 93.1\%, while keeping the poisoned text fluent and natural. The second attack, named Universal KPA (UKPA), exploits linguistic cues such as pronouns and dependency relations to disrupt the structural integrity of the generated graph by altering globally influential words. With fewer than 0.05\% of full text modified, the QA accuracy collapses from 95\% to 50\%. Furthermore, experiments show that state-of-the-art defense methods fail to detect these attacks, highlighting that securing GraphRAG pipelines against knowledge poisoning remains largely unexplored.

摘要: 基于图的检索增强生成（GraphRAG）最近成为一种有前途的范式，可以通过将原始文本转换为结构化知识图来增强大型语言模型（LLM），提高准确性和可解释性。然而，GraphRAG在图形构建过程中依赖LLM从原始文本中提取知识，而这个过程可能会被恶意操纵以植入误导性信息。针对这一攻击面，我们提出了两种知识中毒攻击（KPA），并证明仅修改源文本中的几个单词就可以显着改变所构建的图、毒害GraphRAG并严重误导下游推理。第一次攻击名为Target KPA（TKPA），利用图形理论分析在生成的图形中定位脆弱的节点，并使用LLM重写相应的叙述，实现对特定问答（QA）结果的精确控制，成功率为93.1%，同时保持有毒文本流畅自然。第二种攻击名为Universal KPA（UKPA），利用代词和依赖关系等语言线索通过改变具有全球影响力的单词来破坏生成图的结构完整性。修改的全文少于0.05%，QA准确性从95%下降到50%。此外，实验表明，最先进的防御方法无法检测到这些攻击，这凸显了保护GraphRAG管道免受知识中毒的保护在很大程度上仍然没有被探索。



## **20. Chimera: Harnessing Multi-Agent LLMs for Automatic Insider Threat Simulation**

Chimera：利用多代理LLM进行自动内部威胁模拟 cs.CR

23 pages

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.07745v2) [paper-pdf](http://arxiv.org/pdf/2508.07745v2)

**Authors**: Jiongchi Yu, Xiaofei Xie, Qiang Hu, Yuhan Ma, Ziming Zhao

**Abstract**: Insider threats, which can lead to severe losses, remain a major security concern. While machine learning-based insider threat detection (ITD) methods have shown promising results, their progress is hindered by the scarcity of high-quality data. Enterprise data is sensitive and rarely accessible, while publicly available datasets, when limited in scale due to cost, lack sufficient real-world coverage; and when purely synthetic, they fail to capture rich semantics and realistic user behavior. To address this, we propose Chimera, the first large language model (LLM)-based multi-agent framework that automatically simulates both benign and malicious insider activities and collects diverse logs across diverse enterprise environments. Chimera models each employee with agents that have role-specific behavior and integrates modules for group meetings, pairwise interactions, and autonomous scheduling, capturing realistic organizational dynamics. It incorporates 15 types of insider attacks (e.g., IP theft, system sabotage) and has been deployed to simulate activities in three sensitive domains: technology company, finance corporation, and medical institution, producing a new dataset, ChimeraLog. We assess ChimeraLog via human studies and quantitative analysis, confirming its diversity, realism, and presence of explainable threat patterns. Evaluations of existing ITD methods show an average F1-score of 0.83, which is significantly lower than 0.99 on the CERT dataset, demonstrating ChimeraLog's higher difficulty and utility for advancing ITD research.

摘要: 内部威胁可能导致严重损失，但仍然是一个主要的安全问题。虽然基于机器学习的内部威胁检测（ITD）方法已显示出令人鼓舞的结果，但其进展因高质量数据的稀缺而受到阻碍。企业数据很敏感，很少可访问，而公开可用的数据集，当由于成本而规模有限时，缺乏足够的现实世界覆盖范围;当纯粹合成时，它们无法捕捉丰富的语义和现实的用户行为。为了解决这个问题，我们提出了Chimera，这是第一个基于大型语言模型（LLM）的多代理框架，可以自动模拟良性和恶意的内部活动，并在不同的企业环境中收集不同的日志。Chimera用具有特定角色行为的代理为每位员工建模，并集成了小组会议、成对互动和自主调度模块，捕捉现实的组织动态。它包含15种类型的内部攻击（例如IP盗窃、系统破坏），并已被部署来模拟三个敏感领域的活动：科技公司、金融公司和医疗机构，生成新的数据集ChimeraLog。我们通过人类研究和定量分析评估ChimeraLog，确认其多样性、现实性以及可解释的威胁模式的存在。对现有ITD方法的评估显示F1平均评分为0.83，显着低于CERT数据集的0.99，这表明ChimeraLog在推进ITD研究方面具有更高的难度和实用性。



## **21. Securing Educational LLMs: A Generalised Taxonomy of Attacks on LLMs and DREAD Risk Assessment**

保护教育LLM：LLM攻击的一般分类和DREAD风险评估 cs.CY

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2508.08629v1) [paper-pdf](http://arxiv.org/pdf/2508.08629v1)

**Authors**: Farzana Zahid, Anjalika Sewwandi, Lee Brandon, Vimal Kumar, Roopak Sinha

**Abstract**: Due to perceptions of efficiency and significant productivity gains, various organisations, including in education, are adopting Large Language Models (LLMs) into their workflows. Educator-facing, learner-facing, and institution-facing LLMs, collectively, Educational Large Language Models (eLLMs), complement and enhance the effectiveness of teaching, learning, and academic operations. However, their integration into an educational setting raises significant cybersecurity concerns. A comprehensive landscape of contemporary attacks on LLMs and their impact on the educational environment is missing. This study presents a generalised taxonomy of fifty attacks on LLMs, which are categorized as attacks targeting either models or their infrastructure. The severity of these attacks is evaluated in the educational sector using the DREAD risk assessment framework. Our risk assessment indicates that token smuggling, adversarial prompts, direct injection, and multi-step jailbreak are critical attacks on eLLMs. The proposed taxonomy, its application in the educational environment, and our risk assessment will help academic and industrial practitioners to build resilient solutions that protect learners and institutions.

摘要: 由于人们对效率和生产力的显着提高，包括教育在内的各种组织正在将大型语言模型（LLM）纳入其工作流程中。面向教育者、面向学习者和面向机构的LLM（统称为教育大型语言模型（eLLM）），补充和增强教学、学习和学术运营的有效性。然而，它们融入教育环境会引发严重的网络安全问题。缺乏当代针对法学硕士的攻击及其对教育环境影响的全面景观。本研究提出了针对LLM的五十种攻击的一般分类，这些攻击被归类为针对模型或其基础设施的攻击。教育部门使用DREAD风险评估框架评估这些攻击的严重性。我们的风险评估表明，代币走私、对抗性提示、直接注入和多步越狱是对eLLM的严重攻击。拟议的分类法、其在教育环境中的应用以及我们的风险评估将帮助学术和行业从业者构建保护学习者和机构的弹性解决方案。



## **22. Few-Shot Adversarial Low-Rank Fine-Tuning of Vision-Language Models**

视觉语言模型的少镜头对抗低秩微调 cs.LG

**SubmitDate**: 2025-08-12    [abs](http://arxiv.org/abs/2505.15130v2) [paper-pdf](http://arxiv.org/pdf/2505.15130v2)

**Authors**: Sajjad Ghiasvand, Haniyeh Ehsani Oskouie, Mahnoosh Alizadeh, Ramtin Pedarsani

**Abstract**: Vision-Language Models (VLMs) such as CLIP have shown remarkable performance in cross-modal tasks through large-scale contrastive pre-training. To adapt these large transformer-based models efficiently for downstream tasks, Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA have emerged as scalable alternatives to full fine-tuning, especially in few-shot scenarios. However, like traditional deep neural networks, VLMs are highly vulnerable to adversarial attacks, where imperceptible perturbations can significantly degrade model performance. Adversarial training remains the most effective strategy for improving model robustness in PEFT. In this work, we propose AdvCLIP-LoRA, the first algorithm designed to enhance the adversarial robustness of CLIP models fine-tuned with LoRA in few-shot settings. Our method formulates adversarial fine-tuning as a minimax optimization problem and provides theoretical guarantees for convergence under smoothness and nonconvex-strong-concavity assumptions. Empirical results across eight datasets using ViT-B/16 and ViT-B/32 models show that AdvCLIP-LoRA significantly improves robustness against common adversarial attacks (e.g., FGSM, PGD), without sacrificing much clean accuracy. These findings highlight AdvCLIP-LoRA as a practical and theoretically grounded approach for robust adaptation of VLMs in resource-constrained settings.

摘要: 通过大规模对比预训练，CLIP等视觉语言模型（VLM）在跨模式任务中表现出了出色的表现。为了有效地调整这些基于变压器的大型模型以适应下游任务，LoRA等参数高效微调（PEFT）技术已成为完全微调的可扩展替代方案，尤其是在少量场景中。然而，与传统的深度神经网络一样，VLM非常容易受到对抗攻击，其中不可感知的扰动可能会显着降低模型性能。对抗训练仍然是提高PEFT模型稳健性的最有效策略。在这项工作中，我们提出了AdvCLIP-LoRA，这是第一个旨在增强在少数镜头设置中使用LoRA微调的CLIP模型的对抗鲁棒性的算法。我们的方法将对抗性微调表述为极小极大优化问题，并为光滑性和非凸强插值假设下的收敛提供理论保证。使用ViT-B/16和ViT-B/32模型的八个数据集的经验结果表明，AdvCLIP-LoRA显着提高了针对常见对抗攻击（例如，FGSM、PVD），而不会牺牲太多干净的准确性。这些发现凸显了AdvCLIP-LoRA是一种实用且具有理论依据的方法，用于在资源有限的环境中稳健地适应VLM。



## **23. Selective KV-Cache Sharing to Mitigate Timing Side-Channels in LLM Inference**

LLM推理中的选择性KV缓存共享以缓解定时侧通道 cs.CR

17 pages,17 figures

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.08438v1) [paper-pdf](http://arxiv.org/pdf/2508.08438v1)

**Authors**: Kexin Chu, Zecheng Lin, Dawei Xiang, Zixu Shen, Jianchang Su, Cheng Chu, Yiwei Yang, Wenhui Zhang, Wenfei Wu, Wei Zhang

**Abstract**: Global KV-cache sharing has emerged as a key optimization for accelerating large language model (LLM) inference. However, it exposes a new class of timing side-channel attacks, enabling adversaries to infer sensitive user inputs via shared cache entries. Existing defenses, such as per-user isolation, eliminate leakage but degrade performance by up to 38.9% in time-to-first-token (TTFT), making them impractical for high-throughput deployment. To address this gap, we introduce SafeKV (Secure and Flexible KV Cache Sharing), a privacy-aware KV-cache management framework that selectively shares non-sensitive entries while confining sensitive content to private caches. SafeKV comprises three components: (i) a hybrid, multi-tier detection pipeline that integrates rule-based pattern matching, a general-purpose privacy detector, and context-aware validation; (ii) a unified radix-tree index that manages public and private entries across heterogeneous memory tiers (HBM, DRAM, SSD); and (iii) entropy-based access monitoring to detect and mitigate residual information leakage. Our evaluation shows that SafeKV mitigates 94% - 97% of timing-based side-channel attacks. Compared to per-user isolation method, SafeKV improves TTFT by up to 40.58% and throughput by up to 2.66X across diverse LLMs and workloads. SafeKV reduces cache-induced TTFT overhead from 50.41% to 11.74% on Qwen3-235B. By combining fine-grained privacy control with high cache reuse efficiency, SafeKV reclaims the performance advantages of global sharing while providing robust runtime privacy guarantees for LLM inference.

摘要: 全局KV缓存共享已成为加速大型语言模型（LLM）推理的关键优化。然而，它暴露了一类新的定时侧通道攻击，使对手能够通过共享缓存条目推断敏感用户输入。现有的防御措施（例如按用户隔离）可以消除泄漏，但在首次令牌时间（TTFT）方面性能会降低高达38.9%，因此对于高吞吐量部署来说不切实际。为了解决这一差距，我们引入了SafeKV（安全且灵活的KV缓存共享），这是一种隐私感知的KV缓存管理框架，可以选择性地共享非敏感条目，同时将敏感内容限制在私人缓存中。SafeKV由三个部分组成：（i）混合、多层检测管道，集成了基于规则的模式匹配、通用隐私检测器和上下文感知验证;（ii）统一的根树索引，管理跨异类存储器层（HBM、RAM、SSD）的公共和私有条目;和（iii）基于熵的访问监控，以检测和减轻剩余信息泄露。我们的评估表明，SafeKV可以缓解94% - 97%的基于计时的侧通道攻击。与按用户隔离方法相比，SafeKN将TTFT提高了40.58%，将各种LLM和工作负载的吞吐量提高了2.66倍。SafeKV将Qwen 3 - 235 B上高速缓存引起的TTFT费用从50.41%减少到11.74%。通过将细粒度隐私控制与高缓存重复使用效率相结合，SafeKN充分利用了全局共享的性能优势，同时为LLM推断提供强大的运行时隐私保证。



## **24. Towards Effective MLLM Jailbreaking Through Balanced On-Topicness and OOD-Intensity**

通过平衡的话题性和OOD强度实现有效的MLLM越狱 cs.CV

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.09218v1) [paper-pdf](http://arxiv.org/pdf/2508.09218v1)

**Authors**: Zuoou Li, Weitong Zhang, Jingyuan Wang, Shuyuan Zhang, Wenjia Bai, Bernhard Kainz, Mengyun Qiao

**Abstract**: Multimodal large language models (MLLMs) are widely used in vision-language reasoning tasks. However, their vulnerability to adversarial prompts remains a serious concern, as safety mechanisms often fail to prevent the generation of harmful outputs. Although recent jailbreak strategies report high success rates, many responses classified as "successful" are actually benign, vague, or unrelated to the intended malicious goal. This mismatch suggests that current evaluation standards may overestimate the effectiveness of such attacks. To address this issue, we introduce a four-axis evaluation framework that considers input on-topicness, input out-of-distribution (OOD) intensity, output harmfulness, and output refusal rate. This framework identifies truly effective jailbreaks. In a substantial empirical study, we reveal a structural trade-off: highly on-topic prompts are frequently blocked by safety filters, whereas those that are too OOD often evade detection but fail to produce harmful content. However, prompts that balance relevance and novelty are more likely to evade filters and trigger dangerous output. Building on this insight, we develop a recursive rewriting strategy called Balanced Structural Decomposition (BSD). The approach restructures malicious prompts into semantically aligned sub-tasks, while introducing subtle OOD signals and visual cues that make the inputs harder to detect. BSD was tested across 13 commercial and open-source MLLMs, where it consistently led to higher attack success rates, more harmful outputs, and fewer refusals. Compared to previous methods, it improves success rates by $67\%$ and harmfulness by $21\%$, revealing a previously underappreciated weakness in current multimodal safety systems.

摘要: 多模式大型语言模型（MLLM）广泛用于视觉语言推理任务。然而，它们对对抗提示的脆弱性仍然是一个严重问题，因为安全机制往往无法防止有害输出的产生。尽管最近的越狱策略报告了很高的成功率，但许多被归类为“成功”的响应实际上是良性的、模糊的，或与预期的恶意目标无关。这种不匹配表明当前的评估标准可能高估了此类攻击的有效性。为了解决这个问题，我们引入了一个四轴评估框架，该框架考虑了输入的主题性、输入未分配（OOD）强度、输出危害性和输出拒绝率。该框架确定了真正有效的越狱。在一项实质性的实证研究中，我们揭示了一种结构性权衡：高度切中主题的提示经常被安全过滤器阻止，而那些过于OOD的提示经常逃避检测，但无法产生有害内容。然而，平衡相关性和新颖性的提示更有可能逃避过滤器并触发危险的输出。基于这一见解，我们开发了一种名为平衡结构分解（BCD）的循环重写策略。该方法将恶意提示重组为语义对齐的子任务，同时引入微妙的OOD信号和视觉线索，使输入更难检测。BDS在13个商业和开源MLLM上进行了测试，它始终导致更高的攻击成功率、更多的有害输出和更少的拒绝。与以前的方法相比，它的成功率提高了67美元，危害性提高了21美元，揭示了当前多模式安全系统中以前被低估的弱点。



## **25. Assessing LLM Text Detection in Educational Contexts: Does Human Contribution Affect Detection?**

评估教育上下文中的LLM文本检测：人类贡献会影响检测吗？ cs.CL

Preprint as provided by the authors (19 pages, 12 figures, 9 tables)

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.08096v1) [paper-pdf](http://arxiv.org/pdf/2508.08096v1)

**Authors**: Lukas Gehring, Benjamin Paaßen

**Abstract**: Recent advancements in Large Language Models (LLMs) and their increased accessibility have made it easier than ever for students to automatically generate texts, posing new challenges for educational institutions. To enforce norms of academic integrity and ensure students' learning, learning analytics methods to automatically detect LLM-generated text appear increasingly appealing. This paper benchmarks the performance of different state-of-the-art detectors in educational contexts, introducing a novel dataset, called Generative Essay Detection in Education (GEDE), containing over 900 student-written essays and over 12,500 LLM-generated essays from various domains. To capture the diversity of LLM usage practices in generating text, we propose the concept of contribution levels, representing students' contribution to a given assignment. These levels range from purely human-written texts, to slightly LLM-improved versions, to fully LLM-generated texts, and finally to active attacks on the detector by "humanizing" generated texts. We show that most detectors struggle to accurately classify texts of intermediate student contribution levels, like LLM-improved human-written texts. Detectors are particularly likely to produce false positives, which is problematic in educational settings where false suspicions can severely impact students' lives. Our dataset, code, and additional supplementary materials are publicly available at https://github.com/lukasgehring/Assessing-LLM-Text-Detection-in-Educational-Contexts.

摘要: 大型语言模型（LLM）的最新进步及其可访问性的提高使学生比以往任何时候都更容易自动生成文本，这给教育机构带来了新的挑战。为了执行学术诚信规范并确保学生的学习，自动检测LLM生成的文本的学习分析方法似乎越来越有吸引力。本文对教育环境中不同最先进检测器的性能进行了基准测试，引入了一种名为教育生成性论文检测（GEDE）的新型数据集，其中包含900多篇学生撰写的论文和12，500多篇来自各个领域的LLM生成的论文。为了捕捉LLM在生成文本时使用实践的多样性，我们提出了贡献水平的概念，代表学生对给定作业的贡献。这些级别的范围从纯粹的人类编写的文本，到稍微改进的LLM版本，再到完全LLM生成的文本，最后到通过“人性化”生成的文本对检测器进行主动攻击。我们表明，大多数检测器都很难准确分类中等学生贡献水平的文本，例如LLM改进的人类书面文本。检测器特别有可能产生假阳性，这在教育环境中是一个问题，因为错误的怀疑可能会严重影响学生的生活。我们的数据集、代码和其他补充材料可在https://github.com/lukasgehring/Assessing-LLM-Text-Detection-in-Educational-Contexts上公开获取。



## **26. BadPromptFL: A Novel Backdoor Threat to Prompt-based Federated Learning in Multimodal Models**

BadminutFL：对多模式模型中基于预算的联邦学习的新型后门威胁 cs.LG

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.08040v1) [paper-pdf](http://arxiv.org/pdf/2508.08040v1)

**Authors**: Maozhen Zhang, Mengnan Zhao, Bo Wang

**Abstract**: Prompt-based tuning has emerged as a lightweight alternative to full fine-tuning in large vision-language models, enabling efficient adaptation via learned contextual prompts. This paradigm has recently been extended to federated learning settings (e.g., PromptFL), where clients collaboratively train prompts under data privacy constraints. However, the security implications of prompt-based aggregation in federated multimodal learning remain largely unexplored, leaving a critical attack surface unaddressed. In this paper, we introduce \textbf{BadPromptFL}, the first backdoor attack targeting prompt-based federated learning in multimodal contrastive models. In BadPromptFL, compromised clients jointly optimize local backdoor triggers and prompt embeddings, injecting poisoned prompts into the global aggregation process. These prompts are then propagated to benign clients, enabling universal backdoor activation at inference without modifying model parameters. Leveraging the contextual learning behavior of CLIP-style architectures, BadPromptFL achieves high attack success rates (e.g., \(>90\%\)) with minimal visibility and limited client participation. Extensive experiments across multiple datasets and aggregation protocols validate the effectiveness, stealth, and generalizability of our attack, raising critical concerns about the robustness of prompt-based federated learning in real-world deployments.

摘要: 基于预算的调优已成为大型视觉语言模型中完全微调的轻量级替代方案，可以通过学习的上下文提示进行高效调整。该范式最近已扩展到联邦学习环境（例如，AtlantFL），客户在数据隐私限制下协作训练提示。然而，联邦多模式学习中基于预算的聚合的安全影响在很大程度上仍未得到探索，导致关键的攻击表面尚未得到解决。本文中，我们介绍了\textBF{BadoutFL}，这是第一个针对多模式对比模型中基于预算的联邦学习的后门攻击。在BadoutFL中，受影响的客户端联合优化本地后门触发器和提示嵌入，将有毒提示注入到全球聚合流程中。然后，这些提示被传播到良性客户端，从而在推理时启用通用后门，而无需修改模型参数。利用CLIP风格架构的上下文学习行为，BadminutFL实现了高攻击成功率（例如，\（>90\%\））可见性最低且客户参与有限。跨多个数据集和聚合协议的广泛实验验证了我们攻击的有效性、隐蔽性和可推广性，引发了人们对现实世界部署中基于预算的联邦学习稳健性的严重担忧。



## **27. Robust Anomaly Detection in O-RAN: Leveraging LLMs against Data Manipulation Attacks**

O-RAN中的鲁棒异常检测：利用LLM对抗数据操纵攻击 cs.CR

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.08029v1) [paper-pdf](http://arxiv.org/pdf/2508.08029v1)

**Authors**: Thusitha Dayaratne, Ngoc Duy Pham, Viet Vo, Shangqi Lai, Sharif Abuadbba, Hajime Suzuki, Xingliang Yuan, Carsten Rudolph

**Abstract**: The introduction of 5G and the Open Radio Access Network (O-RAN) architecture has enabled more flexible and intelligent network deployments. However, the increased complexity and openness of these architectures also introduce novel security challenges, such as data manipulation attacks on the semi-standardised Shared Data Layer (SDL) within the O-RAN platform through malicious xApps. In particular, malicious xApps can exploit this vulnerability by introducing subtle Unicode-wise alterations (hypoglyphs) into the data that are being used by traditional machine learning (ML)-based anomaly detection methods. These Unicode-wise manipulations can potentially bypass detection and cause failures in anomaly detection systems based on traditional ML, such as AutoEncoders, which are unable to process hypoglyphed data without crashing. We investigate the use of Large Language Models (LLMs) for anomaly detection within the O-RAN architecture to address this challenge. We demonstrate that LLM-based xApps maintain robust operational performance and are capable of processing manipulated messages without crashing. While initial detection accuracy requires further improvements, our results highlight the robustness of LLMs to adversarial attacks such as hypoglyphs in input data. There is potential to use their adaptability through prompt engineering to further improve the accuracy, although this requires further research. Additionally, we show that LLMs achieve low detection latency (under 0.07 seconds), making them suitable for Near-Real-Time (Near-RT) RIC deployments.

摘要: 5G和开放式无线电接入网络（O-RAN）架构的引入使网络部署更加灵活和智能。然而，这些体系结构复杂性和开放性的增加也带来了新颖的安全挑战，例如通过恶意xApp对O-RAN平台内的半标准化共享数据层（SDF）进行数据操纵攻击。特别是，恶意xApp可以通过在传统基于机器学习（ML）的异常检测方法使用的数据中引入微妙的Unicode更改（次字形）来利用此漏洞。这些基于Unicode的操作可能会绕过检测并导致基于传统ML的异常检测系统（例如AutoEncoders）出现故障，这些系统无法在不崩溃的情况下处理次字母数据。我们研究了在O-RAN架构内使用大型语言模型（LLM）进行异常检测的情况，以应对这一挑战。我们证明基于LLM的xApp可以保持稳健的操作性能，并且能够处理被操纵的消息而不会崩溃。虽然初始检测准确性需要进一步提高，但我们的结果强调了LLM对对抗攻击（例如输入数据中的副字形）的鲁棒性。有可能通过及时的工程利用它们的适应性来进一步提高准确性，尽管这需要进一步的研究。此外，我们表明LLM可以实现低检测延迟（低于0.07秒），使其适合近实时（Near-RT）RIC部署。



## **28. Toward Intelligent and Secure Cloud: Large Language Model Empowered Proactive Defense**

迈向智能和安全的云：大型语言模型增强主动防御 cs.CR

7 pages; Major Revision for IEEE Communications Magazine

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2412.21051v3) [paper-pdf](http://arxiv.org/pdf/2412.21051v3)

**Authors**: Yuyang Zhou, Guang Cheng, Kang Du, Zihan Chen, Yuyu Zhao

**Abstract**: The rapid evolution of cloud computing technologies and the increasing number of cloud applications have provided numerous benefits in our daily lives. However, the diversity and complexity of different components pose a significant challenge to cloud security, especially when dealing with sophisticated and advanced cyberattacks such as Denial of Service (DoS). Recent advancements in the large language models (LLMs) offer promising solutions for security intelligence. By exploiting the powerful capabilities in language understanding, data analysis, task inference, action planning, and code generation, we present LLM-PD, a novel defense architecture that proactively mitigates various DoS threats in cloud networks. LLM-PD can efficiently make decisions through comprehensive data analysis and sequential reasoning, as well as dynamically create and deploy actionable defense mechanisms. Furthermore, it can flexibly self-evolve based on experience learned from previous interactions and adapt to new attack scenarios without additional training. Our case study on three distinct DoS attacks demonstrates its remarkable ability in terms of defense effectiveness and efficiency when compared with other existing methods.

摘要: 云计算技术的快速发展和云应用程序数量的不断增加为我们的日常生活带来了诸多好处。然而，不同组件的多样性和复杂性对云安全构成了重大挑战，特别是在处理拒绝服务（Doc）等复杂且高级的网络攻击时。大型语言模型（LLM）的最新进展为安全情报提供了有前途的解决方案。通过利用语言理解、数据分析、任务推理、动作规划和代码生成方面的强大功能，我们提出了LLM-PD，这是一种新型防御架构，可以主动缓解云网络中的各种NOS威胁。LLM-PD可以通过全面的数据分析和顺序推理有效地做出决策，并动态创建和部署可操作的防御机制。此外，它可以根据从之前的交互中学到的经验灵活地自我进化，并在无需额外训练的情况下适应新的攻击场景。我们对三种不同的DPS攻击的案例研究表明，与其他现有方法相比，其在防御有效性和效率方面具有出色的能力。



## **29. Can You Trick the Grader? Adversarial Persuasion of LLM Judges**

你能欺骗评分员吗？LLM法官的对抗性说服 cs.CL

19 pages, 8 figures

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.07805v1) [paper-pdf](http://arxiv.org/pdf/2508.07805v1)

**Authors**: Yerin Hwang, Dongryeol Lee, Taegwan Kang, Yongil Kim, Kyomin Jung

**Abstract**: As large language models take on growing roles as automated evaluators in practical settings, a critical question arises: Can individuals persuade an LLM judge to assign unfairly high scores? This study is the first to reveal that strategically embedded persuasive language can bias LLM judges when scoring mathematical reasoning tasks, where correctness should be independent of stylistic variation. Grounded in Aristotle's rhetorical principles, we formalize seven persuasion techniques (Majority, Consistency, Flattery, Reciprocity, Pity, Authority, Identity) and embed them into otherwise identical responses. Across six math benchmarks, we find that persuasive language leads LLM judges to assign inflated scores to incorrect solutions, by up to 8% on average, with Consistency causing the most severe distortion. Notably, increasing model size does not substantially mitigate this vulnerability. Further analysis demonstrates that combining multiple persuasion techniques amplifies the bias, and pairwise evaluation is likewise susceptible. Moreover, the persuasive effect persists under counter prompting strategies, highlighting a critical vulnerability in LLM-as-a-Judge pipelines and underscoring the need for robust defenses against persuasion-based attacks.

摘要: 随着大型语言模型在实际环境中作为自动评估者的作用越来越大，一个关键问题出现了：个人能否说服法学硕士法官给予不公平的高分？这项研究首次揭示了战略性嵌入的说服性语言在给数学推理任务打分时会产生偏见，而正确性应该独立于文体差异。以亚里斯多德的修辞原则为基础，我们正式化了七种说服技巧（多数、一致、奉承、互惠、怜悯、权威、身份），并将它们嵌入到其他相同的回应中。在六个数学基准中，我们发现有说服力的语言导致LLM评委对不正确的解决方案给予夸大的分数，平均高达8%，其中一致性会造成最严重的扭曲。值得注意的是，增加模型大小并不能极大地缓解此漏洞。进一步的分析表明，结合多种说服技术会放大偏见，并且成对评估也很容易受到影响。此外，在反提示策略下，说服效果仍然存在，凸显了LLM作为法官管道中的一个关键漏洞，并强调了针对基于说服的攻击的强大防御的必要性。



## **30. Multi-Turn Jailbreaks Are Simpler Than They Seem**

多次越狱比看起来简单 cs.LG

25 pages, 15 figures. Accepted at COLM 2025 SoLaR Workshop

**SubmitDate**: 2025-08-11    [abs](http://arxiv.org/abs/2508.07646v1) [paper-pdf](http://arxiv.org/pdf/2508.07646v1)

**Authors**: Xiaoxue Yang, Jaeha Lee, Anna-Katharina Dick, Jasper Timm, Fei Xie, Diogo Cruz

**Abstract**: While defenses against single-turn jailbreak attacks on Large Language Models (LLMs) have improved significantly, multi-turn jailbreaks remain a persistent vulnerability, often achieving success rates exceeding 70% against models optimized for single-turn protection. This work presents an empirical analysis of automated multi-turn jailbreak attacks across state-of-the-art models including GPT-4, Claude, and Gemini variants, using the StrongREJECT benchmark. Our findings challenge the perceived sophistication of multi-turn attacks: when accounting for the attacker's ability to learn from how models refuse harmful requests, multi-turn jailbreaking approaches are approximately equivalent to simply resampling single-turn attacks multiple times. Moreover, attack success is correlated among similar models, making it easier to jailbreak newly released ones. Additionally, for reasoning models, we find surprisingly that higher reasoning effort often leads to higher attack success rates. Our results have important implications for AI safety evaluation and the design of jailbreak-resistant systems. We release the source code at https://github.com/diogo-cruz/multi_turn_simpler

摘要: 虽然针对大型语言模型（LLM）的单轮越狱攻击的防御已经显着提高，但多轮越狱仍然是一个持久的漏洞，针对针对单轮保护优化的模型，成功率通常超过70%。这项工作使用StrongRESYS基准，对最先进模型（包括GPT-4、Claude和Gemini变体）中的自动多回合越狱攻击进行了实证分析。我们的研究结果挑战了多回合攻击的复杂性：当考虑到攻击者从模型如何拒绝有害请求中学习的能力时，多回合越狱方法大约相当于简单地多次重新加载单回合攻击。此外，攻击成功在类似型号之间是相互关联的，这使得越狱新发布的型号变得更容易。此外，对于推理模型，我们令人惊讶地发现，更高的推理工作量往往会导致更高的攻击成功率。我们的结果对人工智能安全评估和防越狱系统的设计具有重要影响。我们在https://github.com/diogo-cruz/multi_turn_simpler上发布源代码



## **31. Pentest-R1: Towards Autonomous Penetration Testing Reasoning Optimized via Two-Stage Reinforcement Learning**

Pentest-R1：通过两阶段强化学习优化的自主渗透测试推理 cs.AI

**SubmitDate**: 2025-08-10    [abs](http://arxiv.org/abs/2508.07382v1) [paper-pdf](http://arxiv.org/pdf/2508.07382v1)

**Authors**: He Kong, Die Hu, Jingguo Ge, Liangxiong Li, Hui Li, Tong Li

**Abstract**: Automating penetration testing is crucial for enhancing cybersecurity, yet current Large Language Models (LLMs) face significant limitations in this domain, including poor error handling, inefficient reasoning, and an inability to perform complex end-to-end tasks autonomously. To address these challenges, we introduce Pentest-R1, a novel framework designed to optimize LLM reasoning capabilities for this task through a two-stage reinforcement learning pipeline. We first construct a dataset of over 500 real-world, multi-step walkthroughs, which Pentest-R1 leverages for offline reinforcement learning (RL) to instill foundational attack logic. Subsequently, the LLM is fine-tuned via online RL in an interactive Capture The Flag (CTF) environment, where it learns directly from environmental feedback to develop robust error self-correction and adaptive strategies. Our extensive experiments on the Cybench and AutoPenBench benchmarks demonstrate the framework's effectiveness. On AutoPenBench, Pentest-R1 achieves a 24.2\% success rate, surpassing most state-of-the-art models and ranking second only to Gemini 2.5 Flash. On Cybench, it attains a 15.0\% success rate in unguided tasks, establishing a new state-of-the-art for open-source LLMs and matching the performance of top proprietary models. Ablation studies confirm that the synergy of both training stages is critical to its success.

摘要: 自动化渗透测试对于增强网络安全至关重要，但当前的大型语言模型（LLM）在该领域面临着显着的局限性，包括错误处理不良、推理效率低下以及无法自主执行复杂的端到端任务。为了应对这些挑战，我们引入了Pentest-R1，这是一个新颖的框架，旨在通过两阶段强化学习管道优化该任务的LLM推理能力。我们首先构建了一个包含500多个现实世界的多步骤步行表的数据集，Pentest-R1利用该数据集进行离线强化学习（RL）来灌输基础攻击逻辑。随后，LLM在交互式捕获旗帜（CTF）环境中通过在线RL进行微调，直接从环境反馈中学习，以制定稳健的错误自我纠正和自适应策略。我们在Cybench和AutoPenBench基准测试上的大量实验证明了该框架的有效性。在AutoPenBench上，Pentest-R1的成功率达到24.2%，超过了大多数最先进的型号，仅次于Gemini 2.5 Flash。在Cybench上，它在无指导任务中达到了15.0%的成功率，为开源LLM建立了一个新的最先进的水平，并与顶级专有模型的性能相匹配。消融研究证实，两个训练阶段的协同作用是其成功的关键。



## **32. Multi-task Adversarial Attacks against Black-box Model with Few-shot Queries**

针对具有少次触发器的黑匣子模型的多任务对抗攻击 cs.CR

**SubmitDate**: 2025-08-10    [abs](http://arxiv.org/abs/2508.10039v1) [paper-pdf](http://arxiv.org/pdf/2508.10039v1)

**Authors**: Wenqiang Wang, Yan Xiao, Hao Lin, Yangshijie Zhang, Xiaochun Cao

**Abstract**: Current multi-task adversarial text attacks rely on abundant access to shared internal features and numerous queries, often limited to a single task type. As a result, these attacks are less effective against practical scenarios involving black-box feedback APIs, limited queries, or multiple task types. To bridge this gap, we propose \textbf{C}luster and \textbf{E}nsemble \textbf{M}ulti-task Text Adversarial \textbf{A}ttack (\textbf{CEMA}), an effective black-box attack that exploits the transferability of adversarial texts across different tasks. CEMA simplifies complex multi-task scenarios by using a \textit{deep-level substitute model} trained in a \textit{plug-and-play} manner for text classification, enabling attacks without mimicking the victim model. This approach requires only a few queries for training, converting multi-task attacks into classification attacks and allowing attacks across various tasks.   CEMA generates multiple adversarial candidates using different text classification methods and selects the one that most effectively attacks substitute models.   In experiments involving multi-task models with two, three, or six tasks--spanning classification, translation, summarization, and text-to-image generation--CEMA demonstrates significant attack success with as few as 100 queries. Furthermore, CEMA can target commercial APIs (e.g., Baidu and Google Translate), large language models (e.g., ChatGPT 4o), and image-generation models (e.g., Stable Diffusion V2), showcasing its versatility and effectiveness in real-world applications.

摘要: 当前的多任务对抗性文本攻击依赖于对共享内部功能和大量查询的大量访问，通常仅限于单个任务类型。因此，这些攻击对涉及黑匣子反馈API、有限查询或多种任务类型的实际场景效果较差。为了弥合这一差距，我们提出了\textBF{C}luster和\textBF{E} n\textBF{M}多任务文本对抗性\textBF{A}ttack（\textBF{CEMA}），这是一种有效的黑匣子攻击，利用对抗性文本在不同任务之间的可转移性。CEMA通过使用以\textit{即插即用}方式训练的\textit{deep-level替代模型}进行文本分类，简化了复杂的多任务场景，从而在不模仿受害者模型的情况下启动攻击。这种方法只需要几个查询即可进行训练，将多任务攻击转化为分类攻击，并允许跨各种任务进行攻击。   CEMA使用不同的文本分类方法生成多个对抗候选者，并选择最有效攻击替代模型的候选者。   在涉及具有两个、三个或六个任务（跨越分类、翻译、摘要和文本到图像生成）的多任务模型的实验中，CEMA仅用100个查询就表现出了显着的攻击成功。此外，CEMA可以针对商业API（例如，百度和谷歌翻译）、大型语言模型（例如，ChatGPT 4 o）和图像生成模型（例如，稳定扩散V2），展示了其在现实应用中的多功能性和有效性。



## **33. Omni-SafetyBench: A Benchmark for Safety Evaluation of Audio-Visual Large Language Models**

Omni-SafetyBench：视听大语言模型安全评估基准 cs.CL

20 pages, 8 figures, 12 tables

**SubmitDate**: 2025-08-10    [abs](http://arxiv.org/abs/2508.07173v1) [paper-pdf](http://arxiv.org/pdf/2508.07173v1)

**Authors**: Leyi Pan, Zheyu Fu, Yunpeng Zhai, Shuchang Tao, Sheng Guan, Shiyu Huang, Lingzhe Zhang, Zhaoyang Liu, Bolin Ding, Felix Henry, Lijie Wen, Aiwei Liu

**Abstract**: The rise of Omni-modal Large Language Models (OLLMs), which integrate visual and auditory processing with text, necessitates robust safety evaluations to mitigate harmful outputs. However, no dedicated benchmarks currently exist for OLLMs, and prior benchmarks designed for other LLMs lack the ability to assess safety performance under audio-visual joint inputs or cross-modal safety consistency. To fill this gap, we introduce Omni-SafetyBench, the first comprehensive parallel benchmark for OLLM safety evaluation, featuring 24 modality combinations and variations with 972 samples each, including dedicated audio-visual harm cases. Considering OLLMs' comprehension challenges with complex omni-modal inputs and the need for cross-modal consistency evaluation, we propose tailored metrics: a Safety-score based on conditional Attack Success Rate (C-ASR) and Refusal Rate (C-RR) to account for comprehension failures, and a Cross-Modal Safety Consistency Score (CMSC-score) to measure consistency across modalities. Evaluating 6 open-source and 4 closed-source OLLMs reveals critical vulnerabilities: (1) no model excels in both overall safety and consistency, with only 3 models achieving over 0.6 in both metrics and top performer scoring around 0.8; (2) safety defenses weaken with complex inputs, especially audio-visual joints; (3) severe weaknesses persist, with some models scoring as low as 0.14 on specific modalities. Our benchmark and metrics highlight urgent needs for enhanced OLLM safety, providing a foundation for future improvements.

摘要: 全模态大型语言模型（OLLM）的兴起，将视觉和听觉处理与文本集成在一起，需要强大的安全评估来减轻有害的输出。然而，目前没有专门的基准存在的OLLM，和其他LLMs设计的基准缺乏评估视听联合输入下的安全性能或跨模态安全一致性的能力。为了填补这一空白，我们引入了Omni-SafetyBench，这是OLLM安全评估的第一个全面的平行基准，具有24种模态组合和变体，每个样本972个，包括专用的视听伤害案例。考虑到OLLM在复杂的全模态输入下的理解挑战以及跨模态一致性评估的需求，我们提出了定制的度量标准：基于条件攻击成功率（C-ASR）和拒绝率（C-RR）的安全分数来解释理解失败，以及跨模态安全一致性分数（CMSC分数）来衡量跨模态的一致性。评估6个开源OLLM和4个开源OLLM揭示了关键漏洞：（1）没有一个模型在整体安全性和一致性方面表现出色，只有3个模型在两项指标上都达到了0.6以上，最佳表现得分在0.8左右;（2）安全防御随着复杂的输入而减弱，尤其是视听关节;（3）严重的弱点仍然存在，一些模型在特定模式上的得分低至0.14。我们的基准和指标强调了增强OLLM安全性的迫切需求，为未来的改进提供了基础。



## **34. Model-Agnostic Sentiment Distribution Stability Analysis for Robust LLM-Generated Texts Detection**

用于鲁棒LLM生成文本检测的模型不可知情绪分布稳定性分析 cs.CL

**SubmitDate**: 2025-08-09    [abs](http://arxiv.org/abs/2508.06913v1) [paper-pdf](http://arxiv.org/pdf/2508.06913v1)

**Authors**: Siyuan Li, Xi Lin, Guangyan Li, Zehao Liu, Aodu Wulianghai, Li Ding, Jun Wu, Jianhua Li

**Abstract**: The rapid advancement of large language models (LLMs) has resulted in increasingly sophisticated AI-generated content, posing significant challenges in distinguishing LLM-generated text from human-written language. Existing detection methods, primarily based on lexical heuristics or fine-tuned classifiers, often suffer from limited generalizability and are vulnerable to paraphrasing, adversarial perturbations, and cross-domain shifts. In this work, we propose SentiDetect, a model-agnostic framework for detecting LLM-generated text by analyzing the divergence in sentiment distribution stability. Our method is motivated by the empirical observation that LLM outputs tend to exhibit emotionally consistent patterns, whereas human-written texts display greater emotional variability. To capture this phenomenon, we define two complementary metrics: sentiment distribution consistency and sentiment distribution preservation, which quantify stability under sentiment-altering and semantic-preserving transformations. We evaluate SentiDetect on five diverse datasets and a range of advanced LLMs,including Gemini-1.5-Pro, Claude-3, GPT-4-0613, and LLaMa-3.3. Experimental results demonstrate its superiority over state-of-the-art baselines, with over 16% and 11% F1 score improvements on Gemini-1.5-Pro and GPT-4-0613, respectively. Moreover, SentiDetect also shows greater robustness to paraphrasing, adversarial attacks, and text length variations, outperforming existing detectors in challenging scenarios.

摘要: 大型语言模型（LLM）的快速发展导致人工智能生成的内容越来越复杂，这给区分LLM生成的文本与人类书面语言带来了重大挑战。现有的检测方法主要基于词汇启发法或微调分类器，通常具有有限的概括性，并且容易受到重述、对抗性扰动和跨域转移的影响。在这项工作中，我们提出了SentiDetect，这是一个模型不可知的框架，用于通过分析情感分布稳定性的差异来检测LLM生成的文本。我们的方法的动机是经验观察，即LLM输出往往表现出情感一致的模式，而人类书面文本则表现出更大的情感变异性。为了捕捉这一现象，我们定义了两个补充指标：情感分布一致性和情感分布保持性，它们量化了情感改变和语义保持转换下的稳定性。我们在五个不同的数据集和一系列高级LLM上评估SentiDetect，包括Gemini-1.5-Pro、Claude-3、GPT-4-0613和LLaMa-3.3。实验结果证明，其优于最先进的基线，Gemini-1.5-Pro和GPT-4-0613的F1评分分别提高了16%和11%以上。此外，SentiDetect还对重述、对抗性攻击和文本长度变化表现出更强的鲁棒性，在具有挑战性的场景中优于现有的检测器。



## **35. Context Misleads LLMs: The Role of Context Filtering in Maintaining Safe Alignment of LLMs**

上下文误导LLM：上下文过滤在维护LLM安全一致中的作用 cs.CR

13 pages, 2 figures

**SubmitDate**: 2025-08-09    [abs](http://arxiv.org/abs/2508.10031v1) [paper-pdf](http://arxiv.org/pdf/2508.10031v1)

**Authors**: Jinhwa Kim, Ian G. Harris

**Abstract**: While Large Language Models (LLMs) have shown significant advancements in performance, various jailbreak attacks have posed growing safety and ethical risks. Malicious users often exploit adversarial context to deceive LLMs, prompting them to generate responses to harmful queries. In this study, we propose a new defense mechanism called Context Filtering model, an input pre-processing method designed to filter out untrustworthy and unreliable context while identifying the primary prompts containing the real user intent to uncover concealed malicious intent. Given that enhancing the safety of LLMs often compromises their helpfulness, potentially affecting the experience of benign users, our method aims to improve the safety of the LLMs while preserving their original performance. We evaluate the effectiveness of our model in defending against jailbreak attacks through comparative analysis, comparing our approach with state-of-the-art defense mechanisms against six different attacks and assessing the helpfulness of LLMs under these defenses. Our model demonstrates its ability to reduce the Attack Success Rates of jailbreak attacks by up to 88% while maintaining the original LLMs' performance, achieving state-of-the-art Safety and Helpfulness Product results. Notably, our model is a plug-and-play method that can be applied to all LLMs, including both white-box and black-box models, to enhance their safety without requiring any fine-tuning of the models themselves. We will make our model publicly available for research purposes.

摘要: 虽然大型语言模型（LLM）在性能方面表现出了显着的进步，但各种越狱攻击也带来了越来越大的安全和道德风险。恶意用户经常利用对抗上下文来欺骗LLM，促使他们生成对有害查询的响应。在这项研究中，我们提出了一种新的防御机制，称为上下文过滤模型，这是一种输入预处理方法，旨在过滤掉不可信和不可靠的上下文，同时识别包含真实用户意图的主要提示，以揭露隐藏的恶意意图。鉴于增强LLM的安全性往往会损害其帮助性，可能影响良性用户的体验，我们的方法旨在提高LLM的安全性，同时保留其原始性能。我们通过比较分析来评估我们的模型在防御越狱攻击方面的有效性，将我们的方法与针对六种不同攻击的最先进防御机制进行比较，并评估LLM在这些防御下的帮助性。我们的模型证明了它能够将越狱攻击的攻击成功率降低高达88%，同时保持原始LLM的性能，实现最先进的安全性和有用性产品结果。值得注意的是，我们的模型是一种即插即用方法，可以应用于所有LLM（包括白盒和黑匣子模型），以增强其安全性，而无需对模型本身进行任何微调。我们将公开我们的模型用于研究目的。



## **36. Towards Robust Red-Green Watermarking for Autoregressive Image Generators**

自回归图像生成器的鲁棒红-绿水印 cs.CV

**SubmitDate**: 2025-08-08    [abs](http://arxiv.org/abs/2508.06656v1) [paper-pdf](http://arxiv.org/pdf/2508.06656v1)

**Authors**: Denis Lukovnikov, Andreas Müller, Erwin Quiring, Asja Fischer

**Abstract**: In-generation watermarking for detecting and attributing generated content has recently been explored for latent diffusion models (LDMs), demonstrating high robustness. However, the use of in-generation watermarks in autoregressive (AR) image models has not been explored yet. AR models generate images by autoregressively predicting a sequence of visual tokens that are then decoded into pixels using a vector-quantized decoder. Inspired by red-green watermarks for large language models, we examine token-level watermarking schemes that bias the next-token prediction based on prior tokens. We find that a direct transfer of these schemes works in principle, but the detectability of the watermarks decreases considerably under common image perturbations. As a remedy, we propose two novel watermarking methods that rely on visual token clustering to assign similar tokens to the same set. Firstly, we investigate a training-free approach that relies on a cluster lookup table, and secondly, we finetune VAE encoders to predict token clusters directly from perturbed images. Overall, our experiments show that cluster-level watermarks improve robustness against perturbations and regeneration attacks while preserving image quality. Cluster classification further boosts watermark detectability, outperforming a set of baselines. Moreover, our methods offer fast verification runtime, comparable to lightweight post-hoc watermarking methods.

摘要: 最近，针对潜在扩散模型（LDM）探索了用于检测生成内容和属性的代内水印，证明了高鲁棒性。然而，尚未探索在自回归（AR）图像模型中使用代内水印。AR模型通过自回归预测视觉标记序列来生成图像，然后使用载体量化解码器将其解码为像素。受大型语言模型的红-绿水印的启发，我们研究了基于先前标记来偏置下一个标记预测的标记级水印方案。我们发现，这些方案的直接转移原则上是有效的，但在常见的图像扰动下，水印的检测能力显着下降。作为补救措施，我们提出了两种新颖的水印方法，这些方法依赖于视觉令牌集群来将相似的令牌分配给同一集合。首先，我们研究了一种依赖于集群查找表的免训练方法，其次，我们对VAE编码器进行微调，以直接从扰动图像中预测令牌集群。总体而言，我们的实验表明，集群级水印在保持图像质量的同时提高了对扰动和再生攻击的鲁棒性。集群分类进一步提高了水印的可检测性，优于一组基线。此外，我们的方法提供快速验证运行时，与轻量级事后水印方法相当。



## **37. Latent Fusion Jailbreak: Blending Harmful and Harmless Representations to Elicit Unsafe LLM Outputs**

潜在的融合越狱：将有害和无害的代表混合起来，以激发不安全的法学硕士产出 cs.CL

**SubmitDate**: 2025-08-08    [abs](http://arxiv.org/abs/2508.10029v1) [paper-pdf](http://arxiv.org/pdf/2508.10029v1)

**Authors**: Wenpeng Xing, Mohan Li, Chunqiang Hu, Haitao XuNingyu Zhang, Bo Lin, Meng Han

**Abstract**: Large language models (LLMs) demonstrate impressive capabilities in various language tasks but are susceptible to jailbreak attacks that circumvent their safety alignments. This paper introduces Latent Fusion Jailbreak (LFJ), a representation-based attack that interpolates hidden states from harmful and benign query pairs to elicit prohibited responses. LFJ begins by selecting query pairs with high thematic and syntactic similarity, then performs gradient-guided interpolation at influential layers and tokens, followed by optimization to balance attack success, output fluency, and computational efficiency. Evaluations on models such as Vicuna and LLaMA-2 across benchmarks like AdvBench and MaliciousInstruct yield an average attack success rate (ASR) of 94.01%, outperforming existing methods. To mitigate LFJ, we propose an adversarial training defense that fine-tunes models on interpolated examples, reducing ASR by over 80% without degrading performance on benign inputs. Ablation studies validate the importance of query pair selection, hidden state interpolation components, and optimization strategies in LFJ's effectiveness.

摘要: 大型语言模型（LLM）在各种语言任务中表现出令人印象深刻的能力，但容易受到绕过其安全对齐的越狱攻击。本文介绍了潜在融合越狱（LFJ），一种基于表示的攻击，从有害和良性的查询对中插入隐藏状态，以引起禁止的响应。LFJ首先选择具有高主题和语法相似性的查询对，然后在影响层和令牌上执行梯度引导的插值，然后进行优化以平衡攻击成功率，输出流畅性和计算效率。在AdvBench和MaliciousInstruct等基准上对Vicuna和LLaMA-2等模型进行评估，平均攻击成功率（ASR）为94.01%，优于现有方法。为了减轻LFJ，我们提出了一种对抗性训练防御，可以对内插示例进行微调，将ASB降低80%以上，而不会降低良性输入的性能。消融研究验证了查询对选择、隐藏状态插值组件和优化策略对LFJ有效性的重要性。



## **38. Turning Logic Against Itself : Probing Model Defenses Through Contrastive Questions**

将逻辑与自身对立：通过对比问题探索模型辩护 cs.CL

Our code is publicly available at  https://github.com/UKPLab/arxiv2025-poate-attack

**SubmitDate**: 2025-08-08    [abs](http://arxiv.org/abs/2501.01872v3) [paper-pdf](http://arxiv.org/pdf/2501.01872v3)

**Authors**: Rachneet Sachdeva, Rima Hazra, Iryna Gurevych

**Abstract**: Large language models, despite extensive alignment with human values and ethical principles, remain vulnerable to sophisticated jailbreak attacks that exploit their reasoning abilities. Existing safety measures often detect overt malicious intent but fail to address subtle, reasoning-driven vulnerabilities. In this work, we introduce POATE (Polar Opposite query generation, Adversarial Template construction, and Elaboration), a novel jailbreak technique that harnesses contrastive reasoning to provoke unethical responses. POATE crafts semantically opposing intents and integrates them with adversarial templates, steering models toward harmful outputs with remarkable subtlety. We conduct extensive evaluation across six diverse language model families of varying parameter sizes to demonstrate the robustness of the attack, achieving significantly higher attack success rates (~44%) compared to existing methods. To counter this, we propose Intent-Aware CoT and Reverse Thinking CoT, which decompose queries to detect malicious intent and reason in reverse to evaluate and reject harmful responses. These methods enhance reasoning robustness and strengthen the model's defense against adversarial exploits.

摘要: 尽管大型语言模型与人类价值观和道德原则广泛一致，但仍然容易受到利用其推理能力的复杂越狱攻击。现有的安全措施通常检测到明显的恶意意图，但无法解决微妙的、推理驱动的漏洞。在这项工作中，我们引入了POATE（极反相查询生成、对抗模板构建和搜索），这是一种新颖的越狱技术，利用对比推理来引发不道德的反应。POATE精心设计了语义上相反的意图，并将它们与对抗模板集成，以非凡的微妙性引导模型走向有害的输出。我们对参数大小不同的六个不同语言模型家族进行了广泛的评估，以证明攻击的稳健性，与现有方法相比，实现了显着更高的攻击成功率（~44%）。为了解决这个问题，我们提出了意图感知CoT和反向思维CoT，它们分解查询以检测恶意意图，并反向推理以评估和拒绝有害响应。这些方法增强了推理的稳健性并加强了模型对对抗性利用的防御。



## **39. Learning to Detect Unknown Jailbreak Attacks in Large Vision-Language Models: A Unified and Accurate Approach**

学习在大型视觉语言模型中检测未知越狱攻击：统一准确的方法 cs.CR

**SubmitDate**: 2025-08-08    [abs](http://arxiv.org/abs/2508.09201v1) [paper-pdf](http://arxiv.org/pdf/2508.09201v1)

**Authors**: Shuang Liang, Zhihao Xu, Jialing Tao, Hui Xue, Xiting Wang

**Abstract**: Despite extensive alignment efforts, Large Vision-Language Models (LVLMs) remain vulnerable to jailbreak attacks, posing serious safety risks. Although recent detection works have shifted to internal representations due to their rich cross-modal information, most methods rely on heuristic rules rather than principled objectives, resulting in suboptimal performance. To address these limitations, we propose Learning to Detect (LoD), a novel unsupervised framework that formulates jailbreak detection as anomaly detection. LoD introduces two key components: Multi-modal Safety Concept Activation Vectors (MSCAV), which capture layer-wise safety-related representations across modalities, and the Safety Pattern Auto-Encoder, which models the distribution of MSCAV derived from safe inputs and detects anomalies via reconstruction errors. By training the auto-encoder (AE) solely on safe samples without attack labels, LoD naturally identifies jailbreak inputs as distributional anomalies, enabling accurate and unified detection of jailbreak attacks. Comprehensive experiments on three different LVLMs and five benchmarks demonstrate that LoD achieves state-of-the-art performance, with an average AUROC of 0.9951 and an improvement of up to 38.89% in the minimum AUROC over the strongest baselines.

摘要: 尽管做出了广泛的协调努力，大型视觉语言模型（LVLM）仍然容易受到越狱攻击，从而构成严重的安全风险。尽管最近的检测工作由于其丰富的跨模式信息而转向内部表示，但大多数方法依赖于启发式规则而不是原则性目标，从而导致性能次优。为了解决这些局限性，我们提出了学习检测（Lo），这是一种新型的无监督框架，它将越狱检测制定为异常检测。DID引入了两个关键组件：多模式安全概念激活Vectors（MSCAB），它捕获跨模式的分层安全相关表示，以及安全模式自动编码器，它对从安全输入获得的MSCAB的分布进行建模，并通过重建错误检测异常。通过仅在没有攻击标签的安全样本上训练自动编码器（AE），DID自然地将越狱输入识别为分布异常，从而能够准确、统一地检测越狱攻击。对三种不同LVLM和五个基准的综合实验表明，DID实现了最先进的性能，平均AUROC为0.9951，最小AUROC比最强基线提高了38.89%。



## **40. In-Training Defenses against Emergent Misalignment in Language Models**

训练中防止语言模型中出现的失调 cs.LG

Under review

**SubmitDate**: 2025-08-08    [abs](http://arxiv.org/abs/2508.06249v1) [paper-pdf](http://arxiv.org/pdf/2508.06249v1)

**Authors**: David Kaczér, Magnus Jørgenvåg, Clemens Vetter, Lucie Flek, Florian Mai

**Abstract**: Fine-tuning lets practitioners repurpose aligned large language models (LLMs) for new domains, yet recent work reveals emergent misalignment (EMA): Even a small, domain-specific fine-tune can induce harmful behaviors far outside the target domain. Even in the case where model weights are hidden behind a fine-tuning API, this gives attackers inadvertent access to a broadly misaligned model in a way that can be hard to detect from the fine-tuning data alone. We present the first systematic study of in-training safeguards against EMA that are practical for providers who expose fine-tuning via an API. We investigate four training regularization interventions: (i) KL-divergence regularization toward a safe reference model, (ii) $\ell_2$ distance in feature space, (iii) projecting onto a safe subspace (SafeLoRA), and (iv) interleaving of a small amount of safe training examples from a general instruct-tuning dataset. We first evaluate the methods' emergent misalignment effect across four malicious, EMA-inducing tasks. Second, we assess the methods' impacts on benign tasks. We conclude with a discussion of open questions in emergent misalignment research.

摘要: 微调让从业者可以将对齐的大型语言模型（LLM）重新用于新领域，但最近的工作揭示了紧急失调（EMA）：即使是小型的、特定于领域的微调也可能会在目标领域之外引发有害行为。即使模型权重隐藏在微调API后面，这也会让攻击者无意中访问广泛失调的模型，而仅通过微调数据很难检测到。我们首次对针对EMA的培训中保障措施进行了系统性研究，这些措施对于通过API进行微调的提供商来说很实用。我们研究了四种训练正规化干预措施：（i）针对安全参考模型的KL-分歧正规化，（ii）特征空间中的$\ell_2 $距离，（iii）投影到安全子空间（SafeLoRA），以及（iv）交叉来自一般预算调整数据集的少量安全训练示例。我们首先评估这些方法在四个恶意、EMA诱导任务中的紧急失调效应。其次，我们评估这些方法对良性任务的影响。我们最后讨论了紧急失调研究中的未决问题。



## **41. Feedback-Guided Extraction of Knowledge Base from Retrieval-Augmented LLM Applications**

反馈引导从检索增强LLM应用程序中提取知识库 cs.CR

**SubmitDate**: 2025-08-08    [abs](http://arxiv.org/abs/2411.14110v2) [paper-pdf](http://arxiv.org/pdf/2411.14110v2)

**Authors**: Changyue Jiang, Xudong Pan, Geng Hong, Chenfu Bao, Yang Chen, Min Yang

**Abstract**: Retrieval-Augmented Generation (RAG) expands the knowledge boundary of large language models (LLMs) by integrating external knowledge bases, whose construction is often time-consuming and laborious. If an adversary extracts the knowledge base verbatim, it not only severely infringes the owner's intellectual property but also enables the adversary to replicate the application's functionality for unfair competition. Previous works on knowledge base extraction are limited either by low extraction coverage (usually less than 4%) in query-based attacks or by impractical assumptions of white-box access in embedding-based optimization methods. In this work, we propose CopyBreakRAG, an agent-based black-box attack that reasons from feedback and adaptively generates new adversarial queries for progressive extraction. By balancing exploration and exploitation through curiosity-driven queries and feedback-guided query refinement, our method overcomes the limitations of prior approaches and achieves significantly higher extraction coverage in realistic black-box settings. Experimental results show that CopyBreakRAG outperforms the state-of-the-art black-box approach by 45% on average in terms of chunk extraction ratio from applications built with mainstream RAG frameworks, and extracts over 70% of the data from the knowledge base in applications on commercial platforms including OpenAI's GPTs and ByteDance's Coze when essential protection is in place.

摘要: 检索增强生成（RAG）通过集成外部知识库来扩展大型语言模型（LLM）的知识边界，而外部知识库的构建通常既耗时又费力。如果对手逐字提取知识库，不仅会严重侵犯所有者的知识产权，还会使对手能够复制应用程序的功能以进行不公平竞争。之前关于知识库提取的工作要么受到基于查询的攻击中提取覆盖率低（通常低于4%）的限制，要么受到基于嵌入的优化方法中白盒访问不切实际的假设的限制。在这项工作中，我们提出了CopyBreakRAG，这是一种基于代理的黑匣子攻击，它从反馈中推理并自适应地生成新的对抗性查询以进行渐进式提取。通过通过好奇心驱动的查询和反馈引导的查询细化来平衡探索和利用，我们的方法克服了现有方法的局限性，并在现实的黑匣子设置中实现了显着更高的提取覆盖率。实验结果表明，CopyBreakRAG在主流RAG框架构建的应用程序中的块提取率方面平均优于最先进的黑盒方法45%，并且在必要保护到位的情况下，从OpenAI的GPT和字节跳动的Coze等商业平台上的应用程序的知识库中提取超过70%的数据。



## **42. LeakAgent: RL-based Red-teaming Agent for LLM Privacy Leakage**

LeakAgent：基于RL的LLM隐私泄露红色团队代理 cs.CR

Accepted by COLM 2025

**SubmitDate**: 2025-08-08    [abs](http://arxiv.org/abs/2412.05734v2) [paper-pdf](http://arxiv.org/pdf/2412.05734v2)

**Authors**: Yuzhou Nie, Zhun Wang, Ye Yu, Xian Wu, Xuandong Zhao, Wenbo Guo, Dawn Song

**Abstract**: Recent studies have discovered that large language models (LLM) may be ``fooled'' to output private information, including training data, system prompts, and personally identifiable information, under carefully crafted adversarial prompts. Existing red-teaming approaches for privacy leakage either rely on manual efforts or focus solely on system prompt extraction, making them ineffective for severe risks of training data leakage. We propose LeakAgent, a novel black-box red-teaming framework for LLM privacy leakage. Our framework trains an open-source LLM through reinforcement learning as the attack agent to generate adversarial prompts for both training data extraction and system prompt extraction. To achieve this, we propose a novel reward function to provide effective and fine-grained rewards and design novel mechanisms to balance exploration and exploitation during learning and enhance the diversity of adversarial prompts. Through extensive evaluations, we first show that LeakAgent significantly outperforms existing rule-based approaches in training data extraction and automated methods in system prompt leakage. We also demonstrate the effectiveness of LeakAgent in extracting system prompts from real-world applications in OpenAI's GPT Store. We further demonstrate LeakAgent's effectiveness in evading the existing guardrail defense and its helpfulness in enabling better safety alignment. Finally, we validate our customized designs through a detailed ablation study. We release our code here https://github.com/rucnyz/LeakAgent.

摘要: 最近的研究发现，大型语言模型（LLM）可能会“愚蠢”地在精心设计的对抗性提示下输出私人信息，包括训练数据、系统提示和个人可识别信息。现有的隐私泄露红色团队方法要么依赖手动工作，要么仅关注系统即时提取，这使得它们对训练数据泄露的严重风险无效。我们提出了LeakAgent，这是一种用于LLM隐私泄露的新型黑匣子红团队框架。我们的框架通过强化学习训练开源LLM作为攻击代理，为训练数据提取和系统提示提取生成对抗提示。为了实现这一目标，我们提出了一种新颖的奖励函数来提供有效且细粒度的奖励，并设计新颖的机制来平衡学习期间的探索和利用，并增强对抗提示的多样性。通过广泛的评估，我们首先表明LeakAgent在训练数据提取方面的表现显着优于现有的基于规则的方法和系统提示泄漏的自动化方法。我们还展示了LeakAgent从OpenAI的GPT Store中的现实应用程序中提取系统提示的有效性。我们进一步证明了LeakAgent在规避现有护栏防御方面的有效性及其在实现更好的安全对齐方面的帮助。最后，我们通过详细的消融研究验证了我们的定制设计。我们在此处发布我们的代码https://github.com/rucnyz/LeakAgent。



## **43. SLIP: Soft Label Mechanism and Key-Extraction-Guided CoT-based Defense Against Instruction Backdoor in APIs**

SLIP：软标签机制和密钥提取引导的CoT API指令后门防御 cs.CR

**SubmitDate**: 2025-08-08    [abs](http://arxiv.org/abs/2508.06153v1) [paper-pdf](http://arxiv.org/pdf/2508.06153v1)

**Authors**: Zhengxian Wu, Juan Wen, Wanli Peng, Haowei Chang, Yinghan Zhou, Yiming Xue

**Abstract**: With the development of customized large language model (LLM) agents, a new threat of black-box backdoor attacks has emerged, where malicious instructions are injected into hidden system prompts. These attacks easily bypass existing defenses that rely on white-box access, posing a serious security challenge. To address this, we propose SLIP, a Soft Label mechanism and key-extraction-guided CoT-based defense against Instruction backdoors in APIs. SLIP is designed based on two key insights. First, to counteract the model's oversensitivity to triggers, we propose a Key-extraction-guided Chain-of-Thought (KCoT). Instead of only considering the single trigger or the input sentence, KCoT prompts the agent to extract task-relevant key phrases. Second, to guide the LLM toward correct answers, our proposed Soft Label Mechanism (SLM) prompts the agent to quantify the semantic correlation between key phrases and candidate answers. Crucially, to mitigate the influence of residual triggers or misleading content in phrases extracted by KCoT, which typically causes anomalous scores, SLM excludes anomalous scores deviating significantly from the mean and subsequently averages the remaining scores to derive a more reliable semantic representation. Extensive experiments on classification and question-answer (QA) tasks demonstrate that SLIP is highly effective, reducing the average attack success rate (ASR) from 90.2% to 25.13% while maintaining high accuracy on clean data and outperforming state-of-the-art defenses. Our code are available in https://github.com/CAU-ISS-Lab/Backdoor-Attack-Defense-LLMs/tree/main/SLIP.

摘要: 随着定制化大语言模型（LLM）代理的发展，出现了一种新的黑盒后门攻击威胁，其中恶意指令被注入隐藏的系统提示符中。这些攻击很容易绕过依赖白盒访问的现有防御，构成了严重的安全挑战。为了解决这个问题，我们提出了SLIP，一个软标签机制和密钥提取引导的CoT为基础的防御API中的指令后门。SLIP的设计基于两个关键的见解。首先，为了抵消模型对触发器的过度敏感，我们提出了一种密钥提取引导的思想链（KCoT）。KCoT不是只考虑单个触发器或输入句子，而是提示代理提取与任务相关的关键短语。其次，为了引导LLM找到正确的答案，我们提出的软标签机制（LAM）提示代理量化关键短语和候选答案之间的语义相关性。至关重要的是，为了减轻KCoT提取的短语中的残留触发或误导性内容（通常会导致异常分数）的影响，STM排除了与平均值显着偏离的异常分数，然后对剩余分数进行平均以得出更可靠的语义表示。在分类和问答（QA）任务上的大量实验表明，SLIP非常有效，将平均攻击成功率（ASR）从90.2%降低到25.13%，同时保持对干净数据的高准确性，并优于最先进的防御。我们的代码可在https://github.com/CAU-ISS-Lab/Backdoor-Attack-Defense-LLMs/tree/main/SLIP上获取。



## **44. Fine-Grained Safety Neurons with Training-Free Continual Projection to Reduce LLM Fine Tuning Risks**

具有免训练连续投影的细粒度安全神经元，以降低LLM微调风险 cs.LG

**SubmitDate**: 2025-08-08    [abs](http://arxiv.org/abs/2508.09190v1) [paper-pdf](http://arxiv.org/pdf/2508.09190v1)

**Authors**: Bing Han, Feifei Zhao, Dongcheng Zhao, Guobin Shen, Ping Wu, Yu Shi, Yi Zeng

**Abstract**: Fine-tuning as service injects domain-specific knowledge into large language models (LLMs), while challenging the original alignment mechanisms and introducing safety risks. A series of defense strategies have been proposed for the alignment, fine-tuning, and post-fine-tuning phases, where most post-fine-tuning defenses rely on coarse-grained safety layer mapping. These methods lack a comprehensive consideration of both safety layers and fine-grained neurons, limiting their ability to efficiently balance safety and utility. To address this, we propose the Fine-Grained Safety Neurons (FGSN) with Training-Free Continual Projection method to reduce the fine-tuning safety risks. FGSN inherently integrates the multi-scale interactions between safety layers and neurons, localizing sparser and more precise fine-grained safety neurons while minimizing interference with downstream task neurons. We then project the safety neuron parameters onto safety directions, improving model safety while aligning more closely with human preferences. Extensive experiments across multiple fine-tuned LLM models demonstrate that our method significantly reduce harmfulness scores and attack success rates with minimal parameter modifications, while preserving the model's utility. Furthermore, by introducing a task-specific, multi-dimensional heterogeneous safety neuron cluster optimization mechanism, we achieve continual defense and generalization capability against unforeseen emerging safety concerns.

摘要: 微调即服务将特定领域的知识注入到大型语言模型（LLM）中，同时挑战了原始的对齐机制并引入了安全风险。针对对齐、微调和微调后阶段提出了一系列防御策略，其中大多数微调后防御依赖于粗粒度安全层映射。这些方法缺乏对安全层和细粒度神经元的综合考虑，限制了它们有效平衡安全性和实用性的能力。为了解决这个问题，我们提出了细粒度安全神经元（FGSN）与训练免费连续投影方法，以减少微调的安全风险。FGSN固有地集成了安全层和神经元之间的多尺度交互，定位更稀疏和更精确的细粒度安全神经元，同时最大限度地减少对下游任务神经元的干扰。然后，我们将安全神经元参数投影到安全方向上，提高模型的安全性，同时更紧密地与人类偏好保持一致。在多个微调的LLM模型上进行的广泛实验表明，我们的方法在保持模型实用性的同时，以最小的参数修改显着降低了危害分数和攻击成功率。此外，通过引入特定于任务的多维异构安全神经元簇优化机制，我们实现了对不可预见的新出现的安全问题的持续防御和泛化能力。



## **45. Safety of Embodied Navigation: A Survey**

水下航行安全：调查 cs.AI

**SubmitDate**: 2025-08-07    [abs](http://arxiv.org/abs/2508.05855v1) [paper-pdf](http://arxiv.org/pdf/2508.05855v1)

**Authors**: Zixia Wang, Jia Hu, Ronghui Mu

**Abstract**: As large language models (LLMs) continue to advance and gain influence, the development of embodied AI has accelerated, drawing significant attention, particularly in navigation scenarios. Embodied navigation requires an agent to perceive, interact with, and adapt to its environment while moving toward a specified target in unfamiliar settings. However, the integration of embodied navigation into critical applications raises substantial safety concerns. Given their deployment in dynamic, real-world environments, ensuring the safety of such systems is critical. This survey provides a comprehensive analysis of safety in embodied navigation from multiple perspectives, encompassing attack strategies, defense mechanisms, and evaluation methodologies. Beyond conducting a comprehensive examination of existing safety challenges, mitigation technologies, and various datasets and metrics that assess effectiveness and robustness, we explore unresolved issues and future research directions in embodied navigation safety. These include potential attack methods, mitigation strategies, more reliable evaluation techniques, and the implementation of verification frameworks. By addressing these critical gaps, this survey aims to provide valuable insights that can guide future research toward the development of safer and more reliable embodied navigation systems. Furthermore, the findings of this study have broader implications for enhancing societal safety and increasing industrial efficiency.

摘要: 随着大型语言模型（LLM）的不断进步并获得影响力，体现式人工智能的发展加速，引起了人们的广泛关注，特别是在导航场景中。同步导航需要代理感知、交互和适应其环境，同时在不熟悉的环境中向指定目标移动。然而，将具体导航集成到关键应用程序中会引发巨大的安全问题。鉴于它们部署在动态的现实世界环境中，确保此类系统的安全至关重要。这项调查从多个角度对嵌入式导航的安全性进行了全面分析，包括攻击策略、防御机制和评估方法。除了对现有的安全挑战、缓解技术以及评估有效性和稳健性的各种数据集和指标进行全面检查外，我们还探索了具体导航安全方面未解决的问题和未来的研究方向。其中包括潜在的攻击方法、缓解策略、更可靠的评估技术以及验证框架的实施。通过解决这些关键差距，这项调查旨在提供有价值的见解，指导未来的研究开发更安全、更可靠的嵌入式导航系统。此外，这项研究的结果对加强社会安全和提高工业效率具有更广泛的影响。



## **46. No Query, No Access**

无查询，无访问 cs.CL

**SubmitDate**: 2025-08-07    [abs](http://arxiv.org/abs/2505.07258v2) [paper-pdf](http://arxiv.org/pdf/2505.07258v2)

**Authors**: Wenqiang Wang, Siyuan Liang, Yangshijie Zhang, Xiaojun Jia, Hao Lin, Xiaochun Cao

**Abstract**: Textual adversarial attacks mislead NLP models, including Large Language Models (LLMs), by subtly modifying text. While effective, existing attacks often require knowledge of the victim model, extensive queries, or access to training data, limiting real-world feasibility. To overcome these constraints, we introduce the \textbf{Victim Data-based Adversarial Attack (VDBA)}, which operates using only victim texts. To prevent access to the victim model, we create a shadow dataset with publicly available pre-trained models and clustering methods as a foundation for developing substitute models. To address the low attack success rate (ASR) due to insufficient information feedback, we propose the hierarchical substitution model design, generating substitute models to mitigate the failure of a single substitute model at the decision boundary.   Concurrently, we use diverse adversarial example generation, employing various attack methods to generate and select the adversarial example with better similarity and attack effectiveness. Experiments on the Emotion and SST5 datasets show that VDBA outperforms state-of-the-art methods, achieving an ASR improvement of 52.08\% while significantly reducing attack queries to 0. More importantly, we discover that VDBA poses a significant threat to LLMs such as Qwen2 and the GPT family, and achieves the highest ASR of 45.99% even without access to the API, confirming that advanced NLP models still face serious security risks. Our codes can be found at https://anonymous.4open.science/r/VDBA-Victim-Data-based-Adversarial-Attack-36EC/

摘要: 文本对抗攻击通过微妙地修改文本来误导NLP模型，包括大型语言模型（LLM）。虽然有效，但现有的攻击通常需要了解受害者模型、广泛的查询或访问训练数据，从而限制了现实世界的可行性。为了克服这些限制，我们引入了\textBF{基于受害者数据的对抗攻击（VDBA）}，它仅使用受害者文本来操作。为了防止访问受害者模型，我们创建了一个影子数据集，其中包含公开可用的预训练模型和集群方法，作为开发替代模型的基础。为了解决由于信息反馈不足而导致的攻击成功率（ASB）低的问题，我们提出了分层替代模型设计，生成替代模型以减轻单个替代模型在决策边界的失败。   同时，我们使用多样化的对抗性示例生成，采用各种攻击方法来生成并选择具有更好相似性和攻击有效性的对抗性示例。Emoy和CST 5数据集的实验表明，VDBA优于最先进的方法，实现了52.08%的ASB改进，同时将攻击查询显着减少到0。更重要的是，我们发现VDBA对Qwen 2和GPT系列等LLM构成了重大威胁，即使在不访问API的情况下也能达到45.99%的最高ASB，证实高级NLP模型仍然面临严重的安全风险。我们的代码可在https://anonymous.4open.science/r/VDBA-Victim-Data-based-Adversarial-Attack-36EC/上找到



## **47. JULI: Jailbreak Large Language Models by Self-Introspection**

JULI：通过自我内省越狱大型语言模型 cs.LG

**SubmitDate**: 2025-08-07    [abs](http://arxiv.org/abs/2505.11790v3) [paper-pdf](http://arxiv.org/pdf/2505.11790v3)

**Authors**: Jesson Wang, Zhanhao Hu, David Wagner

**Abstract**: Large Language Models (LLMs) are trained with safety alignment to prevent generating malicious content. Although some attacks have highlighted vulnerabilities in these safety-aligned LLMs, they typically have limitations, such as necessitating access to the model weights or the generation process. Since proprietary models through API-calling do not grant users such permissions, these attacks find it challenging to compromise them. In this paper, we propose Jailbreaking Using LLM Introspection (JULI), which jailbreaks LLMs by manipulating the token log probabilities, using a tiny plug-in block, BiasNet. JULI relies solely on the knowledge of the target LLM's predicted token log probabilities. It can effectively jailbreak API-calling LLMs under a black-box setting and knowing only top-$5$ token log probabilities. Our approach demonstrates superior effectiveness, outperforming existing state-of-the-art (SOTA) approaches across multiple metrics.

摘要: 大型语言模型（LLM）经过安全调整训练，以防止生成恶意内容。尽管一些攻击凸显了这些安全一致的LLM中的漏洞，但它们通常具有局限性，例如需要访问模型权重或生成过程。由于通过API调用的专有模型不会向用户授予此类权限，因此这些攻击发现很难损害它们。在本文中，我们提出了使用LLM内省越狱（JULI），它通过使用一个微型插件块BiasNet操纵令牌日志概率来越狱LLM。JULI仅依赖于目标LLM的预测令牌日志概率的知识。它可以在黑盒设置下有效地越狱API调用LLM，并且只知道top-$5$token日志概率。我们的方法表现出卓越的有效性，在多个指标上优于现有的最先进（SOTA）方法。



## **48. From Detection to Correction: Backdoor-Resilient Face Recognition via Vision-Language Trigger Detection and Noise-Based Neutralization**

从检测到纠正：通过视觉语言触发检测和基于噪音的中和来实现后门弹性人脸识别 cs.CV

19 Pages, 24 Figures

**SubmitDate**: 2025-08-07    [abs](http://arxiv.org/abs/2508.05409v1) [paper-pdf](http://arxiv.org/pdf/2508.05409v1)

**Authors**: Farah Wahida, M. A. P. Chamikara, Yashothara Shanmugarasa, Mohan Baruwal Chhetri, Thilina Ranbaduge, Ibrahim Khalil

**Abstract**: Biometric systems, such as face recognition systems powered by deep neural networks (DNNs), rely on large and highly sensitive datasets. Backdoor attacks can subvert these systems by manipulating the training process. By inserting a small trigger, such as a sticker, make-up, or patterned mask, into a few training images, an adversary can later present the same trigger during authentication to be falsely recognized as another individual, thereby gaining unauthorized access. Existing defense mechanisms against backdoor attacks still face challenges in precisely identifying and mitigating poisoned images without compromising data utility, which undermines the overall reliability of the system. We propose a novel and generalizable approach, TrueBiometric: Trustworthy Biometrics, which accurately detects poisoned images using a majority voting mechanism leveraging multiple state-of-the-art large vision language models. Once identified, poisoned samples are corrected using targeted and calibrated corrective noise. Our extensive empirical results demonstrate that TrueBiometric detects and corrects poisoned images with 100\% accuracy without compromising accuracy on clean images. Compared to existing state-of-the-art approaches, TrueBiometric offers a more practical, accurate, and effective solution for mitigating backdoor attacks in face recognition systems.

摘要: 生物识别系统，例如由深度神经网络（DNN）支持的面部识别系统，依赖于大型且高度敏感的数据集。后门攻击可以通过操纵训练过程来颠覆这些系统。通过将小触发器（例如贴纸、化妆品或图案化的面具）插入到一些训练图像中，对手稍后可以在认证期间呈现相同的触发器，从而被错误地识别为另一个人，从而获得未经授权的访问。现有的针对后门攻击的防御机制在准确识别和减轻中毒图像而不损害数据实用性方面仍然面临挑战，这会损害系统的整体可靠性。我们提出了一种新颖且可推广的方法：TrueBimetric：Trustworthy Bimetrics，它使用多数投票机制利用多个最先进的大视觉语言模型来准确检测有毒图像。一旦被识别出来，有毒样本就会使用有针对性和校准的纠正噪音进行纠正。我们广泛的经验结果表明，TrueBimetric可以以100%的准确性检测和纠正有毒图像，而不会影响干净图像的准确性。与现有的最先进方法相比，TrueBimetric提供了一种更实用、准确和有效的解决方案，用于减轻面部识别系统中的后门攻击。



## **49. PhysPatch: A Physically Realizable and Transferable Adversarial Patch Attack for Multimodal Large Language Models-based Autonomous Driving Systems**

物理补丁：一种针对基于多模式大语言模型的自动驾驶系统的物理可实现且可转移的对抗补丁攻击 cs.CV

**SubmitDate**: 2025-08-07    [abs](http://arxiv.org/abs/2508.05167v1) [paper-pdf](http://arxiv.org/pdf/2508.05167v1)

**Authors**: Qi Guo, Xiaojun Jia, Shanmin Pang, Simeng Qin, Lin Wang, Ju Jia, Yang Liu, Qing Guo

**Abstract**: Multimodal Large Language Models (MLLMs) are becoming integral to autonomous driving (AD) systems due to their strong vision-language reasoning capabilities. However, MLLMs are vulnerable to adversarial attacks, particularly adversarial patch attacks, which can pose serious threats in real-world scenarios. Existing patch-based attack methods are primarily designed for object detection models and perform poorly when transferred to MLLM-based systems due to the latter's complex architectures and reasoning abilities. To address these limitations, we propose PhysPatch, a physically realizable and transferable adversarial patch framework tailored for MLLM-based AD systems. PhysPatch jointly optimizes patch location, shape, and content to enhance attack effectiveness and real-world applicability. It introduces a semantic-based mask initialization strategy for realistic placement, an SVD-based local alignment loss with patch-guided crop-resize to improve transferability, and a potential field-based mask refinement method. Extensive experiments across open-source, commercial, and reasoning-capable MLLMs demonstrate that PhysPatch significantly outperforms prior methods in steering MLLM-based AD systems toward target-aligned perception and planning outputs. Moreover, PhysPatch consistently places adversarial patches in physically feasible regions of AD scenes, ensuring strong real-world applicability and deployability.

摘要: 多模式大型语言模型（MLLM）因其强大的视觉语言推理能力而成为自动驾驶（AD）系统不可或缺的一部分。然而，MLLM很容易受到对抗性攻击，尤其是对抗性补丁攻击，这可能会在现实世界场景中构成严重威胁。现有的基于补丁的攻击方法主要是为对象检测模型设计的，并且由于后者复杂的架构和推理能力，当转移到基于MLLM的系统时性能较差。为了解决这些限制，我们提出了Physpatch，这是一种物理上可实现且可转移的对抗性补丁框架，专为基于MLLM的AD系统量身定制。Physpatch联合优化补丁位置、形状和内容，以增强攻击有效性和现实世界的适用性。它引入了基于语义的掩模初始化策略，用于真实放置、基于ASD的局部对齐损失和贴片引导剪裁调整大小以提高可移植性，以及潜在的基于场的掩模细化方法。跨开源、商业和具有推理能力的MLLM的广泛实验表明，Physpatch在引导基于MLLM的AD系统实现目标一致的感知和规划输出方面显着优于先前的方法。此外，Physpatch始终将对抗补丁放置在AD场景的物理可行区域中，确保强大的现实适用性和可部署性。



## **50. AI Agent Smart Contract Exploit Generation**

AI Agent智能合同漏洞生成 cs.CR

**SubmitDate**: 2025-08-07    [abs](http://arxiv.org/abs/2507.05558v3) [paper-pdf](http://arxiv.org/pdf/2507.05558v3)

**Authors**: Arthur Gervais, Liyi Zhou

**Abstract**: Smart contract vulnerabilities have led to billions in losses, yet finding actionable exploits remains challenging. Traditional fuzzers rely on rigid heuristics and struggle with complex attacks, while human auditors are thorough but slow and don't scale. Large Language Models offer a promising middle ground, combining human-like reasoning with machine speed.   However, early studies show that simply prompting LLMs generates unverified vulnerability speculations with high false positive rates. To address this, we present A1, an agentic system that transforms any LLM into an end-to-end exploit generator. A1 provides agents with six domain-specific tools for autonomous vulnerability discovery, from understanding contract behavior to testing strategies on real blockchain states. All outputs are concretely validated through execution, ensuring only profitable proof-of-concept exploits are reported. We evaluate A1 across 36 real-world vulnerable contracts on Ethereum and Binance Smart Chain. A1 achieves a 63% success rate on the VERITE benchmark. Across all successful cases, A1 extracts up to \$8.59 million per exploit and \$9.33 million total. Through 432 experiments across six LLMs, we show that most exploits emerge within five iterations, with costs ranging \$0.01-\$3.59 per attempt.   Using Monte Carlo analysis of historical attacks, we demonstrate that immediate vulnerability detection yields 86-89% success probability, dropping to 6-21% with week-long delays. Our economic analysis reveals a troubling asymmetry: attackers achieve profitability at \$6,000 exploit values while defenders require \$60,000 -- raising fundamental questions about whether AI agents inevitably favor exploitation over defense.

摘要: 智能合同漏洞已导致数十亿美元的损失，但发现可采取行动的漏洞仍然具有挑战性。传统的模糊器依赖于严格的启发式方法，并与复杂的攻击作斗争，而人类审计员很彻底，但速度缓慢，而且无法扩展。大型语言模型提供了一个有希望的中间立场，将类人推理与机器速度相结合。   然而，早期研究表明，简单地提示LLM会产生未经验证且误报率很高的漏洞猜测。为了解决这个问题，我们提出了A1，这是一个代理系统，可以将任何LLM转换为端到端漏洞利用生成器。A1为代理提供了六种特定于领域的工具，用于自主漏洞发现，从理解合同行为到测试真实区块链状态的策略。所有输出都通过执行进行具体验证，确保仅报告有利可图的概念验证漏洞。我们评估了以太坊和币安智能链上36个现实世界的脆弱合同的A1。A1在VERITE基准测试上实现了63%的成功率。在所有成功的案例中，A1每次利用可提取高达859万英镑，总计可提取933万英镑。通过六个LLM的432个实验，我们表明大多数漏洞利用会在五次迭代内出现，每次尝试的成本范围为0.01 - 3.59美元。   使用对历史攻击的蒙特卡洛分析，我们证明立即漏洞检测的成功率为86-89%，在长达一周的延迟后，成功率下降到6-21%。我们的经济分析揭示了一种令人不安的不对称性：攻击者以6，000英镑的剥削价值实现盈利，而防御者则需要60，000英镑--这引发了关于人工智能代理是否不可避免地更喜欢剥削而不是防御的基本问题。



