# Latest Adversarial Attack Papers
**update at 2025-06-18 11:07:13**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. IP Leakage Attacks Targeting LLM-Based Multi-Agent Systems**

针对基于LLM的多代理系统的IP泄露攻击 cs.CR

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2505.12442v3) [paper-pdf](http://arxiv.org/pdf/2505.12442v3)

**Authors**: Liwen Wang, Wenxuan Wang, Shuai Wang, Zongjie Li, Zhenlan Ji, Zongyi Lyu, Daoyuan Wu, Shing-Chi Cheung

**Abstract**: The rapid advancement of Large Language Models (LLMs) has led to the emergence of Multi-Agent Systems (MAS) to perform complex tasks through collaboration. However, the intricate nature of MAS, including their architecture and agent interactions, raises significant concerns regarding intellectual property (IP) protection. In this paper, we introduce MASLEAK, a novel attack framework designed to extract sensitive information from MAS applications. MASLEAK targets a practical, black-box setting, where the adversary has no prior knowledge of the MAS architecture or agent configurations. The adversary can only interact with the MAS through its public API, submitting attack query $q$ and observing outputs from the final agent. Inspired by how computer worms propagate and infect vulnerable network hosts, MASLEAK carefully crafts adversarial query $q$ to elicit, propagate, and retain responses from each MAS agent that reveal a full set of proprietary components, including the number of agents, system topology, system prompts, task instructions, and tool usages. We construct the first synthetic dataset of MAS applications with 810 applications and also evaluate MASLEAK against real-world MAS applications, including Coze and CrewAI. MASLEAK achieves high accuracy in extracting MAS IP, with an average attack success rate of 87% for system prompts and task instructions, and 92% for system architecture in most cases. We conclude by discussing the implications of our findings and the potential defenses.

摘要: 大型语言模型（LLM）的快速发展导致了通过协作执行复杂任务的多智能体系统（MAS）的出现。然而，MAS的复杂性质，包括其架构和代理交互，引发了有关知识产权（IP）保护的严重担忧。本文介绍MASLEAK，这是一种新型攻击框架，旨在从MAS应用程序中提取敏感信息。MASLEAK针对的是实用的黑匣子设置，其中对手不了解MAS架构或代理配置。对手只能通过其公共API与MAS交互，提交攻击查询$q$并观察最终代理的输出。受计算机蠕虫传播和感染脆弱网络主机的方式的启发，MASLEAK精心设计了对抗性查询$q$，以引发、传播和保留每个MAS代理的响应，这些响应揭示了全套专有组件，包括代理数量、系统布局、系统提示、任务指令和工具使用。我们构建了包含810个应用程序的第一个MAS应用程序合成数据集，并根据现实世界的MAS应用程序（包括Coze和CrewAI）评估MASLEAK。MASLEAK在提取MAS IP方面实现了高准确性，系统提示和任务指令的平均攻击成功率为87%，大多数情况下系统架构的平均攻击成功率为92%。最后，我们讨论了我们发现的影响和潜在的防御措施。



## **2. Variational Bayesian Bow tie Neural Networks with Shrinkage**

具有收缩性的变分Bayesian领结神经网络 stat.ML

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2411.11132v3) [paper-pdf](http://arxiv.org/pdf/2411.11132v3)

**Authors**: Alisa Sheinkman, Sara Wade

**Abstract**: Despite the dominant role of deep models in machine learning, limitations persist, including overconfident predictions, susceptibility to adversarial attacks, and underestimation of variability in predictions. The Bayesian paradigm provides a natural framework to overcome such issues and has become the gold standard for uncertainty estimation with deep models, also providing improved accuracy and a framework for tuning critical hyperparameters. However, exact Bayesian inference is challenging, typically involving variational algorithms that impose strong independence and distributional assumptions. Moreover, existing methods are sensitive to the architectural choice of the network. We address these issues by focusing on a stochastic relaxation of the standard feed-forward rectified neural network and using sparsity-promoting priors on the weights of the neural network for increased robustness to architectural design. Thanks to Polya-Gamma data augmentation tricks, which render a conditionally linear and Gaussian model, we derive a fast, approximate variational inference algorithm that avoids distributional assumptions and independence across layers. Suitable strategies to further improve scalability and account for multimodality are considered.

摘要: 尽管深度模型在机器学习中发挥主导作用，但局限性仍然存在，包括过度自信的预测、容易受到对抗攻击以及低估预测的变异性。Bayesian范式提供了克服此类问题的自然框架，并已成为深度模型不确定性估计的黄金标准，还提供了更高的准确性和调整关键超参数的框架。然而，精确的Bayesian推断具有挑战性，通常涉及施加强独立性和分布假设的变分算法。此外，现有方法对网络的架构选择很敏感。我们通过关注标准前向修正神经网络的随机松弛并在神经网络的权重上使用稀疏促进先验来提高建筑设计的鲁棒性来解决这些问题。得益于Polya-Gamma数据增强技巧，它呈现了有条件的线性和高斯模型，我们推导出了一种快速、近似的变分推理算法，可以避免分布假设和跨层的独立性。考虑了进一步提高可扩展性并考虑多模式的适当策略。



## **3. Busting the Paper Ballot: Voting Meets Adversarial Machine Learning**

打破纸质选票：投票遇到对抗性机器学习 cs.CR

18 Pages. Author version of article to appear at CCS 2025

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2506.14582v1) [paper-pdf](http://arxiv.org/pdf/2506.14582v1)

**Authors**: Kaleel Mahmood, Caleb Manicke, Ethan Rathbun, Aayushi Verma, Sohaib Ahmad, Nicholas Stamatakis, Laurent Michel, Benjamin Fuller

**Abstract**: We show the security risk associated with using machine learning classifiers in United States election tabulators. The central classification task in election tabulation is deciding whether a mark does or does not appear on a bubble associated to an alternative in a contest on the ballot. Barretto et al. (E-Vote-ID 2021) reported that convolutional neural networks are a viable option in this field, as they outperform simple feature-based classifiers.   Our contributions to election security can be divided into four parts. To demonstrate and analyze the hypothetical vulnerability of machine learning models on election tabulators, we first introduce four new ballot datasets. Second, we train and test a variety of different models on our new datasets. These models include support vector machines, convolutional neural networks (a basic CNN, VGG and ResNet), and vision transformers (Twins and CaiT). Third, using our new datasets and trained models, we demonstrate that traditional white box attacks are ineffective in the voting domain due to gradient masking. Our analyses further reveal that gradient masking is a product of numerical instability. We use a modified difference of logits ratio loss to overcome this issue (Croce and Hein, ICML 2020). Fourth, in the physical world, we conduct attacks with the adversarial examples generated using our new methods. In traditional adversarial machine learning, a high (50% or greater) attack success rate is ideal. However, for certain elections, even a 5% attack success rate can flip the outcome of a race. We show such an impact is possible in the physical domain. We thoroughly discuss attack realism, and the challenges and practicality associated with printing and scanning ballot adversarial examples.

摘要: 我们展示了在美国选举制表器中使用机器学习分类器相关的安全风险。选举制表中的核心分类任务是决定标记是否出现在与选票竞争中的替代方案相关的气泡上。Barretto等人（E-Vote-ID 2021）报告称，卷积神经网络是该领域的一个可行选择，因为它们的性能优于简单的基于特征的分类器。   我们对选举安全的贡献可以分为四个部分。为了证明和分析机器学习模型在选举制表机上的假设脆弱性，我们首先介绍了四个新的选票数据集。其次，我们在新数据集上训练和测试各种不同的模型。这些模型包括支持向量机，卷积神经网络（基本的CNN，VGG和ResNet）和视觉转换器（Twins和CaiT）。第三，使用我们的新数据集和训练模型，我们证明了由于梯度掩蔽，传统的白盒攻击在投票域中是无效的。我们的分析进一步揭示了梯度掩蔽是数值不稳定性的产物。我们使用修改后的logits比损失差异来克服这个问题（Croce和Hein，ICML 2020）。第四，在物理世界中，我们使用使用新方法生成的对抗示例进行攻击。在传统的对抗性机器学习中，高（50%或更高）的攻击成功率是理想的。然而，对于某些选举来说，即使是5%的攻击成功率也可能会改变竞选结果。我们表明这种影响在物理领域是可能的。我们彻底讨论了攻击现实主义，以及与打印和扫描选票对抗性例子相关的挑战和实用性。



## **4. Doppelgänger Method: Breaking Role Consistency in LLM Agent via Prompt-based Transferable Adversarial Attack**

Doppelgänger方法：通过基于预算的可转移对抗攻击打破LLM代理中的角色一致性 cs.AI

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2506.14539v1) [paper-pdf](http://arxiv.org/pdf/2506.14539v1)

**Authors**: Daewon Kang, YeongHwan Shin, Doyeon Kim, Kyu-Hwan Jung, Meong Hi Son

**Abstract**: Since the advent of large language models, prompt engineering now enables the rapid, low-effort creation of diverse autonomous agents that are already in widespread use. Yet this convenience raises urgent concerns about the safety, robustness, and behavioral consistency of the underlying prompts, along with the pressing challenge of preventing those prompts from being exposed to user's attempts. In this paper, we propose the ''Doppelg\"anger method'' to demonstrate the risk of an agent being hijacked, thereby exposing system instructions and internal information. Next, we define the ''Prompt Alignment Collapse under Adversarial Transfer (PACAT)'' level to evaluate the vulnerability to this adversarial transfer attack. We also propose a ''Caution for Adversarial Transfer (CAT)'' prompt to counter the Doppelg\"anger method. The experimental results demonstrate that the Doppelg\"anger method can compromise the agent's consistency and expose its internal information. In contrast, CAT prompts enable effective defense against this adversarial attack.

摘要: 自从大型语言模型的出现以来，即时工程现在可以快速、低努力地创建已经广泛使用的各种自治代理。然而，这种便利性引发了人们对底层提示的安全性、稳健性和行为一致性的紧迫担忧，以及防止这些提示暴露于用户尝试的紧迫挑战。在本文中，我们提出了“Doppelg”愤怒方法来演示代理被劫持从而暴露系统指令和内部信息的风险。接下来，我们定义“对抗性转移下的提示对齐崩溃（PACAT RST）”级别来评估这种对抗性转移攻击的脆弱性。我们还提出了一个“对抗性转移警告（CAT）”提示来对抗Doppelg '愤怒方法。实验结果表明，Doppelg愤怒方法会损害代理人的一致性并暴露其内部信息。相比之下，CAT提示可以有效防御这种对抗性攻击。



## **5. A Comparative Study on Proactive and Passive Detection of Deepfake Speech**

Deepfake语音主动和被动检测的比较研究 cs.SD

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2506.14398v1) [paper-pdf](http://arxiv.org/pdf/2506.14398v1)

**Authors**: Chia-Hua Wu, Wanying Ge, Xin Wang, Junichi Yamagishi, Yu Tsao, Hsin-Min Wang

**Abstract**: Solutions for defending against deepfake speech fall into two categories: proactive watermarking models and passive conventional deepfake detectors. While both address common threats, their differences in training, optimization, and evaluation prevent a unified protocol for joint evaluation and selecting the best solutions for different cases. This work proposes a framework to evaluate both model types in deepfake speech detection. To ensure fair comparison and minimize discrepancies, all models were trained and tested on common datasets, with performance evaluated using a shared metric. We also analyze their robustness against various adversarial attacks, showing that different models exhibit distinct vulnerabilities to different speech attribute distortions. Our training and evaluation code is available at Github.

摘要: 防御Deepfake语音的解决方案分为两类：主动水印模型和被动传统Deepfake检测器。虽然两者都解决了共同的威胁，但它们在训练、优化和评估方面的差异阻碍了统一的联合评估协议和为不同情况选择最佳解决方案。这项工作提出了一个框架来评估Deepfake语音检测中的两种模型类型。为了确保公平比较并最大限度地减少差异，所有模型都在通用数据集上进行了训练和测试，并使用共享指标评估性能。我们还分析了它们对各种对抗性攻击的鲁棒性，表明不同的模型对不同的语音属性失真表现出不同的脆弱性。我们的培训和评估代码可以在Github上找到。



## **6. Excessive Reasoning Attack on Reasoning LLMs**

对推理LLM的过度推理攻击 cs.CR

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2506.14374v1) [paper-pdf](http://arxiv.org/pdf/2506.14374v1)

**Authors**: Wai Man Si, Mingjie Li, Michael Backes, Yang Zhang

**Abstract**: Recent reasoning large language models (LLMs), such as OpenAI o1 and DeepSeek-R1, exhibit strong performance on complex tasks through test-time inference scaling. However, prior studies have shown that these models often incur significant computational costs due to excessive reasoning, such as frequent switching between reasoning trajectories (e.g., underthinking) or redundant reasoning on simple questions (e.g., overthinking). In this work, we expose a novel threat: adversarial inputs can be crafted to exploit excessive reasoning behaviors and substantially increase computational overhead without compromising model utility. Therefore, we propose a novel loss framework consisting of three components: (1) Priority Cross-Entropy Loss, a modification of the standard cross-entropy objective that emphasizes key tokens by leveraging the autoregressive nature of LMs; (2) Excessive Reasoning Loss, which encourages the model to initiate additional reasoning paths during inference; and (3) Delayed Termination Loss, which is designed to extend the reasoning process and defer the generation of final outputs. We optimize and evaluate our attack for the GSM8K and ORCA datasets on DeepSeek-R1-Distill-LLaMA and DeepSeek-R1-Distill-Qwen. Empirical results demonstrate a 3x to 9x increase in reasoning length with comparable utility performance. Furthermore, our crafted adversarial inputs exhibit transferability, inducing computational overhead in o3-mini, o1-mini, DeepSeek-R1, and QWQ models.

摘要: 最近的推理大型语言模型（LLM），例如OpenAI o 1和DeepSeek-R1，通过测试时推理扩展，在复杂任务上表现出出色的性能。然而，之前的研究表明，这些模型通常会由于过度推理而产生显着的计算成本，例如推理轨迹之间的频繁切换（例如，思考不足）或对简单问题进行重复推理（例如，思考过度）。在这项工作中，我们暴露了一个新的威胁：对抗性输入可以被精心制作，以利用过度的推理行为，并在不影响模型效用的情况下大幅增加计算开销。因此，我们提出了一个新的损失框架，包括三个组成部分：（1）优先级交叉熵损失，通过利用LM的自回归性质强调关键令牌的标准交叉熵目标的修改;（2）过度推理损失，鼓励模型在推理过程中启动额外的推理路径;以及（3）延迟终止损失，其被设计为扩展推理过程并延迟最终输出的生成。我们针对DeepSeek-R1-Distill-LLaMA和DeepSeek-R1-Distill-Qwen上的GSM 8 K和ORCA数据集优化和评估了我们的攻击。经验结果表明，推理长度增加了3倍到9倍，而效用性能相当。此外，我们精心设计的对抗性输入具有可移植性，从而在o3-mini、o 1-mini、DeepSeek-R1和QWQ模型中引发计算负担。



## **7. T2V-OptJail: Discrete Prompt Optimization for Text-to-Video Jailbreak Attacks**

T2 V-OptJail：文本到视频越狱攻击的离散提示优化 cs.CV

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2505.06679v2) [paper-pdf](http://arxiv.org/pdf/2505.06679v2)

**Authors**: Jiayang Liu, Siyuan Liang, Shiqian Zhao, Rongcheng Tu, Wenbo Zhou, Aishan Liu, Dacheng Tao, Siew Kei Lam

**Abstract**: In recent years, fueled by the rapid advancement of diffusion models, text-to-video (T2V) generation models have achieved remarkable progress, with notable examples including Pika, Luma, Kling, and Open-Sora. Although these models exhibit impressive generative capabilities, they also expose significant security risks due to their vulnerability to jailbreak attacks, where the models are manipulated to produce unsafe content such as pornography, violence, or discrimination. Existing works such as T2VSafetyBench provide preliminary benchmarks for safety evaluation, but lack systematic methods for thoroughly exploring model vulnerabilities. To address this gap, we are the first to formalize the T2V jailbreak attack as a discrete optimization problem and propose a joint objective-based optimization framework, called T2V-OptJail. This framework consists of two key optimization goals: bypassing the built-in safety filtering mechanisms to increase the attack success rate, preserving semantic consistency between the adversarial prompt and the unsafe input prompt, as well as between the generated video and the unsafe input prompt, to enhance content controllability. In addition, we introduce an iterative optimization strategy guided by prompt variants, where multiple semantically equivalent candidates are generated in each round, and their scores are aggregated to robustly guide the search toward optimal adversarial prompts. We conduct large-scale experiments on several T2V models, covering both open-source models and real commercial closed-source models. The experimental results show that the proposed method improves 11.4% and 10.0% over the existing state-of-the-art method in terms of attack success rate assessed by GPT-4, attack success rate assessed by human accessors, respectively, verifying the significant advantages of the method in terms of attack effectiveness and content control.

摘要: 近年来，在扩散模型的快速发展的推动下，文本转视频（T2 V）生成模型取得了显着的进步，其中著名的例子包括Pika、Luma、Kling和Open-Sora。尽管这些模型表现出令人印象深刻的生成能力，但由于它们容易受到越狱攻击，它们也暴露了巨大的安全风险，这些模型被操纵以产生色情、暴力或歧视等不安全内容。T2 VSafetyBench等现有作品为安全评估提供了初步基准，但缺乏彻底探索模型漏洞的系统方法。为了解决这一差距，我们是第一个将T2 V越狱攻击形式化为离散优化问题的人，并提出了一个联合的基于目标的优化框架，称为T2 V-OptJail。该框架包含两个关键优化目标：绕过内置的安全过滤机制以提高攻击成功率，保持对抗提示与不安全输入提示之间以及生成的视频与不安全输入提示之间的语义一致性，以增强内容的可控性。此外，我们还引入了一种由提示变体指导的迭代优化策略，其中每轮都会生成多个语义等效的候选项，并汇总它们的分数，以稳健地引导搜索走向最佳对抗提示。我们对多种T2 V模型进行了大规模实验，涵盖开源模型和真正的商业闭源模型。实验结果表明，在GPT-4评估的攻击成功率和人工访问者评估的攻击成功率方面，该方法比现有最先进的方法分别提高了11.4%和10.0%，验证了该方法在攻击有效性和内容控制方面的显着优势。



## **8. RL-Obfuscation: Can Language Models Learn to Evade Latent-Space Monitors?**

RL混淆：语言模型能否学会规避潜在空间预设？ cs.LG

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2506.14261v1) [paper-pdf](http://arxiv.org/pdf/2506.14261v1)

**Authors**: Rohan Gupta, Erik Jenner

**Abstract**: Latent-space monitors aim to detect undesirable behaviours in large language models by leveraging internal model representations rather than relying solely on black-box outputs. These methods have shown promise in identifying behaviours such as deception and unsafe completions, but a critical open question remains: can LLMs learn to evade such monitors? To study this, we introduce RL-Obfuscation, in which LLMs are finetuned via reinforcement learning to bypass latent-space monitors while maintaining coherent generations. We apply RL-Obfuscation to LLMs ranging from 7B to 14B parameters and evaluate evasion success against a suite of monitors. We find that token-level latent-space monitors are highly vulnerable to this attack. More holistic monitors, such as max-pooling or attention-based probes, remain robust. Moreover, we show that adversarial policies trained to evade a single static monitor generalise to unseen monitors of the same type. Finally, we study how the policy learned by RL bypasses these monitors and find that the model can also learn to repurpose tokens to mean something different internally.

摘要: 潜伏空间监视器旨在通过利用内部模型表示而不是仅依赖黑匣子输出来检测大型语言模型中的不良行为。这些方法在识别欺骗和不安全完成等行为方面表现出了希望，但一个关键的悬而未决的问题仍然存在：LLM能否学会逃避此类监控？为了研究这一点，我们引入了RL模糊，其中LLM通过强化学习进行微调，以绕过潜在空间监视器，同时保持连贯的世代。我们对7 B到14 B参数的LLM应用RL模糊，并针对一套监视器评估规避成功率。我们发现，令牌级的潜在空间监视器非常容易受到这种攻击。更全面的监控器，如最大池或基于注意力的探测器，仍然很强大。此外，我们表明，对抗性的政策训练，以避免一个单一的静态监视器推广到看不见的监视器的相同类型。最后，我们研究了RL学习的策略如何绕过这些监视器，并发现该模型还可以学习重新使用令牌，以在内部表示不同的含义。



## **9. CAPTURE: Context-Aware Prompt Injection Testing and Robustness Enhancement**

捕获：上下文感知提示注入测试和稳健性增强 cs.CL

Accepted in ACL LLMSec Workshop 2025

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2505.12368v2) [paper-pdf](http://arxiv.org/pdf/2505.12368v2)

**Authors**: Gauri Kholkar, Ratinder Ahuja

**Abstract**: Prompt injection remains a major security risk for large language models. However, the efficacy of existing guardrail models in context-aware settings remains underexplored, as they often rely on static attack benchmarks. Additionally, they have over-defense tendencies. We introduce CAPTURE, a novel context-aware benchmark assessing both attack detection and over-defense tendencies with minimal in-domain examples. Our experiments reveal that current prompt injection guardrail models suffer from high false negatives in adversarial cases and excessive false positives in benign scenarios, highlighting critical limitations. To demonstrate our framework's utility, we train CaptureGuard on our generated data. This new model drastically reduces both false negative and false positive rates on our context-aware datasets while also generalizing effectively to external benchmarks, establishing a path toward more robust and practical prompt injection defenses.

摘要: 提示注入仍然是大型语言模型的主要安全风险。然而，现有护栏模型在上下文感知环境中的功效仍然没有得到充分的探索，因为它们通常依赖于静态攻击基准。此外，他们还有过度防御的倾向。我们引入了CAPTURE，这是一种新型的上下文感知基准，通过最少的领域内示例来评估攻击检测和过度防御倾向。我们的实验表明，当前的即时注射护栏模型在对抗性情况下存在高假阴性，在良性情况下存在过多假阳性，凸显了严重的局限性。为了展示我们框架的实用性，我们在生成的数据上训练CaptureGuard。这种新模型大大降低了我们的上下文感知数据集上的假阴性和假阳性率，同时还有效地推广到外部基准，为更强大和实用的即时注入防御建立了一条道路。



## **10. Mitigating Data Poisoning Attacks to Local Differential Privacy**

缓解对本地差异隐私的数据中毒攻击 cs.CR

The 32nd ACM Conference on Computer and Communications Security (CCS  2025)

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.02156v2) [paper-pdf](http://arxiv.org/pdf/2506.02156v2)

**Authors**: Xiaolin Li, Ninghui Li, Boyang Wang, Wenhai Sun

**Abstract**: The distributed nature of local differential privacy (LDP) invites data poisoning attacks and poses unforeseen threats to the underlying LDP-supported applications. In this paper, we propose a comprehensive mitigation framework for popular frequency estimation, which contains a suite of novel defenses, including malicious user detection, attack pattern recognition, and damaged utility recovery. In addition to existing attacks, we explore new adaptive adversarial activities for our mitigation design. For detection, we present a new method to precisely identify bogus reports and thus LDP aggregation can be performed over the ``clean'' data. When the attack behavior becomes stealthy and direct filtering out malicious users is difficult, we further propose a detection that can effectively recognize hidden adversarial patterns, thus facilitating the decision-making of service providers. These detection methods require no additional data and attack information and incur minimal computational cost. Our experiment demonstrates their excellent performance and substantial improvement over previous work in various settings. In addition, we conduct an empirical analysis of LDP post-processing for corrupted data recovery and propose a new post-processing method, through which we reveal new insights into protocol recommendations in practice and key design principles for future research.

摘要: 局部差异隐私（LDP）的分布式特性会引发数据中毒攻击，并对底层LDP支持的应用程序构成不可预见的威胁。在本文中，我们提出了一个全面的缓解框架，流行的频率估计，其中包含一套新颖的防御，包括恶意用户检测，攻击模式识别和损坏的实用程序恢复。除了现有的攻击，我们还为我们的缓解设计探索了新的自适应对抗活动。对于检测，我们提出了一种新的方法来精确地识别虚假的报告，从而LDP聚合可以在“干净”的数据。当攻击行为变得隐蔽且难以直接过滤恶意用户时，我们进一步提出一种可以有效识别隐藏的对抗模式的检测方法，从而方便服务提供商的决策。这些检测方法不需要额外的数据和攻击信息，并且计算成本最低。我们的实验证明了它们在各种环境下的出色性能和与之前的工作相比的显着改进。此外，我们还对损坏数据恢复的SDP后处理进行了实证分析，并提出了一种新的后处理方法，通过该方法我们揭示了对实践中协议建议的新见解以及未来研究的关键设计原则。



## **11. Weakest Link in the Chain: Security Vulnerabilities in Advanced Reasoning Models**

链中最薄弱的环节：高级推理模型中的安全漏洞 cs.AI

Accepted to LLMSEC 2025

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.13726v1) [paper-pdf](http://arxiv.org/pdf/2506.13726v1)

**Authors**: Arjun Krishna, Aaditya Rastogi, Erick Galinkin

**Abstract**: The introduction of advanced reasoning capabilities have improved the problem-solving performance of large language models, particularly on math and coding benchmarks. However, it remains unclear whether these reasoning models are more or less vulnerable to adversarial prompt attacks than their non-reasoning counterparts. In this work, we present a systematic evaluation of weaknesses in advanced reasoning models compared to similar non-reasoning models across a diverse set of prompt-based attack categories. Using experimental data, we find that on average the reasoning-augmented models are \emph{slightly more robust} than non-reasoning models (42.51\% vs 45.53\% attack success rate, lower is better). However, this overall trend masks significant category-specific differences: for certain attack types the reasoning models are substantially \emph{more vulnerable} (e.g., up to 32 percentage points worse on a tree-of-attacks prompt), while for others they are markedly \emph{more robust} (e.g., 29.8 points better on cross-site scripting injection). Our findings highlight the nuanced security implications of advanced reasoning in language models and emphasize the importance of stress-testing safety across diverse adversarial techniques.

摘要: 高级推理能力的引入提高了大型语言模型的问题解决性能，特别是在数学和编码基准方面。然而，目前尚不清楚这些推理模型是否比非推理模型更容易受到对抗性提示攻击。在这项工作中，我们对高级推理模型与不同的基于预算的攻击类别的类似非推理模型相比的弱点进行了系统评估。使用实验数据，我们发现，平均而言，推理增强的模型比非推理模型的攻击成功率略高（42.51比45.53，越低越好）。然而，这种总体趋势掩盖了显著的类别特定差异：对于某些攻击类型，推理模型基本上更脆弱（例如，在攻击树提示上差达32个百分点），而对于其他攻击树提示，它们明显更健壮（例如，29.8在跨站点脚本注入方面更好）。我们的研究结果强调了语言模型中高级推理的微妙安全影响，并强调了各种对抗技术压力测试安全性的重要性。



## **12. Deceptive Path Planning: A Bayesian Game Approach**

欺骗性路径规划：Bayesian Game方法 eess.SY

8 pages, 9 figures. This work has been submitted to the IEEE for  possible publication

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.13650v1) [paper-pdf](http://arxiv.org/pdf/2506.13650v1)

**Authors**: Violetta Rostobaya, James Berneburg, Yue Guan, Michael Dorothy, Daigo Shishika

**Abstract**: This paper investigates how an autonomous agent can transmit information through its motion in an adversarial setting. We consider scenarios where an agent must reach its goal while deceiving an intelligent observer about its destination. We model this interaction as a dynamic Bayesian game between a mobile Attacker with a privately known goal and a Defender who infers the Attacker's intent to allocate defensive resources effectively. We use Perfect Bayesian Nash Equilibrium (PBNE) as our solution concept and propose a computationally efficient approach to find it. In the resulting equilibrium, the Defender employs a simple Markovian strategy, while the Attacker strategically balances deception and goal efficiency by stochastically mixing shortest and non-shortest paths to manipulate the Defender's beliefs. Numerical experiments demonstrate the advantages of our PBNE-based strategies over existing methods based on one-sided optimization.

摘要: 本文研究了在对抗环境中，自主智能体如何通过运动传递信息。我们考虑的情况下，代理必须达到其目标，同时欺骗一个智能观察员关于其目的地。我们将这种互动建模为一个动态的贝叶斯游戏之间的移动攻击者与一个私人已知的目标和防御者推断攻击者的意图，有效地分配防御资源。我们使用完美贝叶斯纳什均衡（PBNE）作为我们的解决方案的概念，并提出了一个计算效率高的方法来找到它。在所产生的均衡中，防御者采用一个简单的马尔可夫策略，而攻击者的战略平衡欺骗和目标效率的随机混合最短和非最短路径来操纵防御者的信念。数值实验证明了我们基于PBNE的策略相对于基于单边优化的现有方法的优势。



## **13. Unlearning-Enhanced Website Fingerprinting Attack: Against Backdoor Poisoning in Anonymous Networks**

无学习增强型网站指纹攻击：对抗匿名网络中的后门中毒 cs.CR

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.13563v1) [paper-pdf](http://arxiv.org/pdf/2506.13563v1)

**Authors**: Yali Yuan, Kai Xu, Ruolin Ma, Yuchen Zhang

**Abstract**: Website Fingerprinting (WF) is an effective tool for regulating and governing the dark web. However, its performance can be significantly degraded by backdoor poisoning attacks in practical deployments. This paper aims to address the problem of hidden backdoor poisoning attacks faced by Website Fingerprinting attack, and designs a feasible mothed that integrates unlearning technology to realize detection of automatic poisoned points and complete removal of its destructive effects, requiring only a small number of known poisoned test points. Taking Tor onion routing as an example, our method evaluates the influence value of each training sample on these known poisoned test points as the basis for judgment. We optimize the use of influence scores to identify poisoned samples within the training dataset. Furthermore, by quantifying the difference between the contribution of model parameters on the taining data and the clean data, the target parameters are dynamically adjusted to eliminate the impact of the backdoor attacks. Experiments on public datasets under the assumptions of closed-world (CW) and open-world (OW) verify the effectiveness of the proposed method. In complex scenes containing both clean website fingerprinting features and backdoor triggers, the accuracy of the model on the poisoned dataset and the test dataset is stable at about 80%, significantly outperforming the traditional WF attack models. In addition, the proposed method achieves a 2-3 times speedup in runtime efficiency compared to baseline methods. By incorporating machine unlearning, we realize a WF attack model that exhibits enhanced resistance to backdoor poisoning and faster execution speeds in adversarial settings.

摘要: 网站指纹识别（WF）是监管和治理暗网的有效工具。然而，在实际部署中，其性能可能会因后门中毒攻击而显着降低。本文旨在解决网站指纹攻击面临的隐藏后门中毒攻击问题，设计了一种可行的方法，集成去学习技术，实现自动检测毒点并完全消除其破坏性影响，只需少量已知的中毒测试点即可。我们的方法以Tor洋葱路由为例，评估每个训练样本对这些已知中毒测试点的影响值，作为判断的基础。我们优化了影响力分数的使用，以识别训练数据集中的中毒样本。此外，通过量化模型参数对taining数据和clean数据的贡献之间的差异，动态调整目标参数以消除后门攻击的影响。在封闭世界（CW）和开放世界（OW）假设下在公共数据集上进行的实验验证了所提方法的有效性。在包含干净网站指纹识别功能和后门触发器的复杂场景中，模型对中毒数据集和测试数据集的准确性稳定在80%左右，显着优于传统的WF攻击模型。此外，所提出的方法实现了2-3倍的运行时效率相比，基线方法的加速。通过结合机器非学习，我们实现了一个WF攻击模型，该模型在对抗性设置中表现出对后门中毒的增强抵抗力和更快的执行速度。



## **14. Fast Second-Order Online Kernel Learning through Incremental Matrix Sketching and Decomposition**

通过增量矩阵绘制和分解进行快速二阶在线核心学习 cs.LG

Accepted by IJCAI 2025

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2410.11188v2) [paper-pdf](http://arxiv.org/pdf/2410.11188v2)

**Authors**: Dongxie Wen, Xiao Zhang, Zhewei Wei, Chenping Hou, Shuai Li, Weinan Zhang

**Abstract**: Online Kernel Learning (OKL) has attracted considerable research interest due to its promising predictive performance in streaming environments. Second-order approaches are particularly appealing for OKL as they often offer substantial improvements in regret guarantees. However, existing second-order OKL approaches suffer from at least quadratic time complexity with respect to the pre-set budget, rendering them unsuitable for meeting the real-time demands of large-scale streaming recommender systems. The singular value decomposition required to obtain explicit feature mapping is also computationally expensive due to the complete decomposition process. Moreover, the absence of incremental updates to manage approximate kernel space causes these algorithms to perform poorly in adversarial environments and real-world streaming recommendation datasets. To address these issues, we propose FORKS, a fast incremental matrix sketching and decomposition approach tailored for second-order OKL. FORKS constructs an incremental maintenance paradigm for second-order kernelized gradient descent, which includes incremental matrix sketching for kernel approximation and incremental matrix decomposition for explicit feature mapping construction. Theoretical analysis demonstrates that FORKS achieves a logarithmic regret guarantee on par with other second-order approaches while maintaining a linear time complexity w.r.t. the budget, significantly enhancing efficiency over existing approaches. We validate the performance of FORKS through extensive experiments conducted on real-world streaming recommendation datasets, demonstrating its superior scalability and robustness against adversarial attacks.

摘要: 在线核心学习（OKL）因其在流媒体环境中具有良好的预测性能而引起了相当大的研究兴趣。二阶方法对OKL特别有吸引力，因为它们通常在后悔保证方面提供了实质性改进。然而，现有的二阶OKL方法相对于预设预算至少具有二次的时间复杂度，使它们不适合满足大规模流推荐系统的实时需求。由于完整的分解过程，获得显式特征映射所需的奇异值分解在计算上也很昂贵。此外，缺乏管理大致内核空间的增量更新导致这些算法在对抗环境和现实世界的流媒体推荐数据集中表现不佳。为了解决这些问题，我们提出了FORKS，这是一种为二阶OKL量身定制的快速增量矩阵绘制和分解方法。FORKS构建了二阶核化梯度下降的增量维护范式，其中包括用于核逼近的增量矩阵草图和用于显式特征映射构建的增量矩阵分解。理论分析表明，FORKS实现了与其他二阶方法同等的log遗憾保证，同时保持线性时间复杂度wr.t.预算，比现有方法显着提高效率。我们通过对现实世界的流媒体推荐数据集进行的广泛实验来验证FORKS的性能，展示了其卓越的可扩展性和对抗性攻击的鲁棒性。



## **15. Blockchain Address Poisoning**

区块链地址中毒 cs.CR

To appear in Proceedings of the 34th USENIX Security Symposium  (USENIX Security'25)

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2501.16681v2) [paper-pdf](http://arxiv.org/pdf/2501.16681v2)

**Authors**: Taro Tsuchiya, Jin-Dong Dong, Kyle Soska, Nicolas Christin

**Abstract**: In many blockchains, e.g., Ethereum, Binance Smart Chain (BSC), the primary representation used for wallet addresses is a hardly memorable 40-digit hexadecimal string. As a result, users often select addresses from their recent transaction history, which enables blockchain address poisoning. The adversary first generates lookalike addresses similar to one with which the victim has previously interacted, and then engages with the victim to ``poison'' their transaction history. The goal is to have the victim mistakenly send tokens to the lookalike address, as opposed to the intended recipient. Compared to contemporary studies, this paper provides four notable contributions. First, we develop a detection system and perform measurements over two years on both Ethereum and BSC. We identify 13~times more attack attempts than reported previously -- totaling 270M on-chain attacks targeting 17M victims. 6,633 incidents have caused at least 83.8M USD in losses, which makes blockchain address poisoning one of the largest cryptocurrency phishing schemes observed in the wild. Second, we analyze a few large attack entities using improved clustering techniques, and model attacker profitability and competition. Third, we reveal attack strategies -- targeted populations, success conditions (address similarity, timing), and cross-chain attacks. Fourth, we mathematically define and simulate the lookalike address generation process across various software- and hardware-based implementations, and identify a large-scale attacker group that appears to use GPUs. We also discuss defensive countermeasures.

摘要: 在许多区块链中，例如，以太坊，币安智能链（BSC），用于钱包地址的主要表示是一个几乎令人难忘的40位十六进制字符串。因此，用户经常从最近的交易历史记录中选择地址，这导致区块链地址中毒。对手首先生成与受害者之前互动过的地址相似的地址，然后与受害者互动以“毒害”他们的交易历史记录。目标是让受害者错误地将代币发送到外观相似的地址，而不是预期的收件人。与当代研究相比，本文提供了四个值得注意的贡献。首先，我们开发一个检测系统，并在两年内对以太坊和BSC进行测量。我们发现的攻击尝试比之前报告的多13倍--总计针对1700万受害者的2.7亿次链上攻击。6，633起事件已造成至少8，380万美元的损失，这使得区块链地址中毒成为野外观察到的最大的加密货币网络钓鱼计划之一。其次，我们使用改进的集群技术分析一些大型攻击实体，并对攻击者的盈利能力和竞争进行建模。第三，我们揭示了攻击策略--目标人群、成功条件（地址相似性、时机）和跨链攻击。第四，我们以数学方式定义和模拟各种基于软件和硬件的实现中相似的地址生成过程，并识别似乎使用图形处理器的大规模攻击者群体。我们还讨论防御对策。



## **16. From Promise to Peril: Rethinking Cybersecurity Red and Blue Teaming in the Age of LLMs**

从承诺到危险：重新思考法学硕士时代的网络安全红蓝合作 cs.CR

10 pages

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.13434v1) [paper-pdf](http://arxiv.org/pdf/2506.13434v1)

**Authors**: Alsharif Abuadbba, Chris Hicks, Kristen Moore, Vasilios Mavroudis, Burak Hasircioglu, Diksha Goel, Piers Jennings

**Abstract**: Large Language Models (LLMs) are set to reshape cybersecurity by augmenting red and blue team operations. Red teams can exploit LLMs to plan attacks, craft phishing content, simulate adversaries, and generate exploit code. Conversely, blue teams may deploy them for threat intelligence synthesis, root cause analysis, and streamlined documentation. This dual capability introduces both transformative potential and serious risks.   This position paper maps LLM applications across cybersecurity frameworks such as MITRE ATT&CK and the NIST Cybersecurity Framework (CSF), offering a structured view of their current utility and limitations. While LLMs demonstrate fluency and versatility across various tasks, they remain fragile in high-stakes, context-heavy environments. Key limitations include hallucinations, limited context retention, poor reasoning, and sensitivity to prompts, which undermine their reliability in operational settings.   Moreover, real-world integration raises concerns around dual-use risks, adversarial misuse, and diminished human oversight. Malicious actors could exploit LLMs to automate reconnaissance, obscure attack vectors, and lower the technical threshold for executing sophisticated attacks.   To ensure safer adoption, we recommend maintaining human-in-the-loop oversight, enhancing model explainability, integrating privacy-preserving mechanisms, and building systems robust to adversarial exploitation. As organizations increasingly adopt AI driven cybersecurity, a nuanced understanding of LLMs' risks and operational impacts is critical to securing their defensive value while mitigating unintended consequences.

摘要: 大型语言模型（LLM）将通过增强红色和蓝色团队运营来重塑网络安全。红色团队可以利用LLM来计划攻击、制作网络钓鱼内容、模拟对手并生成利用代码。相反，蓝色团队可能会部署它们来进行威胁情报合成、根本原因分析和简化文档。这种双重能力既带来了变革潜力，也带来了严重的风险。   这份立场文件绘制了跨MITRE ATT & CK和NIH网络安全框架（CSF）等网络安全框架的LLM应用程序，提供了其当前效用和局限性的结构化视图。虽然LLM在各种任务中表现出流畅性和多功能性，但它们在高风险、上下文密集的环境中仍然很脆弱。主要局限性包括幻觉、上下文保留有限、推理能力差以及对提示的敏感性，这些都削弱了其在操作环境中的可靠性。   此外，现实世界的一体化引发了人们对两用风险、对抗性滥用和人类监督减少的担忧。恶意行为者可以利用LLM来自动侦察、掩盖攻击载体并降低执行复杂攻击的技术门槛。   为了确保更安全的采用，我们建议保持人在循环中的监督、增强模型的可解释性、集成隐私保护机制并构建对对抗性剥削稳健的系统。随着组织越来越多地采用人工智能驱动的网络安全，对LLM的风险和运营影响的细致了解对于确保其防御价值同时减轻意外后果至关重要。



## **17. Navigating the Black Box: Leveraging LLMs for Effective Text-Level Graph Injection Attacks**

导航黑匣子：利用LLM进行有效的文本级图形注入攻击 cs.AI

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.13276v1) [paper-pdf](http://arxiv.org/pdf/2506.13276v1)

**Authors**: Yuefei Lyu, Chaozhuo Li, Xi Zhang, Tianle Zhang

**Abstract**: Text-attributed graphs (TAGs) integrate textual data with graph structures, providing valuable insights in applications such as social network analysis and recommendation systems. Graph Neural Networks (GNNs) effectively capture both topological structure and textual information in TAGs but are vulnerable to adversarial attacks. Existing graph injection attack (GIA) methods assume that attackers can directly manipulate the embedding layer, producing non-explainable node embeddings. Furthermore, the effectiveness of these attacks often relies on surrogate models with high training costs. Thus, this paper introduces ATAG-LLM, a novel black-box GIA framework tailored for TAGs. Our approach leverages large language models (LLMs) to generate interpretable text-level node attributes directly, ensuring attacks remain feasible in real-world scenarios. We design strategies for LLM prompting that balance exploration and reliability to guide text generation, and propose a similarity assessment method to evaluate attack text effectiveness in disrupting graph homophily. This method efficiently perturbs the target node with minimal training costs in a strict black-box setting, ensuring a text-level graph injection attack for TAGs. Experiments on real-world TAG datasets validate the superior performance of ATAG-LLM compared to state-of-the-art embedding-level and text-level attack methods.

摘要: 文本属性图（TAG）将文本数据与图结构集成，为社交网络分析和推荐系统等应用程序提供有价值的见解。图形神经网络（GNN）有效捕获TAG中的拓扑结构和文本信息，但很容易受到对抗攻击。现有的图注入攻击（GIA）方法假设攻击者可以直接操纵嵌入层，从而产生不可解释的节点嵌入。此外，这些攻击的有效性通常依赖于具有高训练成本的代理模型。因此，本文介绍了ATAG-LLM，这是一种为TAG量身定制的新型黑匣子GIA框架。我们的方法利用大型语言模型（LLM）直接生成可解释的文本级节点属性，确保攻击在现实世界场景中仍然可行。我们为LLM设计了策略，促使平衡探索性和可靠性来指导文本生成，并提出了一种相似性评估方法来评估攻击文本破坏图同质性的有效性。该方法在严格的黑匣子设置中以最小的训练成本有效地扰乱目标节点，确保对TAG的文本级图注入攻击。现实世界TAG数据集上的实验验证了ATAG-LLM与最先进的嵌入级和文本级攻击方法相比的优越性能。



## **18. CAT: Contrastive Adversarial Training for Evaluating the Robustness of Protective Perturbations in Latent Diffusion Models**

CAT：对比对抗训练用于评估潜在扩散模型中保护扰动的鲁棒性 cs.CV

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2502.07225v2) [paper-pdf](http://arxiv.org/pdf/2502.07225v2)

**Authors**: Sen Peng, Mingyue Wang, Jianfei He, Jijia Yang, Xiaohua Jia

**Abstract**: Latent diffusion models have recently demonstrated superior capabilities in many downstream image synthesis tasks. However, customization of latent diffusion models using unauthorized data can severely compromise the privacy and intellectual property rights of data owners. Adversarial examples as protective perturbations have been developed to defend against unauthorized data usage by introducing imperceptible noise to customization samples, preventing diffusion models from effectively learning them. In this paper, we first reveal that the primary reason adversarial examples are effective as protective perturbations in latent diffusion models is the distortion of their latent representations, as demonstrated through qualitative and quantitative experiments. We then propose the Contrastive Adversarial Training (CAT) utilizing lightweight adapters as an adaptive attack against these protection methods, highlighting their lack of robustness. Extensive experiments demonstrate that our CAT method significantly reduces the effectiveness of protective perturbations in customization, urging the community to reconsider and improve the robustness of existing protective perturbations. The code is available at https://github.com/senp98/CAT.

摘要: 潜在扩散模型最近在许多下游图像合成任务中展示了卓越的能力。然而，使用未经授权的数据定制潜在扩散模型可能会严重损害数据所有者的隐私和知识产权。已经开发了作为保护性扰动的对抗性示例，通过向定制样本引入难以感知的噪音来防止未经授权的数据使用，防止扩散模型有效地学习它们。在本文中，我们首先揭示了对抗性例子作为潜在扩散模型中的保护性扰动有效的主要原因是其潜在表示的扭曲，正如定性和定量实验所证明的那样。然后，我们提出利用轻量级适配器的对比对抗训练（CAT）作为针对这些保护方法的自适应攻击，强调了它们缺乏稳健性。大量实验表明，我们的CAT方法显着降低了定制中保护性扰动的有效性，敦促社区重新考虑并提高现有保护性扰动的鲁棒性。该代码可在https://github.com/senp98/CAT上获取。



## **19. Detecting Hard-Coded Credentials in Software Repositories via LLMs**

通过LLM检测软件存储库中的硬编码凭据 cs.CR

Accepted to the ACM Digital Threats: Research and Practice (DTRAP)

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2506.13090v1) [paper-pdf](http://arxiv.org/pdf/2506.13090v1)

**Authors**: Chidera Biringa, Gokhan Kul

**Abstract**: Software developers frequently hard-code credentials such as passwords, generic secrets, private keys, and generic tokens in software repositories, even though it is strictly advised against due to the severe threat to the security of the software. These credentials create attack surfaces exploitable by a potential adversary to conduct malicious exploits such as backdoor attacks. Recent detection efforts utilize embedding models to vectorize textual credentials before passing them to classifiers for predictions. However, these models struggle to discriminate between credentials with contextual and complex sequences resulting in high false positive predictions. Context-dependent Pre-trained Language Models (PLMs) or Large Language Models (LLMs) such as Generative Pre-trained Transformers (GPT) tackled this drawback by leveraging the transformer neural architecture capacity for self-attention to capture contextual dependencies between words in input sequences. As a result, GPT has achieved wide success in several natural language understanding endeavors. Hence, we assess LLMs to represent these observations and feed extracted embedding vectors to a deep learning classifier to detect hard-coded credentials. Our model outperforms the current state-of-the-art by 13% in F1 measure on the benchmark dataset. We have made all source code and data publicly available to facilitate the reproduction of all results presented in this paper.

摘要: 软件开发人员经常在软件存储库中硬编码凭证，例如密码、通用秘密、私有密钥和通用令牌，尽管由于对软件安全性的严重威胁而被严格建议不要这样做。这些凭证创建了可被潜在对手利用的攻击表面，以实施后门攻击等恶意利用。最近的检测工作利用嵌入模型来对文本凭证进行载体化，然后将其传递给分类器进行预测。然而，这些模型很难区分具有上下文和复杂序列的证书，从而导致高假阳性预测。上下文相关的预训练语言模型（PLM）或大型语言模型（LLM），如生成式预训练转换器（GPT），通过利用Transformer神经架构的自我注意能力来捕获输入序列中单词之间的上下文依赖关系，解决了这个缺点。因此，GPT在多项自然语言理解工作中取得了广泛成功。因此，我们评估LLM来表示这些观察，并将提取的嵌入载体提供给深度学习分类器以检测硬编码凭证。在基准数据集的F1指标中，我们的模型比当前最先进的模型高出13%。我们已公开所有源代码和数据，以促进本文中提出的所有结果的复制。



## **20. Detecting Adversarial Examples**

检测对抗示例 cs.LG

**SubmitDate**: 2025-06-16    [abs](http://arxiv.org/abs/2410.17442v2) [paper-pdf](http://arxiv.org/pdf/2410.17442v2)

**Authors**: Furkan Mumcu, Yasin Yilmaz

**Abstract**: Deep Neural Networks (DNNs) have been shown to be vulnerable to adversarial examples. While numerous successful adversarial attacks have been proposed, defenses against these attacks remain relatively understudied. Existing defense approaches either focus on negating the effects of perturbations caused by the attacks to restore the DNNs' original predictions or use a secondary model to detect adversarial examples. However, these methods often become ineffective due to the continuous advancements in attack techniques. We propose a novel universal and lightweight method to detect adversarial examples by analyzing the layer outputs of DNNs. Our method trains a lightweight regression model that predicts deeper-layer features from early-layer features, and uses the prediction error to detect adversarial samples. Through theoretical justification and extensive experiments, we demonstrate that our detection method is highly effective, compatible with any DNN architecture, and applicable across different domains, such as image, video, and audio.

摘要: 深度神经网络（DNN）已被证明容易受到对抗性示例的影响。虽然已经提出了许多成功的对抗性攻击，但针对这些攻击的防御研究仍然相对不足。现有的防御方法要么专注于抵消攻击引起的扰动的影响，以恢复DNN的原始预测，要么使用二级模型来检测对抗性示例。然而，由于攻击技术的不断进步，这些方法往往变得无效。我们提出了一种新颖的通用和轻量级方法来通过分析DNN的层输出来检测对抗性示例。我们的方法训练一个轻量级回归模型，该模型从早期层特征预测更深层特征，并使用预测误差来检测对抗样本。通过理论论证和大量实验，我们证明我们的检测方法非常有效，与任何DNN架构兼容，并且适用于图像、视频和音频等不同领域。



## **21. Constraint-Guided Prediction Refinement via Deterministic Diffusion Trajectories**

通过确定性扩散轨迹的约束引导预测细化 cs.AI

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12911v1) [paper-pdf](http://arxiv.org/pdf/2506.12911v1)

**Authors**: Pantelis Dogoulis, Fabien Bernier, Félix Fourreau, Karim Tit, Maxime Cordy

**Abstract**: Many real-world machine learning tasks require outputs that satisfy hard constraints, such as physical conservation laws, structured dependencies in graphs, or column-level relationships in tabular data. Existing approaches rely either on domain-specific architectures and losses or on strong assumptions on the constraint space, restricting their applicability to linear or convex constraints. We propose a general-purpose framework for constraint-aware refinement that leverages denoising diffusion implicit models (DDIMs). Starting from a coarse prediction, our method iteratively refines it through a deterministic diffusion trajectory guided by a learned prior and augmented by constraint gradient corrections. The approach accommodates a wide class of non-convex and nonlinear equality constraints and can be applied post hoc to any base model. We demonstrate the method in two representative domains: constrained adversarial attack generation on tabular data with column-level dependencies and in AC power flow prediction under Kirchhoff's laws. Across both settings, our diffusion-guided refinement improves both constraint satisfaction and performance while remaining lightweight and model-agnostic.

摘要: 许多现实世界的机器学习任务需要满足硬约束的输出，例如物理保守定律、图形中的结构化依赖关系或表格数据中的列级关系。现有的方法要么依赖于特定领域的架构和损失，要么依赖于对约束空间的强假设，从而将其适用性限制在线性或凸约束中。我们提出了一个用于约束感知细化的通用框架，该框架利用去噪扩散隐式模型（DDIM）。我们的方法从粗略预测开始，通过由学习先验引导并通过约束梯度修正增强的确定性扩散轨迹来迭代细化它。该方法适应广泛的非凸和非线性等式约束，并且可以事后应用于任何基本模型。我们在两个代表性领域展示了该方法：对具有列级依赖关系的表格数据产生约束对抗攻击，以及根据基尔霍夫定律进行交流潮流预测。在这两种设置中，我们的扩散引导细化可以提高约束满意度和性能，同时保持轻量级和模型不可知。



## **22. Universal Jailbreak Suffixes Are Strong Attention Hijackers**

通用越狱后缀是强烈的注意力劫持者 cs.CR

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12880v1) [paper-pdf](http://arxiv.org/pdf/2506.12880v1)

**Authors**: Matan Ben-Tov, Mor Geva, Mahmood Sharif

**Abstract**: We study suffix-based jailbreaks$\unicode{x2013}$a powerful family of attacks against large language models (LLMs) that optimize adversarial suffixes to circumvent safety alignment. Focusing on the widely used foundational GCG attack (Zou et al., 2023), we observe that suffixes vary in efficacy: some markedly more universal$\unicode{x2013}$generalizing to many unseen harmful instructions$\unicode{x2013}$than others. We first show that GCG's effectiveness is driven by a shallow, critical mechanism, built on the information flow from the adversarial suffix to the final chat template tokens before generation. Quantifying the dominance of this mechanism during generation, we find GCG irregularly and aggressively hijacks the contextualization process. Crucially, we tie hijacking to the universality phenomenon, with more universal suffixes being stronger hijackers. Subsequently, we show that these insights have practical implications: GCG universality can be efficiently enhanced (up to $\times$5 in some cases) at no additional computational cost, and can also be surgically mitigated, at least halving attack success with minimal utility loss. We release our code and data at http://github.com/matanbt/interp-jailbreak.

摘要: 我们研究基于后缀的越狱$\unicode{x2013}$这是一个针对大型语言模型（LLM）的强大攻击家族，这些模型优化对抗性后缀以规避安全对齐。专注于广泛使用的基础GCG攻击（Zou等人，2023），我们观察到后缀的功效各不相同：有些后缀明显更通用的$\unicode{x2013}$一般化为许多不可见的有害指令$\unicode{x2013}$。我们首先表明，GCG的有效性是由一种肤浅的关键机制驱动的，该机制建立在从对抗性后缀到生成之前的最终聊天模板令牌的信息流之上。量化这种机制在生成过程中的主导地位，我们发现GCG不规则且积极地劫持了情境化过程。至关重要的是，我们将劫持与普遍现象联系起来，更普遍的后缀意味着更强大的劫持者。随后，我们证明了这些见解具有实际意义：GCG通用性可以在不需要额外计算成本的情况下有效增强（在某些情况下高达5美元），并且还可以通过手术来减轻，至少将攻击成功率减半，并将效用损失最小。我们在http://github.com/matanbt/interp-jailbreak上发布我们的代码和数据。



## **23. Intriguing Frequency Interpretation of Adversarial Robustness for CNNs and ViTs**

CNN和ViT对抗稳健性的有趣频率解释 cs.CV

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12875v1) [paper-pdf](http://arxiv.org/pdf/2506.12875v1)

**Authors**: Lu Chen, Han Yang, Hu Wang, Yuxin Cao, Shaofeng Li, Yuan Luo

**Abstract**: Adversarial examples have attracted significant attention over the years, yet understanding their frequency-based characteristics remains insufficient. In this paper, we investigate the intriguing properties of adversarial examples in the frequency domain for the image classification task, with the following key findings. (1) As the high-frequency components increase, the performance gap between adversarial and natural examples becomes increasingly pronounced. (2) The model performance against filtered adversarial examples initially increases to a peak and declines to its inherent robustness. (3) In Convolutional Neural Networks, mid- and high-frequency components of adversarial examples exhibit their attack capabilities, while in Transformers, low- and mid-frequency components of adversarial examples are particularly effective. These results suggest that different network architectures have different frequency preferences and that differences in frequency components between adversarial and natural examples may directly influence model robustness. Based on our findings, we further conclude with three useful proposals that serve as a valuable reference to the AI model security community.

摘要: 多年来，对抗性例子引起了人们的极大关注，但对它们基于频率的特征的理解仍然不够。在本文中，我们研究了对抗性示例在频域中用于图像分类任务的有趣特性，主要发现如下。(1)随着高频分量的增加，对抗性和自然示例之间的性能差距变得越来越明显。(2)针对过滤的对抗性示例的模型性能最初会增加到峰值，并下降到其固有的鲁棒性。(3)在卷积神经网络中，对抗性示例的中高频分量表现出其攻击能力，而在变形金刚中，对抗性示例的低频和中频分量尤其有效。这些结果表明，不同的网络架构具有不同的频率偏好，并且对抗示例和自然示例之间频率分量的差异可能会直接影响模型的稳健性。根据我们的调查结果，我们进一步提出了三个有用的建议，作为人工智能模型安全社区的宝贵参考。



## **24. Active Adversarial Noise Suppression for Image Forgery Localization**

图像伪造定位中的主动对抗噪音抑制 cs.CV

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12871v1) [paper-pdf](http://arxiv.org/pdf/2506.12871v1)

**Authors**: Rongxuan Peng, Shunquan Tan, Xianbo Mo, Alex C. Kot, Jiwu Huang

**Abstract**: Recent advances in deep learning have significantly propelled the development of image forgery localization. However, existing models remain highly vulnerable to adversarial attacks: imperceptible noise added to forged images can severely mislead these models. In this paper, we address this challenge with an Adversarial Noise Suppression Module (ANSM) that generate a defensive perturbation to suppress the attack effect of adversarial noise. We observe that forgery-relevant features extracted from adversarial and original forged images exhibit distinct distributions. To bridge this gap, we introduce Forgery-relevant Features Alignment (FFA) as a first-stage training strategy, which reduces distributional discrepancies by minimizing the channel-wise Kullback-Leibler divergence between these features. To further refine the defensive perturbation, we design a second-stage training strategy, termed Mask-guided Refinement (MgR), which incorporates a dual-mask constraint. MgR ensures that the perturbation remains effective for both adversarial and original forged images, recovering forgery localization accuracy to their original level. Extensive experiments across various attack algorithms demonstrate that our method significantly restores the forgery localization model's performance on adversarial images. Notably, when ANSM is applied to original forged images, the performance remains nearly unaffected. To our best knowledge, this is the first report of adversarial defense in image forgery localization tasks. We have released the source code and anti-forensics dataset.

摘要: 深度学习的最新进展显着推动了图像伪造本地化的发展。然而，现有模型仍然极易受到对抗攻击：添加到伪造图像中的不可感知的噪音可能会严重误导这些模型。在本文中，我们通过对抗性噪音抑制模块（ANSM）来解决这一挑战，该模块生成防御性扰动以抑制对抗性噪音的攻击效应。我们观察到，从对抗性和原始伪造图像中提取的伪造相关特征表现出不同的分布。为了弥合这一差距，我们引入了伪造相关特征对齐（FFA）作为第一阶段的训练策略，通过最小化这些特征之间的通道Kullback-Leibler分歧来减少分布差异。为了进一步完善防御扰动，我们设计了一个第二阶段的训练策略，称为面具指导细化（MgR），其中包括一个双面具约束。MgR确保扰动对对抗和原始伪造图像都有效，将伪造定位精度恢复到原始水平。在各种攻击算法的广泛实验表明，我们的方法显着恢复伪造定位模型的对抗性图像的性能。值得注意的是，当ANSM应用于原始伪造图像时，性能几乎不受影响。据我们所知，这是图像伪造本地化任务中对抗性防御的第一份报告。我们已经发布了源代码和反取证数据集。



## **25. Privacy-Preserving Federated Learning against Malicious Clients Based on Verifiable Functional Encryption**

基于可验证功能加密的针对恶意客户端的隐私保护联邦学习 cs.CR

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12846v1) [paper-pdf](http://arxiv.org/pdf/2506.12846v1)

**Authors**: Nina Cai, Jinguang Han

**Abstract**: Federated learning is a promising distributed learning paradigm that enables collaborative model training without exposing local client data, thereby protect data privacy. However, it also brings new threats and challenges. The advancement of model inversion attacks has rendered the plaintext transmission of local models insecure, while the distributed nature of federated learning makes it particularly vulnerable to attacks raised by malicious clients. To protect data privacy and prevent malicious client attacks, this paper proposes a privacy-preserving federated learning framework based on verifiable functional encryption, without a non-colluding dual-server setup or additional trusted third-party. Specifically, we propose a novel decentralized verifiable functional encryption (DVFE) scheme that enables the verification of specific relationships over multi-dimensional ciphertexts. This scheme is formally treated, in terms of definition, security model and security proof. Furthermore, based on the proposed DVFE scheme, we design a privacy-preserving federated learning framework VFEFL that incorporates a novel robust aggregation rule to detect malicious clients, enabling the effective training of high-accuracy models under adversarial settings. Finally, we provide formal analysis and empirical evaluation of the proposed schemes. The results demonstrate that our approach achieves the desired privacy protection, robustness, verifiability and fidelity, while eliminating the reliance on non-colluding dual-server settings or trusted third parties required by existing methods.

摘要: 联邦学习是一种有前途的分布式学习范式，可以在不暴露本地客户数据的情况下实现协作模型训练，从而保护数据隐私。然而，它也带来了新的威胁和挑战。模型倒置攻击的发展使本地模型的明文传输变得不安全，而联邦学习的分布式性质使其特别容易受到恶意客户端发起的攻击。为了保护数据隐私并防止恶意客户端攻击，本文提出了一种基于可验证功能加密的隐私保护联邦学习框架，无需非勾结双服务器设置或额外的受信任第三方。具体来说，我们提出了一种新型的去中心化可验证功能加密（DVFE）方案，该方案能够验证多维密文上的特定关系。从定义、安全模型和安全证明方面对该方案进行了正式处理。此外，基于提出的DVFE方案，我们设计了一个保护隐私的联邦学习框架VFEFL，该框架结合了一种新颖的鲁棒聚合规则来检测恶意客户端，从而能够在对抗环境下有效训练高准确度模型。最后，我们对所提出的方案进行了形式分析和实证评估。结果表明，我们的方法实现了预期的隐私保护、稳健性、可验证性和保真度，同时消除了对现有方法所需的非勾结双服务器设置或受信任第三方的依赖。



## **26. Transforming Chatbot Text: A Sequence-to-Sequence Approach**

改造聊天机器人文本：序列到序列的方法 cs.CL

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12843v1) [paper-pdf](http://arxiv.org/pdf/2506.12843v1)

**Authors**: Natesh Reddy, Mark Stamp

**Abstract**: Due to advances in Large Language Models (LLMs) such as ChatGPT, the boundary between human-written text and AI-generated text has become blurred. Nevertheless, recent work has demonstrated that it is possible to reliably detect GPT-generated text. In this paper, we adopt a novel strategy to adversarially transform GPT-generated text using sequence-to-sequence (Seq2Seq) models, with the goal of making the text more human-like. We experiment with the Seq2Seq models T5-small and BART which serve to modify GPT-generated sentences to include linguistic, structural, and semantic components that may be more typical of human-authored text. Experiments show that classification models trained to distinguish GPT-generated text are significantly less accurate when tested on text that has been modified by these Seq2Seq models. However, after retraining classification models on data generated by our Seq2Seq technique, the models are able to distinguish the transformed GPT-generated text from human-generated text with high accuracy. This work adds to the accumulating knowledge of text transformation as a tool for both attack -- in the sense of defeating classification models -- and defense -- in the sense of improved classifiers -- thereby advancing our understanding of AI-generated text.

摘要: 由于ChatGPT等大型语言模型（LLM）的进步，人类书面文本和人工智能生成文本之间的界限变得模糊。尽管如此，最近的工作表明，可以可靠地检测GPT生成的文本。在本文中，我们采用了一种新颖的策略，使用序列到序列（Seq 2Seq）模型对GPT生成的文本进行反向转换，目标是使文本更像人。我们实验了Seq 2Seq模型T5-small和BART，这些模型用于修改GPT生成的句子，以包括可能更典型的人类创作文本的语言、结构和语义成分。实验表明，当对已由这些Seq 2Seq模型修改的文本进行测试时，为区分GPT生成的文本而训练的分类模型的准确性明显较差。然而，在对我们的Seq 2Seq技术生成的数据重新训练分类模型后，这些模型能够高准确性区分转换后的GPT生成文本与人类生成文本。这项工作增加了文本转换作为攻击工具（从击败分类模型的意义上来说）和防御（从改进分类器的意义上来说）的知识的积累，从而促进我们对人工智能生成的文本的理解。



## **27. I Know What You Said: Unveiling Hardware Cache Side-Channels in Local Large Language Model Inference**

我知道你说什么：揭开本地大型语言模型推理中的硬件缓存侧通道 cs.CR

Submitted for review in January 22, 2025, revised under shepherding

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2505.06738v3) [paper-pdf](http://arxiv.org/pdf/2505.06738v3)

**Authors**: Zibo Gao, Junjie Hu, Feng Guo, Yixin Zhang, Yinglong Han, Siyuan Liu, Haiyang Li, Zhiqiang Lv

**Abstract**: Large Language Models (LLMs) that can be deployed locally have recently gained popularity for privacy-sensitive tasks, with companies such as Meta, Google, and Intel playing significant roles in their development. However, the security of local LLMs through the lens of hardware cache side-channels remains unexplored. In this paper, we unveil novel side-channel vulnerabilities in local LLM inference: token value and token position leakage, which can expose both the victim's input and output text, thereby compromising user privacy. Specifically, we found that adversaries can infer the token values from the cache access patterns of the token embedding operation, and deduce the token positions from the timing of autoregressive decoding phases. To demonstrate the potential of these leaks, we design a novel eavesdropping attack framework targeting both open-source and proprietary LLM inference systems. The attack framework does not directly interact with the victim's LLM and can be executed without privilege.   We evaluate the attack on a range of practical local LLM deployments (e.g., Llama, Falcon, and Gemma), and the results show that our attack achieves promising accuracy. The restored output and input text have an average edit distance of 5.2% and 17.3% to the ground truth, respectively. Furthermore, the reconstructed texts achieve average cosine similarity scores of 98.7% (input) and 98.0% (output).

摘要: 可以在本地部署的大型语言模型（LLM）最近在隐私敏感任务中越来越受欢迎，Meta、谷歌和英特尔等公司在其开发中发挥了重要作用。然而，通过硬件缓存侧通道的视角来探讨本地LLM的安全性仍然有待探索。在本文中，我们揭示了本地LLM推断中的新型侧通道漏洞：令牌值和令牌位置泄露，它可以暴露受害者的输入和输出文本，从而损害用户隐私。具体来说，我们发现对手可以从令牌嵌入操作的缓存访问模式中推断令牌值，并从自回归解码阶段的时间推断令牌位置。为了证明这些泄漏的潜力，我们设计了一个新的窃听攻击框架，针对开源和专有的LLM推理系统。攻击框架不直接与受害者的LLM交互，并且可以在没有特权的情况下执行。   我们评估了对一系列实际本地LLM部署的攻击（例如，Llama、Falcon和Gemma），结果表明我们的攻击达到了令人满意的准确性。恢复的输出和输入文本与地面真相的平均编辑距离分别为5.2%和17.3%。此外，重建的文本的平均cos相似度评分为98.7%（输入）和98.0%（输出）。



## **28. SecurityLingua: Efficient Defense of LLM Jailbreak Attacks via Security-Aware Prompt Compression**

SecureLingua：通过安全意识提示压缩有效防御LLM越狱攻击 cs.CR

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12707v1) [paper-pdf](http://arxiv.org/pdf/2506.12707v1)

**Authors**: Yucheng Li, Surin Ahn, Huiqiang Jiang, Amir H. Abdi, Yuqing Yang, Lili Qiu

**Abstract**: Large language models (LLMs) have achieved widespread adoption across numerous applications. However, many LLMs are vulnerable to malicious attacks even after safety alignment. These attacks typically bypass LLMs' safety guardrails by wrapping the original malicious instructions inside adversarial jailbreaks prompts. Previous research has proposed methods such as adversarial training and prompt rephrasing to mitigate these safety vulnerabilities, but these methods often reduce the utility of LLMs or lead to significant computational overhead and online latency. In this paper, we propose SecurityLingua, an effective and efficient approach to defend LLMs against jailbreak attacks via security-oriented prompt compression. Specifically, we train a prompt compressor designed to discern the "true intention" of the input prompt, with a particular focus on detecting the malicious intentions of adversarial prompts. Then, in addition to the original prompt, the intention is passed via the system prompt to the target LLM to help it identify the true intention of the request. SecurityLingua ensures a consistent user experience by leaving the original input prompt intact while revealing the user's potentially malicious intention and stimulating the built-in safety guardrails of the LLM. Moreover, thanks to prompt compression, SecurityLingua incurs only a negligible overhead and extra token cost compared to all existing defense methods, making it an especially practical solution for LLM defense. Experimental results demonstrate that SecurityLingua can effectively defend against malicious attacks and maintain utility of the LLM with negligible compute and latency overhead. Our code is available at https://aka.ms/SecurityLingua.

摘要: 大型语言模型（LLM）已经在许多应用程序中得到了广泛采用。然而，许多LLM即使在安全对齐之后也容易受到恶意攻击。这些攻击通常通过将原始恶意指令包装在对抗性越狱提示中来绕过LLM的安全护栏。以前的研究已经提出了对抗性训练和快速改写等方法来减轻这些安全漏洞，但这些方法通常会降低LLM的实用性或导致显着的计算开销和在线延迟。在本文中，我们提出了SecureLingua，这是一种通过面向安全的即时压缩来保护LLM免受越狱攻击的有效且高效的方法。具体来说，我们训练了一个提示压缩器，旨在识别输入提示的“真实意图”，特别关注检测对抗提示的恶意意图。然后，除了原始提示之外，意图还通过系统提示传递给目标LLM，以帮助其识别请求的真实意图。SecureLingua通过保持原始输入提示不变，同时揭露用户潜在的恶意意图并刺激LLM的内置安全护栏，确保一致的用户体验。此外，由于快速压缩，与所有现有防御方法相比，SecureLingua仅产生可忽略不计的费用和额外的令牌成本，使其成为LLM防御特别实用的解决方案。实验结果表明，SecureLingua可以有效防御恶意攻击并保持LLM的实用性，而计算和延迟负担可以忽略不计。我们的代码可在https://aka.ms/SecurityLingua上获取。



## **29. NAP-Tuning: Neural Augmented Prompt Tuning for Adversarially Robust Vision-Language Models**

NAP调优：对抗鲁棒视觉语言模型的神经增强提示调优 cs.CV

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12706v1) [paper-pdf](http://arxiv.org/pdf/2506.12706v1)

**Authors**: Jiaming Zhang, Xin Wang, Xingjun Ma, Lingyu Qiu, Yu-Gang Jiang, Jitao Sang

**Abstract**: Vision-Language Models (VLMs) such as CLIP have demonstrated remarkable capabilities in understanding relationships between visual and textual data through joint embedding spaces. Despite their effectiveness, these models remain vulnerable to adversarial attacks, particularly in the image modality, posing significant security concerns. Building upon our previous work on Adversarial Prompt Tuning (AdvPT), which introduced learnable text prompts to enhance adversarial robustness in VLMs without extensive parameter training, we present a significant extension by introducing the Neural Augmentor framework for Multi-modal Adversarial Prompt Tuning (NAP-Tuning).Our key innovations include: (1) extending AdvPT from text-only to multi-modal prompting across both text and visual modalities, (2) expanding from single-layer to multi-layer prompt architectures, and (3) proposing a novel architecture-level redesign through our Neural Augmentor approach, which implements feature purification to directly address the distortions introduced by adversarial attacks in feature space. Our NAP-Tuning approach incorporates token refiners that learn to reconstruct purified features through residual connections, allowing for modality-specific and layer-specific feature correction.Comprehensive experiments demonstrate that NAP-Tuning significantly outperforms existing methods across various datasets and attack types. Notably, our approach shows significant improvements over the strongest baselines under the challenging AutoAttack benchmark, outperforming them by 33.5% on ViT-B16 and 33.0% on ViT-B32 architectures while maintaining competitive clean accuracy.

摘要: 视觉语言模型（VLM），如CLIP，通过联合嵌入空间，在理解视觉和文本数据之间的关系方面表现出显着的能力。尽管这些模型很有效，但它们仍然容易受到对抗性攻击，特别是在图像模式中，这带来了重大的安全问题。在我们之前关于对抗提示调整（AdvPT）的工作的基础上，我们引入了可学习的文本提示，以增强VLM中的对抗鲁棒性，而无需大量的参数训练，我们通过引入用于多模态对抗提示调整的神经增强器框架进行了重大扩展我们的主要创新包括：（1）将AdvPT从纯文本扩展到跨文本和视觉模态的多模态提示，（2）从单层扩展到多层提示架构，以及（3）通过我们的神经增强器方法提出了一种新的架构级重新设计，该方法实现了特征纯化，以直接解决对抗性攻击在特征空间中引入的扭曲。我们的NAP-Tuning方法结合了令牌精炼器，这些令牌精炼器通过残余连接学习重建纯化的特征，允许特定于模式和特定于层的特征校正。综合实验表明，NAP-Tuning在各种数据集和攻击类型上显著优于现有方法。值得注意的是，我们的方法在具有挑战性的AutoAttack基准测试中显示出了显著的改进，在ViT-B16和ViT-B32架构上的性能分别超过了33.5%和33.0%，同时保持了具有竞争力的干净准确性。



## **30. Zeroth-Order primal-dual Alternating Projection Gradient Algorithms for Nonconvex Minimax Problems with Coupled linear Constraints**

具有耦合线性约束的非凸极小极大问题的零阶原始-二元交替投影梯度算法 math.OC

arXiv admin note: text overlap with arXiv:2212.04672

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2402.03352v2) [paper-pdf](http://arxiv.org/pdf/2402.03352v2)

**Authors**: Huiling Zhang, Zi Xu, Yuhong Dai

**Abstract**: In this paper, we study zeroth-order algorithms for nonconvex minimax problems with coupled linear constraints under the deterministic and stochastic settings, which have attracted wide attention in machine learning, signal processing and many other fields in recent years, e.g., adversarial attacks in resource allocation problems and network flow problems etc. We propose two single-loop algorithms, namely the zeroth-order primal-dual alternating projected gradient (ZO-PDAPG) algorithm and the zeroth-order regularized momentum primal-dual projected gradient algorithm (ZO-RMPDPG), for solving deterministic and stochastic nonconvex-(strongly) concave minimax problems with coupled linear constraints. The iteration complexity of the two proposed algorithms to obtain an $\varepsilon$-stationary point are proved to be $\mathcal{O}(\varepsilon ^{-2})$ (resp. $\mathcal{O}(\varepsilon ^{-4})$) for solving nonconvex-strongly concave (resp. nonconvex-concave) minimax problems with coupled linear constraints under deterministic settings and $\tilde{\mathcal{O}}(\varepsilon ^{-3})$ (resp. $\tilde{\mathcal{O}}(\varepsilon ^{-6.5})$) under stochastic settings respectively. To the best of our knowledge, they are the first two zeroth-order algorithms with iterative complexity guarantees for solving nonconvex-(strongly) concave minimax problems with coupled linear constraints under the deterministic and stochastic settings.

摘要: 本文研究了确定性和随机设置下具有耦合线性约束的非凸极小极大问题的零阶算法，近年来在机器学习、信号处理等许多领域引起了广泛关注，例如，资源分配问题和网络流问题等中的对抗性攻击。我们提出了两种单回路算法，即零阶原始-二元交替投影梯度算法（ZO-PDATG）和零阶正规动量原始-二元投影梯度算法（ZO-RMPDPG），用于求解具有耦合线性约束的确定性和随机非凸-（强）凹极小极大问题。获得$\varepð $-稳定点的两种提出的算法的迭代复杂度被证明为$\mathCal{O}（\varepð$\mathCal{O}（\varepð非凹凸）确定性设置和$\tilde{\mathCal{O}}（\varepð分别在随机设置下的$\tilde{\mathCal{O}}（\varepð ^{-6.5}）$）。据我们所知，它们是前两种具有迭代复杂性保证的零阶算法，用于在确定性和随机性设置下解决具有耦合线性约束的非凸-（强）凹极小极大问题。



## **31. Alphabet Index Mapping: Jailbreaking LLMs through Semantic Dissimilarity**

字母索引映射：通过语义差异越狱LLM cs.CR

10 pages, 2 figures, 3 tables

**SubmitDate**: 2025-06-15    [abs](http://arxiv.org/abs/2506.12685v1) [paper-pdf](http://arxiv.org/pdf/2506.12685v1)

**Authors**: Bilal Saleh Husain

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities, yet their susceptibility to adversarial attacks, particularly jailbreaking, poses significant safety and ethical concerns. While numerous jailbreak methods exist, many suffer from computational expense, high token usage, or complex decoding schemes. Liu et al. (2024) introduced FlipAttack, a black-box method that achieves high attack success rates (ASR) through simple prompt manipulation. This paper investigates the underlying mechanisms of FlipAttack's effectiveness by analyzing the semantic changes induced by its flipping modes. We hypothesize that semantic dissimilarity between original and manipulated prompts is inversely correlated with ASR. To test this, we examine embedding space visualizations (UMAP, KDE) and cosine similarities for FlipAttack's modes. Furthermore, we introduce a novel adversarial attack, Alphabet Index Mapping (AIM), designed to maximize semantic dissimilarity while maintaining simple decodability. Experiments on GPT-4 using a subset of AdvBench show AIM and its variant AIM+FWO achieve a 94% ASR, outperforming FlipAttack and other methods on this subset. Our findings suggest that while high semantic dissimilarity is crucial, a balance with decoding simplicity is key for successful jailbreaking. This work contributes to a deeper understanding of adversarial prompt mechanics and offers a new, effective jailbreak technique.

摘要: 大型语言模型（LLM）已表现出非凡的能力，但它们对对抗攻击（尤其是越狱）的敏感性带来了重大的安全和道德问题。虽然存在多种越狱方法，但许多方法都面临计算成本、令牌使用率高或解码方案复杂的问题。Liu等人（2024）介绍了FlipAttack，这是一种黑匣子方法，通过简单的提示操作来实现高攻击成功率（ASB）。本文通过分析FlipAttack翻转模式引发的语义变化，探讨了FlipAttack有效性的潜在机制。我们假设原始提示和操纵提示之间的语义差异与ASB呈负相关。为了测试这一点，我们检查了FlipAttack模式的嵌入空间可视化（UMAP、TEK）和cos相似性。此外，我们还引入了一种新型的对抗攻击，即字母索引映射（AIM），旨在最大化语义差异，同时保持简单的可解码性。使用AdvBench的一个子集对GPT-4进行的实验表明，AIM及其变体AIM+ FBO实现了94%的ASB，优于FlipAttack和该子集的其他方法。我们的研究结果表明，虽然高度的语义差异至关重要，但与解码简单性的平衡是成功越狱的关键。这项工作有助于更深入地理解对抗提示机制，并提供了一种新的、有效的越狱技术。



## **32. FAIR-TAT: Improving Model Fairness Using Targeted Adversarial Training**

FAIR-TAT：使用针对性对抗训练提高模型公平性 cs.LG

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2410.23142v3) [paper-pdf](http://arxiv.org/pdf/2410.23142v3)

**Authors**: Tejaswini Medi, Steffen Jung, Margret Keuper

**Abstract**: Deep neural networks are susceptible to adversarial attacks and common corruptions, which undermine their robustness. In order to enhance model resilience against such challenges, Adversarial Training (AT) has emerged as a prominent solution. Nevertheless, adversarial robustness is often attained at the expense of model fairness during AT, i.e., disparity in class-wise robustness of the model. While distinctive classes become more robust towards such adversaries, hard to detect classes suffer. Recently, research has focused on improving model fairness specifically for perturbed images, overlooking the accuracy of the most likely non-perturbed data. Additionally, despite their robustness against the adversaries encountered during model training, state-of-the-art adversarial trained models have difficulty maintaining robustness and fairness when confronted with diverse adversarial threats or common corruptions. In this work, we address the above concerns by introducing a novel approach called Fair Targeted Adversarial Training (FAIR-TAT). We show that using targeted adversarial attacks for adversarial training (instead of untargeted attacks) can allow for more favorable trade-offs with respect to adversarial fairness. Empirical results validate the efficacy of our approach.

摘要: 深度神经网络容易受到对抗攻击和常见腐败的影响，从而削弱了其稳健性。为了增强模型应对此类挑战的弹性，对抗训练（AT）已成为一种重要的解决方案。然而，对抗鲁棒性通常是以AT期间牺牲模型公平性为代价的，即模型的类别稳健性存在差异。虽然独特的类别对此类对手变得更加强大，但难以检测的类别却会受到影响。最近，研究的重点是提高专门针对受干扰图像的模型公平性，而忽视了最有可能的未受干扰数据的准确性。此外，尽管最先进的对抗训练模型对模型训练期间遇到的对手具有鲁棒性，但在面对各种对抗威胁或常见腐败时，它们很难保持鲁棒性和公平性。在这项工作中，我们通过引入一种名为公平定向对抗训练（FAIR-STAT）的新颖方法来解决上述问题。我们表明，使用有针对性的对抗攻击进行对抗训练（而不是无针对性的攻击）可以在对抗公平性方面进行更有利的权衡。经验结果验证了我们方法的有效性。



## **33. Control Tax: The Price of Keeping AI in Check**

控制税：控制人工智能的代价 cs.AI

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2506.05296v2) [paper-pdf](http://arxiv.org/pdf/2506.05296v2)

**Authors**: Mikhail Terekhov, Zhen Ning David Liu, Caglar Gulcehre, Samuel Albanie

**Abstract**: The rapid integration of agentic AI into high-stakes real-world applications requires robust oversight mechanisms. The emerging field of AI Control (AIC) aims to provide such an oversight mechanism, but practical adoption depends heavily on implementation overhead. To study this problem better, we introduce the notion of Control tax -- the operational and financial cost of integrating control measures into AI pipelines. Our work makes three key contributions to the field of AIC: (1) we introduce a theoretical framework that quantifies the Control Tax and maps classifier performance to safety assurances; (2) we conduct comprehensive evaluations of state-of-the-art language models in adversarial settings, where attacker models insert subtle backdoors into code while monitoring models attempt to detect these vulnerabilities; and (3) we provide empirical financial cost estimates for control protocols and develop optimized monitoring strategies that balance safety and cost-effectiveness while accounting for practical constraints like auditing budgets. Our framework enables practitioners to make informed decisions by systematically connecting safety guarantees with their costs, advancing AIC through principled economic feasibility assessment across different deployment contexts.

摘要: 将代理人工智能快速集成到高风险的现实世界应用程序中需要强大的监督机制。新兴的AI控制（AIC）领域旨在提供这样的监督机制，但实际采用在很大程度上取决于实现开销。为了更好地研究这个问题，我们引入了控制税的概念--将控制措施整合到人工智能管道中的运营和财务成本。我们的工作为AIC领域做出了三个关键贡献：（1）我们引入了一个理论框架，可以量化控制税并将分类器性能映射到安全保证;（2）我们在对抗环境中对最先进的语言模型进行了全面评估，其中攻击者模型在监控模型试图检测这些漏洞时将微妙的后门插入代码中;（3）我们为控制协议提供经验财务成本估计，并开发优化的监控策略，平衡安全性和成本效益，同时考虑审计预算等实际限制。我们的框架使从业者能够通过系统地将安全保证与成本联系起来，做出明智的决策，并通过不同部署背景下的原则性经济可行性评估来推进AIC。



## **34. Exploiting AI for Attacks: On the Interplay between Adversarial AI and Offensive AI**

利用人工智能进行攻击：对抗性人工智能和进攻性人工智能之间的相互作用 cs.CR

under submission

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2506.12519v1) [paper-pdf](http://arxiv.org/pdf/2506.12519v1)

**Authors**: Saskia Laura Schröer, Luca Pajola, Alberto Castagnaro, Giovanni Apruzzese, Mauro Conti

**Abstract**: As Artificial Intelligence (AI) continues to evolve, it has transitioned from a research-focused discipline to a widely adopted technology, enabling intelligent solutions across various sectors. In security, AI's role in strengthening organizational resilience has been studied for over two decades. While much attention has focused on AI's constructive applications, the increasing maturity and integration of AI have also exposed its darker potentials. This article explores two emerging AI-related threats and the interplay between them: AI as a target of attacks (`Adversarial AI') and AI as a means to launch attacks on any target (`Offensive AI') -- potentially even on another AI. By cutting through the confusion and explaining these threats in plain terms, we introduce the complex and often misunderstood interplay between Adversarial AI and Offensive AI, offering a clear and accessible introduction to the challenges posed by these threats.

摘要: 随着人工智能（AI）的不断发展，它已经从一个以研究为重点的学科转变为一种广泛采用的技术，使智能解决方案能够跨越各个领域。在安全方面，人工智能在加强组织弹性方面的作用已经研究了二十多年。虽然人们的注意力都集中在人工智能的建设性应用上，但人工智能的日益成熟和整合也暴露了其更黑暗的潜力。本文探讨了两种新兴的与AI相关的威胁以及它们之间的相互作用：AI作为攻击的目标（“对抗性AI”）和AI作为对任何目标发动攻击的手段（“进攻性AI”）-甚至可能对另一个AI发动攻击。通过消除混乱并简单地解释这些威胁，我们介绍了对抗性AI和进攻性AI之间复杂且经常被误解的相互作用，为这些威胁带来的挑战提供了清晰且易于理解的介绍。



## **35. On the existence of consistent adversarial attacks in high-dimensional linear classification**

关于多维线性分类中一致对抗攻击的存在性 stat.ML

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2506.12454v1) [paper-pdf](http://arxiv.org/pdf/2506.12454v1)

**Authors**: Matteo Vilucchio, Lenka Zdeborová, Bruno Loureiro

**Abstract**: What fundamentally distinguishes an adversarial attack from a misclassification due to limited model expressivity or finite data? In this work, we investigate this question in the setting of high-dimensional binary classification, where statistical effects due to limited data availability play a central role. We introduce a new error metric that precisely capture this distinction, quantifying model vulnerability to consistent adversarial attacks -- perturbations that preserve the ground-truth labels. Our main technical contribution is an exact and rigorous asymptotic characterization of these metrics in both well-specified models and latent space models, revealing different vulnerability patterns compared to standard robust error measures. The theoretical results demonstrate that as models become more overparameterized, their vulnerability to label-preserving perturbations grows, offering theoretical insight into the mechanisms underlying model sensitivity to adversarial attacks.

摘要: 对抗性攻击与由于模型表达能力有限或数据有限而导致的错误分类的根本区别是什么？在这项工作中，我们在多维二元分类的背景下研究了这个问题，其中由于数据可用性有限而产生的统计效应发挥着核心作用。我们引入了一种新的错误指标，可以准确地捕捉这种区别，量化模型对一致对抗攻击的脆弱性--保留地面真相标签的扰动。我们的主要技术贡献是在明确指定的模型和潜在空间模型中对这些指标进行了精确而严格的渐进描述，揭示了与标准稳健错误测量相比不同的脆弱性模式。理论结果表明，随着模型变得更加过度参数化，它们对标签保留扰动的脆弱性也会增加，从而为模型对对抗性攻击敏感性的潜在机制提供了理论见解。



## **36. Pushing the Limits of Safety: A Technical Report on the ATLAS Challenge 2025**

突破安全极限：2025年ATLAS挑战赛技术报告 cs.CR

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2506.12430v1) [paper-pdf](http://arxiv.org/pdf/2506.12430v1)

**Authors**: Zonghao Ying, Siyang Wu, Run Hao, Peng Ying, Shixuan Sun, Pengyu Chen, Junze Chen, Hao Du, Kaiwen Shen, Shangkun Wu, Jiwei Wei, Shiyuan He, Yang Yang, Xiaohai Xu, Ke Ma, Qianqian Xu, Qingming Huang, Shi Lin, Xun Wang, Changting Lin, Meng Han, Yilei Jiang, Siqi Lai, Yaozhi Zheng, Yifei Song, Xiangyu Yue, Zonglei Jing, Tianyuan Zhang, Zhilei Zhu, Aishan Liu, Jiakai Wang, Siyuan Liang, Xianglong Kong, Hainan Li, Junjie Mu, Haotong Qin, Yue Yu, Lei Chen, Felix Juefei-Xu, Qing Guo, Xinyun Chen, Yew Soon Ong, Xianglong Liu, Dawn Song, Alan Yuille, Philip Torr, Dacheng Tao

**Abstract**: Multimodal Large Language Models (MLLMs) have enabled transformative advancements across diverse applications but remain susceptible to safety threats, especially jailbreak attacks that induce harmful outputs. To systematically evaluate and improve their safety, we organized the Adversarial Testing & Large-model Alignment Safety Grand Challenge (ATLAS) 2025}. This technical report presents findings from the competition, which involved 86 teams testing MLLM vulnerabilities via adversarial image-text attacks in two phases: white-box and black-box evaluations. The competition results highlight ongoing challenges in securing MLLMs and provide valuable guidance for developing stronger defense mechanisms. The challenge establishes new benchmarks for MLLM safety evaluation and lays groundwork for advancing safer multimodal AI systems. The code and data for this challenge are openly available at https://github.com/NY1024/ATLAS_Challenge_2025.

摘要: 多模式大型语言模型（MLLM）在不同的应用程序中实现了变革性的进步，但仍然容易受到安全威胁，尤其是引发有害输出的越狱攻击。为了系统地评估和提高其安全性，我们组织了对抗性测试和大模型对齐安全大挑战赛（ATLAS）2025。本技术报告介绍了比赛的结果，其中86个团队通过对抗性图像文本攻击分两个阶段测试MLLM漏洞：白盒和黑匣子评估。竞赛结果凸显了确保MLLM方面持续存在的挑战，并为开发更强大的防御机制提供了宝贵的指导。该挑战为MLLM安全评估建立了新的基准，并为推进更安全的多模式人工智能系统奠定了基础。此挑战的代码和数据可在https://github.com/NY1024/ATLAS_Challenge_2025上公开获取。



## **37. InverTune: Removing Backdoors from Multimodal Contrastive Learning Models via Trigger Inversion and Activation Tuning**

InverButton：通过触发倒置和激活调整从多模式对比学习模型中删除后门 cs.CR

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2506.12411v1) [paper-pdf](http://arxiv.org/pdf/2506.12411v1)

**Authors**: Mengyuan Sun, Yu Li, Yuchen Liu, Bo Du, Yunjie Ge

**Abstract**: Multimodal contrastive learning models like CLIP have demonstrated remarkable vision-language alignment capabilities, yet their vulnerability to backdoor attacks poses critical security risks. Attackers can implant latent triggers that persist through downstream tasks, enabling malicious control of model behavior upon trigger presentation. Despite great success in recent defense mechanisms, they remain impractical due to strong assumptions about attacker knowledge or excessive clean data requirements. In this paper, we introduce InverTune, the first backdoor defense framework for multimodal models under minimal attacker assumptions, requiring neither prior knowledge of attack targets nor access to the poisoned dataset. Unlike existing defense methods that rely on the same dataset used in the poisoning stage, InverTune effectively identifies and removes backdoor artifacts through three key components, achieving robust protection against backdoor attacks. Specifically, InverTune first exposes attack signatures through adversarial simulation, probabilistically identifying the target label by analyzing model response patterns. Building on this, we develop a gradient inversion technique to reconstruct latent triggers through activation pattern analysis. Finally, a clustering-guided fine-tuning strategy is employed to erase the backdoor function with only a small amount of arbitrary clean data, while preserving the original model capabilities. Experimental results show that InverTune reduces the average attack success rate (ASR) by 97.87% against the state-of-the-art (SOTA) attacks while limiting clean accuracy (CA) degradation to just 3.07%. This work establishes a new paradigm for securing multimodal systems, advancing security in foundation model deployment without compromising performance.

摘要: CLIP等多模式对比学习模型已经表现出出色的视觉语言对齐能力，但它们对后门攻击的脆弱性构成了严重的安全风险。攻击者可以植入在下游任务中持续存在的潜在触发器，从而在触发器呈现后对模型行为进行恶意控制。尽管最近的防御机制取得了巨大成功，但由于对攻击者知识的强烈假设或过度的干净数据要求，它们仍然不切实际。在本文中，我们介绍了InverTune，这是在最小攻击者假设下用于多模式模型的第一个后门防御框架，既不需要攻击目标的先验知识，也不需要访问中毒数据集。与依赖于中毒阶段使用的相同数据集的现有防御方法不同，InverTune通过三个关键组件有效识别和删除后门伪影，实现针对后门攻击的强大保护。具体来说，InverTune首先通过对抗模拟暴露攻击特征，通过分析模型响应模式概率地识别目标标签。在此基础上，我们开发了一种梯度倒置技术，通过激活模式分析重建潜在触发因素。最后，采用集群引导的微调策略，仅使用少量任意干净数据即可擦除后门函数，同时保留原始模型能力。实验结果表明，InverTune将针对最先进的（SOTA）攻击的平均攻击成功率（ASB）降低了97.87%，同时将干净准确率（CA）下降限制在仅3.07%。这项工作为保护多模式系统建立了一个新范式，在不影响性能的情况下提高基础模型部署的安全性。



## **38. Exploring the Secondary Risks of Large Language Models**

探索大型语言模型的次要风险 cs.LG

18 pages, 5 figures

**SubmitDate**: 2025-06-14    [abs](http://arxiv.org/abs/2506.12382v1) [paper-pdf](http://arxiv.org/pdf/2506.12382v1)

**Authors**: Jiawei Chen, Zhengwei Fang, Xiao Yang, Chao Yu, Zhaoxia Yin, Hang Su

**Abstract**: Ensuring the safety and alignment of Large Language Models is a significant challenge with their growing integration into critical applications and societal functions. While prior research has primarily focused on jailbreak attacks, less attention has been given to non-adversarial failures that subtly emerge during benign interactions. We introduce secondary risks a novel class of failure modes marked by harmful or misleading behaviors during benign prompts. Unlike adversarial attacks, these risks stem from imperfect generalization and often evade standard safety mechanisms. To enable systematic evaluation, we introduce two risk primitives verbose response and speculative advice that capture the core failure patterns. Building on these definitions, we propose SecLens, a black-box, multi-objective search framework that efficiently elicits secondary risk behaviors by optimizing task relevance, risk activation, and linguistic plausibility. To support reproducible evaluation, we release SecRiskBench, a benchmark dataset of 650 prompts covering eight diverse real-world risk categories. Experimental results from extensive evaluations on 16 popular models demonstrate that secondary risks are widespread, transferable across models, and modality independent, emphasizing the urgent need for enhanced safety mechanisms to address benign yet harmful LLM behaviors in real-world deployments.

摘要: 随着大型语言模型越来越多地集成到关键应用程序和社会功能中，确保大型语言模型的安全性和一致性是一项重大挑战。虽然之前的研究主要集中在越狱攻击上，但对良性互动中微妙出现的非对抗性失败的关注较少。我们引入了二级风险，这是一种新型的失败模式，其特征是良性提示期间的有害或误导行为。与对抗性攻击不同，这些风险源于不完美的概括，并且常常逃避标准安全机制。为了使系统的评估，我们引入了两个风险原语详细的响应和投机性的意见，捕捉核心故障模式。在这些定义的基础上，我们提出了SecLens，一个黑盒子，多目标搜索框架，通过优化任务相关性，风险激活，和语言的可扩展性，有效地消除二次风险行为。为了支持可重复的评估，我们发布了SecRiskBench，这是一个包含650个提示的基准数据集，涵盖了8个不同的现实风险类别。对16种流行模型进行广泛评估的实验结果表明，次级风险是普遍存在的，可跨模型转移，并且与模态无关，强调迫切需要增强安全机制，以解决现实世界部署中的良性但有害的LLM行为。



## **39. InfoFlood: Jailbreaking Large Language Models with Information Overload**

InfoFlood：用信息过载破解大型语言模型 cs.CR

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.12274v1) [paper-pdf](http://arxiv.org/pdf/2506.12274v1)

**Authors**: Advait Yadav, Haibo Jin, Man Luo, Jun Zhuang, Haohan Wang

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various domains. However, their potential to generate harmful responses has raised significant societal and regulatory concerns, especially when manipulated by adversarial techniques known as "jailbreak" attacks. Existing jailbreak methods typically involve appending carefully crafted prefixes or suffixes to malicious prompts in order to bypass the built-in safety mechanisms of these models.   In this work, we identify a new vulnerability in which excessive linguistic complexity can disrupt built-in safety mechanisms-without the need for any added prefixes or suffixes-allowing attackers to elicit harmful outputs directly. We refer to this phenomenon as Information Overload.   To automatically exploit this vulnerability, we propose InfoFlood, a jailbreak attack that transforms malicious queries into complex, information-overloaded queries capable of bypassing built-in safety mechanisms. Specifically, InfoFlood: (1) uses linguistic transformations to rephrase malicious queries, (2) identifies the root cause of failure when an attempt is unsuccessful, and (3) refines the prompt's linguistic structure to address the failure while preserving its malicious intent.   We empirically validate the effectiveness of InfoFlood on four widely used LLMs-GPT-4o, GPT-3.5-turbo, Gemini 2.0, and LLaMA 3.1-by measuring their jailbreak success rates. InfoFlood consistently outperforms baseline attacks, achieving up to 3 times higher success rates across multiple jailbreak benchmarks. Furthermore, we demonstrate that commonly adopted post-processing defenses, including OpenAI's Moderation API, Perspective API, and SmoothLLM, fail to mitigate these attacks. This highlights a critical weakness in traditional AI safety guardrails when confronted with information overload-based jailbreaks.

摘要: 大型语言模型（LLM）在各个领域都表现出了非凡的能力。然而，它们产生有害反应的可能性引起了严重的社会和监管担忧，特别是当被称为“越狱”攻击的对抗性技术操纵时。现有的越狱方法通常涉及在恶意提示中添加精心制作的前置或后缀，以绕过这些模型的内置安全机制。   在这项工作中，我们发现了一个新的漏洞，其中过度的语言复杂性可能会破坏内置的安全机制，而不需要添加任何前置或后缀，从而使攻击者能够直接获取有害输出。我们将这种现象称为信息过载。   为了自动利用此漏洞，我们提出了InfoFlood，这是一种越狱攻击，可将恶意查询转换为复杂的、信息超载的查询，能够绕过内置安全机制。具体来说，InfoFlood：（1）使用语言转换来重新表达恶意查询，（2）在尝试不成功时识别失败的根本原因，以及（3）细化提示的语言结构以解决失败，同时保留其恶意意图。   我们通过测量四种广泛使用的LLMS（GPT-4 o、GPT-3.5-涡轮、Gemini 2.0和LLaMA 3.1）的越狱成功率，以经验验证了InfoFlood对它们的有效性。InfoFlood的性能始终优于基线攻击，在多个越狱基准中实现了高达3倍的成功率。此外，我们还证明，常用的后处理防御措施（包括OpenAI的Moderation API、Perspective API和SmoothLLM）无法缓解这些攻击。这凸显了传统人工智能安全护栏在面临基于信息过载的越狱时的一个严重弱点。



## **40. AegisLLM: Scaling Agentic Systems for Self-Reflective Defense in LLM Security**

AegisLLM：在LLM安全中扩展自反射防御的防御系统 cs.LG

ICLR 2025 Workshop BuildingTrust

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2504.20965v2) [paper-pdf](http://arxiv.org/pdf/2504.20965v2)

**Authors**: Zikui Cai, Shayan Shabihi, Bang An, Zora Che, Brian R. Bartoldson, Bhavya Kailkhura, Tom Goldstein, Furong Huang

**Abstract**: We introduce AegisLLM, a cooperative multi-agent defense against adversarial attacks and information leakage. In AegisLLM, a structured workflow of autonomous agents - orchestrator, deflector, responder, and evaluator - collaborate to ensure safe and compliant LLM outputs, while self-improving over time through prompt optimization. We show that scaling agentic reasoning system at test-time - both by incorporating additional agent roles and by leveraging automated prompt optimization (such as DSPy)- substantially enhances robustness without compromising model utility. This test-time defense enables real-time adaptability to evolving attacks, without requiring model retraining. Comprehensive evaluations across key threat scenarios, including unlearning and jailbreaking, demonstrate the effectiveness of AegisLLM. On the WMDP unlearning benchmark, AegisLLM achieves near-perfect unlearning with only 20 training examples and fewer than 300 LM calls. For jailbreaking benchmarks, we achieve 51% improvement compared to the base model on StrongReject, with false refusal rates of only 7.9% on PHTest compared to 18-55% for comparable methods. Our results highlight the advantages of adaptive, agentic reasoning over static defenses, establishing AegisLLM as a strong runtime alternative to traditional approaches based on model modifications. Code is available at https://github.com/zikuicai/aegisllm

摘要: 我们引入了AegisLLM，这是一种针对对抗攻击和信息泄露的协作多代理防御系统。在AegisLLM中，由自治代理（协调器、偏转器、响应者和评估器）组成的结构化工作流程相互协作，以确保安全合规的LLM输出，同时通过及时优化随着时间的推移进行自我改进。我们表明，在测试时扩展代理推理系统--通过合并额外的代理角色和利用自动化提示优化（例如DSPy）--可以在不损害模型效用的情况下大幅增强稳健性。这种测试时防御能够实时适应不断发展的攻击，而无需模型重新训练。对关键威胁场景（包括取消学习和越狱）的全面评估展示了AegisLLM的有效性。在WMDP取消学习基准上，AegisLLM仅使用20个训练示例和少于300个LM调用即可实现近乎完美的取消学习。对于越狱基准，与Strongestival上的基本模型相比，我们实现了51%的改进，PHTest上的错误拒绝率仅为7.9%，而类似方法的错误拒绝率为18-55%。我们的结果强调了自适应、代理推理相对于静态防御的优势，将AegisLLM确立为基于模型修改的传统方法的强大运行时替代方案。代码可访问https://github.com/zikuicai/aegisllm



## **41. Learning Causality for Modern Machine Learning**

现代机器学习的学习因果关系 cs.LG

PhD thesis

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.12226v1) [paper-pdf](http://arxiv.org/pdf/2506.12226v1)

**Authors**: Yongqiang Chen

**Abstract**: In the past decades, machine learning with Empirical Risk Minimization (ERM) has demonstrated great capability in learning and exploiting the statistical patterns from data, or even surpassing humans. Despite the success, ERM avoids the modeling of causality the way of understanding and handling changes, which is fundamental to human intelligence. When deploying models beyond the training environment, distribution shifts are everywhere. For example, an autopilot system often needs to deal with new weather conditions that have not been seen during training, An Al-aided drug discovery system needs to predict the biochemical properties of molecules with respect to new viruses such as COVID-19. It renders the problem of Out-of-Distribution (OOD) generalization challenging to conventional machine learning.   In this thesis, we investigate how to incorporate and realize the causality for broader tasks in modern machine learning. In particular, we exploit the invariance implied by the principle of independent causal mechanisms (ICM), that is, the causal mechanisms generating the effects from causes do not inform or influence each other. Therefore, the conditional distribution between the target variable given its causes is invariant under distribution shifts. With the causal invariance principle, we first instantiate it to graphs -- a general data structure ubiquitous in many real-world industry and scientific applications, such as financial networks and molecules. Then, we shall see how learning the causality benefits many of the desirable properties of modern machine learning, in terms of (i) OOD generalization capability; (ii) interpretability; and (iii) robustness to adversarial attacks.   Realizing the causality in machine learning, on the other hand, raises a dilemma for optimization in conventional machine learning, as it often contradicts the objective of ERM...

摘要: 在过去的几十年里，具有经验风险最小化（ERM）的机器学习在学习和利用数据统计模式方面表现出了强大的能力，甚至超越了人类。尽管取得了成功，但ERM避免了因果关系的建模，即理解和处理变化的方式，而这是人类智能的基础。当在培训环境之外部署模型时，分布转移无处不在。例如，自动驾驶系统通常需要处理训练期间未观察到的新天气条件，Al辅助药物发现系统需要预测分子针对COVID-19等新病毒的生化性质。它使得分布外（OOD）概括问题对传统机器学习构成挑战。   在本文中，我们研究如何在现代机器学习中纳入和实现更广泛任务的因果关系。特别是，我们利用了独立因果机制（ICM）原则所暗示的不变性，即产生原因影响的因果机制不会相互告知或影响。因此，给定其原因的目标变量之间的条件分布在分布变化下是不变的。利用因果不变性原则，我们首先将其实例化为图形--这是一种普遍存在于许多现实世界的工业和科学应用中的通用数据结构，例如金融网络和分子。然后，我们将看到学习因果关系如何有利于现代机器学习的许多理想属性，包括（i）OOD概括能力;（ii）可解释性;（iii）对抗攻击的鲁棒性。   另一方面，实现机器学习中的因果关系会给传统机器学习中的优化带来困境，因为它经常与ERM的目标相矛盾..



## **42. Self-interpreting Adversarial Images**

自我解释对抗图像 cs.CR

in USENIX Security 2025

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2407.08970v4) [paper-pdf](http://arxiv.org/pdf/2407.08970v4)

**Authors**: Tingwei Zhang, Collin Zhang, John X. Morris, Eugene Bagdasarian, Vitaly Shmatikov

**Abstract**: We introduce a new type of indirect, cross-modal injection attacks against visual language models that enable creation of self-interpreting images. These images contain hidden "meta-instructions" that control how models answer users' questions about the image and steer models' outputs to express an adversary-chosen style, sentiment, or point of view.   Self-interpreting images act as soft prompts, conditioning the model to satisfy the adversary's (meta-)objective while still producing answers based on the image's visual content. Meta-instructions are thus a stronger form of prompt injection. Adversarial images look natural and the model's answers are coherent and plausible, yet they also follow the adversary-chosen interpretation, e.g., political spin, or even objectives that are not achievable with explicit text instructions.   We evaluate the efficacy of self-interpreting images for a variety of models, interpretations, and user prompts. We describe how these attacks could cause harm by enabling creation of self-interpreting content that carries spam, misinformation, or spin. Finally, we discuss defenses.

摘要: 我们针对视觉语言模型引入了一种新型的间接、跨模式注入攻击，可以创建自我解释的图像。这些图像包含隐藏的“元指令”，这些指令控制模型如何回答用户有关图像的问题，并引导模型的输出来表达对手选择的风格、情感或观点。   自我解释图像充当软提示，调节模型以满足对手的（Meta）目标，同时仍然根据图像的视觉内容生成答案。因此，元指令是一种更强的提示注入形式。对抗图像看起来很自然，模型的答案连贯且可信，但它们也遵循对抗选择的解释，例如，政治旋转，甚至是通过明确的文本指令无法实现的目标。   我们评估各种模型、解释和用户提示的自我解释图像的有效性。我们描述了这些攻击如何通过创建携带垃圾邮件、错误信息或旋转的自我解释内容来造成伤害。最后，我们谈谈防御。



## **43. Improving Large Language Model Safety with Contrastive Representation Learning**

通过对比表示学习提高大型语言模型安全性 cs.CL

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11938v1) [paper-pdf](http://arxiv.org/pdf/2506.11938v1)

**Authors**: Samuel Simko, Mrinmaya Sachan, Bernhard Schölkopf, Zhijing Jin

**Abstract**: Large Language Models (LLMs) are powerful tools with profound societal impacts, yet their ability to generate responses to diverse and uncontrolled inputs leaves them vulnerable to adversarial attacks. While existing defenses often struggle to generalize across varying attack types, recent advancements in representation engineering offer promising alternatives. In this work, we propose a defense framework that formulates model defense as a contrastive representation learning (CRL) problem. Our method finetunes a model using a triplet-based loss combined with adversarial hard negative mining to encourage separation between benign and harmful representations. Our experimental results across multiple models demonstrate that our approach outperforms prior representation engineering-based defenses, improving robustness against both input-level and embedding-space attacks without compromising standard performance. Our code is available at https://github.com/samuelsimko/crl-llm-defense

摘要: 大型语言模型（LLM）是具有深远社会影响的强大工具，但它们对多样化且不受控制的输入产生响应的能力使它们容易受到对抗性攻击。虽然现有的防御通常很难在不同的攻击类型中进行概括，但表示工程的最新进展提供了有希望的替代方案。在这项工作中，我们提出了一个防御框架，将模型防御制定为对比表示学习（RTL）问题。我们的方法使用基于三重组的损失结合对抗性硬负挖掘来微调模型，以鼓励良性和有害表示之间的分离。我们跨多个模型的实验结果表明，我们的方法优于基于先验表示工程的防御，在不损害标准性能的情况下提高了针对输入级和嵌入空间攻击的鲁棒性。我们的代码可在https://github.com/samuelsimko/crl-llm-defense上获取



## **44. Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation**

带有修剪的攻击图：优化隐形越狱提示生成以增强的LLM内容审核 cs.CR

14 pages, 5 figures

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2501.18638v2) [paper-pdf](http://arxiv.org/pdf/2501.18638v2)

**Authors**: Daniel Schwartz, Dmitriy Bespalov, Zhe Wang, Ninad Kulkarni, Yanjun Qi

**Abstract**: As large language models (LLMs) become increasingly prevalent, ensuring their robustness against adversarial misuse is crucial. This paper introduces the GAP (Graph of Attacks with Pruning) framework, an advanced approach for generating stealthy jailbreak prompts to evaluate and enhance LLM safeguards. GAP addresses limitations in existing tree-based LLM jailbreak methods by implementing an interconnected graph structure that enables knowledge sharing across attack paths. Our experimental evaluation demonstrates GAP's superiority over existing techniques, achieving a 20.8% increase in attack success rates while reducing query costs by 62.7%. GAP consistently outperforms state-of-the-art methods for attacking both open and closed LLMs, with attack success rates of >96%. Additionally, we present specialized variants like GAP-Auto for automated seed generation and GAP-VLM for multimodal attacks. GAP-generated prompts prove highly effective in improving content moderation systems, increasing true positive detection rates by 108.5% and accuracy by 183.6% when used for fine-tuning. Our implementation is available at https://github.com/dsbuddy/GAP-LLM-Safety.

摘要: 随着大型语言模型（LLM）变得越来越普遍，确保其针对对抗性滥用的鲁棒性至关重要。本文介绍了GAP（带有修剪的攻击图）框架，这是一种生成隐形越狱提示以评估和增强LLM保障措施的高级方法。GAP通过实现互连的图结构来解决现有基于树的LLM越狱方法的局限性，该结构能够实现跨攻击路径的知识共享。我们的实验评估证明了GAP相对于现有技术的优越性，攻击成功率提高了20.8%，同时将查询成本降低了62.7%。对于攻击开放式和封闭式LLM，RAP始终优于最先进的方法，攻击成功率> 96%。此外，我们还提供了专门的变体，例如用于自动种子生成的GAP-Auto和用于多模式攻击的GAP-VLM。事实证明，由间隙生成的提示在改进内容审核系统方面非常有效，用于微调时，真阳性检测率可提高108.5%，准确率可提高183.6%。我们的实施可在https://github.com/dsbuddy/GAP-LLM-Safety上获取。



## **45. Attention-based Adversarial Robust Distillation in Radio Signal Classifications for Low-Power IoT Devices**

低功耗物联网设备无线信号分类中基于注意力的对抗鲁棒蒸馏 cs.LG

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11892v1) [paper-pdf](http://arxiv.org/pdf/2506.11892v1)

**Authors**: Lu Zhang, Sangarapillai Lambotharan, Gan Zheng, Guisheng Liao, Basil AsSadhan, Fabio Roli

**Abstract**: Due to great success of transformers in many applications such as natural language processing and computer vision, transformers have been successfully applied in automatic modulation classification. We have shown that transformer-based radio signal classification is vulnerable to imperceptible and carefully crafted attacks called adversarial examples. Therefore, we propose a defense system against adversarial examples in transformer-based modulation classifications. Considering the need for computationally efficient architecture particularly for Internet of Things (IoT)-based applications or operation of devices in environment where power supply is limited, we propose a compact transformer for modulation classification. The advantages of robust training such as adversarial training in transformers may not be attainable in compact transformers. By demonstrating this, we propose a novel compact transformer that can enhance robustness in the presence of adversarial attacks. The new method is aimed at transferring the adversarial attention map from the robustly trained large transformer to a compact transformer. The proposed method outperforms the state-of-the-art techniques for the considered white-box scenarios including fast gradient method and projected gradient descent attacks. We have provided reasoning of the underlying working mechanisms and investigated the transferability of the adversarial examples between different architectures. The proposed method has the potential to protect the transformer from the transferability of adversarial examples.

摘要: 由于转换器在自然语言处理和计算机视觉等许多应用中取得了巨大成功，转换器已成功应用于自动调制分类。我们已经表明，基于变压器的无线电信号分类很容易受到难以察觉且精心设计的攻击，称为对抗性示例。因此，我们提出了一种针对基于变压器的调制分类中对抗性示例的防御系统。考虑到特别是对于基于物联网（IoT）的应用或在电源受限的环境中的设备的操作的计算高效架构的需要，我们提出了用于调制分类的紧凑型Transformer。强大的训练，如变压器中的对抗训练的优点可能无法在紧凑的变压器。通过证明这一点，我们提出了一种新型的紧凑型Transformer，可以增强对抗攻击的鲁棒性。新方法的目的是将对抗性注意力地图从经过稳健训练的大型Transformer转移到紧凑Transformer。对于所考虑的白盒场景，包括快速梯度方法和投影梯度下降攻击，所提出的方法优于最新技术。我们提供了底层工作机制的推理，并研究了不同架构之间对抗性示例的可移植性。所提出的方法有可能保护Transformer免受对抗性示例的可移植性的影响。



## **46. Black-Box Adversarial Attacks on LLM-Based Code Completion**

基于LLM的代码补全黑盒对抗攻击 cs.CR

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2408.02509v2) [paper-pdf](http://arxiv.org/pdf/2408.02509v2)

**Authors**: Slobodan Jenko, Niels Mündler, Jingxuan He, Mark Vero, Martin Vechev

**Abstract**: Modern code completion engines, powered by large language models (LLMs), assist millions of developers with their strong capabilities to generate functionally correct code. Due to this popularity, it is crucial to investigate the security implications of relying on LLM-based code completion. In this work, we demonstrate that state-of-the-art black-box LLM-based code completion engines can be stealthily biased by adversaries to significantly increase their rate of insecure code generation. We present the first attack, named INSEC, that achieves this goal. INSEC works by injecting an attack string as a short comment in the completion input. The attack string is crafted through a query-based optimization procedure starting from a set of carefully designed initialization schemes. We demonstrate INSEC's broad applicability and effectiveness by evaluating it on various state-of-the-art open-source models and black-box commercial services (e.g., OpenAI API and GitHub Copilot). On a diverse set of security-critical test cases, covering 16 CWEs across 5 programming languages, INSEC increases the rate of generated insecure code by more than 50%, while maintaining the functional correctness of generated code. We consider INSEC practical -- it requires low resources and costs less than 10 US dollars to develop on commodity hardware. Moreover, we showcase the attack's real-world deployability, by developing an IDE plug-in that stealthily injects INSEC into the GitHub Copilot extension.

摘要: 由大型语言模型（LLM）支持的现代代码完成引擎可以帮助数百万开发人员以其强大的能力生成功能正确的代码。由于这种受欢迎程度，研究依赖基于LLM的代码完成的安全影响至关重要。在这项工作中，我们证明了最先进的基于LLM的黑匣子代码完成引擎可能会受到对手的悄悄偏见，以显着提高其不安全代码生成率。我们介绍了第一个攻击，名为INSEC，可以实现这一目标。INSEC的工作原理是在完成输入中注入攻击字符串作为简短注释。攻击字符串是通过基于查询的优化过程从一组精心设计的初始化方案开始精心设计的。我们通过在各种最先进的开源模型和黑匣子商业服务上进行评估来展示INSEC的广泛适用性和有效性（例如，OpenAI API和GitHub Copilot）。在一组多样化的安全关键测试用例中，涵盖5种编程语言的16个CWE，INSEC将生成的不安全代码的比率提高了50%以上，同时保持生成代码的功能正确性。我们认为INSEC是可行的--在商品硬件上进行开发所需的资源较少，成本不到10美元。此外，我们通过开发一个将INSEC秘密注入GitHub Copilot扩展的IDE插件，展示了攻击在现实世界中的可部署性。



## **47. TrustGLM: Evaluating the Robustness of GraphLLMs Against Prompt, Text, and Structure Attacks**

TrustGLM：评估GraphLLM针对提示、文本和结构攻击的稳健性 cs.LG

12 pages, 5 figures, in KDD 2025

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11844v1) [paper-pdf](http://arxiv.org/pdf/2506.11844v1)

**Authors**: Qihai Zhang, Xinyue Sheng, Yuanfu Sun, Qiaoyu Tan

**Abstract**: Inspired by the success of large language models (LLMs), there is a significant research shift from traditional graph learning methods to LLM-based graph frameworks, formally known as GraphLLMs. GraphLLMs leverage the reasoning power of LLMs by integrating three key components: the textual attributes of input nodes, the structural information of node neighborhoods, and task-specific prompts that guide decision-making. Despite their promise, the robustness of GraphLLMs against adversarial perturbations remains largely unexplored-a critical concern for deploying these models in high-stakes scenarios. To bridge the gap, we introduce TrustGLM, a comprehensive study evaluating the vulnerability of GraphLLMs to adversarial attacks across three dimensions: text, graph structure, and prompt manipulations. We implement state-of-the-art attack algorithms from each perspective to rigorously assess model resilience. Through extensive experiments on six benchmark datasets from diverse domains, our findings reveal that GraphLLMs are highly susceptible to text attacks that merely replace a few semantically similar words in a node's textual attribute. We also find that standard graph structure attack methods can significantly degrade model performance, while random shuffling of the candidate label set in prompt templates leads to substantial performance drops. Beyond characterizing these vulnerabilities, we investigate defense techniques tailored to each attack vector through data-augmented training and adversarial training, which show promising potential to enhance the robustness of GraphLLMs. We hope that our open-sourced library will facilitate rapid, equitable evaluation and inspire further innovative research in this field.

摘要: 受大型语言模型（LLM）成功的启发，研究从传统的图学习方法发生了重大转变，转向基于LLM的图框架（正式称为GraphLLM）。GraphLLM通过集成三个关键组件来利用LLM的推理能力：输入节点的文本属性、节点邻居的结构信息以及指导决策的特定任务提示。尽管它们有希望，但GraphLLM对对抗性扰动的稳健性在很大程度上仍然没有被开发--这是在高风险场景中部署这些模型的一个关键问题。为了弥合这一差距，我们引入了TrustGLM，这是一项综合研究，评估了GraphLLM在三个维度（文本、图形结构和提示操作）中对对抗攻击的脆弱性。我们从各个角度实施最先进的攻击算法，以严格评估模型弹性。通过对来自不同领域的六个基准数据集的广泛实验，我们的研究结果表明，GraphLLM非常容易受到文本攻击，这些攻击只是替换节点文本属性中的一些语义相似的单词。我们还发现，标准图结构攻击方法会显着降低模型性能，而提示模板中候选标签集的随机洗牌会导致性能大幅下降。除了描述这些漏洞之外，我们还通过数据增强训练和对抗训练研究了针对每个攻击载体量身定制的防御技术，这些技术在增强GraphLLM稳健性方面表现出了广阔的潜力。我们希望我们的开源图书馆能够促进快速、公平的评估，并激发该领域的进一步创新研究。



## **48. Differential Privacy in Machine Learning: From Symbolic AI to LLMs**

机器学习中的差异隐私：从符号人工智能到LLM cs.CR

arXiv admin note: text overlap with arXiv:2303.00654 by other authors

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11687v1) [paper-pdf](http://arxiv.org/pdf/2506.11687v1)

**Authors**: Francisco Aguilera-Martínez, Fernando Berzal

**Abstract**: Machine learning models should not reveal particular information that is not otherwise accessible. Differential privacy provides a formal framework to mitigate privacy risks by ensuring that the inclusion or exclusion of any single data point does not significantly alter the output of an algorithm, thus limiting the exposure of private information. This survey paper explores the foundational definitions of differential privacy, reviews its original formulations and tracing its evolution through key research contributions. It then provides an in-depth examination of how DP has been integrated into machine learning models, analyzing existing proposals and methods to preserve privacy when training ML models. Finally, it describes how DP-based ML techniques can be evaluated in practice. %Finally, it discusses the broader implications of DP, highlighting its potential for public benefit, its real-world applications, and the challenges it faces, including vulnerabilities to adversarial attacks. By offering a comprehensive overview of differential privacy in machine learning, this work aims to contribute to the ongoing development of secure and responsible AI systems.

摘要: 机器学习模型不应透露以其他方式无法访问的特定信息。差异隐私提供了一个正式的框架，可以通过确保任何单个数据点的包含或排除不会显着改变算法的输出来降低隐私风险，从而限制私人信息的暴露。这篇调查论文探讨了差异隐私的基本定义，回顾了其原始公式，并通过关键研究贡献追踪其演变。然后，它深入研究了DP如何集成到机器学习模型中，分析了在训练ML模型时保护隐私的现有提案和方法。最后，它描述了如何在实践中评估基于DP的ML技术。%最后，它讨论了DP的更广泛影响，强调了其对公共利益的潜力、其在现实世界中的应用以及它面临的挑战，包括对抗性攻击的脆弱性。通过全面概述机器学习中的差异隐私，这项工作旨在为安全且负责任的人工智能系统的持续开发做出贡献。



## **49. KCES: Training-Free Defense for Robust Graph Neural Networks via Kernel Complexity**

KCES：通过核复杂性为鲁棒图神经网络提供免训练防御 cs.LG

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11611v1) [paper-pdf](http://arxiv.org/pdf/2506.11611v1)

**Authors**: Yaning Jia, Shenyang Deng, Chiyu Ma, Yaoqing Yang, Soroush Vosoughi

**Abstract**: Graph Neural Networks (GNNs) have achieved impressive success across a wide range of graph-based tasks, yet they remain highly vulnerable to small, imperceptible perturbations and adversarial attacks. Although numerous defense methods have been proposed to address these vulnerabilities, many rely on heuristic metrics, overfit to specific attack patterns, and suffer from high computational complexity. In this paper, we propose Kernel Complexity-Based Edge Sanitization (KCES), a training-free, model-agnostic defense framework. KCES leverages Graph Kernel Complexity (GKC), a novel metric derived from the graph's Gram matrix that characterizes GNN generalization via its test error bound. Building on GKC, we define a KC score for each edge, measuring the change in GKC when the edge is removed. Edges with high KC scores, typically introduced by adversarial perturbations, are pruned to mitigate their harmful effects, thereby enhancing GNNs' robustness. KCES can also be seamlessly integrated with existing defense strategies as a plug-and-play module without requiring training. Theoretical analysis and extensive experiments demonstrate that KCES consistently enhances GNN robustness, outperforms state-of-the-art baselines, and amplifies the effectiveness of existing defenses, offering a principled and efficient solution for securing GNNs.

摘要: 图神经网络（GNN）在广泛的基于图的任务中取得了令人印象深刻的成功，但它们仍然极易受到小的、难以察觉的扰动和对抗性攻击的影响。尽管已经提出了许多防御方法来解决这些漏洞，但许多方法依赖于启发式指标，过度适合特定的攻击模式，并且计算复杂性很高。在本文中，我们提出了基于核心复杂性的边缘清理（KCES），这是一种免训练、模型不可知的防御框架。KCES利用图核复杂性（GKC），这是一种从图的Gram矩阵派生的新型指标，通过其测试误差界限来描述GNN一般化的特征。在GKC的基础上，我们为每条边定义了KC分数，测量删除边时GKC的变化。通常由对抗性扰动引入的具有高KC分数的边被修剪以减轻其有害影响，从而增强GNN的鲁棒性。KCES还可以作为即插即用模块与现有防御策略无缝集成，无需培训。理论分析和大量实验表明，KCES持续增强GNN稳健性，优于最先进的基线，并放大了现有防御的有效性，为保护GNN提供了原则性且高效的解决方案。



## **50. Investigating Vulnerabilities and Defenses Against Audio-Visual Attacks: A Comprehensive Survey Emphasizing Multimodal Models**

调查漏洞和针对视听攻击的防御：强调多模式模型的全面调查 cs.CR

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11521v1) [paper-pdf](http://arxiv.org/pdf/2506.11521v1)

**Authors**: Jinming Wen, Xinyi Wu, Shuai Zhao, Yanhao Jia, Yuwen Li

**Abstract**: Multimodal large language models (MLLMs), which bridge the gap between audio-visual and natural language processing, achieve state-of-the-art performance on several audio-visual tasks. Despite the superior performance of MLLMs, the scarcity of high-quality audio-visual training data and computational resources necessitates the utilization of third-party data and open-source MLLMs, a trend that is increasingly observed in contemporary research. This prosperity masks significant security risks. Empirical studies demonstrate that the latest MLLMs can be manipulated to produce malicious or harmful content. This manipulation is facilitated exclusively through instructions or inputs, including adversarial perturbations and malevolent queries, effectively bypassing the internal security mechanisms embedded within the models. To gain a deeper comprehension of the inherent security vulnerabilities associated with audio-visual-based multimodal models, a series of surveys investigates various types of attacks, including adversarial and backdoor attacks. While existing surveys on audio-visual attacks provide a comprehensive overview, they are limited to specific types of attacks, which lack a unified review of various types of attacks. To address this issue and gain insights into the latest trends in the field, this paper presents a comprehensive and systematic review of audio-visual attacks, which include adversarial attacks, backdoor attacks, and jailbreak attacks. Furthermore, this paper also reviews various types of attacks in the latest audio-visual-based MLLMs, a dimension notably absent in existing surveys. Drawing upon comprehensive insights from a substantial review, this paper delineates both challenges and emergent trends for future research on audio-visual attacks and defense.

摘要: 多模式大型语言模型（MLLM）弥合了视听和自然语言处理之间的差距，在多项视听任务上实现了最先进的性能。尽管MLLM性能优越，但高质量视听训练数据和计算资源的稀缺使得需要利用第三方数据和开源MLLM，这是当代研究中越来越多地观察到的趋势。这种繁荣掩盖了巨大的安全风险。实证研究表明，最新的MLLM可能会被操纵以产生恶意或有害内容。这种操纵完全通过指令或输入（包括对抗性扰动和恶意查询）来促进，有效地绕过了模型中嵌入的内部安全机制。为了更深入地了解与基于视听的多模式模型相关的固有安全漏洞，一系列调查调查了各种类型的攻击，包括对抗性攻击和后门攻击。虽然现有的关于视听攻击的调查提供了全面的概述，但仅限于特定类型的攻击，缺乏对各种类型的攻击的统一审查。为了解决这个问题并深入了解该领域的最新趋势，本文对视听攻击进行了全面、系统的回顾，其中包括对抗性攻击、后门攻击和越狱攻击。此外，本文还审查了最新的基于视听的MLLM中的各种类型的攻击，这是现有调查中明显缺乏的一个方面。本文从大量的评论中得出了全面的见解，描绘了未来视听攻击和防御研究的挑战和新兴趋势。



