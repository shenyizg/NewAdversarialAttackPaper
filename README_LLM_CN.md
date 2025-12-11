# Latest Large Language Model Attack Papers
**update at 2025-12-11 09:52:35**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. When Tables Leak: Attacking String Memorization in LLM-Based Tabular Data Generation**

当表泄露时：攻击基于LLM的表格数据生成中的字符串重新同步 cs.LG

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08875v1) [paper-pdf](https://arxiv.org/pdf/2512.08875v1)

**Authors**: Joshua Ward, Bochao Gu, Chi-Hua Wang, Guang Cheng

**Abstract**: Large Language Models (LLMs) have recently demonstrated remarkable performance in generating high-quality tabular synthetic data. In practice, two primary approaches have emerged for adapting LLMs to tabular data generation: (i) fine-tuning smaller models directly on tabular datasets, and (ii) prompting larger models with examples provided in context. In this work, we show that popular implementations from both regimes exhibit a tendency to compromise privacy by reproducing memorized patterns of numeric digits from their training data. To systematically analyze this risk, we introduce a simple No-box Membership Inference Attack (MIA) called LevAtt that assumes adversarial access to only the generated synthetic data and targets the string sequences of numeric digits in synthetic observations. Using this approach, our attack exposes substantial privacy leakage across a wide range of models and datasets, and in some cases, is even a perfect membership classifier on state-of-the-art models. Our findings highlight a unique privacy vulnerability of LLM-based synthetic data generation and the need for effective defenses. To this end, we propose two methods, including a novel sampling strategy that strategically perturbs digits during generation. Our evaluation demonstrates that this approach can defeat these attacks with minimal loss of fidelity and utility of the synthetic data.

摘要: 大型语言模型（LLM）最近在生成高质量表格合成数据方面表现出色。在实践中，出现了两种用于使LLM适应表格数据生成的主要方法：（i）直接在表格数据集上微调较小的模型，以及（ii）通过上下文中提供的示例来提示较大的模型。在这项工作中，我们表明，这两种制度的流行实现都表现出通过从训练数据中复制记忆的数字模式来损害隐私的倾向。为了系统性地分析这种风险，我们引入了一种名为LevAtt的简单无箱成员推断攻击（MIA），该攻击假设仅对生成的合成数据进行对抗访问，并针对合成观察中的数字字符串序列。使用这种方法，我们的攻击暴露了广泛的模型和数据集中的大量隐私泄露，在某些情况下，甚至是最先进模型上的完美成员资格分类器。我们的研究结果强调了基于LLM的合成数据生成的独特隐私漏洞以及有效防御的必要性。为此，我们提出了两种方法，包括一种新颖的采样策略，可以在生成期间战略性地扰乱数字。我们的评估表明，这种方法可以在合成数据的保真度和实用性损失最小的情况下击败这些攻击。



## **2. PrivTune: Efficient and Privacy-Preserving Fine-Tuning of Large Language Models via Device-Cloud Collaboration**

PrivButton：通过设备云协作对大型语言模型进行高效且保护隐私的微调 cs.CR

Accepted at IEEE INFOCOM 2026 (full version)

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08809v1) [paper-pdf](https://arxiv.org/pdf/2512.08809v1)

**Authors**: Yi Liu, Weixiang Han, Chengjun Cai, Xingliang Yuan, Cong Wang

**Abstract**: With the rise of large language models, service providers offer language models as a service, enabling users to fine-tune customized models via uploaded private datasets. However, this raises concerns about sensitive data leakage. Prior methods, relying on differential privacy within device-cloud collaboration frameworks, struggle to balance privacy and utility, exposing users to inference attacks or degrading fine-tuning performance. To address this, we propose PrivTune, an efficient and privacy-preserving fine-tuning framework via Split Learning (SL). The key idea of PrivTune is to inject crafted noise into token representations from the SL bottom model, making each token resemble the $n$-hop indirect neighbors. PrivTune formulates this as an optimization problem to compute the optimal noise vector, aligning with defense-utility goals. On this basis, it then adjusts the parameters (i.e., mean) of the $d_χ$-Privacy noise distribution to align with the optimization direction and scales the noise according to token importance to minimize distortion. Experiments on five datasets (covering both classification and generation tasks) against three embedding inversion and three attribute inference attacks show that, using RoBERTa on the Stanford Sentiment Treebank dataset, PrivTune reduces the attack success rate to 10% with only a 3.33% drop in utility performance, outperforming state-of-the-art baselines.

摘要: 随着大型语言模型的兴起，服务提供商提供语言模型作为服务，使用户能够通过上传的私人数据集微调定制模型。然而，这引发了对敏感数据泄露的担忧。先前的方法依赖于设备云协作框架内的差异隐私，难以平衡隐私和实用性，从而使用户面临推理攻击或降低微调性能。为了解决这个问题，我们提出了PrivButton，这是一个通过Split Learning（SL）进行的高效且保护隐私的微调框架。PrivTune的关键想法是将精心设计的噪音注入SL底部模型的令牌表示中，使每个令牌类似于$n$-hop间接邻居。PrivTune将其定义为一个优化问题，以计算最佳噪音载体，与防御效用目标保持一致。在此基础上，它然后调整参数（即，$d_x $-隐私噪音分布的平均值）以与优化方向保持一致，并根据令牌重要性缩放噪音以最大限度地减少失真。针对三种嵌入倒置和三种属性推断攻击的五个数据集（涵盖分类和生成任务）的实验表明，在斯坦福大学Sentiment Treebank数据集上使用RoBERTa，PrivTune将攻击成功率降低至10%，而实用程序性能仅下降3.33%，表现优于最先进的基线。



## **3. Attention is All You Need to Defend Against Indirect Prompt Injection Attacks in LLMs**

防御LLM中的间接即时注入攻击只需注意力 cs.CR

Accepted by Network and Distributed System Security (NDSS) Symposium 2026

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08417v1) [paper-pdf](https://arxiv.org/pdf/2512.08417v1)

**Authors**: Yinan Zhong, Qianhao Miao, Yanjiao Chen, Jiangyi Deng, Yushi Cheng, Wenyuan Xu

**Abstract**: Large Language Models (LLMs) have been integrated into many applications (e.g., web agents) to perform more sophisticated tasks. However, LLM-empowered applications are vulnerable to Indirect Prompt Injection (IPI) attacks, where instructions are injected via untrustworthy external data sources. This paper presents Rennervate, a defense framework to detect and prevent IPI attacks. Rennervate leverages attention features to detect the covert injection at a fine-grained token level, enabling precise sanitization that neutralizes IPI attacks while maintaining LLM functionalities. Specifically, the token-level detector is materialized with a 2-step attentive pooling mechanism, which aggregates attention heads and response tokens for IPI detection and sanitization. Moreover, we establish a fine-grained IPI dataset, FIPI, to be open-sourced to support further research. Extensive experiments verify that Rennervate outperforms 15 commercial and academic IPI defense methods, achieving high precision on 5 LLMs and 6 datasets. We also demonstrate that Rennervate is transferable to unseen attacks and robust against adaptive adversaries.

摘要: 大型语言模型（LLM）已集成到许多应用程序中（例如，Web代理）来执行更复杂的任务。然而，LLM授权的应用程序很容易受到间接提示注入（IPI）攻击，其中指令是通过不可信的外部数据源注入的。本文介绍了Rennervate，一个用于检测和防止IPI攻击的防御框架。Rennervate利用注意力功能来检测细粒度代币级别的隐蔽注入，从而实现精确的清理，以中和IPI攻击，同时维护LLM功能。具体来说，代币级检测器通过两步注意池机制实现，该机制聚集注意力头和响应代币以进行IPI检测和清理。此外，我们还建立了一个细粒度的IPI数据集FIPI，将其开源以支持进一步的研究。大量实验证实，Rennervate优于15种商业和学术IPI防御方法，在5个LLM和6个数据集上实现了高精度。我们还证明，Rennervate可以转移到不可见的攻击中，并且对适应性对手具有强大的鲁棒性。



## **4. A Practical Framework for Evaluating Medical AI Security: Reproducible Assessment of Jailbreaking and Privacy Vulnerabilities Across Clinical Specialties**

评估医疗人工智能安全性的实用框架：跨临床专业越狱和隐私漏洞的可重复性评估 cs.CR

6 pages, 1 figure, framework proposal

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2512.08185v1) [paper-pdf](https://arxiv.org/pdf/2512.08185v1)

**Authors**: Jinghao Wang, Ping Zhang, Carter Yagemann

**Abstract**: Medical Large Language Models (LLMs) are increasingly deployed for clinical decision support across diverse specialties, yet systematic evaluation of their robustness to adversarial misuse and privacy leakage remains inaccessible to most researchers. Existing security benchmarks require GPU clusters, commercial API access, or protected health data -- barriers that limit community participation in this critical research area. We propose a practical, fully reproducible framework for evaluating medical AI security under realistic resource constraints. Our framework design covers multiple medical specialties stratified by clinical risk -- from high-risk domains such as emergency medicine and psychiatry to general practice -- addressing jailbreaking attacks (role-playing, authority impersonation, multi-turn manipulation) and privacy extraction attacks. All evaluation utilizes synthetic patient records requiring no IRB approval. The framework is designed to run entirely on consumer CPU hardware using freely available models, eliminating cost barriers. We present the framework specification including threat models, data generation methodology, evaluation protocols, and scoring rubrics. This proposal establishes a foundation for comparative security assessment of medical-specialist models and defense mechanisms, advancing the broader goal of ensuring safe and trustworthy medical AI systems.

摘要: 医学大型语言模型（LLM）越来越多地被部署用于不同专业的临床决策支持，但大多数研究人员仍然无法对其对抗性滥用和隐私泄露的稳健性进行系统评估。现有的安全基准测试需要图形处理器集群、商业API访问或受保护的健康数据--这些障碍限制了社区参与这一关键研究领域。我们提出了一个实用、完全可重复的框架，用于在现实资源限制下评估医疗人工智能安全性。我们的框架设计涵盖了按临床风险分层的多个医学专业--从急诊医学和精神病学等高风险领域到全科医学--解决越狱攻击（角色扮演、权威模仿、多回合操纵）和隐私提取攻击。所有评估均使用合成患者记录，无需获得机构审核委员会批准。该框架旨在完全在使用免费型号的消费者中央处理器硬件上运行，消除了成本障碍。我们提出了框架规范，包括威胁模型、数据生成方法、评估协议和评分规则。该提案为医疗专家模型和防御机制的比较安全评估奠定了基础，推进确保医疗人工智能系统安全可信的更广泛目标。



## **5. Detecting Ambiguity Aversion in Cyberattack Behavior to Inform Cognitive Defense Strategies**

检测网络攻击行为中的模糊厌恶以告知认知防御策略 cs.CR

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.08107v1) [paper-pdf](https://arxiv.org/pdf/2512.08107v1)

**Authors**: Stephan Carney, Soham Hans, Sofia Hirschmann, Stacey Marsella, Yvonne Fonken, Peggy Wu, Nikolos Gurney

**Abstract**: Adversaries (hackers) attempting to infiltrate networks frequently face uncertainty in their operational environments. This research explores the ability to model and detect when they exhibit ambiguity aversion, a cognitive bias reflecting a preference for known (versus unknown) probabilities. We introduce a novel methodological framework that (1) leverages rich, multi-modal data from human-subjects red-team experiments, (2) employs a large language model (LLM) pipeline to parse unstructured logs into MITRE ATT&CK-mapped action sequences, and (3) applies a new computational model to infer an attacker's ambiguity aversion level in near-real time. By operationalizing this cognitive trait, our work provides a foundational component for developing adaptive cognitive defense strategies.

摘要: 试图渗透网络的对手（黑客）经常面临其操作环境的不确定性。这项研究探索了建模和检测他们何时表现出歧义厌恶的能力，这是一种反映对已知（与未知）概率偏好的认知偏见。我们引入了一种新颖的方法论框架，（1）利用来自人类受试者红队实验的丰富、多模式数据，（2）采用大型语言模型（LLM）管道将非结构化日志解析为MITRE ATA和CK映射的动作序列，（3）应用新的计算模型来近实时地推断攻击者的歧义厌恶水平。通过操作这种认知特征，我们的工作为开发适应性认知防御策略提供了基础组成部分。



## **6. RL-MTJail: Reinforcement Learning for Automated Black-Box Multi-Turn Jailbreaking of Large Language Models**

RL-MTJail：用于大型语言模型自动黑匣子多回合越狱的强化学习 cs.AI

19 pages, 15 figures

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07761v1) [paper-pdf](https://arxiv.org/pdf/2512.07761v1)

**Authors**: Xiqiao Xiong, Ouxiang Li, Zhuo Liu, Moxin Li, Wentao Shi, Fuli Feng, Xiangnan He

**Abstract**: Large language models are vulnerable to jailbreak attacks, threatening their safe deployment in real-world applications. This paper studies black-box multi-turn jailbreaks, aiming to train attacker LLMs to elicit harmful content from black-box models through a sequence of prompt-output interactions. Existing approaches typically rely on single turn optimization, which is insufficient for learning long-term attack strategies. To bridge this gap, we formulate the problem as a multi-turn reinforcement learning task, directly optimizing the harmfulness of the final-turn output as the outcome reward. To mitigate sparse supervision and promote long-term attack strategies, we propose two heuristic process rewards: (1) controlling the harmfulness of intermediate outputs to prevent triggering the black-box model's rejection mechanisms, and (2) maintaining the semantic relevance of intermediate outputs to avoid drifting into irrelevant content. Experimental results on multiple benchmarks show consistently improved attack success rates across multiple models, highlighting the effectiveness of our approach. The code is available at https://github.com/xxiqiao/RL-MTJail. Warning: This paper contains examples of harmful content.

摘要: 大型语言模型容易受到越狱攻击，威胁其在现实世界应用程序中的安全部署。本文研究黑匣子多回合越狱，旨在训练攻击者LLM通过一系列预算-输出交互从黑匣子模型中获取有害内容。现有的方法通常依赖于单轮优化，这不足以学习长期攻击策略。为了弥合这一差距，我们将问题制定为多回合强化学习任务，直接优化最终回合输出的危害性作为结果奖励。为了减轻稀疏监督并促进长期攻击策略，我们提出了两个启发式过程奖励：（1）控制中间输出的危害性，以防止触发黑匣子模型的拒绝机制，（2）维护中间输出的语义相关性，以避免陷入不相关的内容。多个基准测试的实验结果显示，多个模型的攻击成功率持续提高，凸显了我们方法的有效性。该代码可在https://github.com/xxiqiao/RL-MTJail上获取。警告：本文包含有害内容的示例。



## **7. When Large Language Models Do Not Work: Online Incivility Prediction through Graph Neural Networks**

当大型语言模型不起作用时：通过图神经网络进行在线不文明预测 cs.CL

10 pages

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07684v1) [paper-pdf](https://arxiv.org/pdf/2512.07684v1)

**Authors**: Zihan Chen, Lanyu Yu

**Abstract**: Online incivility has emerged as a widespread and persistent problem in digital communities, imposing substantial social and psychological burdens on users. Although many platforms attempt to curb incivility through moderation and automated detection, the performance of existing approaches often remains limited in both accuracy and efficiency. To address this challenge, we propose a Graph Neural Network (GNN) framework for detecting three types of uncivil behavior (i.e., toxicity, aggression, and personal attacks) within the English Wikipedia community. Our model represents each user comment as a node, with textual similarity between comments defining the edges, allowing the network to jointly learn from both linguistic content and relational structures among comments. We also introduce a dynamically adjusted attention mechanism that adaptively balances nodal and topological features during information aggregation. Empirical evaluations demonstrate that our proposed architecture outperforms 12 state-of-the-art Large Language Models (LLMs) across multiple metrics while requiring significantly lower inference cost. These findings highlight the crucial role of structural context in detecting online incivility and address the limitations of text-only LLM paradigms in behavioral prediction. All datasets and comparative outputs will be publicly available in our repository to support further research and reproducibility.

摘要: 在线不文明已成为数字社区中一个普遍且持续存在的问题，给用户带来了沉重的社会和心理负担。尽管许多平台试图通过审核和自动检测来遏制不文明行为，但现有方法的性能在准确性和效率方面往往仍然有限。为了应对这一挑战，我们提出了一个图形神经网络（GNN）框架来检测三种类型的不文明行为（即毒性、攻击性和人身攻击）在英语维基百科社区中。我们的模型将每个用户评论表示为一个节点，评论之间的文本相似性定义了边缘，允许网络共同从评论之间的语言内容和关系结构中学习。我们还引入了一种动态调整的注意力机制，可以在信息聚合期间自适应地平衡节点和拓扑特征。经验评估表明，我们提出的架构在多个指标上优于12个最先进的大型语言模型（LLM），同时需要显着更低的推理成本。这些发现强调了结构性上下文在检测在线不文明方面的关键作用，并解决了纯文本LLM范式在行为预测中的局限性。所有数据集和比较输出都将在我们的存储库中公开，以支持进一步的研究和重现性。



## **8. Think-Reflect-Revise: A Policy-Guided Reflective Framework for Safety Alignment in Large Vision Language Models**

思考-反思-修订：大视野语言模型中安全一致的政策引导反思框架 cs.CV

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07141v1) [paper-pdf](https://arxiv.org/pdf/2512.07141v1)

**Authors**: Fenghua Weng, Chaochao Lu, Xia Hu, Wenqi Shao, Wenjie Wang

**Abstract**: As multimodal reasoning improves the overall capabilities of Large Vision Language Models (LVLMs), recent studies have begun to explore safety-oriented reasoning, aiming to enhance safety awareness by analyzing potential safety risks during the reasoning process before generating the final response. Although such approaches improve safety awareness and interpretability, this single-pass think-then-answer paradigm remains vulnerable to contextual or visual jailbreak attacks. This reveals a critical flaw: single-pass reasoning may overlook explicit harmful content in its own output. Our key insight is to exploit this wasted signal through reflection, which can effectively leverage the malicious content revealed in the first-pass reasoning to enable genuine self-correction and prevent unsafe generations. Motivated by this, we propose Think-Reflect-Revise (TRR), a three-stage training framework designed to enhance the safety alignment of LVLMs through policy-guided self-reflection. We first build a Reflective Safety Reasoning (ReSafe) dataset with 5,000 examples that follow a think-reflect-revise process. We then fine-tune the target model using the ReSafe dataset to initialize reflective behavior, and finally reinforce policy-guided reflection through reinforcement learning. Experimental results show that TRR substantially improves the safety performance of LVLMs across both safety-awareness benchmarks and jailbreak attack evaluations, increasing the overall safe response rate from 42.8% to 87.7% on Qwen2.5-VL-7B, while preserving stable performance on general benchmarks such as MMMU and MMStar. The project page is available at https://think-reflect-revise.github.io/.

摘要: 随着多模式推理提高了大视觉语言模型（LVLM）的整体能力，最近的研究开始探索以安全为导向的推理，旨在通过在生成最终响应之前分析推理过程中的潜在安全风险来增强安全意识。尽管此类方法提高了安全意识和可解释性，但这种单程思考然后回答的范式仍然容易受到上下文或视觉越狱攻击。这揭示了一个关键缺陷：单程推理可能会忽视其输出中明显的有害内容。我们的主要见解是通过反射利用这种浪费的信号，这可以有效地利用第一遍推理中揭示的恶意内容，以实现真正的自我纠正并防止不安全的世代。出于此动机，我们提出了思考-反思-修订（TRR），这是一个三阶段培训框架，旨在通过政策引导的自我反思来增强LVLM的安全一致性。我们首先构建一个反思安全推理（ReSafe）数据集，包含5，000个示例，遵循思考-反思-修改过程。然后，我们使用ReSafe数据集微调目标模型以初始化反射行为，并最终通过强化学习加强政策引导的反射。实验结果表明，TRR在安全意识基准和越狱攻击评估中大幅提高了LVLM的安全性能，将Qwen 2.5-BL-7 B的总体安全响应率从42.8%提高到87.7%，同时在MMMU和MMStar等通用基准上保持稳定的性能。该项目页面可访问https://think-reflect-revise.github.io/。



## **9. ThinkTrap: Denial-of-Service Attacks against Black-box LLM Services via Infinite Thinking**

ThinkTrap：通过无限思维对黑匣子LLM服务进行拒绝服务攻击 cs.CR

This version includes the final camera-ready manuscript accepted by NDSS 2026

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07086v1) [paper-pdf](https://arxiv.org/pdf/2512.07086v1)

**Authors**: Yunzhe Li, Jianan Wang, Hongzi Zhu, James Lin, Shan Chang, Minyi Guo

**Abstract**: Large Language Models (LLMs) have become foundational components in a wide range of applications, including natural language understanding and generation, embodied intelligence, and scientific discovery. As their computational requirements continue to grow, these models are increasingly deployed as cloud-based services, allowing users to access powerful LLMs via the Internet. However, this deployment model introduces a new class of threat: denial-of-service (DoS) attacks via unbounded reasoning, where adversaries craft specially designed inputs that cause the model to enter excessively long or infinite generation loops. These attacks can exhaust backend compute resources, degrading or denying service to legitimate users. To mitigate such risks, many LLM providers adopt a closed-source, black-box setting to obscure model internals. In this paper, we propose ThinkTrap, a novel input-space optimization framework for DoS attacks against LLM services even in black-box environments. The core idea of ThinkTrap is to first map discrete tokens into a continuous embedding space, then undertake efficient black-box optimization in a low-dimensional subspace exploiting input sparsity. The goal of this optimization is to identify adversarial prompts that induce extended or non-terminating generation across several state-of-the-art LLMs, achieving DoS with minimal token overhead. We evaluate the proposed attack across multiple commercial, closed-source LLM services. Our results demonstrate that, even far under the restrictive request frequency limits commonly enforced by these platforms, typically capped at ten requests per minute (10 RPM), the attack can degrade service throughput to as low as 1% of its original capacity, and in some cases, induce complete service failure.

摘要: 大型语言模型（LLM）已经成为广泛应用的基础组件，包括自然语言理解和生成，体现智能和科学发现。随着计算需求的不断增长，这些模型越来越多地部署为基于云的服务，允许用户通过互联网访问强大的LLM。然而，这种部署模型引入了一类新的威胁：通过无限推理的拒绝服务（DoS）攻击，其中攻击者精心设计了特别设计的输入，导致模型进入过长或无限的生成循环。这些攻击可能耗尽后端计算资源，降低或拒绝向合法用户提供服务。为了降低此类风险，许多LLM提供商采用闭源、黑匣子设置来掩盖模型内部内容。在本文中，我们提出了ThinkTrap，这是一种新型的输入空间优化框架，即使在黑匣子环境中也可以对LLM服务进行NOS攻击。ThinkTrap的核心思想是首先将离散令牌映射到连续嵌入空间，然后在利用输入稀疏性的低维子空间中进行高效的黑匣子优化。此优化的目标是识别在多个最先进的LLM之间引发扩展或非终止生成的对抗提示，以最小的令牌负载实现拒绝服务。我们评估了跨多个商业、闭源LLM服务的拟议攻击。我们的结果表明，即使远低于这些平台通常强制执行的限制性请求频率限制（通常限制为每分钟10个请求（10转/分钟）），攻击也可以将服务吞吐量降低至低至其原始容量的1%，在某些情况下，会导致完全的服务故障。



## **10. Replicating TEMPEST at Scale: Multi-Turn Adversarial Attacks Against Trillion-Parameter Frontier Models**

大规模复制TEMPST：针对万亿参数前沿模型的多轮对抗攻击 cs.CL

30 pages, 11 figures, 5 tables. Code and data: https://github.com/ricyoung/tempest-replication

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.07059v1) [paper-pdf](https://arxiv.org/pdf/2512.07059v1)

**Authors**: Richard Young

**Abstract**: Despite substantial investment in safety alignment, the vulnerability of large language models to sophisticated multi-turn adversarial attacks remains poorly characterized, and whether model scale or inference mode affects robustness is unknown. This study employed the TEMPEST multi-turn attack framework to evaluate ten frontier models from eight vendors across 1,000 harmful behaviors, generating over 97,000 API queries across adversarial conversations with automated evaluation by independent safety classifiers. Results demonstrated a spectrum of vulnerability: six models achieved 96% to 100% attack success rate (ASR), while four showed meaningful resistance, with ASR ranging from 42% to 78%; enabling extended reasoning on identical architecture reduced ASR from 97% to 42%. These findings indicate that safety alignment quality varies substantially across vendors, that model scale does not predict adversarial robustness, and that thinking mode provides a deployable safety enhancement. Collectively, this work establishes that current alignment techniques remain fundamentally vulnerable to adaptive multi-turn attacks regardless of model scale, while identifying deliberative inference as a promising defense direction.

摘要: 尽管在安全对齐方面投入了大量资金，但大型语言模型对复杂多轮对抗攻击的脆弱性仍然很难描述，并且模型规模或推理模式是否会影响稳健性尚不清楚。这项研究采用TEMPEST多回合攻击框架来评估来自8家供应商的10个前沿模型，涵盖1，000种有害行为，通过独立安全分类器的自动评估，在对抗性对话中生成超过97，000个API查询。结果展示了一系列脆弱性：六个模型实现了96%至100%的攻击成功率（ASB），四个模型表现出有意义的抵抗力，ASB范围从42%至78%;在相同的架构上启用扩展推理将ASC从97%降低到42%。这些发现表明，不同供应商的安全对齐质量存在很大差异，模型规模无法预测对抗稳健性，并且思维模式提供了可部署的安全增强。总的来说，这项工作确定了，无论模型规模如何，当前的对齐技术仍然从根本上容易受到自适应多转弯攻击的影响，同时将刻意推理确定为一个有希望的防御方向。



## **11. SoK: Trust-Authorization Mismatch in LLM Agent Interactions**

SoK：LLM代理交互中的信任授权不匹配 cs.CR

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2512.06914v1) [paper-pdf](https://arxiv.org/pdf/2512.06914v1)

**Authors**: Guanquan Shi, Haohua Du, Zhiqiang Wang, Xiaoyu Liang, Weiwenpei Liu, Song Bian, Zhenyu Guan

**Abstract**: Large Language Models (LLMs) are rapidly evolving into autonomous agents capable of interacting with the external world, significantly expanding their capabilities through standardized interaction protocols. However, this paradigm revives the classic cybersecurity challenges of agency and authorization in a novel and volatile context. As decision-making shifts from deterministic code logic to probabilistic inference driven by natural language, traditional security mechanisms designed for deterministic behavior fail. It is fundamentally challenging to establish trust for unpredictable AI agents and to enforce the Principle of Least Privilege (PoLP) when instructions are ambiguous. Despite the escalating threat landscape, the academic community's understanding of this emerging domain remains fragmented, lacking a systematic framework to analyze its root causes. This paper provides a unifying formal lens for agent-interaction security.   We observed that most security threats in this domain stem from a fundamental mismatch between trust evaluation and authorization policies. We introduce a novel risk analysis model centered on this trust-authorization gap. Using this model as a unifying lens, we survey and classify the implementation paths of existing, often seemingly isolated, attacks and defenses. This new framework not only unifies the field but also allows us to identify critical research gaps. Finally, we leverage our analysis to suggest a systematic research direction toward building robust, trusted agents and dynamic authorization mechanisms.

摘要: 大型语言模型（LLM）正在迅速演变为能够与外部世界交互的自治代理，通过标准化的交互协议显着扩展其能力。然而，这种范式在新颖且不稳定的背景下重新焕发了代理和授权的经典网络安全挑战。随着决策从确定性代码逻辑转向自然语言驱动的概率推理，为确定性行为设计的传统安全机制就会失败。为不可预测的人工智能代理建立信任并在指令模糊时执行最小特权原则（PoLP）是一个根本性的挑战。尽管威胁格局不断升级，但学术界对这一新兴领域的理解仍然支离破碎，缺乏系统性的框架来分析其根本原因。本文为代理交互安全性提供了统一的正式视角。   我们观察到，该领域中的大多数安全威胁源于信任评估和授权策略之间的根本不匹配。我们以这种信任-授权差距为中心引入了一种新颖的风险分析模型。使用这个模型作为统一的镜头，我们调查和分类了现有的（通常看似孤立的）攻击和防御的实施路径。这个新框架不仅统一了该领域，还使我们能够识别关键的研究差距。最后，我们利用我们的分析提出了一个系统性的研究方向，以构建稳健、可信的代理和动态授权机制。



## **12. From Description to Score: Can LLMs Quantify Vulnerabilities?**

从描述到评分：LLM可以量化漏洞吗？ cs.CR

10 pages

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2512.06781v1) [paper-pdf](https://arxiv.org/pdf/2512.06781v1)

**Authors**: Sima Jafarikhah, Daniel Thompson, Eva Deans, Hossein Siadati, Yi Liu

**Abstract**: Manual vulnerability scoring, such as assigning Common Vulnerability Scoring System (CVSS) scores, is a resource-intensive process that is often influenced by subjective interpretation. This study investigates the potential of general-purpose large language models (LLMs), namely ChatGPT, Llama, Grok, DeepSeek, and Gemini, to automate this process by analyzing over 31{,}000 recent Common Vulnerabilities and Exposures (CVE) entries. The results show that LLMs substantially outperform the baseline on certain metrics (e.g., \textit{Availability Impact}), while offering more modest gains on others (e.g., \textit{Attack Complexity}). Moreover, model performance varies across both LLM families and individual CVSS metrics, with ChatGPT-5 attaining the highest precision. Our analysis reveals that LLMs tend to misclassify many of the same CVEs, and ensemble-based meta-classifiers only marginally improve performance. Further examination shows that CVE descriptions often lack critical context or contain ambiguous phrasing, which contributes to systematic misclassifications. These findings underscore the importance of enhancing vulnerability descriptions and incorporating richer contextual details to support more reliable automated reasoning and alleviate the growing backlog of CVEs awaiting triage.

摘要: 手动漏洞评分，例如分配通用漏洞评分系统（CVD）分数，是一个资源密集型的过程，通常受到主观解释的影响。本研究调查了通用大型语言模型（LLM）（即ChatGPT、Llama、Grok、DeepSeek和Gemini）的潜力，通过分析超过31{，}000个最近的常见漏洞和暴露（UTE）条目来自动化该过程。结果表明，LLM在某些指标上的表现大大优于基线（例如，\textit{可用性影响}），同时为其他人提供更适度的收益（例如，\textit{Attack Complexity}）。此外，模型性能因LLM系列和单个CVD指标而异，其中ChatGPT-5达到了最高的精确度。我们的分析表明，LLM往往会对许多相同的CVE进行错误分类，而基于集成的元分类器只能略微提高性能。进一步的检查表明，UTE描述通常缺乏关键上下文或包含模棱两可的措辞，这导致了系统性的错误分类。这些发现强调了增强漏洞描述和整合更丰富的上下文细节的重要性，以支持更可靠的自动化推理并减轻等待分诊的CVS不断增加的积压。



## **13. Cognitive Control Architecture (CCA): A Lifecycle Supervision Framework for Robustly Aligned AI Agents**

认知控制架构（CAA）：一个针对稳健一致的人工智能代理的生命周期监督框架 cs.AI

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2512.06716v1) [paper-pdf](https://arxiv.org/pdf/2512.06716v1)

**Authors**: Zhibo Liang, Tianze Hu, Zaiye Chen, Mingjie Tang

**Abstract**: Autonomous Large Language Model (LLM) agents exhibit significant vulnerability to Indirect Prompt Injection (IPI) attacks. These attacks hijack agent behavior by polluting external information sources, exploiting fundamental trade-offs between security and functionality in existing defense mechanisms. This leads to malicious and unauthorized tool invocations, diverting agents from their original objectives. The success of complex IPIs reveals a deeper systemic fragility: while current defenses demonstrate some effectiveness, most defense architectures are inherently fragmented. Consequently, they fail to provide full integrity assurance across the entire task execution pipeline, forcing unacceptable multi-dimensional compromises among security, functionality, and efficiency. Our method is predicated on a core insight: no matter how subtle an IPI attack, its pursuit of a malicious objective will ultimately manifest as a detectable deviation in the action trajectory, distinct from the expected legitimate plan. Based on this, we propose the Cognitive Control Architecture (CCA), a holistic framework achieving full-lifecycle cognitive supervision. CCA constructs an efficient, dual-layered defense system through two synergistic pillars: (i) proactive and preemptive control-flow and data-flow integrity enforcement via a pre-generated "Intent Graph"; and (ii) an innovative "Tiered Adjudicator" that, upon deviation detection, initiates deep reasoning based on multi-dimensional scoring, specifically designed to counter complex conditional attacks. Experiments on the AgentDojo benchmark substantiate that CCA not only effectively withstands sophisticated attacks that challenge other advanced defense methods but also achieves uncompromised security with notable efficiency and robustness, thereby reconciling the aforementioned multi-dimensional trade-off.

摘要: 自治大型语言模型（LLM）代理对间接提示注入（IPI）攻击表现出明显的脆弱性。这些攻击通过污染外部信息源、利用现有防御机制中安全性和功能之间的基本权衡来劫持代理行为。这会导致恶意和未经授权的工具调用，转移代理人的原始目标。复杂知识产权的成功揭示了更深层次的系统脆弱性：虽然当前的防御表现出一定的有效性，但大多数防御架构本质上是碎片化的。因此，它们未能在整个任务执行管道中提供完全的完整性保证，从而迫使安全性、功能和效率之间出现不可接受的多维妥协。我们的方法基于一个核心见解：无论IPI攻击多么微妙，其对恶意目标的追求最终都会表现为行动轨迹中可检测到的偏差，与预期的合法计划不同。在此基础上，我们提出了认知控制架构（CAA），这是一个实现全生命周期认知监督的整体框架。CAA通过两个协同支柱构建了一个高效的双层防御系统：（i）通过预生成的“意图图”主动和先发制人地实施控制流和数据流完整性;（ii）创新的“分层裁决器”，在检测偏差后，它会启动基于多维评分的深度推理，专门设计用于对抗复杂的条件攻击。在AgentDojo基准测试上的实验证实，CCA不仅有效地抵御了挑战其他先进防御方法的复杂攻击，而且还以显著的效率和鲁棒性实现了不折不扣的安全性，从而调和了上述多维权衡。



## **14. GSAE: Graph-Regularized Sparse Autoencoders for Robust LLM Safety Steering**

GAE：图形正规化稀疏自动编码器，用于稳健的LLM安全转向 cs.LG

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2512.06655v1) [paper-pdf](https://arxiv.org/pdf/2512.06655v1)

**Authors**: Jehyeok Yeon, Federico Cinus, Yifan Wu, Luca Luceri

**Abstract**: Large language models (LLMs) face critical safety challenges, as they can be manipulated to generate harmful content through adversarial prompts and jailbreak attacks. Many defenses are typically either black-box guardrails that filter outputs, or internals-based methods that steer hidden activations by operationalizing safety as a single latent feature or dimension. While effective for simple concepts, this assumption is limiting, as recent evidence shows that abstract concepts such as refusal and temporality are distributed across multiple features rather than isolated in one. To address this limitation, we introduce Graph-Regularized Sparse Autoencoders (GSAEs), which extends SAEs with a Laplacian smoothness penalty on the neuron co-activation graph. Unlike standard SAEs that assign each concept to a single latent feature, GSAEs recover smooth, distributed safety representations as coherent patterns spanning multiple features. We empirically demonstrate that GSAE enables effective runtime safety steering, assembling features into a weighted set of safety-relevant directions and controlling them with a two-stage gating mechanism that activates interventions only when harmful prompts or continuations are detected during generation. This approach enforces refusals adaptively while preserving utility on benign queries. Across safety and QA benchmarks, GSAE steering achieves an average 82% selective refusal rate, substantially outperforming standard SAE steering (42%), while maintaining strong task accuracy (70% on TriviaQA, 65% on TruthfulQA, 74% on GSM8K). Robustness experiments further show generalization across LLaMA-3, Mistral, Qwen, and Phi families and resilience against jailbreak attacks (GCG, AutoDAN), consistently maintaining >= 90% refusal of harmful content.

摘要: 大型语言模型（LLM）面临着严峻的安全挑战，因为它们可能会被操纵以通过对抗性提示和越狱攻击生成有害内容。许多防御通常是过滤输出的黑匣子护栏，或者是基于内部的方法，通过将安全性作为单个潜在特征或维度来操作来引导隐藏激活。虽然对简单概念有效，但这种假设具有局限性，因为最近的证据表明，拒绝和时间性等抽象概念分布在多个特征中，而不是孤立在一个特征中。为了解决这一局限性，我们引入了图正规化稀疏自动编码器（GSAEs），它在神经元共激活图上使用拉普拉斯光滑罚分来扩展SAEs。与将每个概念分配给一个潜在特征的标准严重不良事件不同，G严重不良事件恢复为跨越多个特征的连贯模式的平滑、分布式的安全性表示。我们经验证明，GAE能够实现有效的运行时安全转向，将功能组装到一组加权的安全相关方向中，并通过两级门控机制控制它们，该机制仅在生成期间检测到有害提示或延续时激活干预。这种方法自适应地强制拒绝，同时保留对良性查询的实用性。在安全和QA基准中，GSE转向平均实现了82%的选择拒绝率，大大优于标准的SAS转向（42%），同时保持了强大的任务准确性（TriviaQA为70%，TruthfulQA为65%，GSM 8 K为74%）。鲁棒性实验进一步显示了LLaMA-3、Mistral、Qwen和Phi家族的普遍性以及对越狱攻击的韧性（GCG、AutoDAN），始终保持>= 90%拒绝有害内容。



## **15. OmniSafeBench-MM: A Unified Benchmark and Toolbox for Multimodal Jailbreak Attack-Defense Evaluation**

OmniSafeBench-MM：多模式越狱攻击防御评估的统一基准和工作组 cs.CR

**SubmitDate**: 2025-12-06    [abs](http://arxiv.org/abs/2512.06589v1) [paper-pdf](https://arxiv.org/pdf/2512.06589v1)

**Authors**: Xiaojun Jia, Jie Liao, Qi Guo, Teng Ma, Simeng Qin, Ranjie Duan, Tianlin Li, Yihao Huang, Zhitao Zeng, Dongxian Wu, Yiming Li, Wenqi Ren, Xiaochun Cao, Yang Liu

**Abstract**: Recent advances in multi-modal large language models (MLLMs) have enabled unified perception-reasoning capabilities, yet these systems remain highly vulnerable to jailbreak attacks that bypass safety alignment and induce harmful behaviors. Existing benchmarks such as JailBreakV-28K, MM-SafetyBench, and HADES provide valuable insights into multi-modal vulnerabilities, but they typically focus on limited attack scenarios, lack standardized defense evaluation, and offer no unified, reproducible toolbox. To address these gaps, we introduce OmniSafeBench-MM, which is a comprehensive toolbox for multi-modal jailbreak attack-defense evaluation. OmniSafeBench-MM integrates 13 representative attack methods, 15 defense strategies, and a diverse dataset spanning 9 major risk domains and 50 fine-grained categories, structured across consultative, imperative, and declarative inquiry types to reflect realistic user intentions. Beyond data coverage, it establishes a three-dimensional evaluation protocol measuring (1) harmfulness, distinguished by a granular, multi-level scale ranging from low-impact individual harm to catastrophic societal threats, (2) intent alignment between responses and queries, and (3) response detail level, enabling nuanced safety-utility analysis. We conduct extensive experiments on 10 open-source and 8 closed-source MLLMs to reveal their vulnerability to multi-modal jailbreak. By unifying data, methodology, and evaluation into an open-source, reproducible platform, OmniSafeBench-MM provides a standardized foundation for future research. The code is released at https://github.com/jiaxiaojunQAQ/OmniSafeBench-MM.

摘要: 多模式大型语言模型（MLLM）的最新进展已经实现了统一的感知推理能力，但这些系统仍然极易受到绕过安全对齐并引发有害行为的越狱攻击。JailBreakV-28 K、MM-SafetyBench和HADES等现有基准测试为多模式漏洞提供了宝贵的见解，但它们通常专注于有限的攻击场景，缺乏标准化的防御评估，并且没有提供统一的、可重复的工具箱。为了解决这些差距，我们引入了OmniSafeBench-MM，这是一个用于多模式越狱攻击防御评估的综合工具箱。OmniSafeBench-MM集成了13种代表性攻击方法、15种防御策略以及涵盖9个主要风险领域和50个细粒度类别的多元化数据集，结构跨越咨询性、命令性和声明性询问类型，以反映现实的用户意图。除了数据覆盖之外，它还建立了一个三维评估协议，测量（1）危害性，通过从低影响的个人伤害到灾难性社会威胁的细致、多级别的量表区分，（2）响应和查询之间的意图一致性，（3）响应细节级别，能够进行细致入微的安全-效用分析。我们对10个开源和8个开源MLLM进行了广泛的实验，以揭示它们对多模式越狱的脆弱性。通过将数据、方法和评估统一到开源、可重复的平台中，OmniSafeBench-MM为未来的研究提供了标准化的基础。该代码发布于https://github.com/jiaxiaojunQAQ/OmniSafeBench-MM。



## **16. Securing the Model Context Protocol: Defending LLMs Against Tool Poisoning and Adversarial Attacks**

保护模型上下文协议：保护LLM免受工具中毒和对抗性攻击 cs.CR

**SubmitDate**: 2025-12-06    [abs](http://arxiv.org/abs/2512.06556v1) [paper-pdf](https://arxiv.org/pdf/2512.06556v1)

**Authors**: Saeid Jamshidi, Kawser Wazed Nafi, Arghavan Moradi Dakhel, Negar Shahabi, Foutse Khomh, Naser Ezzati-Jivan

**Abstract**: The Model Context Protocol (MCP) enables Large Language Models to integrate external tools through structured descriptors, increasing autonomy in decision-making, task execution, and multi-agent workflows. However, this autonomy creates a largely overlooked security gap. Existing defenses focus on prompt-injection attacks and fail to address threats embedded in tool metadata, leaving MCP-based systems exposed to semantic manipulation. This work analyzes three classes of semantic attacks on MCP-integrated systems: (1) Tool Poisoning, where adversarial instructions are hidden in tool descriptors; (2) Shadowing, where trusted tools are indirectly compromised through contaminated shared context; and (3) Rug Pulls, where descriptors are altered after approval to subvert behavior. To counter these threats, we introduce a layered security framework with three components: RSA-based manifest signing to enforce descriptor integrity, LLM-on-LLM semantic vetting to detect suspicious tool definitions, and lightweight heuristic guardrails that block anomalous tool behavior at runtime. Through evaluation of GPT-4, DeepSeek, and Llama-3.5 across eight prompting strategies, we find that security performance varies widely by model architecture and reasoning method. GPT-4 blocks about 71 percent of unsafe tool calls, balancing latency and safety. DeepSeek shows the highest resilience to Shadowing attacks but with greater latency, while Llama-3.5 is fastest but least robust. Our results show that the proposed framework reduces unsafe tool invocation rates without model fine-tuning or internal modification.

摘要: 模型上下文协议（HCP）使大型语言模型能够通过结构化描述符集成外部工具，增加决策、任务执行和多代理工作流程的自主性。然而，这种自主性造成了一个很大程度上被忽视的安全漏洞。现有的防御措施专注于预算注入攻击，无法解决工具元数据中嵌入的威胁，从而使基于MVP的系统面临语义操纵。这项工作分析了对HCP集成系统的三类语义攻击：（1）工具中毒，其中对抗性指令隐藏在工具描述符中;（2）影子，其中受信任的工具通过受污染的共享上下文间接受到损害;（3）Rug Pull，其中描述符在批准后被更改以颠覆行为。为了应对这些威胁，我们引入了一个由三个组件组成的分层安全框架：基于RSA的清单签名以强制描述符完整性、LLM对LLM语义审查以检测可疑工具定义，以及在运行时阻止异常工具行为的轻量级启发式护栏。通过对八种提示策略的GPT-4、DeepSeek和Llama-3.5进行评估，我们发现安全性能因模型架构和推理方法而存在很大差异。GPT-4阻止约71%的不安全工具调用，平衡延迟和安全性。DeepSeek对Shadowing攻击表现出最高的弹性，但延迟更大，而Llama-3.5速度最快，但最不稳健。我们的结果表明，所提出的框架无需模型微调或内部修改即可降低不安全工具调用率。



## **17. VRSA: Jailbreaking Multimodal Large Language Models through Visual Reasoning Sequential Attack**

VRSA：通过视觉推理序列攻击破解多模式大型语言模型 cs.CV

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.05853v2) [paper-pdf](https://arxiv.org/pdf/2512.05853v2)

**Authors**: Shiji Zhao, Shukun Xiong, Yao Huang, Yan Jin, Zhenyu Wu, Jiyang Guan, Ranjie Duan, Jialing Tao, Hui Xue, Xingxing Wei

**Abstract**: Multimodal Large Language Models (MLLMs) are widely used in various fields due to their powerful cross-modal comprehension and generation capabilities. However, more modalities bring more vulnerabilities to being utilized for jailbreak attacks, which induces MLLMs to output harmful content. Due to the strong reasoning ability of MLLMs, previous jailbreak attacks try to explore reasoning safety risk in text modal, while similar threats have been largely overlooked in the visual modal. To fully evaluate potential safety risks in the visual reasoning task, we propose Visual Reasoning Sequential Attack (VRSA), which induces MLLMs to gradually externalize and aggregate complete harmful intent by decomposing the original harmful text into several sequentially related sub-images. In particular, to enhance the rationality of the scene in the image sequence, we propose Adaptive Scene Refinement to optimize the scene most relevant to the original harmful query. To ensure the semantic continuity of the generated image, we propose Semantic Coherent Completion to iteratively rewrite each sub-text combined with contextual information in this scene. In addition, we propose Text-Image Consistency Alignment to keep the semantical consistency. A series of experiments demonstrates that the VRSA can achieve a higher attack success rate compared with the state-of-the-art jailbreak attack methods on both the open-source and closed-source MLLMs such as GPT-4o and Claude-4.5-Sonnet.

摘要: 多模式大型语言模型（MLLM）因其强大的跨模式理解和生成能力而被广泛应用于各个领域。然而，更多的模式会带来更多被用于越狱攻击的漏洞，从而导致MLLM输出有害内容。由于MLLM推理能力强，之前的越狱攻击试图在文本模式中探索推理安全风险，而类似的威胁在视觉模式中基本上被忽视。为了充分评估视觉推理任务中潜在的安全风险，我们提出了视觉推理序列攻击（VRSA），通过将原始有害文本分解为几个顺序相关的子图像，诱导MLLM逐渐外部化和聚合完整的有害意图。特别是，为了增强图像序列中场景的合理性，我们提出自适应场景细化来优化与原始有害查询最相关的场景。为了确保生成图像的语义连续性，我们提出了语义连贯完成来迭代重写该场景中结合上下文信息的每个子文本。此外，我们还提出了文本-图像一致性对齐来保持语义一致性。一系列实验表明，与最先进的越狱攻击方法相比，VRSA可以在GPT-4 o和Claude-4.5-Sonnet等开源和闭源MLLM上实现更高的攻击成功率。



## **18. ARGUS: Defending Against Multimodal Indirect Prompt Injection via Steering Instruction-Following Behavior**

ARGucci：通过转向指令遵循行为防御多模式间接提示注射 cs.CR

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.05745v1) [paper-pdf](https://arxiv.org/pdf/2512.05745v1)

**Authors**: Weikai Lu, Ziqian Zeng, Kehua Zhang, Haoran Li, Huiping Zhuang, Ruidong Wang, Cen Chen, Hao Peng

**Abstract**: Multimodal Large Language Models (MLLMs) are increasingly vulnerable to multimodal Indirect Prompt Injection (IPI) attacks, which embed malicious instructions in images, videos, or audio to hijack model behavior. Existing defenses, designed primarily for text-only LLMs, are unsuitable for countering these multimodal threats, as they are easily bypassed, modality-dependent, or generalize poorly. Inspired by activation steering researches, we hypothesize that a robust, general defense independent of modality can be achieved by steering the model's behavior in the representation space. Through extensive experiments, we discover that the instruction-following behavior of MLLMs is encoded in a subspace. Steering along directions within this subspace can enforce adherence to user instructions, forming the basis of a defense. However, we also found that a naive defense direction could be coupled with a utility-degrading direction, and excessive intervention strength harms model performance. To address this, we propose ARGUS, which searches for an optimal defense direction within the safety subspace that decouples from the utility degradation direction, further combining adaptive strength steering to achieve a better safety-utility trade-off. ARGUS also introduces lightweight injection detection stage to activate the defense on-demand, and a post-filtering stage to verify defense success. Experimental results show that ARGUS can achieve robust defense against multimodal IPI while maximally preserving the MLLM's utility.

摘要: 多模式大型语言模型（MLLM）越来越容易受到多模式间接提示注入（IPI）攻击的影响，这些攻击将恶意指令嵌入图像、视频或音频中以劫持模型行为。现有的防御主要为纯文本的LLM设计，不适合对抗这些多模式威胁，因为它们很容易被绕过、依赖于模式或概括性较差。受激活引导研究的启发，我们假设可以通过引导模型在表示空间中的行为来实现独立于形态的稳健、通用防御。通过大量实验，我们发现MLLM的描述跟随行为被编码在子空间中。沿着该子空间内的方向转向可以强制遵守用户指令，从而形成防御的基础。然而，我们还发现，天真的防御方向可能会与效用下降的方向相结合，过度的干预强度会损害模型性能。为了解决这个问题，我们提出了ARGucci，它在安全子空间内搜索与效用降级方向并行的最佳防御方向，进一步结合自适应强度引导以实现更好的安全-效用权衡。ARGucci还引入了轻量级注入检测阶段以按需激活防御，以及后过滤阶段以验证防御成功。实验结果表明，ARGUS可以实现强大的防御多模态IPI，同时最大限度地保持MLLM的效用。



## **19. Matching Ranks Over Probability Yields Truly Deep Safety Alignment**

超越概率的匹配排名产生真正深入的安全一致 cs.CR

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.05518v1) [paper-pdf](https://arxiv.org/pdf/2512.05518v1)

**Authors**: Jason Vega, Gagandeep Singh

**Abstract**: A frustratingly easy technique known as the prefilling attack has been shown to effectively circumvent the safety alignment of frontier LLMs by simply prefilling the assistant response with an affirmative prefix before decoding. In response, recent work proposed a supervised fine-tuning (SFT) defense using data augmentation to achieve a \enquote{deep} safety alignment, allowing the model to generate natural language refusals immediately following harmful prefills. Unfortunately, we show in this work that the "deep" safety alignment produced by such an approach is in fact not very deep. A generalization of the prefilling attack, which we refer to as the Rank-Assisted Prefilling (RAP) attack, can effectively extract harmful content from models fine-tuned with the data augmentation defense by selecting low-probability "harmful" tokens from the top 20 predicted next tokens at each step (thus ignoring high-probability "refusal" tokens). We argue that this vulnerability is enabled due to the "gaming" of the SFT objective when the target distribution entropies are low, where low fine-tuning loss is achieved by shifting large probability mass to a small number of refusal tokens while neglecting the high ranks of harmful tokens. We then propose a new perspective on achieving deep safety alignment by matching the token ranks of the target distribution, rather than their probabilities. This perspective yields a surprisingly simple fix to the data augmentation defense based on regularizing the attention placed on harmful prefill tokens, an approach we call PRefill attEntion STOpping (PRESTO). Adding PRESTO yields up to a 4.7x improvement in the mean StrongREJECT score under RAP attacks across three popular open-source LLMs, with low impact to model utility.

摘要: 一种名为预填充攻击的简单技术已被证明可以通过在解码之前简单地预填充辅助响应来有效规避前沿LLM的安全对齐。作为回应，最近的工作提出了一种监督式微调（SFT）防御，使用数据增强来实现安全对齐，允许模型在有害预填充后立即生成自然语言拒绝。不幸的是，我们在这项工作中表明，这种方法产生的“深度”安全调整实际上并不很深。预填充攻击的概括（我们称之为排名辅助预填充（RAP）攻击）可以通过从每一步预测的前20个下一个令牌中选择低概率的“有害”令牌，从用数据增强防御微调的模型中有效地提取有害内容（从而忽略高概率的“拒绝”令牌）。我们认为，这种漏洞是启用由于“游戏”的SFT目标时，目标分布熵低，其中低微调损失是通过转移大概率质量的拒绝令牌的数量少，而忽略了高排名的有害令牌。然后，我们提出了一个新的视角，通过匹配目标分布的令牌等级，而不是它们的概率，来实现深度安全对齐。这种观点产生了一个令人惊讶的简单的修复数据增强防御的基础上正规化的注意力放在有害的预填充令牌，一种方法，我们称之为预填充attEntion停止（PRESTO）。在三种流行的开源LLM中，在RAP攻击下，添加PRESTO可以使平均StrongRESISTANCE得分提高4.7倍，对模型实用性的影响很小。



## **20. TeleAI-Safety: A comprehensive LLM jailbreaking benchmark towards attacks, defenses, and evaluations**

TeleAI-Safety：针对攻击、防御和评估的全面LLM越狱基准 cs.CR

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2512.05485v2) [paper-pdf](https://arxiv.org/pdf/2512.05485v2)

**Authors**: Xiuyuan Chen, Jian Zhao, Yuxiang He, Yuan Xun, Xinwei Liu, Yanshu Li, Huilin Zhou, Wei Cai, Ziyan Shi, Yuchen Yuan, Tianle Zhang, Chi Zhang, Xuelong Li

**Abstract**: While the deployment of large language models (LLMs) in high-value industries continues to expand, the systematic assessment of their safety against jailbreak and prompt-based attacks remains insufficient. Existing safety evaluation benchmarks and frameworks are often limited by an imbalanced integration of core components (attack, defense, and evaluation methods) and an isolation between flexible evaluation frameworks and standardized benchmarking capabilities. These limitations hinder reliable cross-study comparisons and create unnecessary overhead for comprehensive risk assessment. To address these gaps, we present TeleAI-Safety, a modular and reproducible framework coupled with a systematic benchmark for rigorous LLM safety evaluation. Our framework integrates a broad collection of 19 attack methods (including one self-developed method), 29 defense methods, and 19 evaluation methods (including one self-developed method). With a curated attack corpus of 342 samples spanning 12 distinct risk categories, the TeleAI-Safety benchmark conducts extensive evaluations across 14 target models. The results reveal systematic vulnerabilities and model-specific failure cases, highlighting critical trade-offs between safety and utility, and identifying potential defense patterns for future optimization. In practical scenarios, TeleAI-Safety can be flexibly adjusted with customized attack, defense, and evaluation combinations to meet specific demands. We release our complete code and evaluation results to facilitate reproducible research and establish unified safety baselines.

摘要: 尽管大型语言模型（LLM）在高价值行业的部署持续扩大，但对其针对越狱和预算攻击的安全性的系统评估仍然不足。现有的安全评估基准和框架通常受到核心组件（攻击、防御和评估方法）的不平衡集成以及灵活评估框架和标准化基准能力之间的隔离的限制。这些限制阻碍了可靠的交叉研究比较，并为全面风险评估带来了不必要的费用。为了解决这些差距，我们提出了TeleAI-Safety，这是一个模块化、可重复的框架，结合了严格LLM安全评估的系统基准。我们的框架集成了19种攻击方法（包括一种自主开发的方法），29种防御方法和19种评估方法（包括一种自主开发的方法）。TeleAI-Safety基准测试使用了涵盖12个不同风险类别的342个样本的策划攻击语料库，对14个目标模型进行了广泛的评估。结果揭示了系统漏洞和特定于模型的故障案例，突出了安全性和实用性之间的关键权衡，并确定了未来优化的潜在防御模式。在实际场景中，TeleAI-Safety可以灵活调整自定义攻击、防御和评估组合，以满足特定需求。我们发布完整的代码和评估结果，以促进可重复的研究并建立统一的安全基线。



## **21. Exposing Pink Slime Journalism: Linguistic Signatures and Robust Detection Against LLM-Generated Threats**

揭露Pink Slime新闻：语言签名和针对LLM生成的威胁的稳健检测 cs.CL

Published in RANLP 2025

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2512.05331v1) [paper-pdf](https://arxiv.org/pdf/2512.05331v1)

**Authors**: Sadat Shahriar, Navid Ayoobi, Arjun Mukherjee, Mostafa Musharrat, Sai Vishnu Vamsi

**Abstract**: The local news landscape, a vital source of reliable information for 28 million Americans, faces a growing threat from Pink Slime Journalism, a low-quality, auto-generated articles that mimic legitimate local reporting. Detecting these deceptive articles requires a fine-grained analysis of their linguistic, stylistic, and lexical characteristics. In this work, we conduct a comprehensive study to uncover the distinguishing patterns of Pink Slime content and propose detection strategies based on these insights. Beyond traditional generation methods, we highlight a new adversarial vector: modifications through large language models (LLMs). Our findings reveal that even consumer-accessible LLMs can significantly undermine existing detection systems, reducing their performance by up to 40% in F1-score. To counter this threat, we introduce a robust learning framework specifically designed to resist LLM-based adversarial attacks and adapt to the evolving landscape of automated pink slime journalism, and showed and improvement by up to 27%.

摘要: 当地新闻格局是2800万美国人可靠信息的重要来源，但面临着来自Pink Slime Journalism的日益增长的威胁，Pink Slime Journalism是一种模仿合法当地报道的低质量自动生成文章。检测这些欺骗性文章需要对其语言、文体和词汇特征进行细粒度分析。在这项工作中，我们进行了一项全面的研究，以揭示Pink Slime内容的区别模式，并根据这些见解提出检测策略。除了传统的生成方法之外，我们还强调了一种新的对抗性载体：通过大型语言模型（LLM）进行修改。我们的研究结果表明，即使是消费者可访问的LLM也会显着破坏现有的检测系统，使其F1评分的性能降低高达40%。为了应对这一威胁，我们引入了一个强大的学习框架，专门设计用于抵抗基于LLM的对抗攻击并适应自动化粉红粘液新闻不断变化的格局，并表现出高达27%的改进。



## **22. Chameleon: Adaptive Adversarial Agents for Scaling-Based Visual Prompt Injection in Multimodal AI Systems**

Chameleon：用于多模式人工智能系统中基于扩展的视觉提示注入的自适应对抗代理 cs.AI

5 pages, 2 figures, IEEE Transactions on Dependable and Secure Computing

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2512.04895v1) [paper-pdf](https://arxiv.org/pdf/2512.04895v1)

**Authors**: M Zeeshan, Saud Satti

**Abstract**: Multimodal Artificial Intelligence (AI) systems, particularly Vision-Language Models (VLMs), have become integral to critical applications ranging from autonomous decision-making to automated document processing. As these systems scale, they rely heavily on preprocessing pipelines to handle diverse inputs efficiently. However, this dependency on standard preprocessing operations, specifically image downscaling, creates a significant yet often overlooked security vulnerability. While intended for computational optimization, scaling algorithms can be exploited to conceal malicious visual prompts that are invisible to human observers but become active semantic instructions once processed by the model. Current adversarial strategies remain largely static, failing to account for the dynamic nature of modern agentic workflows. To address this gap, we propose Chameleon, a novel, adaptive adversarial framework designed to expose and exploit scaling vulnerabilities in production VLMs. Unlike traditional static attacks, Chameleon employs an iterative, agent-based optimization mechanism that dynamically refines image perturbations based on the target model's real-time feedback. This allows the framework to craft highly robust adversarial examples that survive standard downscaling operations to hijack downstream execution. We evaluate Chameleon against Gemini 2.5 Flash model. Our experiments demonstrate that Chameleon achieves an Attack Success Rate (ASR) of 84.5% across varying scaling factors, significantly outperforming static baseline attacks which average only 32.1%. Furthermore, we show that these attacks effectively compromise agentic pipelines, reducing decision-making accuracy by over 45% in multi-step tasks. Finally, we discuss the implications of these vulnerabilities and propose multi-scale consistency checks as a necessary defense mechanism.

摘要: 多模式人工智能（AI）系统，特别是视觉语言模型（VLM），已成为从自主决策到自动文档处理等关键应用不可或缺的一部分。随着这些系统的扩展，它们严重依赖预处理管道来有效处理不同的输入。然而，这种对标准预处理操作（特别是图像缩减）的依赖会产生一个严重但经常被忽视的安全漏洞。虽然缩放算法旨在实现计算优化，但可以利用缩放算法来隐藏恶意视觉提示，这些提示对人类观察者来说是不可见的，但一旦被模型处理就变成了活动的语义指令。当前的对抗策略在很大程度上仍然是静态的，未能考虑到现代代理工作流程的动态性质。为了解决这一差距，我们提出了Chameleon，这是一种新颖的、自适应的对抗框架，旨在暴露和利用生产VLM中的扩展漏洞。与传统的静态攻击不同，Chameleon采用基于代理的迭代优化机制，该机制根据目标模型的实时反馈动态细化图像扰动。这使得该框架能够制作高度稳健的对抗性示例，这些示例能够经受住标准缩减操作以劫持下游执行的考验。我们根据Gemini 2.5 Flash模型评估Chameleon。我们的实验表明，Chameleon在不同的缩放因子下实现了84.5%的攻击成功率（ASB），显着优于平均仅为32.1%的静态基线攻击。此外，我们表明这些攻击有效地损害了代理管道，使多步骤任务中的决策准确性降低了45%以上。最后，我们讨论了这些漏洞的影响，并提出多规模一致性检查作为必要的防御机制。



## **23. SoK: a Comprehensive Causality Analysis Framework for Large Language Model Security**

SoK：大型语言模型安全性的全面因果分析框架 cs.CR

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2512.04841v1) [paper-pdf](https://arxiv.org/pdf/2512.04841v1)

**Authors**: Wei Zhao, Zhe Li, Jun Sun

**Abstract**: Large Language Models (LLMs) exhibit remarkable capabilities but remain vulnerable to adversarial manipulations such as jailbreaking, where crafted prompts bypass safety mechanisms. Understanding the causal factors behind such vulnerabilities is essential for building reliable defenses.   In this work, we introduce a unified causality analysis framework that systematically supports all levels of causal investigation in LLMs, ranging from token-level, neuron-level, and layer-level interventions to representation-level analysis. The framework enables consistent experimentation and comparison across diverse causality-based attack and defense methods. Accompanying this implementation, we provide the first comprehensive survey of causality-driven jailbreak studies and empirically evaluate the framework on multiple open-weight models and safety-critical benchmarks including jailbreaks, hallucination detection, backdoor identification, and fairness evaluation. Our results reveal that: (1) targeted interventions on causally critical components can reliably modify safety behavior; (2) safety-related mechanisms are highly localized (i.e., concentrated in early-to-middle layers with only 1--2\% of neurons exhibiting causal influence); and (3) causal features extracted from our framework achieve over 95\% detection accuracy across multiple threat types.   By bridging theoretical causality analysis and practical model safety, our framework establishes a reproducible foundation for research on causality-based attacks, interpretability, and robust attack detection and mitigation in LLMs. Code is available at https://github.com/Amadeuszhao/SOK_Casuality.

摘要: 大型语言模型（LLM）表现出非凡的能力，但仍然容易受到越狱等敌对操纵的影响，其中精心设计的提示绕过了安全机制。了解此类漏洞背后的因果因素对于构建可靠的防御至关重要。   在这项工作中，我们引入了一个统一的因果关系分析框架，该框架系统地支持LLM中所有层面的因果关系调查，从代币层面、神经元层面和层层面干预到代表层面分析。该框架能够在各种基于偶然性的攻击和防御方法之间进行一致的实验和比较。伴随着这一实施，我们对疏忽驱动的越狱研究进行了首次全面调查，并对多个开放权重模型和安全关键基准（包括越狱、幻觉检测、后门识别和公平性评估）的框架进行了实证评估。我们的结果表明：（1）对因果关键部件进行有针对性的干预可以可靠地改变安全行为;（2）安全相关机制高度局部化（即，集中在早期到中层，只有1- 2%的神经元表现出因果影响）;以及（3）从我们的框架中提取的因果特征在多种威胁类型中实现了超过95%的检测准确率。   通过连接理论因果关系分析和实际模型安全性，我们的框架为LLM中基于因果关系的攻击、可解释性以及稳健的攻击检测和缓解的研究奠定了可重复的基础。代码可在https://github.com/Amadeuszhao/SOK_Casuality上获取。



## **24. ASTRIDE: A Security Threat Modeling Platform for Agentic-AI Applications**

ASTRIDE：用于统计人工智能应用的安全威胁建模平台 cs.AI

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2512.04785v1) [paper-pdf](https://arxiv.org/pdf/2512.04785v1)

**Authors**: Eranga Bandara, Amin Hass, Ross Gore, Sachin Shetty, Ravi Mukkamala, Safdar H. Bouk, Xueping Liang, Ng Wee Keong, Kasun De Zoysa, Aruna Withanage, Nilaan Loganathan

**Abstract**: AI agent-based systems are becoming increasingly integral to modern software architectures, enabling autonomous decision-making, dynamic task execution, and multimodal interactions through large language models (LLMs). However, these systems introduce novel and evolving security challenges, including prompt injection attacks, context poisoning, model manipulation, and opaque agent-to-agent communication, that are not effectively captured by traditional threat modeling frameworks. In this paper, we introduce ASTRIDE, an automated threat modeling platform purpose-built for AI agent-based systems. ASTRIDE extends the classical STRIDE framework by introducing a new threat category, A for AI Agent-Specific Attacks, which encompasses emerging vulnerabilities such as prompt injection, unsafe tool invocation, and reasoning subversion, unique to agent-based applications. To automate threat modeling, ASTRIDE combines a consortium of fine-tuned vision-language models (VLMs) with the OpenAI-gpt-oss reasoning LLM to perform end-to-end analysis directly from visual agent architecture diagrams, such as data flow diagrams(DFDs). LLM agents orchestrate the end-to-end threat modeling automation process by coordinating interactions between the VLM consortium and the reasoning LLM. Our evaluations demonstrate that ASTRIDE provides accurate, scalable, and explainable threat modeling for next-generation intelligent systems. To the best of our knowledge, ASTRIDE is the first framework to both extend STRIDE with AI-specific threats and integrate fine-tuned VLMs with a reasoning LLM to fully automate diagram-driven threat modeling in AI agent-based applications.

摘要: 基于人工智能代理的系统越来越成为现代软件架构的组成部分，通过大型语言模型（LLM）实现自主决策、动态任务执行和多模式交互。然而，这些系统引入了新颖且不断发展的安全挑战，包括即时注入攻击、上下文中毒、模型操纵和不透明的代理到代理通信，传统威胁建模框架无法有效捕捉这些挑战。在本文中，我们介绍ASTRIDE，这是一个专门为基于人工智能代理的系统构建的自动化威胁建模平台。ASTRIDE通过引入新的威胁类别A（代表人工智能代理特定攻击）扩展了经典的WRIDE框架，其中包括基于代理的应用程序所独有的新漏洞，例如提示注入、不安全工具调用和推理颠覆。为了自动化威胁建模，ASTRIDE将微调视觉语言模型（VLM）联盟与OpenAI-gpt-oss推理LLM相结合，直接从视觉代理架构图（例如数据流图（DFD））执行端到端分析。LLM代理通过协调VLM联盟和推理LLM之间的交互来协调端到端威胁建模自动化流程。我们的评估表明，ASTRIDE为下一代智能系统提供了准确、可扩展且可解释的威胁建模。据我们所知，ASTRIDE是第一个既可以通过人工智能特定的威胁扩展WRIDE，又可以将微调的VLM与推理LLM集成，以在基于人工智能代理的应用程序中完全自动化任务驱动的威胁建模。



## **25. Automatic Attack Discovery for Few-Shot Class-Incremental Learning via Large Language Models**

通过大型语言模型实现少镜头类增量学习的自动攻击发现 cs.LG

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03882v1) [paper-pdf](https://arxiv.org/pdf/2512.03882v1)

**Authors**: Haidong Kang, Wei Wu, Hanling Wang

**Abstract**: Few-shot class incremental learning (FSCIL) is a more realistic and challenging paradigm in continual learning to incrementally learn unseen classes and overcome catastrophic forgetting on base classes with only a few training examples. Previous efforts have primarily centered around studying more effective FSCIL approaches. By contrast, less attention was devoted to thinking the security issues in contributing to FSCIL. This paper aims to provide a holistic study of the impact of attacks on FSCIL. We first derive insights by systematically exploring how human expert-designed attack methods (i.e., PGD, FGSM) affect FSCIL. We find that those methods either fail to attack base classes, or suffer from huge labor costs due to relying on huge expert knowledge. This highlights the need to craft a specialized attack method for FSCIL. Grounded in these insights, in this paper, we propose a simple yet effective ACraft method to automatically steer and discover optimal attack methods targeted at FSCIL by leveraging Large Language Models (LLMs) without human experts. Moreover, to improve the reasoning between LLMs and FSCIL, we introduce a novel Proximal Policy Optimization (PPO) based reinforcement learning to optimize learning, making LLMs generate better attack methods in the next generation by establishing positive feedback. Experiments on mainstream benchmarks show that our ACraft significantly degrades the performance of state-of-the-art FSCIL methods and dramatically beyond human expert-designed attack methods while maintaining the lowest costs of attack.

摘要: 少镜头课堂增量学习（FSCIL）是持续学习中一种更现实且更具挑战性的范式，可以通过少数训练示例逐步学习未见过的课程并克服基础课程上的灾难性遗忘。之前的工作主要集中在研究更有效的FSCIL方法上。相比之下，在为FSCIL做出贡献时，人们较少关注考虑安全问题。本文旨在对攻击对FSCIL的影响进行全面研究。我们首先通过系统地探索人类专家如何设计的攻击方法（即，PVD、FGSM）影响FSCIL。我们发现这些方法要么无法攻击基本类，要么由于依赖大量的专家知识而面临巨大的劳动力成本。这凸显了为FSCIL设计专门的攻击方法的必要性。基于这些见解，在本文中，我们提出了一种简单而有效的ACraft方法，通过在没有人类专家的情况下利用大型语言模型（LLM）来自动引导和发现针对FSCIL的最佳攻击方法。此外，为了改善LLM和FSCIL之间的推理，我们引入了一种新型的基于近端策略优化（PPO）的强化学习来优化学习，使LLM通过建立正反馈来生成下一代更好的攻击方法。主流基准测试的实验表明，我们的ACraft显着降低了最先进的FSCIL方法的性能，并大大超出了人类专家设计的攻击方法，同时保持了最低的攻击成本。



## **26. In-Context Representation Hijacking**

上下文表示劫持 cs.CL

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2512.03771v2) [paper-pdf](https://arxiv.org/pdf/2512.03771v2)

**Authors**: Itay Yona, Amir Sarid, Michael Karasik, Yossi Gandelsman

**Abstract**: We introduce $\textbf{Doublespeak}$, a simple in-context representation hijacking attack against large language models (LLMs). The attack works by systematically replacing a harmful keyword (e.g., bomb) with a benign token (e.g., carrot) across multiple in-context examples, provided a prefix to a harmful request. We demonstrate that this substitution leads to the internal representation of the benign token converging toward that of the harmful one, effectively embedding the harmful semantics under a euphemism. As a result, superficially innocuous prompts (e.g., "How to build a carrot?") are internally interpreted as disallowed instructions (e.g., "How to build a bomb?"), thereby bypassing the model's safety alignment. We use interpretability tools to show that this semantic overwrite emerges layer by layer, with benign meanings in early layers converging into harmful semantics in later ones. Doublespeak is optimization-free, broadly transferable across model families, and achieves strong success rates on closed-source and open-source systems, reaching 74% ASR on Llama-3.3-70B-Instruct with a single-sentence context override. Our findings highlight a new attack surface in the latent space of LLMs, revealing that current alignment strategies are insufficient and should instead operate at the representation level.

摘要: 我们引入了$\textBF{Douspel peak}$，这是一种针对大型语言模型（LLM）的简单上下文表示劫持攻击。该攻击通过系统性地替换有害关键字（例如，炸弹）具有良性标志（例如，胡萝卜）跨多个上下文示例，为有害请求提供了前置。我们证明，这种替代导致良性标记的内部表示向有害标记的内部表示收敛，有效地将有害语义嵌入在委婉语下。结果，表面上无害的提示（例如，“怎么造胡萝卜？“）在内部被解释为不允许的指令（例如，“如何制造炸弹？”），从而绕过模型的安全对齐。我们使用可解释性工具来表明这种语义覆盖是逐层出现的，早期层中的良性含义会在后期层中收敛为有害的语义。Doublem peak无需优化，可在模型系列中广泛移植，并在闭源和开源系统上实现了很高的成功率，在Llama-3.3- 70 B-Direcct上通过单句上下文覆盖的情况下达到了74%的ASB。我们的研究结果突出了LLM潜在空间中的一个新的攻击面，揭示了当前的对齐策略是不够的，应该在代表性层面上操作。



## **27. Context-Aware Hierarchical Learning: A Two-Step Paradigm towards Safer LLMs**

上下文感知分层学习：实现更安全的LLM的两步范式 cs.CR

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03720v1) [paper-pdf](https://arxiv.org/pdf/2512.03720v1)

**Authors**: Tengyun Ma, Jiaqi Yao, Daojing He, Shihao Peng, Yu Li, Shaohui Liu, Zhuotao Tian

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for diverse applications. However, their uniform token processing paradigm introduces critical vulnerabilities in instruction handling, particularly when exposed to adversarial scenarios. In this work, we identify and propose a novel class of vulnerabilities, termed Tool-Completion Attack (TCA), which exploits function-calling mechanisms to subvert model behavior. To evaluate LLM robustness against such threats, we introduce the Tool-Completion benchmark, a comprehensive security assessment framework, which reveals that even state-of-the-art models remain susceptible to TCA, with surprisingly high attack success rates. To address these vulnerabilities, we introduce Context-Aware Hierarchical Learning (CAHL), a sophisticated mechanism that dynamically balances semantic comprehension with role-specific instruction constraints. CAHL leverages the contextual correlations between different instruction segments to establish a robust, context-aware instruction hierarchy. Extensive experiments demonstrate that CAHL significantly enhances LLM robustness against both conventional attacks and the proposed TCA, exhibiting strong generalization capabilities in zero-shot evaluations while still preserving model performance on generic tasks. Our code is available at https://github.com/S2AILab/CAHL.

摘要: 大型语言模型（LLM）已成为各种应用程序的强大工具。然而，他们的统一令牌处理范式在指令处理中引入了关键漏洞，特别是当暴露于对抗场景时。在这项工作中，我们识别并提出了一类新型漏洞，称为工具完成攻击（MCA），它利用函数调用机制来颠覆模型行为。为了评估LLM针对此类威胁的稳健性，我们引入了工具完成基准，这是一个全面的安全评估框架，它表明即使是最先进的模型仍然容易受到MCA的影响，并且攻击成功率高得惊人。为了解决这些漏洞，我们引入了上下文感知分层学习（CAHL），这是一种复杂的机制，可以动态平衡语义理解与特定角色的指令约束。CAHL利用不同指令段之间的上下文相关性来建立稳健的、上下文感知的指令层次结构。大量实验表明，CAHL显着增强了LLM针对传统攻击和拟议的MCA的鲁棒性，在零激发评估中表现出强大的概括能力，同时仍然保留了通用任务的模型性能。我们的代码可以在https://github.com/S2AILab/CAHL上找到。



## **28. SRPG: Semantically Reconstructed Privacy Guard for Zero-Trust Privacy in Educational Multi-Agent Systems**

SRPG：教育多Agent系统中零信任隐私的语义重构隐私保护 cs.MA

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03694v1) [paper-pdf](https://arxiv.org/pdf/2512.03694v1)

**Authors**: Shuang Guo, Zihui Li

**Abstract**: Multi-Agent Systems (MAS) with large language models (LLMs) enable personalized education but risk leaking minors personally identifiable information (PII) via unstructured dialogue. Existing privacy methods struggle to balance security and utility: role-based access control fails on unstructured text, while naive masking destroys pedagogical context. We propose SRPG, a privacy guard for educational MAS, using a Dual-Stream Reconstruction Mechanism: a strict sanitization stream ensures zero PII leakage, and a context reconstruction stream (LLM driven) recovers mathematical logic. This decouples instructional content from private data, preserving teaching efficacy. Tests on MathDial show SRPG works across models; with GPT-4o, it achieves 0.0000 Attack Success Rate (ASR) (zero leakage) and 0.8267 Exact Match, far outperforming the zero trust Pure LLM baseline (0.2138). SRPG effectively protects minors privacy without sacrificing mathematical instructional quality.

摘要: 具有大型语言模型（LLM）的多智能体系统（MAS）可以实现个性化教育，但存在通过非结构化对话泄露未成年人个人可识别信息（PRI）的风险。现有的隐私方法很难平衡安全性和实用性：基于角色的访问控制对非结构化文本失败，而天真的掩蔽则破坏了教学背景。我们提出SRPG，一种教育MAS的隐私保护，使用双流重建机制：严格的净化流确保零RTI泄漏，而上下文重建流（LLM驱动）恢复数学逻辑。这将教学内容与私人数据分开，从而保持教学效率。MathDial上的测试显示SRPG可以跨模型工作;使用GPT-4 o，它实现了0.0000攻击成功率（ASR）（零泄漏）和0.8267精确匹配，远远超过零信任Pure LLM基线（0.2138）。SRPG在不牺牲数学教学质量的前提下，有效地保护了未成年人的隐私。



## **29. SELF: A Robust Singular Value and Eigenvalue Approach for LLM Fingerprinting**

SELF：一种鲁棒的奇异值和特征值LLM指纹算法 cs.CR

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03620v1) [paper-pdf](https://arxiv.org/pdf/2512.03620v1)

**Authors**: Hanxiu Zhang, Yue Zheng

**Abstract**: The protection of Intellectual Property (IP) in Large Language Models (LLMs) represents a critical challenge in contemporary AI research. While fingerprinting techniques have emerged as a fundamental mechanism for detecting unauthorized model usage, existing methods -- whether behavior-based or structural -- suffer from vulnerabilities such as false claim attacks or susceptible to weight manipulations. To overcome these limitations, we propose SELF, a novel intrinsic weight-based fingerprinting scheme that eliminates dependency on input and inherently resists false claims. SELF achieves robust IP protection through two key innovations: 1) unique, scalable and transformation-invariant fingerprint extraction via singular value and eigenvalue decomposition of LLM attention weights, and 2) effective neural network-based fingerprint similarity comparison based on few-shot learning and data augmentation. Experimental results demonstrate SELF maintains high IP infringement detection accuracy while showing strong robustness against various downstream modifications, including quantization, pruning, and fine-tuning attacks. Our code is available at https://github.com/HanxiuZhang/SELF_v2.

摘要: 大型语言模型（LLM）中的知识产权（IP）保护是当代人工智能研究的一个关键挑战。虽然指纹识别技术已成为检测未经授权的模型使用的基本机制，但现有的方法（无论是基于行为的还是结构性的）都存在虚假声明攻击或容易受到权重操纵等漏洞。为了克服这些限制，我们提出了SELF，这是一种新颖的基于内在权重的指纹识别方案，它消除了对输入的依赖，并从本质上抵制虚假声明。SELF通过两项关键创新实现了稳健的IP保护：1）通过LLM注意力权重的奇异值和特征值分解来进行独特、可扩展和变换不变的指纹提取，2）基于少镜头学习和数据增强的有效基于神经网络的指纹相似性比较。实验结果表明，SELF保持了较高的IP侵权检测精度，同时对各种下游修改，包括量化，修剪和微调攻击表现出较强的鲁棒性。我们的代码可以在https://github.com/HanxiuZhang/SELF_v2上找到。



## **30. Immunity memory-based jailbreak detection: multi-agent adaptive guard for large language models**

基于免疫记忆的越狱检测：大型语言模型的多代理自适应警卫 cs.CR

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03356v1) [paper-pdf](https://arxiv.org/pdf/2512.03356v1)

**Authors**: Jun Leng, Litian Zhang, Xi Zhang

**Abstract**: Large language models (LLMs) have become foundational in AI systems, yet they remain vulnerable to adversarial jailbreak attacks. These attacks involve carefully crafted prompts that bypass safety guardrails and induce models to produce harmful content. Detecting such malicious input queries is therefore critical for maintaining LLM safety. Existing methods for jailbreak detection typically involve fine-tuning LLMs as static safety LLMs using fixed training datasets. However, these methods incur substantial computational costs when updating model parameters to improve robustness, especially in the face of novel jailbreak attacks. Inspired by immunological memory mechanisms, we propose the Multi-Agent Adaptive Guard (MAAG) framework for jailbreak detection. The core idea is to equip guard with memory capabilities: upon encountering novel jailbreak attacks, the system memorizes attack patterns, enabling it to rapidly and accurately identify similar threats in future encounters. Specifically, MAAG first extracts activation values from input prompts and compares them to historical activations stored in a memory bank for quick preliminary detection. A defense agent then simulates responses based on these detection results, and an auxiliary agent supervises the simulation process to provide secondary filtering of the detection outcomes. Extensive experiments across five open-source models demonstrate that MAAG significantly outperforms state-of-the-art (SOTA) methods, achieving 98% detection accuracy and a 96% F1-score across a diverse range of attack scenarios.

摘要: 大型语言模型（LLM）已成为人工智能系统的基础，但它们仍然容易受到敌对越狱攻击。这些攻击涉及精心设计的提示，绕过安全护栏并诱导模型产生有害内容。因此，检测此类恶意输入查询对于维护LLM安全至关重要。现有的越狱检测方法通常涉及使用固定训练数据集将LLM微调为静态安全LLM。然而，这些方法在更新模型参数以提高鲁棒性时会产生巨大的计算成本，尤其是在面对新型越狱攻击时。受免疫记忆机制的启发，我们提出了用于越狱检测的多智能体自适应警卫（MAAG）框架。核心想法是为警卫配备记忆能力：在遇到新颖的越狱攻击时，系统会记住攻击模式，使其能够在未来遇到类似威胁时快速准确地识别出类似威胁。具体来说，MAAG首先从输入提示中提取激活值，并将其与存储在存储库中的历史激活进行比较，以进行快速初步检测。然后，防御代理根据这些检测结果模拟响应，辅助代理监督模拟过程，以提供检测结果的二次过滤。针对五个开源模型的广泛实验表明，MAAG的性能明显优于最先进的（SOTA）方法，在各种攻击场景中实现了98%的检测准确率和96%的F1评分。



## **31. Invasive Context Engineering to Control Large Language Models**

控制大型语言模型的侵入性上下文工程 cs.AI

4 pages

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2512.03001v1) [paper-pdf](https://arxiv.org/pdf/2512.03001v1)

**Authors**: Thomas Rivasseau

**Abstract**: Current research on operator control of Large Language Models improves model robustness against adversarial attacks and misbehavior by training on preference examples, prompting, and input/output filtering. Despite good results, LLMs remain susceptible to abuse, and jailbreak probability increases with context length. There is a need for robust LLM security guarantees in long-context situations. We propose control sentences inserted into the LLM context as invasive context engineering to partially solve the problem. We suggest this technique can be generalized to the Chain-of-Thought process to prevent scheming. Invasive Context Engineering does not rely on LLM training, avoiding data shortage pitfalls which arise in training models for long context situations.

摘要: 当前对大型语言模型操作员控制的研究通过对偏好示例、提示和输入/输出过滤进行训练，提高了模型针对对抗性攻击和不当行为的鲁棒性。尽管结果良好，但LLM仍然容易受到滥用，越狱可能性随着上下文长度的增加而增加。在长期背景下需要强有力的LLM安全保证。我们建议将控制句插入到LLM上下文中，作为侵入性上下文工程，以部分解决问题。我们建议这种技术可以推广到思想链过程中，以防止阴谋。侵入式上下文工程不依赖于LLM培训，从而避免了长期上下文情况的训练模型中出现的数据短缺陷阱。



## **32. Contextual Image Attack: How Visual Context Exposes Multimodal Safety Vulnerabilities**

上下文图像攻击：视觉上下文如何暴露多模式安全漏洞 cs.CV

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2512.02973v1) [paper-pdf](https://arxiv.org/pdf/2512.02973v1)

**Authors**: Yuan Xiong, Ziqi Miao, Lijun Li, Chen Qian, Jie Li, Jing Shao

**Abstract**: While Multimodal Large Language Models (MLLMs) show remarkable capabilities, their safety alignments are susceptible to jailbreak attacks. Existing attack methods typically focus on text-image interplay, treating the visual modality as a secondary prompt. This approach underutilizes the unique potential of images to carry complex, contextual information. To address this gap, we propose a new image-centric attack method, Contextual Image Attack (CIA), which employs a multi-agent system to subtly embeds harmful queries into seemingly benign visual contexts using four distinct visualization strategies. To further enhance the attack's efficacy, the system incorporate contextual element enhancement and automatic toxicity obfuscation techniques. Experimental results on the MMSafetyBench-tiny dataset show that CIA achieves high toxicity scores of 4.73 and 4.83 against the GPT-4o and Qwen2.5-VL-72B models, respectively, with Attack Success Rates (ASR) reaching 86.31\% and 91.07\%. Our method significantly outperforms prior work, demonstrating that the visual modality itself is a potent vector for jailbreaking advanced MLLMs.

摘要: 虽然多模式大型语言模型（MLLM）表现出非凡的能力，但它们的安全排列很容易受到越狱攻击。现有的攻击方法通常专注于文本与图像的相互作用，将视觉形态视为次要提示。这种方法没有充分利用图像承载复杂上下文信息的独特潜力。为了解决这一差距，我们提出了一种新的以图像为中心的攻击方法--上下文图像攻击（CIA），它采用多代理系统，使用四种不同的可视化策略将有害查询巧妙地嵌入到看似良性的视觉上下文中。为了进一步增强攻击的功效，该系统结合了上下文元素增强和自动毒性混淆技术。MMSafetyBench-tiny数据集的实验结果表明，CIA对GPT-4 o和Qwen 2.5-BL-72 B模型的毒性评分分别为4.73和4.83，攻击成功率（ASB）达到86.31%和91.07%。我们的方法显着优于之前的工作，证明视觉形态本身是越狱高级MLLM的有力载体。



## **33. Lost in Modality: Evaluating the Effectiveness of Text-Based Membership Inference Attacks on Large Multimodal Models**

迷失在模式中：评估基于文本的成员推断攻击对大型多模式模型的有效性 cs.CR

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2512.03121v1) [paper-pdf](https://arxiv.org/pdf/2512.03121v1)

**Authors**: Ziyi Tong, Feifei Sun, Le Minh Nguyen

**Abstract**: Large Multimodal Language Models (MLLMs) are emerging as one of the foundational tools in an expanding range of applications. Consequently, understanding training-data leakage in these systems is increasingly critical. Log-probability-based membership inference attacks (MIAs) have become a widely adopted approach for assessing data exposure in large language models (LLMs), yet their effect in MLLMs remains unclear. We present the first comprehensive evaluation of extending these text-based MIA methods to multimodal settings. Our experiments under vision-and-text (V+T) and text-only (T-only) conditions across the DeepSeek-VL and InternVL model families show that in in-distribution settings, logit-based MIAs perform comparably across configurations, with a slight V+T advantage. Conversely, in out-of-distribution settings, visual inputs act as regularizers, effectively masking membership signals.

摘要: 大型多模式语言模型（MLLM）正在成为不断扩大的应用程序的基础工具之一。因此，了解这些系统中的训练数据泄露变得越来越重要。基于日志概率的隶属度推理攻击（MIA）已成为一种广泛采用的评估大型语言模型（LLM）中数据暴露的方法，但其对MLLM的影响仍不清楚。我们首次对将这些基于文本的MIA方法扩展到多模式设置进行了全面评估。我们在DeepSeek-BL和InternVL模型系列中的视觉和文本（V+T）和纯文本（T-仅）条件下进行的实验表明，在分布环境中，基于日志的MIA在跨配置执行切换，具有轻微的V+T优势。相反，在非分布设置中，视觉输入充当规则器，有效地掩盖了成员资格信号。



## **34. COGNITION: From Evaluation to Defense against Multimodal LLM CAPTCHA Solvers**

认知：从评估到防御多模式LLM验证码解决器 cs.CR

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.02318v2) [paper-pdf](https://arxiv.org/pdf/2512.02318v2)

**Authors**: Junyu Wang, Changjia Zhu, Yuanbo Zhou, Lingyao Li, Xu He, Junjie Xiong

**Abstract**: This paper studies how multimodal large language models (MLLMs) undermine the security guarantees of visual CAPTCHA. We identify the attack surface where an adversary can cheaply automate CAPTCHA solving using off-the-shelf models. We evaluate 7 leading commercial and open-source MLLMs across 18 real-world CAPTCHA task types, measuring single-shot accuracy, success under limited retries, end-to-end latency, and per-solve cost. We further analyze the impact of task-specific prompt engineering and few-shot demonstrations on solver effectiveness. We reveal that MLLMs can reliably solve recognition-oriented and low-interaction CAPTCHA tasks at human-like cost and latency, whereas tasks requiring fine-grained localization, multi-step spatial reasoning, or cross-frame consistency remain significantly harder for current models. By examining the reasoning traces of such MLLMs, we investigate the underlying mechanisms of why models succeed/fail on specific CAPTCHA puzzles and use these insights to derive defense-oriented guidelines for selecting and strengthening CAPTCHA tasks. We conclude by discussing implications for platform operators deploying CAPTCHA as part of their abuse-mitigation pipeline.Code Availability (https://anonymous.4open.science/r/Captcha-465E/).

摘要: 本文研究了多模式大型语言模型（MLLM）如何破坏视觉验证码的安全保证。我们确定了对手可以使用现成模型廉价地自动化验证码解决的攻击表面。我们评估了18种现实世界的CAPTCHA任务类型中的7种领先的商业和开源MLLM，衡量单次准确性、有限再试下的成功率、端到端延迟和每次解决的成本。我们进一步分析了特定任务的即时工程和少数镜头演示对求解器有效性的影响。我们发现，MLLM可以以类似于人类的成本和延迟可靠地解决面向认知和低交互性的CAPTCHA任务，而对于当前的模型来说，需要细粒度本地化、多步空间推理或跨框架一致性的任务仍然明显困难。通过检查此类MLLM的推理痕迹，我们研究模型为何在特定验证码难题上成功/失败的潜在机制，并利用这些见解来得出选择和加强验证码任务的防御导向指南。最后，我们讨论了部署CAPTCHA作为其虐待缓解管道的一部分对平台运营商的影响。代码可用性（https：//anonymous.4open.science/r/Captcha-465E/）。



## **35. The Trojan Knowledge: Bypassing Commercial LLM Guardrails via Harmless Prompt Weaving and Adaptive Tree Search**

木马知识：通过无害提示编织和自适应树搜索破解商业LLM护栏 cs.CR

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2512.01353v2) [paper-pdf](https://arxiv.org/pdf/2512.01353v2)

**Authors**: Rongzhe Wei, Peizhi Niu, Xinjie Shen, Tony Tu, Yifan Li, Ruihan Wu, Eli Chien, Pin-Yu Chen, Olgica Milenkovic, Pan Li

**Abstract**: Large language models (LLMs) remain vulnerable to jailbreak attacks that bypass safety guardrails to elicit harmful outputs. Existing approaches overwhelmingly operate within the prompt-optimization paradigm: whether through traditional algorithmic search or recent agent-based workflows, the resulting prompts typically retain malicious semantic signals that modern guardrails are primed to detect. In contrast, we identify a deeper, largely overlooked vulnerability stemming from the highly interconnected nature of an LLM's internal knowledge. This structure allows harmful objectives to be realized by weaving together sequences of benign sub-queries, each of which individually evades detection. To exploit this loophole, we introduce the Correlated Knowledge Attack Agent (CKA-Agent), a dynamic framework that reframes jailbreaking as an adaptive, tree-structured exploration of the target model's knowledge base. The CKA-Agent issues locally innocuous queries, uses model responses to guide exploration across multiple paths, and ultimately assembles the aggregated information to achieve the original harmful objective. Evaluated across state-of-the-art commercial LLMs (Gemini2.5-Flash/Pro, GPT-oss-120B, Claude-Haiku-4.5), CKA-Agent consistently achieves over 95% success rates even against strong guardrails, underscoring the severity of this vulnerability and the urgent need for defenses against such knowledge-decomposition attacks. Our codes are available at https://github.com/Graph-COM/CKA-Agent.

摘要: 大型语言模型（LLM）仍然容易受到越狱攻击，这些攻击绕过安全护栏以引发有害输出。现有方法绝大多数在预算优化范式下运行：无论是通过传统的算法搜索还是最近的基于代理的工作流程，产生的提示通常都会保留现代护栏准备好检测的恶意语义信号。相比之下，我们发现了一个更深层次的、在很大程度上被忽视的漏洞，该漏洞源于法学硕士内部知识的高度相互关联的性质。这种结构允许通过将良性子查询序列编织在一起来实现有害目标，每个子查询都单独逃避检测。为了利用这个漏洞，我们引入了相关知识攻击代理（CKA-Agent），这是一个动态框架，它将越狱重新构建为对目标模型知识库的自适应、树结构化探索。CKA-Agent发出本地无害的查询，使用模型响应来指导跨多个路径的探索，并最终聚集信息以实现最初的有害目标。经过最先进的商业LLM（Gemini 2.5-Flash/Pro、GPT-oss-120 B、Claude-Haiku-4.5）的评估，即使在强大的护栏下，CKA-Agent也始终实现了超过95%的成功率，凸显了该漏洞的严重性以及对此类知识分解攻击的防御的迫切需要。我们的代码可在https://github.com/Graph-COM/CKA-Agent上获取。



## **36. Simplex-Optimized Hybrid Ensemble for Large Language Model Text Detection Under Generative Distribution Drif**

生成性分布驱动下的大型语言模型文本检测的简化优化混合集成 cs.CL

8 pages, 2 Figure, Politeknik Negeri Banyuwangi

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2511.22153v2) [paper-pdf](https://arxiv.org/pdf/2511.22153v2)

**Authors**: Sepyan Purnama Kristanto, Lutfi Hakim, Dianni Yusuf

**Abstract**: The widespread adoption of large language models (LLMs) has made it difficult to distinguish human writing from machine-produced text in many real applications. Detectors that were effective for one generation of models tend to degrade when newer models or modified decoding strategies are introduced. In this work, we study this lack of stability and propose a hybrid ensemble that is explicitly designed to cope with changing generator distributions. The ensemble combines three complementary components: a RoBERTa-based classifier fine-tuned for supervised detection, a curvature-inspired score based on perturbing the input and measuring changes in model likelihood, and a compact stylometric model built on hand-crafted linguistic features. The outputs of these components are fused on the probability simplex, and the weights are chosen via validation-based search. We frame this approach in terms of variance reduction and risk under mixtures of generators, and show that the simplex constraint provides a simple way to trade off the strengths and weaknesses of each branch. Experiments on a 30000 document corpus drawn from several LLM families including models unseen during training and paraphrased attack variants show that the proposed method achieves 94.2% accuracy and an AUC of 0.978. The ensemble also lowers false positives on scientific articles compared to strong baselines, which is critical in educational and research settings where wrongly flagging human work is costly

摘要: 大型语言模型（LLM）的广泛采用使得在许多实际应用中很难区分人类写作与机器生成的文本。当引入新的模型或修改后的解码策略时，对一代模型有效的检测器往往会降级。在这项工作中，我们研究了这种稳定性的缺乏，并提出了一种明确设计来应对不断变化的发电机分布的混合集成。该集成结合了三个补充的组件：针对监督检测进行微调的基于RoBERTa的分类器、基于扰动输入和测量模型可能性变化的弯曲启发评分，以及基于手工语言特征构建的紧凑风格模型。这些组件的输出融合的概率单纯形，并通过基于验证的搜索选择的权重。我们框架这种方法的方差减少和风险的混合发电机，并表明，单纯形约束提供了一个简单的方法来权衡每个分支的优势和劣势。实验结果表明，该方法的准确率达到94.2%，AUC为0.978。与强大的基线相比，该集成还降低了科学文章的误报率，这在错误标记人类工作成本高昂的教育和研究环境中至关重要



## **37. Tight and Practical Privacy Auditing for Differentially Private In-Context Learning**

针对差异私密的上下文学习进行严格而实用的隐私审计 cs.CR

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2511.13502v2) [paper-pdf](https://arxiv.org/pdf/2511.13502v2)

**Authors**: Yuyang Xia, Ruixuan Liu, Li Xiong

**Abstract**: Large language models (LLMs) perform in-context learning (ICL) by adapting to tasks from prompt demonstrations, which in practice often contain private or proprietary data. Although differential privacy (DP) with private voting is a pragmatic mitigation, DP-ICL implementations are error-prone, and worst-case DP bounds may substantially overestimate actual leakage, calling for practical auditing tools. We present a tight and efficient privacy auditing framework for DP-ICL systems that runs membership inference attacks and translates their success rates into empirical privacy guarantees using Gaussian DP. Our analysis of the private voting mechanism identifies vote configurations that maximize the auditing signal, guiding the design of audit queries that reliably reveal whether a canary demonstration is present in the context. The framework supports both black-box (API-only) and white-box (internal vote) threat models, and unifies auditing for classification and generation by reducing both to a binary decision problem. Experiments on standard text classification and generation benchmarks show that our empirical leakage estimates closely match theoretical DP budgets on classification tasks and are consistently lower on generation tasks due to conservative embedding-sensitivity bounds, making our framework a practical privacy auditor and verifier for real-world DP-ICL deployments.

摘要: 大型语言模型（LLM）通过适应即时演示的任务来执行上下文学习（ICL），这些任务在实践中通常包含私人或专有数据。尽管带有私人投票的差异隐私（DP）是一种务实的缓解措施，但DP-ICL的实现很容易出错，而且最坏情况下的DP界限可能会大大高估实际泄漏，因此需要实用的审计工具。我们为DP-ICL系统提供了一个严格而高效的隐私审计框架，该框架运行成员资格推断攻击，并使用高斯DP将其成功率转化为经验隐私保证。我们对私人投票机制的分析确定了最大化审计信号的投票配置，指导审计查询的设计，可靠地揭示上下文中是否存在金丝雀演示。该框架支持黑匣子（仅API）和白盒（内部投票）威胁模型，并通过将两者简化为二元决策问题来统一分类和生成审计。标准文本分类和生成基准的实验表明，我们的经验泄露估计与分类任务的理论DP预算密切匹配，并且由于保守的嵌入敏感性界限，生成任务的泄漏估计始终较低，使我们的框架成为现实世界DP-ICL部署的实用隐私审计器和验证器。



## **38. OpenLVLM-MIA: A Controlled Benchmark Revealing the Limits of Membership Inference Attacks on Large Vision-Language Models**

OpenLVLM-MIA：揭示大型视觉语言模型成员推断攻击局限性的受控基准 cs.CV

WACV2026 Accepted

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2510.16295v2) [paper-pdf](https://arxiv.org/pdf/2510.16295v2)

**Authors**: Ryoto Miyamoto, Xin Fan, Fuyuko Kido, Tsuneo Matsumoto, Hayato Yamana

**Abstract**: OpenLVLM-MIA is a new benchmark that highlights fundamental challenges in evaluating membership inference attacks (MIA) against large vision-language models (LVLMs). While prior work has reported high attack success rates, our analysis suggests that these results often arise from detecting distributional bias introduced during dataset construction rather than from identifying true membership status. To address this issue, we introduce a controlled benchmark of 6{,}000 images where the distributions of member and non-member samples are carefully balanced, and ground-truth membership labels are provided across three distinct training stages. Experiments using OpenLVLM-MIA demonstrated that the performance of state-of-the-art MIA methods approached chance-level. OpenLVLM-MIA, designed to be transparent and unbiased benchmark, clarifies certain limitations of MIA research on LVLMs and provides a solid foundation for developing stronger privacy-preserving techniques.

摘要: OpenLVLM-MIA是一个新基准，强调了评估针对大型视觉语言模型（LVLM）的成员资格推理攻击（MIA）的根本挑战。虽然之前的工作报告了很高的攻击成功率，但我们的分析表明，这些结果通常来自检测数据集构建期间引入的分布偏差，而不是识别真正的成员身份。为了解决这个问题，我们引入了一个由6{，}000张图像组成的受控基准，其中成员和非成员样本的分布经过仔细平衡，并在三个不同的训练阶段提供地面真相成员资格标签。使用OpenLVLM-MIA的实验表明，最先进的MIA方法的性能接近机会水平。OpenLVLM-MIA旨在成为透明和公正的基准，澄清了MIA对LVLM研究的某些局限性，并为开发更强大的隐私保护技术提供了坚实的基础。



## **39. When Ads Become Profiles: Uncovering the Invisible Risk of Web Advertising at Scale with LLMs**

当广告成为简介：利用LLM揭露大规模网络广告的隐形风险 cs.HC

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2509.18874v2) [paper-pdf](https://arxiv.org/pdf/2509.18874v2)

**Authors**: Baiyu Chen, Benjamin Tag, Hao Xue, Daniel Angus, Flora Salim

**Abstract**: Regulatory limits on explicit targeting have not eliminated algorithmic profiling on the Web, as optimisation systems still adapt ad delivery to users' private attributes. The widespread availability of powerful zero-shot multimodal Large Language Models (LLMs) has dramatically lowered the barrier for exploiting these latent signals for adversarial inference. We investigate this emerging societal risk, specifically how adversaries can now exploit these signals to reverse-engineer private attributes from ad exposure alone. We introduce a novel pipeline that leverages LLMs as adversarial inference engines to perform natural language profiling. Applying this method to a longitudinal dataset comprising over 435,000 ad impressions collected from 891 users, we conducted a large-scale study to assess the feasibility and precision of inferring private attributes from passive online ad observations. Our results demonstrate that off-the-shelf LLMs can accurately reconstruct complex user private attributes, including party preference, employment status, and education level, consistently outperforming strong census-based priors and matching or exceeding human social perception, while operating at only a fraction of the cost (223$\times$ lower) and time (52$\times$ faster) required by humans. Critically, actionable profiling is feasible even within short observation windows, indicating that prolonged tracking is not a prerequisite for a successful attack. These findings provide the first empirical evidence that ad streams serve as a high-fidelity digital footprint, enabling off-platform profiling that inherently bypasses current platform safeguards, highlighting a systemic vulnerability in the ad ecosystem and the urgent need for responsible web AI governance in the generative AI era. The code is available at https://github.com/Breezelled/when-ads-become-profiles.

摘要: 对显式定位的监管限制并没有消除网络上的算法分析，因为优化系统仍然根据用户的私人属性调整广告交付。强大的零镜头多模式大型语言模型（LLM）的广泛使用极大地降低了利用这些潜在信号进行对抗性推理的障碍。我们调查这种新出现的社会风险，特别是对手现在如何利用这些信号来仅从广告曝光中反向工程私人属性。我们引入了一种新颖的管道，利用LLM作为对抗推理引擎来执行自然语言分析。将这种方法应用于包含从891名用户收集的超过435，000个广告印象的纵向数据集，我们进行了一项大规模研究，以评估从被动在线广告观察中推断私人属性的可行性和精确性。我们的结果表明，现成的LLM可以准确地重建复杂的用户私人属性，包括政党偏好、就业状况和教育水平，始终优于基于人口普查的强大先验，并匹配或超过人类社会认知，同时运营成本仅为人类所需的一小部分（223美元\倍）和时间（52美元\倍）。至关重要的是，即使在短的观察窗口内，可操作的分析也是可行的，这表明长期跟踪并不是成功攻击的先决条件。这些发现提供了第一个经验证据，证明广告流可以充当高保真数字足迹，实现从本质上绕过当前平台保障措施的平台外分析，凸显了广告生态系统中的系统性漏洞以及生成性人工智能时代对负责任的网络人工智能治理的迫切需要。该代码可在https://github.com/Breezelled/when-ads-become-profiles上获取。



## **40. Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference**

缓存中的影子：在LLM推理中揭示和减轻KV缓存的隐私风险 cs.CR

This paper is accepted by Network and Distributed System Security Symposium (NDSS) 2026

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2508.09442v3) [paper-pdf](https://arxiv.org/pdf/2508.09442v3)

**Authors**: Zhifan Luo, Shuo Shao, Su Zhang, Lijing Zhou, Yuke Hu, Chenxu Zhao, Zhihao Liu, Zhan Qin

**Abstract**: The Key-Value (KV) cache, which stores intermediate attention computations (Key and Value pairs) to avoid redundant calculations, is a fundamental mechanism for accelerating Large Language Model (LLM) inference. However, this efficiency optimization introduces significant yet underexplored privacy risks. This paper provides the first comprehensive analysis of these vulnerabilities, demonstrating that an attacker can reconstruct sensitive user inputs directly from the KV-cache. We design and implement three distinct attack vectors: a direct Inversion Attack, a more broadly applicable and potent Collision Attack, and a semantic-based Injection Attack. These methods demonstrate the practicality and severity of KV-cache privacy leakage issues. To mitigate this, we propose KV-Cloak, a novel, lightweight, and efficient defense mechanism. KV-Cloak uses a reversible matrix-based obfuscation scheme, combined with operator fusion, to secure the KV-cache. Our extensive experiments show that KV-Cloak effectively thwarts all proposed attacks, reducing reconstruction quality to random noise. Crucially, it achieves this robust security with virtually no degradation in model accuracy and minimal performance overhead, offering a practical solution for trustworthy LLM deployment.

摘要: Key-Value（KV）缓存存储中间注意力计算（Key和Value对）以避免冗余计算，是加速大型语言模型（LLM）推理的基本机制。然而，这种效率优化引入了重大但未充分探索的隐私风险。本文首次对这些漏洞进行了全面分析，证明攻击者可以直接从KV缓存重建敏感用户输入。我们设计并实现了三种不同的攻击载体：直接倒置攻击、更广泛适用且更强大的碰撞攻击以及基于语义的注入攻击。这些方法证明了KV缓存隐私泄露问题的实用性和严重性。为了缓解这个问题，我们提出了KV-Cloak，这是一种新颖、轻量级且高效的防御机制。KV-Cloak使用基于可逆矩阵的混淆方案，结合操作符融合来保护KV-缓存。我们广泛的实验表明，KV-Cloak有效地阻止了所有提出的攻击，降低了随机噪音的重建质量。至关重要的是，它实现了这种强大的安全性，模型准确性几乎没有下降，性能负担最小，为值得信赖的LLM部署提供了实用的解决方案。



## **41. Enhancing LLM Watermark Resilience Against Both Scrubbing and Spoofing Attacks**

增强LLM水印针对擦除和欺骗攻击的弹性 cs.CR

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2507.06274v2) [paper-pdf](https://arxiv.org/pdf/2507.06274v2)

**Authors**: Huanming Shen, Baizhou Huang, Xiaojun Wan

**Abstract**: Watermarking is a promising defense against the misuse of large language models (LLMs), yet it remains vulnerable to scrubbing and spoofing attacks. This vulnerability stems from an inherent trade-off governed by watermark window size: smaller windows resist scrubbing better but are easier to reverse-engineer, enabling low-cost statistics-based spoofing attacks. This work breaks this trade-off by introducing a novel mechanism, equivalent texture keys, where multiple tokens within a watermark window can independently support the detection. Based on the redundancy, we propose a novel watermark scheme with Sub-vocabulary decomposed Equivalent tExture Key (SEEK). It achieves a Pareto improvement, increasing the resilience against scrubbing attacks without compromising robustness to spoofing. Experiments demonstrate SEEK's superiority over prior method, yielding spoofing robustness gains of +88.2%/+92.3%/+82.0% and scrubbing robustness gains of +10.2%/+6.4%/+24.6% across diverse dataset settings.

摘要: 水印是防止大型语言模型（LLM）滥用的一种有希望的防御方法，但它仍然容易受到擦洗和欺骗攻击。该漏洞源于由水印窗口大小决定的固有权衡：较小的窗口更难抵抗擦洗，但更容易进行反向工程，从而实现低成本的基于统计学的欺骗攻击。这项工作通过引入一种新颖的机制（等效纹理密钥）打破了这种权衡，其中水印窗口内的多个令牌可以独立支持检测。基于冗余度，我们提出了一种新的子词汇分解等效tExture密钥（SEEK）水印方案。它实现了帕累托改进，提高了针对擦除攻击的弹性，而不会损害欺骗的稳健性。实验证明了SEEK相对于现有方法的优越性，在不同的数据集设置中产生了+88.2%/+92.3%/+82.0%的欺骗鲁棒性收益，并产生了+10.2%/+6.4%/+24.6%的擦洗鲁棒性收益。



## **42. Blackbox Dataset Inference for LLM**

LLM的黑匣子数据集推理 cs.CR

**SubmitDate**: 2025-12-05    [abs](http://arxiv.org/abs/2507.03619v3) [paper-pdf](https://arxiv.org/pdf/2507.03619v3)

**Authors**: Ruikai Zhou, Kang Yang, Xun Chen, Wendy Hui Wang, Guanhong Tao, Jun Xu

**Abstract**: Today, the training of large language models (LLMs) can involve personally identifiable information and copyrighted material, incurring dataset misuse. To mitigate the problem of dataset misuse, this paper explores \textit{dataset inference}, which aims to detect if a suspect model $\mathcal{M}$ used a victim dataset $\mathcal{D}$ in training. Previous research tackles dataset inference by aggregating results of membership inference attacks (MIAs) -- methods to determine whether individual samples are a part of the training dataset. However, restricted by the low accuracy of MIAs, previous research mandates grey-box access to $\mathcal{M}$ to get intermediate outputs (probabilities, loss, perplexity, etc.) for obtaining satisfactory results. This leads to reduced practicality, as LLMs, especially those deployed for profits, have limited incentives to return the intermediate outputs.   In this paper, we propose a new method of dataset inference with only black-box access to the target model (i.e., assuming only the text-based responses of the target model are available). Our method is enabled by two sets of locally built reference models, one set involving $\mathcal{D}$ in training and the other not. By measuring which set of reference model $\mathcal{M}$ is closer to, we determine if $\mathcal{M}$ used $\mathcal{D}$ for training. Evaluations of real-world LLMs in the wild show that our method offers high accuracy in all settings and presents robustness against bypassing attempts.

摘要: 如今，大型语言模型（LLM）的训练可能涉及个人可识别信息和受版权保护的材料，从而导致数据集滥用。为了缓解数据集滥用的问题，本文探讨了\textit{dataset initiation}，其目的是检测可疑模型$\mathCal{M}$是否在训练中使用了受害者数据集$\mathCal{D}$。之前的研究通过聚集隶属度推理攻击（MIA）的结果来解决数据集推理--MIA是确定单个样本是否是训练数据集一部分的方法。然而，受MIA准确性低的限制，之前的研究要求灰箱访问$\mathCal{M}$以获得中间输出（概率、损失、困惑度等）以获得令人满意的结果。这导致实用性降低，因为有限责任公司，特别是那些为利润而部署的有限责任公司，回报中间产出的激励有限。   在本文中，我们提出了一种新的数据集推理方法，只对目标模型进行黑盒访问（即，假设只有目标模型的基于文本的响应可用）。我们的方法由两组本地构建的参考模型来支持，一组在训练中涉及$\mathCal{D}$，另一组不涉及。通过测量$\mathCal{M}$更接近哪一组参考模型，我们确定$\mathCal{M}$是否使用$\mathCal{D}$进行训练。对现实世界LLM的野外评估表明，我们的方法在所有设置中都提供了高准确性，并且具有针对绕过尝试的鲁棒性。



## **43. SafePTR: Token-Level Jailbreak Defense in Multimodal LLMs via Prune-then-Restore Mechanism**

SafeTLR：通过删除然后恢复机制在多模式LLM中进行令牌级越狱防御 cs.CR

Accepted by NeurIPS 2025

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2507.01513v2) [paper-pdf](https://arxiv.org/pdf/2507.01513v2)

**Authors**: Beitao Chen, Xinyu Lyu, Lianli Gao, Jingkuan Song, Heng Tao Shen

**Abstract**: By incorporating visual inputs, Multimodal Large Language Models (MLLMs) extend LLMs to support visual reasoning. However, this integration also introduces new vulnerabilities, making MLLMs susceptible to multimodal jailbreak attacks and hindering their safe deployment.Existing defense methods, including Image-to-Text Translation, Safe Prompting, and Multimodal Safety Tuning, attempt to address this by aligning multimodal inputs with LLMs' built-in safeguards.Yet, they fall short in uncovering root causes of multimodal vulnerabilities, particularly how harmful multimodal tokens trigger jailbreak in MLLMs? Consequently, they remain vulnerable to text-driven multimodal jailbreaks, often exhibiting overdefensive behaviors and imposing heavy training overhead.To bridge this gap, we present an comprehensive analysis of where, how and which harmful multimodal tokens bypass safeguards in MLLMs. Surprisingly, we find that less than 1% tokens in early-middle layers are responsible for inducing unsafe behaviors, highlighting the potential of precisely removing a small subset of harmful tokens, without requiring safety tuning, can still effectively improve safety against jailbreaks. Motivated by this, we propose Safe Prune-then-Restore (SafePTR), an training-free defense framework that selectively prunes harmful tokens at vulnerable layers while restoring benign features at subsequent layers.Without incurring additional computational overhead, SafePTR significantly enhances the safety of MLLMs while preserving efficiency. Extensive evaluations across three MLLMs and five benchmarks demonstrate SafePTR's state-of-the-art performance in mitigating jailbreak risks without compromising utility.

摘要: 通过结合视觉输入，多模式大型语言模型（MLLM）扩展了LLM以支持视觉推理。然而，这种集成也引入了新的漏洞，使MLLM容易受到多模式越狱攻击并阻碍其安全部署。现有的防御方法，包括图像到文本翻译、安全预算处理和多模式安全调优，试图通过将多模式输入与LLM的内置保护措施相一致来解决这个问题。然而，它们未能发现多模式漏洞的根本原因，特别是有害的多模式代币如何触发MLLM越狱？因此，他们仍然容易受到文本驱动的多模式越狱的影响，通常表现出过度防御行为并施加沉重的培训费用。为了弥合这一差距，我们对MLLM中的哪些有害多模式代币在哪里、如何以及哪些方式绕过保障措施进行了全面分析。令人惊讶的是，我们发现早期中层中只有不到1%的代币会引发不安全行为，这凸显了在不需要安全调整的情况下精确删除有害代币的一小部分的潜力，仍然可以有效地提高针对越狱的安全性。出于此动机，我们提出了Safe Prune-then-Restore（SafeTLR），这是一种免训练的防御框架，可以选择性地修剪脆弱层的有害令牌，同时在后续层恢复良性特征。在不产生额外计算负担的情况下，SafeTLR显着增强了MLLM的安全性，同时保持了效率。对三个MLLM和五个基准的广泛评估表明，SafeTLR在缓解越狱风险而不影响实用性方面具有最先进的性能。



## **44. QA-LIGN: Aligning LLMs through Constitutionally Decomposed QA**

QA-LIGN：通过宪法分解的QA调整LLM cs.CL

Findings of the Association for Computational Linguistics: EMNLP 2025, pages 20619-20642, Suzhou, China

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2506.08123v5) [paper-pdf](https://arxiv.org/pdf/2506.08123v5)

**Authors**: Jacob Dineen, Aswin RRV, Qin Liu, Zhikun Xu, Xiao Ye, Ming Shen, Zhaonan Li, Shijie Lu, Chitta Baral, Muhao Chen, Ben Zhou

**Abstract**: Alignment of large language models (LLMs) with principles like helpfulness, honesty, and harmlessness typically relies on scalar rewards that obscure which objectives drive the training signal. We introduce QA-LIGN, which decomposes monolithic rewards into interpretable principle-specific evaluations through structured natural language programs. Models learn through a draft, critique, and revise pipeline, where symbolic evaluation against the rubrics provides transparent feedback for both initial and revised responses during GRPO training. Applied to uncensored Llama-3.1-8B-Instruct, QA-LIGN reduces attack success rates by up to 68.7% while maintaining a 0.67% false refusal rate, achieving Pareto optimal safety-helpfulness performance and outperforming both DPO and GRPO with state-of-the-art reward models given equivalent training. These results demonstrate that making reward signals interpretable and modular improves alignment effectiveness, suggesting transparency enhances LLM safety.

摘要: 大型语言模型（LLM）与乐于助人、诚实和无害等原则的一致通常依赖于量化奖励，这些奖励模糊了哪些目标驱动训练信号。我们引入QA-LIGN，它通过结构化自然语言程序将单一奖励分解为可解释的特定于原则的评估。模型通过起草、评论和修改管道进行学习，其中针对主题的象征性评估为GRPO培训期间的初始和修改响应提供透明的反馈。应用于未经审查的Llama-3.1- 8B-Direct，QA-LIGN将攻击成功率降低高达68.7%，同时保持0.67%的错误拒绝率，实现了帕累托最佳安全帮助性能，并在同等培训的情况下优于DPO和GRPO。这些结果表明，使奖励信号可解释和模块化可以提高对齐有效性，这表明透明度增强了LLM的安全性。



## **45. CoP: Agentic Red-teaming for Large Language Models using Composition of Principles**

CoP：使用原则组合的大型语言模型的大型红色团队 cs.AI

**SubmitDate**: 2025-12-06    [abs](http://arxiv.org/abs/2506.00781v3) [paper-pdf](https://arxiv.org/pdf/2506.00781v3)

**Authors**: Chen Xiong, Pin-Yu Chen, Tsung-Yi Ho

**Abstract**: Recent advances in Large Language Models (LLMs) have spurred transformative applications in various domains, ranging from open-source to proprietary LLMs. However, jailbreak attacks, which aim to break safety alignment and user compliance by tricking the target LLMs into answering harmful and risky responses, are becoming an urgent concern. The practice of red-teaming for LLMs is to proactively explore potential risks and error-prone instances before the release of frontier AI technology. This paper proposes an agentic workflow to automate and scale the red-teaming process of LLMs through the Composition-of-Principles (CoP) framework, where human users provide a set of red-teaming principles as instructions to an AI agent to automatically orchestrate effective red-teaming strategies and generate jailbreak prompts. Distinct from existing red-teaming methods, our CoP framework provides a unified and extensible framework to encompass and orchestrate human-provided red-teaming principles to enable the automated discovery of new red-teaming strategies. When tested against leading LLMs, CoP reveals unprecedented safety risks by finding novel jailbreak prompts and improving the best-known single-turn attack success rate by up to 19.0 times.

摘要: 大型语言模型（LLM）的最新进展激发了各个领域的变革性应用程序，从开源到专有LLM。然而，越狱攻击的目的是通过诱骗目标LLM回答有害和危险的响应来打破安全一致和用户合规性，正在成为一个紧迫的问题。LLM的红色团队实践是在前沿人工智能技术发布之前主动探索潜在风险和容易出错的实例。本文提出了一种代理工作流，通过原则组成（CoP）框架自动化和扩展LLM的红队过程，其中人类用户提供一组红队原则作为AI代理的指令，以自动编排有效的红队策略并生成越狱提示。与现有的红色团队方法不同，我们的CoP框架提供了一个统一且可扩展的框架，以涵盖和编排人类提供的红色团队原则，以实现新的红色团队策略的自动发现。当针对领先的LLM进行测试时，CoP发现了新颖的越狱提示并将最著名的单回合攻击成功率提高了19.0倍，从而揭示了前所未有的安全风险。



## **46. OMNIGUARD: An Efficient Approach for AI Safety Moderation Across Languages and Modalities**

OMNIGUARD：跨语言和模态的AI安全适度的有效方法 cs.CL

**SubmitDate**: 2025-12-09    [abs](http://arxiv.org/abs/2505.23856v2) [paper-pdf](https://arxiv.org/pdf/2505.23856v2)

**Authors**: Sahil Verma, Keegan Hines, Jeff Bilmes, Charlotte Siska, Luke Zettlemoyer, Hila Gonen, Chandan Singh

**Abstract**: The emerging capabilities of large language models (LLMs) have sparked concerns about their immediate potential for harmful misuse. The core approach to mitigate these concerns is the detection of harmful queries to the model. Current detection approaches are fallible, and are particularly susceptible to attacks that exploit mismatched generalization of model capabilities (e.g., prompts in low-resource languages or prompts provided in non-text modalities such as image and audio). To tackle this challenge, we propose Omniguard, an approach for detecting harmful prompts across languages and modalities. Our approach (i) identifies internal representations of an LLM/MLLM that are aligned across languages or modalities and then (ii) uses them to build a language-agnostic or modality-agnostic classifier for detecting harmful prompts. Omniguard improves harmful prompt classification accuracy by 11.57\% over the strongest baseline in a multilingual setting, by 20.44\% for image-based prompts, and sets a new SOTA for audio-based prompts. By repurposing embeddings computed during generation, Omniguard is also very efficient ($\approx\!120 \times$ faster than the next fastest baseline). Code and data are available at: https://github.com/vsahil/OmniGuard.

摘要: 大型语言模型（LLM）的新兴功能引发了人们对其直接潜在有害滥用的担忧。缓解这些担忧的核心方法是检测对模型的有害查询。当前的检测方法是容易出错的，并且特别容易受到利用模型能力不匹配的概括的攻击（例如，低资源语言的提示或以图像和音频等非文本形式提供的提示）。为了应对这一挑战，我们提出了Omniguard，一种用于检测跨语言和模式的有害提示的方法。我们的方法（i）识别跨语言或模式对齐的LLM/MLLM的内部表示，然后（ii）使用它们来构建语言不可知或模式不可知的分类器，用于检测有害提示。Omniguard将有害提示分类准确性提高了11.57\%，超过了多语言设置中最强的基线，对于基于图像的提示提高了20.44\%，并为基于音频的提示设置了新的SOTA。通过重新利用在生成过程中计算的嵌入，Omniguard也非常高效（$\approx\！比下一个最快的基线快120倍）。代码和数据可访问：https://github.com/vsahil/OmniGuard。



## **47. Guard Me If You Know Me: Protecting Specific Face-Identity from Deepfakes**

如果你认识我，请保护我：保护特定面部身份免受Deepfakes的侵害 cs.CV

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2505.19582v2) [paper-pdf](https://arxiv.org/pdf/2505.19582v2)

**Authors**: Kaiqing Lin, Zhiyuan Yan, Ke-Yue Zhang, Li Hao, Yue Zhou, Yuzhen Lin, Weixiang Li, Taiping Yao, Shouhong Ding, Bin Li

**Abstract**: Securing personal identity against deepfake attacks is increasingly critical in the digital age, especially for celebrities and political figures whose faces are easily accessible and frequently targeted. Most existing deepfake detection methods focus on general-purpose scenarios and often ignore the valuable prior knowledge of known facial identities, e.g., "VIP individuals" whose authentic facial data are already available. In this paper, we propose \textbf{VIPGuard}, a unified multimodal framework designed to capture fine-grained and comprehensive facial representations of a given identity, compare them against potentially fake or similar-looking faces, and reason over these comparisons to make accurate and explainable predictions. Specifically, our framework consists of three main stages. First, fine-tune a multimodal large language model (MLLM) to learn detailed and structural facial attributes. Second, we perform identity-level discriminative learning to enable the model to distinguish subtle differences between highly similar faces, including real and fake variations. Finally, we introduce user-specific customization, where we model the unique characteristics of the target face identity and perform semantic reasoning via MLLM to enable personalized and explainable deepfake detection. Our framework shows clear advantages over previous detection works, where traditional detectors mainly rely on low-level visual cues and provide no human-understandable explanations, while other MLLM-based models often lack a detailed understanding of specific face identities. To facilitate the evaluation of our method, we built a comprehensive identity-aware benchmark called \textbf{VIPBench} for personalized deepfake detection, involving the latest 7 face-swapping and 7 entire face synthesis techniques for generation. The code is available at https://github.com/KQL11/VIPGuard .

摘要: 在数字时代，保护个人身份免受Deepfake攻击变得越来越重要，尤其是对于面部易于接触且经常成为攻击目标的名人和政治人物。大多数现有的Deepfake检测方法都专注于通用场景，并且经常忽略已知面部身份的宝贵先验知识，例如，其真实面部数据已经可用的“VIP个人”。在本文中，我们提出了\textBF{VIPGuard}，这是一个统一的多模式框架，旨在捕获给定身份的细粒度和全面的面部表示，将它们与潜在的虚假或相似的面部进行比较，并推理这些比较以做出准确且可解释的预测。具体来说，我们的框架由三个主要阶段组成。首先，微调多模式大型语言模型（MLLM）以学习详细和结构化的面部属性。其次，我们执行身份级别的辨别学习，使模型能够区分高度相似的面孔之间的细微差异，包括真实和虚假的变体。最后，我们引入了特定于用户的定制，其中我们对目标人脸身份的独特特征进行建模，并通过MLLM执行语义推理，以实现个性化和可解释的深度伪造检测。与之前的检测工作相比，我们的框架显示出明显的优势，传统的检测器主要依赖于低级视觉线索，并且不提供人类可理解的解释，而其他基于MLLM的模型通常缺乏对特定面部身份的详细了解。为了促进对我们的方法的评估，我们构建了一个名为\textBF{VIPBench}的全面身份感知基准，用于个性化深度伪造检测，其中涉及最新的7种面部交换和7种完整面部合成技术。该代码可在https://github.com/KQL11/VIPGuard上获取。



## **48. Les Dissonances: Cross-Tool Harvesting and Polluting in Pool-of-Tools Empowered LLM Agents**

Les Dissonance：工具池中的跨工具收获和污染赋予LLM代理人权力 cs.CR

Network and Distributed System Security (NDSS) Symposium 2026

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2504.03111v3) [paper-pdf](https://arxiv.org/pdf/2504.03111v3)

**Authors**: Zichuan Li, Jian Cui, Xiaojing Liao, Luyi Xing

**Abstract**: Large Language Model (LLM) agents are autonomous systems powered by LLMs, capable of reasoning and planning to solve problems by leveraging a set of tools. However, the integration of multi-tool capabilities in LLM agents introduces challenges in securely managing tools, ensuring their compatibility, handling dependency relationships, and protecting control flows within LLM agent workflows. In this paper, we present the first systematic security analysis of task control flows in multi-tool-enabled LLM agents. We identify a novel threat, Cross-Tool Harvesting and Polluting (XTHP), which includes multiple attack vectors to first hijack the normal control flows of agent tasks, and then collect and pollute confidential or private information within LLM agent systems. To understand the impact of this threat, we developed Chord, a dynamic scanning tool designed to automatically detect real-world agent tools susceptible to XTHP attacks. Our evaluation of 66 real-world tools from the repositories of two major LLM agent development frameworks, LangChain and LlamaIndex, revealed a significant security concern: 75% are vulnerable to XTHP attacks, highlighting the prevalence of this threat.

摘要: 大型语言模型（LLM）代理是由LLM支持的自治系统，能够通过利用一组工具进行推理和规划来解决问题。然而，LLM代理中多工具功能的集成在安全管理工具、确保其兼容性、处理依赖关系以及保护LLM代理工作流程中的控制流方面带来了挑战。本文中，我们首次对支持多工具的LLM代理中的任务控制流进行了系统性安全分析。我们识别了一种新型威胁，即跨工具收获和污染（XTHP），它包括多个攻击载体，首先劫持代理任务的正常控制流，然后收集和污染LLM代理系统内的机密或私人信息。为了了解这种威胁的影响，我们开发了Chord，这是一种动态扫描工具，旨在自动检测容易受到XTHP攻击的现实世界代理工具。我们对两个主要LLM代理开发框架LangChain和LlamaIndex存储库中的66个现实工具进行了评估，发现了一个重大的安全问题：75%容易受到XTHP攻击，凸显了这种威胁的普遍性。



## **49. A Fingerprint for Large Language Models**

大型语言模型的指纹 cs.CR

Updated by Hanzhou Wu, 8 pages

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2407.01235v2) [paper-pdf](https://arxiv.org/pdf/2407.01235v2)

**Authors**: Zhiguang Yang, Hanzhou Wu

**Abstract**: Recent advances confirm that large language models (LLMs) can achieve state-of-the-art performance across various tasks. However, due to the resource-intensive nature of training LLMs from scratch, it is urgent and crucial to protect the intellectual property of LLMs against infringement. This has motivated the authors in this paper to propose a novel black-box fingerprinting technique for LLMs. We firstly demonstrate that the outputs of LLMs span a unique vector space associated with each model. We model the problem of fingerprint authentication as the task of evaluating the similarity between the space of the victim model and the space of the suspect model. To tackle with this problem, we introduce two solutions: the first determines whether suspect outputs lie within the victim's subspace, enabling fast infringement detection; the second reconstructs a joint subspace to detect models modified via parameter-efficient fine-tuning (PEFT). Experiments indicate that the proposed method achieves superior performance in fingerprint verification and robustness against the PEFT attacks. This work reveals inherent characteristics of LLMs and provides a promising solution for protecting LLMs, ensuring efficiency, generality and practicality.

摘要: 最近的进展证实，大型语言模型（LLM）可以在各种任务中实现最先进的性能。然而，由于从头开始培训法学硕士的资源密集型性质，保护法学硕士的知识产权免遭侵权显得紧迫而至关重要。这促使本文作者为LLM提出了一种新型的黑匣子指纹技术。我们首先证明LLM的输出跨越与每个模型相关的唯一载体空间。我们将指纹认证问题建模为评估受害者模型空间和嫌疑人模型空间之间相似性的任务。为了解决这个问题，我们引入了两种解决方案：第一种解决方案确定可疑输出是否位于受害者的子空间内，从而实现快速侵权检测;第二种重建联合子空间以检测通过参数高效微调（PEFT）修改的模型。实验表明，该方法在指纹验证方面具有优越的性能和对PEFT攻击的鲁棒性。这项工作揭示了LLM的固有特征，并为保护LLM、确保效率、通用性和实用性提供了一个有前途的解决方案。



## **50. Lockpicking LLMs: A Logit-Based Jailbreak Using Token-level Manipulation**

撬锁LLM：使用代币级操纵的基于日志的越狱 cs.CR

**SubmitDate**: 2025-12-02    [abs](http://arxiv.org/abs/2405.13068v3) [paper-pdf](https://arxiv.org/pdf/2405.13068v3)

**Authors**: Yuxi Li, Yi Liu, Yuekang Li, Ling Shi, Gelei Deng, Shengquan Chen, Kailong Wang

**Abstract**: Large language models (LLMs) have transformed the field of natural language processing, but they remain susceptible to jailbreaking attacks that exploit their capabilities to generate unintended and potentially harmful content. Existing token-level jailbreaking techniques, while effective, face scalability and efficiency challenges, especially as models undergo frequent updates and incorporate advanced defensive measures. In this paper, we introduce JailMine, an innovative token-level manipulation approach that addresses these limitations effectively. JailMine employs an automated "mining" process to elicit malicious responses from LLMs by strategically selecting affirmative outputs and iteratively reducing the likelihood of rejection. Through rigorous testing across multiple well-known LLMs and datasets, we demonstrate JailMine's effectiveness and efficiency, achieving a significant average reduction of 86% in time consumed while maintaining high success rates averaging 95%, even in the face of evolving defensive strategies. Our work contributes to the ongoing effort to assess and mitigate the vulnerability of LLMs to jailbreaking attacks, underscoring the importance of continued vigilance and proactive measures to enhance the security and reliability of these powerful language models.

摘要: 大型语言模型（LLM）已经改变了自然语言处理领域，但它们仍然容易受到越狱攻击，这些攻击利用它们的能力来生成意想不到的和潜在有害的内容。现有的代币级越狱技术虽然有效，但面临可扩展性和效率的挑战，特别是当模型经历频繁更新并纳入先进的防御措施时。在本文中，我们介绍了JailMine，这是一种创新的代币级操纵方法，可以有效地解决这些限制。JailMine采用自动化“挖掘”流程，通过战略性地选择肯定输出并迭代降低拒绝的可能性来引发LLM的恶意响应。通过对多个知名LLM和数据集的严格测试，我们证明了JailMine的有效性和效率，实现了平均86%的时间大幅减少，同时保持了平均95%的高成功率，即使面对不断变化的防御策略。我们的工作有助于评估和减轻LLM对越狱攻击的脆弱性的持续努力，强调了持续保持警惕和采取积极主动措施以增强这些强大语言模型的安全性和可靠性的重要性。



