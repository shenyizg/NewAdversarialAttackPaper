# Latest Adversarial Attack Papers
**update at 2025-05-24 16:13:15**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. When Are Concepts Erased From Diffusion Models?**

概念何时从扩散模型中删除？ cs.LG

Project Page:  https://nyu-dice-lab.github.io/when-are-concepts-erased/

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.17013v1) [paper-pdf](http://arxiv.org/pdf/2505.17013v1)

**Authors**: Kevin Lu, Nicky Kriplani, Rohit Gandikota, Minh Pham, David Bau, Chinmay Hegde, Niv Cohen

**Abstract**: Concept erasure, the ability to selectively prevent a model from generating specific concepts, has attracted growing interest, with various approaches emerging to address the challenge. However, it remains unclear how thoroughly these methods erase the target concept. We begin by proposing two conceptual models for the erasure mechanism in diffusion models: (i) reducing the likelihood of generating the target concept, and (ii) interfering with the model's internal guidance mechanisms. To thoroughly assess whether a concept has been truly erased from the model, we introduce a suite of independent evaluations. Our evaluation framework includes adversarial attacks, novel probing techniques, and analysis of the model's alternative generations in place of the erased concept. Our results shed light on the tension between minimizing side effects and maintaining robustness to adversarial prompts. Broadly, our work underlines the importance of comprehensive evaluation for erasure in diffusion models.

摘要: 概念擦除，即选择性地阻止模型生成特定概念的能力，引起了越来越多的兴趣，各种方法出现了来应对这一挑战。然而，目前尚不清楚这些方法如何彻底消除目标概念。我们首先提出了扩散模型中擦除机制的两个概念模型：（i）降低生成目标概念的可能性，（ii）干扰模型的内部引导机制。为了彻底评估某个概念是否已真正从模型中删除，我们引入了一套独立评估。我们的评估框架包括对抗性攻击、新颖的探测技术以及对模型替代世代的分析，以取代被删除的概念。我们的结果揭示了最大限度地减少副作用和保持对抗提示的鲁棒性之间的紧张关系。从广义上讲，我们的工作强调了对扩散模型中擦除进行全面评估的重要性。



## **2. Harnessing the Computation Redundancy in ViTs to Boost Adversarial Transferability**

利用ViT中的计算冗余来提高对抗性可移植性 cs.CV

15 pages. 7 figures

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2504.10804v2) [paper-pdf](http://arxiv.org/pdf/2504.10804v2)

**Authors**: Jiani Liu, Zhiyuan Wang, Zeliang Zhang, Chao Huang, Susan Liang, Yunlong Tang, Chenliang Xu

**Abstract**: Vision Transformers (ViTs) have demonstrated impressive performance across a range of applications, including many safety-critical tasks. However, their unique architectural properties raise new challenges and opportunities in adversarial robustness. In particular, we observe that adversarial examples crafted on ViTs exhibit higher transferability compared to those crafted on CNNs, suggesting that ViTs contain structural characteristics favorable for transferable attacks. In this work, we investigate the role of computational redundancy in ViTs and its impact on adversarial transferability. Unlike prior studies that aim to reduce computation for efficiency, we propose to exploit this redundancy to improve the quality and transferability of adversarial examples. Through a detailed analysis, we identify two forms of redundancy, including the data-level and model-level, that can be harnessed to amplify attack effectiveness. Building on this insight, we design a suite of techniques, including attention sparsity manipulation, attention head permutation, clean token regularization, ghost MoE diversification, and test-time adversarial training. Extensive experiments on the ImageNet-1k dataset validate the effectiveness of our approach, showing that our methods significantly outperform existing baselines in both transferability and generality across diverse model architectures.

摘要: Vision Transformers（ViT）在一系列应用中表现出令人印象深刻的性能，包括许多安全关键任务。然而，它们独特的架构属性在对抗稳健性方面提出了新的挑战和机遇。特别是，我们观察到，与CNN上制作的对抗示例相比，在ViT上制作的对抗示例表现出更高的可转移性，这表明ViT包含有利于转移攻击的结构特征。在这项工作中，我们研究了计算冗余在ViT中的作用及其对对抗可转移性的影响。与之前旨在减少计算以提高效率的研究不同，我们建议利用这种冗余来提高对抗性示例的质量和可移植性。通过详细的分析，我们确定了两种形式的冗余，包括数据级和模型级，可以利用它们来放大攻击有效性。基于这一见解，我们设计了一套技术，包括注意力稀疏性操纵、注意力头排列、干净令牌正规化、幽灵MoE多样化和测试时对抗训练。ImageNet-1 k数据集上的大量实验验证了我们方法的有效性，表明我们的方法在跨不同模型架构的可移植性和通用性方面都显着优于现有基线。



## **3. Invisible Prompts, Visible Threats: Malicious Font Injection in External Resources for Large Language Models**

看不见的警告，可见的威胁：大型语言模型的外部资源中的恶意字体注入 cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16957v1) [paper-pdf](http://arxiv.org/pdf/2505.16957v1)

**Authors**: Junjie Xiong, Changjia Zhu, Shuhang Lin, Chong Zhang, Yongfeng Zhang, Yao Liu, Lingyao Li

**Abstract**: Large Language Models (LLMs) are increasingly equipped with capabilities of real-time web search and integrated with protocols like Model Context Protocol (MCP). This extension could introduce new security vulnerabilities. We present a systematic investigation of LLM vulnerabilities to hidden adversarial prompts through malicious font injection in external resources like webpages, where attackers manipulate code-to-glyph mapping to inject deceptive content which are invisible to users. We evaluate two critical attack scenarios: (1) "malicious content relay" and (2) "sensitive data leakage" through MCP-enabled tools. Our experiments reveal that indirect prompts with injected malicious font can bypass LLM safety mechanisms through external resources, achieving varying success rates based on data sensitivity and prompt design. Our research underscores the urgent need for enhanced security measures in LLM deployments when processing external content.

摘要: 大型语言模型（LLM）越来越多地配备实时网络搜索功能，并与模型上下文协议（HCP）等协议集成。此扩展可能会引入新的安全漏洞。我们对通过在网页等外部资源中恶意字体注入来隐藏对抗提示的LLM漏洞进行了系统性调查，其中攻击者操纵代码到收件箱的映射来注入用户不可见的欺骗性内容。我们评估了两种关键攻击场景：（1）“恶意内容中继”和（2）通过支持MVP的工具“敏感数据泄露”。我们的实验表明，注入恶意字体的间接提示可以通过外部资源绕过LLM安全机制，根据数据敏感性和提示设计实现不同的成功率。我们的研究强调了处理外部内容时LLM部署中迫切需要增强的安全措施。



## **4. MixAT: Combining Continuous and Discrete Adversarial Training for LLMs**

MixAT：结合LLM的连续和离散对抗训练 cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16947v1) [paper-pdf](http://arxiv.org/pdf/2505.16947v1)

**Authors**: Csaba Dékány, Stefan Balauca, Robin Staab, Dimitar I. Dimitrov, Martin Vechev

**Abstract**: Despite recent efforts in Large Language Models (LLMs) safety and alignment, current adversarial attacks on frontier LLMs are still able to force harmful generations consistently. Although adversarial training has been widely studied and shown to significantly improve the robustness of traditional machine learning models, its strengths and weaknesses in the context of LLMs are less understood. Specifically, while existing discrete adversarial attacks are effective at producing harmful content, training LLMs with concrete adversarial prompts is often computationally expensive, leading to reliance on continuous relaxations. As these relaxations do not correspond to discrete input tokens, such latent training methods often leave models vulnerable to a diverse set of discrete attacks. In this work, we aim to bridge this gap by introducing MixAT, a novel method that combines stronger discrete and faster continuous attacks during training. We rigorously evaluate MixAT across a wide spectrum of state-of-the-art attacks, proposing the At Least One Attack Success Rate (ALO-ASR) metric to capture the worst-case vulnerability of models. We show MixAT achieves substantially better robustness (ALO-ASR < 20%) compared to prior defenses (ALO-ASR > 50%), while maintaining a runtime comparable to methods based on continuous relaxations. We further analyze MixAT in realistic deployment settings, exploring how chat templates, quantization, low-rank adapters, and temperature affect both adversarial training and evaluation, revealing additional blind spots in current methodologies. Our results demonstrate that MixAT's discrete-continuous defense offers a principled and superior robustness-accuracy tradeoff with minimal computational overhead, highlighting its promise for building safer LLMs. We provide our code and models at https://github.com/insait-institute/MixAT.

摘要: 尽管最近在大型语言模型（LLM）的安全性和一致性方面做出了努力，但当前对前沿LLM的对抗性攻击仍然能够持续地迫使有害的世代。尽管对抗训练已得到广泛研究，并被证明可以显着提高传统机器学习模型的鲁棒性，但其在LLM背景下的优点和缺点却知之甚少。具体来说，虽然现有的离散对抗攻击可以有效地产生有害内容，但用具体的对抗提示训练LLM通常计算成本高昂，导致依赖于持续的放松。由于这些松弛不对应于离散输入令牌，因此此类潜在训练方法通常使模型容易受到一系列不同的离散攻击。在这项工作中，我们的目标是通过引入MixAT来弥合这一差距，MixAT是一种新颖的方法，在训练期间结合了更强的离散攻击和更快的连续攻击。我们对MixAT进行了广泛的最先进攻击，提出了至少一次攻击成功率（ALO-ASB）指标来捕捉模型的最坏情况漏洞。我们表明，与之前的防御（ALO-ASB> 50%）相比，MixAT实现了更好的鲁棒性（ALO-ASB < 20%），同时保持与基于连续松弛的方法相当的运行时间。我们进一步分析了现实部署环境中的MixAT，探索聊天模板、量化、低等级适配器和温度如何影响对抗训练和评估，从而揭示了当前方法中的其他盲点。我们的结果表明，MixAT的离散-连续防御以最小的计算负担提供了原则性且卓越的鲁棒性-准确性权衡，凸显了其构建更安全的LLM的承诺。我们在https://github.com/insait-institute/MixAT上提供我们的代码和模型。



## **5. CAIN: Hijacking LLM-Humans Conversations via a Two-Stage Malicious System Prompt Generation and Refining Framework**

CAIN：通过两阶段恶意系统提示生成和精炼框架劫持LLM与人类对话 cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16888v1) [paper-pdf](http://arxiv.org/pdf/2505.16888v1)

**Authors**: Viet Pham, Thai Le

**Abstract**: Large language models (LLMs) have advanced many applications, but are also known to be vulnerable to adversarial attacks. In this work, we introduce a novel security threat: hijacking AI-human conversations by manipulating LLMs' system prompts to produce malicious answers only to specific targeted questions (e.g., "Who should I vote for US President?", "Are Covid vaccines safe?"), while behaving benignly on others. This attack is detrimental as it can enable malicious actors to exercise large-scale information manipulation by spreading harmful but benign-looking system prompts online. To demonstrate such an attack, we develop CAIN, an algorithm that can automatically curate such harmful system prompts for a specific target question in a black-box setting or without the need to access the LLM's parameters. Evaluated on both open-source and commercial LLMs, CAIN demonstrates significant adversarial impact. In untargeted attacks or forcing LLMs to output incorrect answers, CAIN achieves up to 40% F1 degradation on targeted questions while preserving high accuracy on benign inputs. For targeted attacks or forcing LLMs to output specific harmful answers, CAIN achieves over 70% F1 scores on these targeted responses with minimal impact on benign questions. Our results highlight the critical need for enhanced robustness measures to safeguard the integrity and safety of LLMs in real-world applications. All source code will be publicly available.

摘要: 大型语言模型（LLM）先进了许多应用程序，但也容易受到对抗攻击。在这项工作中，我们引入了一种新颖的安全威胁：通过操纵LLM的系统提示来劫持人工智能与人类的对话，以仅对特定目标问题（例如，“我应该投票给谁美国总统？”，“新冠疫苗安全吗？”），同时对他人表现友善。这种攻击是有害的，因为它可以使恶意行为者通过在线传播有害但看起来友善的系统提示来进行大规模信息操纵。为了演示此类攻击，我们开发了CAIN，这是一种算法，可以在黑匣子设置中或无需访问LLM参数的情况下自动策划此类有害系统提示特定目标问题。在开源和商业LLM上进行评估，CAIN表现出显着的对抗影响。在无针对性攻击或迫使LLM输出错误答案中，CAIN对目标问题实现了高达40%的F1降级，同时对良性输入保持高准确性。对于有针对性的攻击或迫使LLM输出特定的有害答案，CAIN在这些有针对性的回答上获得了超过70%的F1分数，而对良性问题的影响最小。我们的结果凸显了对增强稳健性措施的迫切需要，以保障LLM在现实世界应用中的完整性和安全性。所有源代码都将公开。



## **6. Safe RLHF-V: Safe Reinforcement Learning from Multi-modal Human Feedback**

Safe RLHF-V：基于多模态人类反馈的安全强化学习 cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2503.17682v2) [paper-pdf](http://arxiv.org/pdf/2503.17682v2)

**Authors**: Jiaming Ji, Xinyu Chen, Rui Pan, Conghui Zhang, Han Zhu, Jiahao Li, Donghai Hong, Boyuan Chen, Jiayi Zhou, Kaile Wang, Juntao Dai, Chi-Min Chan, Yida Tang, Sirui Han, Yike Guo, Yaodong Yang

**Abstract**: Multimodal large language models (MLLMs) are essential for building general-purpose AI assistants; however, they pose increasing safety risks. How can we ensure safety alignment of MLLMs to prevent undesired behaviors? Going further, it is critical to explore how to fine-tune MLLMs to preserve capabilities while meeting safety constraints. Fundamentally, this challenge can be formulated as a min-max optimization problem. However, existing datasets have not yet disentangled single preference signals into explicit safety constraints, hindering systematic investigation in this direction. Moreover, it remains an open question whether such constraints can be effectively incorporated into the optimization process for multi-modal models. In this work, we present the first exploration of the Safe RLHF-V -- the first multimodal safety alignment framework. The framework consists of: $\mathbf{(I)}$ BeaverTails-V, the first open-source dataset featuring dual preference annotations for helpfulness and safety, supplemented with multi-level safety labels (minor, moderate, severe); $\mathbf{(II)}$ Beaver-Guard-V, a multi-level guardrail system to proactively defend against unsafe queries and adversarial attacks. Applying the guard model over five rounds of filtering and regeneration significantly enhances the precursor model's overall safety by an average of 40.9%. $\mathbf{(III)}$ Based on dual preference, we initiate the first exploration of multi-modal safety alignment within a constrained optimization. Experimental results demonstrate that Safe RLHF effectively improves both model helpfulness and safety. Specifically, Safe RLHF-V enhances model safety by 34.2% and helpfulness by 34.3%.

摘要: 多模式大型语言模型（MLLM）对于构建通用人工智能助手至关重要;然而，它们带来了越来越大的安全风险。我们如何确保MLLM的安全一致以防止不良行为？进一步说，探索如何微调MLLM以在满足安全限制的同时保留功能至关重要。从根本上讲，这个挑战可以被描述为一个最小-最大优化问题。然而，现有的数据集尚未将单一偏好信号分解为明确的安全约束，从而阻碍了这方面的系统性研究。此外，这些约束是否可以有效地纳入多模式模型的优化过程仍然是一个悬而未决的问题。在这项工作中，我们首次探索Safe RLHF-V --第一个多模式安全对齐框架。该框架包括：$\mathBF{（I）}$ BeaverTails-V，第一个开源数据集，具有帮助性和安全性的双重偏好注释，并辅之以多级别安全标签（轻微、中度、严重）; $\mathBF{（II）}$ Beaver-Guard-V，一个多级别护栏系统，用于主动防御不安全的查询和对抗性攻击。经过五轮过滤和再生应用防护模型，前体模型的整体安全性平均显着提高了40.9%。$\mathBF{（III）}$基于双重偏好，我们在约束优化中启动了多模式安全对齐的首次探索。实验结果表明，Safe RL HF有效提高了模型的帮助性和安全性。具体而言，Safe RLHF-V将模型的安全性提高了34.2%，帮助性提高了34.3%。



## **7. Accidental Misalignment: Fine-Tuning Language Models Induces Unexpected Vulnerability**

意外失调：微调语言模型会引发意外漏洞 cs.CL

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16789v1) [paper-pdf](http://arxiv.org/pdf/2505.16789v1)

**Authors**: Punya Syon Pandey, Samuel Simko, Kellin Pelrine, Zhijing Jin

**Abstract**: As large language models gain popularity, their vulnerability to adversarial attacks remains a primary concern. While fine-tuning models on domain-specific datasets is often employed to improve model performance, it can introduce vulnerabilities within the underlying model. In this work, we investigate Accidental Misalignment, unexpected vulnerabilities arising from characteristics of fine-tuning data. We begin by identifying potential correlation factors such as linguistic features, semantic similarity, and toxicity within our experimental datasets. We then evaluate the adversarial performance of these fine-tuned models and assess how dataset factors correlate with attack success rates. Lastly, we explore potential causal links, offering new insights into adversarial defense strategies and highlighting the crucial role of dataset design in preserving model alignment. Our code is available at https://github.com/psyonp/accidental_misalignment.

摘要: 随着大型语言模型越来越受欢迎，它们对对抗攻击的脆弱性仍然是一个主要问题。虽然通常使用特定领域数据集的微调模型来提高模型性能，但它可能会在基础模型中引入漏洞。在这项工作中，我们调查了意外失准，即微调数据特征引起的意外漏洞。我们首先确定潜在的相关因素，如语言特征，语义相似性和毒性在我们的实验数据集。然后，我们评估这些微调模型的对抗性能，并评估数据集因素与攻击成功率的相关性。最后，我们探索了潜在的因果关系，为对抗性防御策略提供了新的见解，并强调了数据集设计在保持模型对齐方面的关键作用。我们的代码可在https://github.com/psyonp/accidental_misalignment上获取。



## **8. Experimental robustness benchmark of quantum neural network on a superconducting quantum processor**

超量子处理器上量子神经网络实验鲁棒性基准 quant-ph

There are 8 pages with 5 figures in the main text and 15 pages with  14 figures in the supplementary information

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16714v1) [paper-pdf](http://arxiv.org/pdf/2505.16714v1)

**Authors**: Hai-Feng Zhang, Zhao-Yun Chen, Peng Wang, Liang-Liang Guo, Tian-Le Wang, Xiao-Yan Yang, Ren-Ze Zhao, Ze-An Zhao, Sheng Zhang, Lei Du, Hao-Ran Tao, Zhi-Long Jia, Wei-Cheng Kong, Huan-Yu Liu, Athanasios V. Vasilakos, Yang Yang, Yu-Chun Wu, Ji Guan, Peng Duan, Guo-Ping Guo

**Abstract**: Quantum machine learning (QML) models, like their classical counterparts, are vulnerable to adversarial attacks, hindering their secure deployment. Here, we report the first systematic experimental robustness benchmark for 20-qubit quantum neural network (QNN) classifiers executed on a superconducting processor. Our benchmarking framework features an efficient adversarial attack algorithm designed for QNNs, enabling quantitative characterization of adversarial robustness and robustness bounds. From our analysis, we verify that adversarial training reduces sensitivity to targeted perturbations by regularizing input gradients, significantly enhancing QNN's robustness. Additionally, our analysis reveals that QNNs exhibit superior adversarial robustness compared to classical neural networks, an advantage attributed to inherent quantum noise. Furthermore, the empirical upper bound extracted from our attack experiments shows a minimal deviation ($3 \times 10^{-3}$) from the theoretical lower bound, providing strong experimental confirmation of the attack's effectiveness and the tightness of fidelity-based robustness bounds. This work establishes a critical experimental framework for assessing and improving quantum adversarial robustness, paving the way for secure and reliable QML applications.

摘要: 量子机器学习（QML）模型与经典模型一样，容易受到对抗攻击，从而阻碍其安全部署。在这里，我们报告了在高温处理器上执行的20量子位量子神经网络（QNN）分类器的第一个系统实验鲁棒性基准。我们的基准测试框架具有为QNN设计的高效对抗攻击算法，能够量化描述对抗稳健性和稳健性界限。根据我们的分析，我们验证了对抗训练通过规范化输入梯度来降低对目标扰动的敏感性，从而显着增强了QNN的鲁棒性。此外，我们的分析表明，与经典神经网络相比，QNN表现出更好的对抗鲁棒性，这一优势归因于固有的量子噪音。此外，从我们的攻击实验中提取的经验上限显示出与理论下限的最小偏差（$3 \x 10 '' s），为攻击的有效性和基于一致性的鲁棒性界限的严格性提供了强有力的实验证实。这项工作建立了一个关键的实验框架，用于评估和改进量子对抗鲁棒性，为安全可靠的QML应用铺平了道路。



## **9. BadVLA: Towards Backdoor Attacks on Vision-Language-Action Models via Objective-Decoupled Optimization**

BadVLA：通过解耦优化实现对视觉-语言-动作模型的后门攻击 cs.CR

19 pages, 12 figures, 6 tables

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16640v1) [paper-pdf](http://arxiv.org/pdf/2505.16640v1)

**Authors**: Xueyang Zhou, Guiyao Tie, Guowen Zhang, Hechang Wang, Pan Zhou, Lichao Sun

**Abstract**: Vision-Language-Action (VLA) models have advanced robotic control by enabling end-to-end decision-making directly from multimodal inputs. However, their tightly coupled architectures expose novel security vulnerabilities. Unlike traditional adversarial perturbations, backdoor attacks represent a stealthier, persistent, and practically significant threat-particularly under the emerging Training-as-a-Service paradigm-but remain largely unexplored in the context of VLA models. To address this gap, we propose BadVLA, a backdoor attack method based on Objective-Decoupled Optimization, which for the first time exposes the backdoor vulnerabilities of VLA models. Specifically, it consists of a two-stage process: (1) explicit feature-space separation to isolate trigger representations from benign inputs, and (2) conditional control deviations that activate only in the presence of the trigger, while preserving clean-task performance. Empirical results on multiple VLA benchmarks demonstrate that BadVLA consistently achieves near-100% attack success rates with minimal impact on clean task accuracy. Further analyses confirm its robustness against common input perturbations, task transfers, and model fine-tuning, underscoring critical security vulnerabilities in current VLA deployments. Our work offers the first systematic investigation of backdoor vulnerabilities in VLA models, highlighting an urgent need for secure and trustworthy embodied model design practices. We have released the project page at https://badvla-project.github.io/.

摘要: 视觉-语言-动作（VLA）模型通过直接从多模式输入进行端到端决策，实现了先进的机器人控制。然而，它们的紧密耦合架构暴露了新型安全漏洞。与传统的对抗性扰动不同，后门攻击代表了一种更隐蔽、持久且实际上重大的威胁--特别是在新兴的“服务培训”范式下--但在VLA模型的背景下，它在很大程度上尚未被探索。为了弥补这一差距，我们提出了BadVLA，这是一种基于Inbox-Decoupled优化的后门攻击方法，首次暴露了VLA模型的后门漏洞。具体来说，它由两阶段过程组成：（1）显式特征空间分离，以将触发器表示与良性输入隔离，以及（2）仅在触发器存在时激活的条件控制偏差，同时保持干净任务性能。多个VLA基准的经验结果表明，BadVLA始终实现接近100%的攻击成功率，对干净任务准确性的影响最小。进一步的分析证实了它对常见输入扰动、任务传输和模型微调的稳健性，凸显了当前VLA部署中的关键安全漏洞。我们的工作首次对VLA模型中的后门漏洞进行了系统性调查，凸显了对安全且值得信赖的具体模型设计实践的迫切需求。我们已在https://badvla-project.github.io/上发布了项目页面。



## **10. Finetuning-Activated Backdoors in LLMs**

LLM中的微调激活后门 cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16567v1) [paper-pdf](http://arxiv.org/pdf/2505.16567v1)

**Authors**: Thibaud Gloaguen, Mark Vero, Robin Staab, Martin Vechev

**Abstract**: Finetuning openly accessible Large Language Models (LLMs) has become standard practice for achieving task-specific performance improvements. Until now, finetuning has been regarded as a controlled and secure process in which training on benign datasets led to predictable behaviors. In this paper, we demonstrate for the first time that an adversary can create poisoned LLMs that initially appear benign but exhibit malicious behaviors once finetuned by downstream users. To this end, our proposed attack, FAB (Finetuning-Activated Backdoor), poisons an LLM via meta-learning techniques to simulate downstream finetuning, explicitly optimizing for the emergence of malicious behaviors in the finetuned models. At the same time, the poisoned LLM is regularized to retain general capabilities and to exhibit no malicious behaviors prior to finetuning. As a result, when users finetune the seemingly benign model on their own datasets, they unknowingly trigger its hidden backdoor behavior. We demonstrate the effectiveness of FAB across multiple LLMs and three target behaviors: unsolicited advertising, refusal, and jailbreakability. Additionally, we show that FAB-backdoors are robust to various finetuning choices made by the user (e.g., dataset, number of steps, scheduler). Our findings challenge prevailing assumptions about the security of finetuning, revealing yet another critical attack vector exploiting the complexities of LLMs.

摘要: 微调可开放访问的大型语言模型（LLM）已成为实现特定任务性能改进的标准实践。到目前为止，微调一直被认为是一个受控且安全的过程，其中对良性数据集的训练会导致可预测的行为。在本文中，我们首次证明对手可以创建有毒的LLM，这些LLM最初看起来是良性的，但一旦被下游用户微调，就会表现出恶意行为。为此，我们提出的攻击FAB（微调激活后门）通过元学习技术毒害LLM，以模拟下游微调，明确优化微调模型中恶意行为的出现。与此同时，有毒的LLM会被规范化，以保留一般能力，并且在微调之前不会表现出恶意行为。因此，当用户在自己的数据集上微调看似良性的模型时，他们会在不知不觉中触发其隐藏的后门行为。我们展示了FAB在多个LLM和三种目标行为中的有效性：未经请求的广告、拒绝和越狱。此外，我们表明FAB后门对于用户做出的各种微调选择是稳健的（例如，数据集、步骤数、调度程序）。我们的发现挑战了有关微调安全性的普遍假设，揭示了另一个利用LLM复杂性的关键攻击载体。



## **11. On the Lack of Robustness of Binary Function Similarity Systems**

论二元函数相似系统的鲁棒性缺乏 cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2412.04163v2) [paper-pdf](http://arxiv.org/pdf/2412.04163v2)

**Authors**: Gianluca Capozzi, Tong Tang, Jie Wan, Ziqi Yang, Daniele Cono D'Elia, Giuseppe Antonio Di Luna, Lorenzo Cavallaro, Leonardo Querzoni

**Abstract**: Binary function similarity, which often relies on learning-based algorithms to identify what functions in a pool are most similar to a given query function, is a sought-after topic in different communities, including machine learning, software engineering, and security. Its importance stems from the impact it has in facilitating several crucial tasks, from reverse engineering and malware analysis to automated vulnerability detection. Whereas recent work cast light around performance on this long-studied problem, the research landscape remains largely lackluster in understanding the resiliency of the state-of-the-art machine learning models against adversarial attacks. As security requires to reason about adversaries, in this work we assess the robustness of such models through a simple yet effective black-box greedy attack, which modifies the topology and the content of the control flow of the attacked functions. We demonstrate that this attack is successful in compromising all the models, achieving average attack success rates of 57.06% and 95.81% depending on the problem settings (targeted and untargeted attacks). Our findings are insightful: top performance on clean data does not necessarily relate to top robustness properties, which explicitly highlights performance-robustness trade-offs one should consider when deploying such models, calling for further research.

摘要: 二进制函数相似性通常依赖于基于学习的算法来识别池中哪些函数与给定查询函数最相似，是不同社区（包括机器学习、软件工程和安全）中备受追捧的话题。它的重要性源于它对促进从反向工程和恶意软件分析到自动漏洞检测等几项关键任务的影响。尽管最近的工作揭示了这个长期研究问题的性能，但研究领域在了解最先进的机器学习模型对抗对抗攻击的弹性方面仍然缺乏活力。由于安全需要对对手进行推理，在这项工作中，我们通过简单而有效的黑匣子贪婪攻击来评估此类模型的稳健性，该攻击修改了被攻击功能的布局和控制流的内容。我们证明，这种攻击成功地破坏了所有模型，根据问题设置（有针对性和无针对性的攻击），平均攻击成功率分别为57.06%和95.81%。我们的发现很有洞察力：干净数据上的顶级性能不一定与顶级稳健性属性相关，这明确强调了在部署此类模型时应该考虑的性能稳健性权衡，需要进一步研究。



## **12. Implicit Jailbreak Attacks via Cross-Modal Information Concealment on Vision-Language Models**

通过视觉语言模型的跨模式信息隐藏进行隐性越狱攻击 cs.LG

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16446v1) [paper-pdf](http://arxiv.org/pdf/2505.16446v1)

**Authors**: Zhaoxin Wang, Handing Wang, Cong Tian, Yaochu Jin

**Abstract**: Multimodal large language models (MLLMs) enable powerful cross-modal reasoning capabilities. However, the expanded input space introduces new attack surfaces. Previous jailbreak attacks often inject malicious instructions from text into less aligned modalities, such as vision. As MLLMs increasingly incorporate cross-modal consistency and alignment mechanisms, such explicit attacks become easier to detect and block. In this work, we propose a novel implicit jailbreak framework termed IJA that stealthily embeds malicious instructions into images via least significant bit steganography and couples them with seemingly benign, image-related textual prompts. To further enhance attack effectiveness across diverse MLLMs, we incorporate adversarial suffixes generated by a surrogate model and introduce a template optimization module that iteratively refines both the prompt and embedding based on model feedback. On commercial models like GPT-4o and Gemini-1.5 Pro, our method achieves attack success rates of over 90% using an average of only 3 queries.

摘要: 多模式大型语言模型（MLLM）实现强大的跨模式推理能力。然而，扩展的输入空间引入了新的攻击面。之前的越狱攻击经常将文本中的恶意指令注入到不一致的模式中，例如视觉。随着MLLM越来越多地结合跨模式一致性和对齐机制，此类显式攻击变得更容易检测和阻止。在这项工作中，我们提出了一种名为IJA的新型隐式越狱框架，该框架通过最低有效位隐写术将恶意指令秘密地嵌入到图像中，并将其与看似良性的图像相关文本提示相结合。为了进一步增强不同MLLM之间的攻击有效性，我们结合了代理模型生成的对抗性后缀，并引入了模板优化模块，该模块根据模型反馈迭代地细化提示和嵌入。在GPT-4 o和Gemini-1.5 Pro等商业型号上，我们的方法平均只需3个查询即可实现超过90%的攻击成功率。



## **13. AdvReal: Adversarial Patch Generation Framework with Application to Adversarial Safety Evaluation of Object Detection Systems**

AdvReal：对抗性补丁生成框架，应用于对象检测系统的对抗性安全评估 cs.CV

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16402v1) [paper-pdf](http://arxiv.org/pdf/2505.16402v1)

**Authors**: Yuanhao Huang, Yilong Ren, Jinlei Wang, Lujia Huo, Xuesong Bai, Jinchuan Zhang, Haiyan Yu

**Abstract**: Autonomous vehicles are typical complex intelligent systems with artificial intelligence at their core. However, perception methods based on deep learning are extremely vulnerable to adversarial samples, resulting in safety accidents. How to generate effective adversarial examples in the physical world and evaluate object detection systems is a huge challenge. In this study, we propose a unified joint adversarial training framework for both 2D and 3D samples to address the challenges of intra-class diversity and environmental variations in real-world scenarios. Building upon this framework, we introduce an adversarial sample reality enhancement approach that incorporates non-rigid surface modeling and a realistic 3D matching mechanism. We compare with 5 advanced adversarial patches and evaluate their attack performance on 8 object detecotrs, including single-stage, two-stage, and transformer-based models. Extensive experiment results in digital and physical environments demonstrate that the adversarial textures generated by our method can effectively mislead the target detection model. Moreover, proposed method demonstrates excellent robustness and transferability under multi-angle attacks, varying lighting conditions, and different distance in the physical world. The demo video and code can be obtained at https://github.com/Huangyh98/AdvReal.git.

摘要: 自动驾驶汽车是典型的复杂智能系统，以人工智能为核心。然而，基于深度学习的感知方法极易受到对抗样本的影响，从而导致安全事故。如何在物理世界中生成有效的对抗示例并评估对象检测系统是一个巨大的挑战。在这项研究中，我们提出了一个针对2D和3D样本的统一联合对抗训练框架，以应对现实世界场景中班级内多样性和环境变化的挑战。在此框架的基础上，我们引入了一种对抗性样本现实增强方法，该方法结合了非刚性表面建模和真实的3D匹配机制。我们与5个高级对抗补丁进行了比较，并评估了它们对8个对象Detecotr的攻击性能，包括单阶段、两阶段和基于变换器的模型。数字和物理环境中的大量实验结果表明，我们的方法生成的对抗纹理可以有效地误导目标检测模型。此外，所提出的方法在多角度攻击、变化的光照条件和物理世界中不同距离下表现出出色的鲁棒性和可移植性。演示视频和代码可在https://github.com/Huangyh98/AdvReal.git上获取。



## **14. Chain-of-Thought Poisoning Attacks against R1-based Retrieval-Augmented Generation Systems**

针对基于R1的检索增强生成系统的思想链中毒攻击 cs.IR

7 pages,3 figures

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16367v1) [paper-pdf](http://arxiv.org/pdf/2505.16367v1)

**Authors**: Hongru Song, Yu-an Liu, Ruqing Zhang, Jiafeng Guo, Yixing Fan

**Abstract**: Retrieval-augmented generation (RAG) systems can effectively mitigate the hallucination problem of large language models (LLMs),but they also possess inherent vulnerabilities. Identifying these weaknesses before the large-scale real-world deployment of RAG systems is of great importance, as it lays the foundation for building more secure and robust RAG systems in the future. Existing adversarial attack methods typically exploit knowledge base poisoning to probe the vulnerabilities of RAG systems, which can effectively deceive standard RAG models. However, with the rapid advancement of deep reasoning capabilities in modern LLMs, previous approaches that merely inject incorrect knowledge are inadequate when attacking RAG systems equipped with deep reasoning abilities. Inspired by the deep thinking capabilities of LLMs, this paper extracts reasoning process templates from R1-based RAG systems, uses these templates to wrap erroneous knowledge into adversarial documents, and injects them into the knowledge base to attack RAG systems. The key idea of our approach is that adversarial documents, by simulating the chain-of-thought patterns aligned with the model's training signals, may be misinterpreted by the model as authentic historical reasoning processes, thus increasing their likelihood of being referenced. Experiments conducted on the MS MARCO passage ranking dataset demonstrate the effectiveness of our proposed method.

摘要: 检索增强生成（RAG）系统可以有效地缓解大型语言模型（LLM）的幻觉问题，但它们也具有固有的漏洞。在RAG系统大规模现实部署之前识别这些弱点非常重要，因为它为未来构建更安全、更强大的RAG系统奠定了基础。现有的对抗攻击方法通常利用知识库中毒来探测RAG系统的漏洞，这可以有效地欺骗标准RAG模型。然而，随着现代LLM深度推理能力的迅速进步，以前仅仅注入错误知识的方法在攻击配备深度推理能力的RAG系统时是不够的。受LLM深度思维能力的启发，本文从基于R1的RAG系统中提取推理过程模板，使用这些模板将错误知识包装到对抗文档中，并将其注入知识库中以攻击RAG系统。我们方法的关键思想是，通过模拟与模型训练信号一致的思维链模式，对抗性文档可能会被模型误解为真实的历史推理过程，从而增加它们被引用的可能性。在MS MARCO通过排名数据集上进行的实验证明了我们提出的方法的有效性。



## **15. SuperPure: Efficient Purification of Localized and Distributed Adversarial Patches via Super-Resolution GAN Models**

SuperPure：通过超分辨率GAN模型有效纯化局部和分布式对抗补丁 cs.CV

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16318v1) [paper-pdf](http://arxiv.org/pdf/2505.16318v1)

**Authors**: Hossein Khalili, Seongbin Park, Venkat Bollapragada, Nader Sehatbakhsh

**Abstract**: As vision-based machine learning models are increasingly integrated into autonomous and cyber-physical systems, concerns about (physical) adversarial patch attacks are growing. While state-of-the-art defenses can achieve certified robustness with minimal impact on utility against highly-concentrated localized patch attacks, they fall short in two important areas: (i) State-of-the-art methods are vulnerable to low-noise distributed patches where perturbations are subtly dispersed to evade detection or masking, as shown recently by the DorPatch attack; (ii) Achieving high robustness with state-of-the-art methods is extremely time and resource-consuming, rendering them impractical for latency-sensitive applications in many cyber-physical systems.   To address both robustness and latency issues, this paper proposes a new defense strategy for adversarial patch attacks called SuperPure. The key novelty is developing a pixel-wise masking scheme that is robust against both distributed and localized patches. The masking involves leveraging a GAN-based super-resolution scheme to gradually purify the image from adversarial patches. Our extensive evaluations using ImageNet and two standard classifiers, ResNet and EfficientNet, show that SuperPure advances the state-of-the-art in three major directions: (i) it improves the robustness against conventional localized patches by more than 20%, on average, while also improving top-1 clean accuracy by almost 10%; (ii) It achieves 58% robustness against distributed patch attacks (as opposed to 0% in state-of-the-art method, PatchCleanser); (iii) It decreases the defense end-to-end latency by over 98% compared to PatchCleanser. Our further analysis shows that SuperPure is robust against white-box attacks and different patch sizes. Our code is open-source.

摘要: 随着基于视觉的机器学习模型越来越多地集成到自主和网络物理系统中，人们对（物理）对抗性补丁攻击的担忧也越来越多。虽然最先进的防御方法可以在对高度集中的局部补丁攻击的效用影响最小的情况下实现经认证的鲁棒性，但它们在两个重要领域存在不足：（i）最先进的方法容易受到低噪声分布式补丁的攻击，其中扰动被巧妙地分散以逃避检测或掩蔽，如最近的DorPatch攻击所示;（ii）用最先进的方法实现高鲁棒性非常耗时和耗费资源，使得它们对于许多网络物理系统中的延迟敏感应用程序来说不切实际。   为了解决鲁棒性和延迟问题，本文提出了一种新的对抗补丁攻击的防御策略，称为SuperPure。关键的新颖性是开发一个像素级掩蔽方案，该方案对分布式和局部补丁都具有鲁棒性。掩蔽涉及利用基于GAN的超分辨率方案来逐渐从对抗补丁中净化图像。我们使用ImageNet和两个标准分类器ResNet和EfficientNet进行的广泛评估表明，SuperPure在三个主要方向上推进了最先进的技术：（i）平均而言，它将传统局部补丁的鲁棒性提高了20%以上，同时还将top-1的清洁准确性提高了近10%;（ii）它对分布式补丁攻击的鲁棒性达到了58%（而最先进的方法PatchCleanser中的鲁棒性为0%）;（iii）与PatchCleanser相比，它将防御端到端延迟降低了98%以上。我们的进一步分析表明，SuperPure对于白盒攻击和不同补丁大小具有强大的鲁棒性。我们的代码是开源的。



## **16. Accelerating Targeted Hard-Label Adversarial Attacks in Low-Query Black-Box Settings**

加速低查询黑匣子设置中的有针对性的硬标签对抗攻击 cs.CV

This paper contains 11 pages, 7 figures and 3 tables. For associated  supplementary code, see https://github.com/mdppml/TEA

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16313v1) [paper-pdf](http://arxiv.org/pdf/2505.16313v1)

**Authors**: Arjhun Swaminathan, Mete Akgün

**Abstract**: Deep neural networks for image classification remain vulnerable to adversarial examples -- small, imperceptible perturbations that induce misclassifications. In black-box settings, where only the final prediction is accessible, crafting targeted attacks that aim to misclassify into a specific target class is particularly challenging due to narrow decision regions. Current state-of-the-art methods often exploit the geometric properties of the decision boundary separating a source image and a target image rather than incorporating information from the images themselves. In contrast, we propose Targeted Edge-informed Attack (TEA), a novel attack that utilizes edge information from the target image to carefully perturb it, thereby producing an adversarial image that is closer to the source image while still achieving the desired target classification. Our approach consistently outperforms current state-of-the-art methods across different models in low query settings (nearly 70\% fewer queries are used), a scenario especially relevant in real-world applications with limited queries and black-box access. Furthermore, by efficiently generating a suitable adversarial example, TEA provides an improved target initialization for established geometry-based attacks.

摘要: 用于图像分类的深度神经网络仍然容易受到对抗性示例的影响--这些小而难以察觉的扰动会导致错误分类。在黑匣子环境中，只有最终预测才能访问，由于决策区域狭窄，精心设计旨在错误分类到特定目标类别的有针对性的攻击尤其具有挑战性。当前最先进的方法通常利用分离源图像和目标图像的决策边界的几何属性，而不是合并来自图像本身的信息。相比之下，我们提出了目标边缘信息攻击（TEA），这是一种新型攻击，利用目标图像的边缘信息仔细扰动它，从而产生更接近源图像的对抗图像，同时仍然实现所需的目标分类。在低查询设置（使用的查询减少了近70%）下，我们的方法在不同模型上始终优于当前最先进的方法，这种情况在查询和黑匣子访问有限的现实世界应用程序中尤其相关。此外，通过有效生成合适的对抗示例，TEA为已建立的基于几何的攻击提供了改进的目标初始化。



## **17. Timestamp Manipulation: Timestamp-based Nakamoto-style Blockchains are Vulnerable**

时间戳操纵：基于时间戳的中本风格区块链很脆弱 cs.CR

26 pages, 6 figures

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.05328v3) [paper-pdf](http://arxiv.org/pdf/2505.05328v3)

**Authors**: Junjie Hu, Na Ruan, Sisi Duan

**Abstract**: Nakamoto consensus are the most widely adopted decentralized consensus mechanism in cryptocurrency systems. Since it was proposed in 2008, many studies have focused on analyzing its security. Most of them focus on maximizing the profit of the adversary. Examples include the selfish mining attack [FC '14] and the recent riskless uncle maker (RUM) attack [CCS '23]. In this work, we introduce the Staircase-Unrestricted Uncle Maker (SUUM), the first block withholding attack targeting the timestamp-based Nakamoto-style blockchain. Through block withholding, timestamp manipulation, and difficulty risk control, SUUM adversaries are capable of launching persistent attacks with zero cost and minimal difficulty risk characteristics, indefinitely exploiting rewards from honest participants. This creates a self-reinforcing cycle that threatens the security of blockchains. We conduct a comprehensive and systematic evaluation of SUUM, including the attack conditions, its impact on blockchains, and the difficulty risks. Finally, we further discuss four feasible mitigation measures against SUUM.

摘要: Nakamoto共识是加密货币系统中最广泛采用的去中心化共识机制。自2008年提出以来，许多研究都集中在分析其安全性上。他们中的大多数都专注于使对手的利润最大化。例子包括自私的采矿攻击[FC ' 14]和最近的无风险叔叔制造商（RUM）攻击[CS ' 23]。在这项工作中，我们介绍了Staircase-Unrestricted Maker（SUUM），这是第一个针对基于时间戳的Nakamoto风格区块链的区块扣留攻击。通过区块扣留、时间戳操纵和难度风险控制，SUUM对手能够以零成本和最小难度风险特征发起持续攻击，无限期地利用诚实参与者的回报。这造成了一个自我强化的循环，威胁区块链的安全。我们对SUUM进行全面、系统的评估，包括攻击条件、对区块链的影响以及难度风险。最后，我们进一步讨论了四个可行的缓解措施，对SUUM。



## **18. Decentralized Nonconvex Robust Optimization over Unsafe Multiagent Systems: System Modeling, Utility, Resilience, and Privacy Analysis**

不安全多智能体系统上的分散非凸鲁棒优化：系统建模、效用、弹性和隐私分析 math.OC

15 pages, 15 figures

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2409.18632v7) [paper-pdf](http://arxiv.org/pdf/2409.18632v7)

**Authors**: Jinhui Hu, Guo Chen, Huaqing Li, Huqiang Cheng, Xiaoyu Guo, Tingwen Huang

**Abstract**: Privacy leakage and Byzantine failures are two adverse factors to the intelligent decision-making process of multi-agent systems (MASs). Considering the presence of these two issues, this paper targets the resolution of a class of nonconvex optimization problems under the Polyak-{\L}ojasiewicz (P-{\L}) condition. To address this problem, we first identify and construct the adversary system model. To enhance the robustness of stochastic gradient descent methods, we mask the local gradients with Gaussian noises and adopt a resilient aggregation method self-centered clipping (SCC) to design a differentially private (DP) decentralized Byzantine-resilient algorithm, namely DP-SCC-PL, which simultaneously achieves differential privacy and Byzantine resilience. The convergence analysis of DP-SCC-PL is challenging since the convergence error can be contributed jointly by privacy-preserving and Byzantine-resilient mechanisms, as well as the nonconvex relaxation, which is addressed via seeking the contraction relationships among the disagreement measure of reliable agents before and after aggregation, together with the optimal gap. Theoretical results reveal that DP-SCC-PL achieves consensus among all reliable agents and sublinear (inexact) convergence with well-designed step-sizes. It has also been proved that if there are no privacy issues and Byzantine agents, then the asymptotic exact convergence can be recovered. Numerical experiments verify the utility, resilience, and differential privacy of DP-SCC-PL by tackling a nonconvex optimization problem satisfying the P-{\L} condition under various Byzantine attacks.

摘要: 隐私泄露和拜占庭式故障是多代理系统（MAS）智能决策过程的两个不利因素。考虑到这两个问题的存在，本文针对Polyak-{\L}ojasiewicz（P-{\L}）条件下的一类非凸优化问题进行了求解。为了解决这个问题，我们首先识别并构建对手系统模型。为了增强随机梯度下降方法的鲁棒性，我们用高斯噪音掩盖局部梯度，并采用弹性聚合方法自中心剪裁（SCC）设计了一种差异私密（DP）去中心化拜占庭弹性算法，即DP-SCC-PL，同时实现了差异隐私和拜占庭弹性。DP-SCC-PL的收敛分析具有挑战性，因为收敛误差可以由隐私保护和拜占庭弹性机制以及非凸松弛共同造成，非凸松弛是通过寻求聚集前后可靠代理人的分歧测量之间的收缩关系来解决的，以及最佳差距。理论结果表明，DP-SCC-PL在所有可靠代理之间实现了共识，并通过精心设计的步进大小实现了亚线性（不精确）收敛。事实还证明，如果不存在隐私问题和拜占庭代理，则可以恢复渐进精确收敛。数值实验通过解决各种拜占庭攻击下满足P-{\L}条件的非凸优化问题来验证DP-SCC-PL的实用性、弹性和差异隐私。



## **19. PoisonArena: Uncovering Competing Poisoning Attacks in Retrieval-Augmented Generation**

PoisonArena：揭露检索增强一代中的竞争中毒攻击 cs.IR

29 pages

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.12574v3) [paper-pdf](http://arxiv.org/pdf/2505.12574v3)

**Authors**: Liuji Chen, Xiaofang Yang, Yuanzhuo Lu, Jinghao Zhang, Xin Sun, Qiang Liu, Shu Wu, Jing Dong, Liang Wang

**Abstract**: Retrieval-Augmented Generation (RAG) systems, widely used to improve the factual grounding of large language models (LLMs), are increasingly vulnerable to poisoning attacks, where adversaries inject manipulated content into the retriever's corpus. While prior research has predominantly focused on single-attacker settings, real-world scenarios often involve multiple, competing attackers with conflicting objectives. In this work, we introduce PoisonArena, the first benchmark to systematically study and evaluate competing poisoning attacks in RAG. We formalize the multi-attacker threat model, where attackers vie to control the answer to the same query using mutually exclusive misinformation. PoisonArena leverages the Bradley-Terry model to quantify each method's competitive effectiveness in such adversarial environments. Through extensive experiments on the Natural Questions and MS MARCO datasets, we demonstrate that many attack strategies successful in isolation fail under competitive pressure. Our findings highlight the limitations of conventional evaluation metrics like Attack Success Rate (ASR) and F1 score and underscore the need for competitive evaluation to assess real-world attack robustness. PoisonArena provides a standardized framework to benchmark and develop future attack and defense strategies under more realistic, multi-adversary conditions. Project page: https://github.com/yxf203/PoisonArena.

摘要: 检索增强生成（RAG）系统，广泛用于改善大型语言模型（LLM）的事实基础，越来越容易受到中毒攻击，其中对手将操纵的内容注入检索器的语料库。虽然以前的研究主要集中在单个攻击者的设置，但现实世界的场景往往涉及多个相互竞争的攻击者，这些攻击者的目标相互冲突。在这项工作中，我们介绍PoisonArena，第一个基准系统地研究和评估竞争中毒攻击在RAG。我们形式化的多攻击者威胁模型，攻击者争夺控制答案相同的查询使用互斥的错误信息。PoisonArena利用Bradley-Terry模型来量化每种方法在此类对抗环境中的竞争有效性。通过对Natural Questions和MS MARCO数据集的广泛实验，我们证明了许多孤立成功的攻击策略在竞争压力下失败。我们的研究结果强调了攻击成功率（SVR）和F1评分等传统评估指标的局限性，并强调了竞争性评估来评估现实世界攻击稳健性的必要性。PoisonArena提供了一个标准化的框架，可以在更现实的多对手条件下基准和开发未来的攻击和防御策略。项目页面：https://github.com/yxf203/PoisonArena。



## **20. ErasableMask: A Robust and Erasable Privacy Protection Scheme against Black-box Face Recognition Models**

ErasableMass：针对黑匣子人脸识别模型的稳健且可擦除的隐私保护方案 cs.CV

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2412.17038v4) [paper-pdf](http://arxiv.org/pdf/2412.17038v4)

**Authors**: Sipeng Shen, Yunming Zhang, Dengpan Ye, Xiuwen Shi, Long Tang, Haoran Duan, Yueyun Shang, Zhihong Tian

**Abstract**: While face recognition (FR) models have brought remarkable convenience in face verification and identification, they also pose substantial privacy risks to the public. Existing facial privacy protection schemes usually adopt adversarial examples to disrupt face verification of FR models. However, these schemes often suffer from weak transferability against black-box FR models and permanently damage the identifiable information that cannot fulfill the requirements of authorized operations such as forensics and authentication. To address these limitations, we propose ErasableMask, a robust and erasable privacy protection scheme against black-box FR models. Specifically, via rethinking the inherent relationship between surrogate FR models, ErasableMask introduces a novel meta-auxiliary attack, which boosts black-box transferability by learning more general features in a stable and balancing optimization strategy. It also offers a perturbation erasion mechanism that supports the erasion of semantic perturbations in protected face without degrading image quality. To further improve performance, ErasableMask employs a curriculum learning strategy to mitigate optimization conflicts between adversarial attack and perturbation erasion. Extensive experiments on the CelebA-HQ and FFHQ datasets demonstrate that ErasableMask achieves the state-of-the-art performance in transferability, achieving over 72% confidence on average in commercial FR systems. Moreover, ErasableMask also exhibits outstanding perturbation erasion performance, achieving over 90% erasion success rate.

摘要: 虽然人脸识别模型在人脸验证和识别方面带来了显着的便利，但它们也对公众构成了重大的隐私风险。现有的人脸隐私保护方案通常采用对抗性样本来破坏FR模型的人脸验证。然而，这些计划往往遭受弱的可转移性对黑箱FR模型和永久损坏的可识别信息，不能满足授权操作，如取证和认证的要求。为了解决这些限制，我们提出了ErasableMask，一个强大的和可擦除的隐私保护计划，对黑盒FR模型。具体来说，通过重新思考代理FR模型之间的内在关系，ErasableMass引入了一种新型的元辅助攻击，该攻击通过在稳定和平衡的优化策略中学习更多通用特征来提高黑匣子的可移植性。它还提供了一种扰动误差机制，该机制支持受保护面部中的语义扰动误差，而不会降低图像质量。为了进一步提高性能，ErasableMass采用课程学习策略来减轻对抗攻击和扰动误差之间的优化冲突。对CelebA-HQ和FFHQ数据集的广泛实验表明，ErasableMass在可移植性方面实现了最先进的性能，在商用FR系统中平均置信度超过72%。此外，ErasableMass还表现出出色的扰动出错性能，出错成功率超过90%。



## **21. PandaGuard: Systematic Evaluation of LLM Safety against Jailbreaking Attacks**

PandaGuard：针对越狱攻击的LLM安全性系统评估 cs.CR

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.13862v2) [paper-pdf](http://arxiv.org/pdf/2505.13862v2)

**Authors**: Guobin Shen, Dongcheng Zhao, Linghao Feng, Xiang He, Jihang Wang, Sicheng Shen, Haibo Tong, Yiting Dong, Jindong Li, Xiang Zheng, Yi Zeng

**Abstract**: Large language models (LLMs) have achieved remarkable capabilities but remain vulnerable to adversarial prompts known as jailbreaks, which can bypass safety alignment and elicit harmful outputs. Despite growing efforts in LLM safety research, existing evaluations are often fragmented, focused on isolated attack or defense techniques, and lack systematic, reproducible analysis. In this work, we introduce PandaGuard, a unified and modular framework that models LLM jailbreak safety as a multi-agent system comprising attackers, defenders, and judges. Our framework implements 19 attack methods and 12 defense mechanisms, along with multiple judgment strategies, all within a flexible plugin architecture supporting diverse LLM interfaces, multiple interaction modes, and configuration-driven experimentation that enhances reproducibility and practical deployment. Built on this framework, we develop PandaBench, a comprehensive benchmark that evaluates the interactions between these attack/defense methods across 49 LLMs and various judgment approaches, requiring over 3 billion tokens to execute. Our extensive evaluation reveals key insights into model vulnerabilities, defense cost-performance trade-offs, and judge consistency. We find that no single defense is optimal across all dimensions and that judge disagreement introduces nontrivial variance in safety assessments. We release the code, configurations, and evaluation results to support transparent and reproducible research in LLM safety.

摘要: 大型语言模型（LLM）已经取得了卓越的能力，但仍然容易受到被称为越狱的对抗性提示的影响，这可能会绕过安全对齐并引发有害的输出。尽管LLM安全研究的努力越来越多，但现有的评估往往是分散的，集中在孤立的攻击或防御技术上，缺乏系统的，可重复的分析。在这项工作中，我们引入了PandaGuard，一个统一的模块化框架，将LLM越狱安全建模为一个由攻击者，防御者和法官组成的多代理系统。我们的框架实现了19种攻击方法和12种防御机制，以及多种判断策略，所有这些都在一个灵活的插件架构中，支持多种LLM接口，多种交互模式和配置驱动的实验，从而增强了可重复性和实际部署。基于这个框架，我们开发了PandaBench，这是一个全面的基准，可评估49个LLM和各种判断方法之间的相互作用，需要超过30亿个代币来执行。我们的广泛评估揭示了对模型漏洞、国防成本-性能权衡和判断一致性的关键见解。我们发现，没有一种防御在所有维度上都是最佳的，而且判断分歧会在安全评估中引入非平凡的方差。我们发布代码、配置和评估结果，以支持LLM安全性方面的透明和可重复研究。



## **22. SafeKey: Amplifying Aha-Moment Insights for Safety Reasoning**

SafeKey：放大啊哈时刻洞察以实现安全推理 cs.AI

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16186v1) [paper-pdf](http://arxiv.org/pdf/2505.16186v1)

**Authors**: Kaiwen Zhou, Xuandong Zhao, Gaowen Liu, Jayanth Srinivasa, Aosong Feng, Dawn Song, Xin Eric Wang

**Abstract**: Large Reasoning Models (LRMs) introduce a new generation paradigm of explicitly reasoning before answering, leading to remarkable improvements in complex tasks. However, they pose great safety risks against harmful queries and adversarial attacks. While recent mainstream safety efforts on LRMs, supervised fine-tuning (SFT), improve safety performance, we find that SFT-aligned models struggle to generalize to unseen jailbreak prompts. After thorough investigation of LRMs' generation, we identify a safety aha moment that can activate safety reasoning and lead to a safe response. This aha moment typically appears in the `key sentence', which follows models' query understanding process and can indicate whether the model will proceed safely. Based on these insights, we propose SafeKey, including two complementary objectives to better activate the safety aha moment in the key sentence: (1) a Dual-Path Safety Head to enhance the safety signal in the model's internal representations before the key sentence, and (2) a Query-Mask Modeling objective to improve the models' attention on its query understanding, which has important safety hints. Experiments across multiple safety benchmarks demonstrate that our methods significantly improve safety generalization to a wide range of jailbreak attacks and out-of-distribution harmful prompts, lowering the average harmfulness rate by 9.6\%, while maintaining general abilities. Our analysis reveals how SafeKey enhances safety by reshaping internal attention and improving the quality of hidden representations.

摘要: 大型推理模型（LRM）引入了新一代的显式推理范式，在回答之前，导致显着改善复杂的任务。然而，它们对有害查询和对抗性攻击构成了巨大的安全风险。虽然最近LRM的主流安全措施，监督微调（SFT），提高了安全性能，我们发现，SFT对齐的模型很难推广到看不见的越狱提示。在对LRM一代进行彻底调查后，我们发现了一个可以激活安全推理并导致安全响应的安全啊哈时刻。这个啊哈时刻通常出现在“关键时刻”中，它遵循模型的查询理解过程，并且可以指示模型是否将安全地继续进行。基于这些见解，我们提出了SafeKey，其中包括两个补充目标，以更好地激活关键句中的安全啊哈时刻：（1）双路径安全头，以增强关键句之前模型内部表示中的安全信号，（2）查询面具建模目标，以提高模型对其查询理解的关注度，这具有重要的安全提示。跨多个安全基准的实验表明，我们的方法显着提高了对各种越狱攻击和分发外有害提示的安全概括性，将平均危害率降低9.6%，同时保持一般能力。我们的分析揭示了SafeKey如何通过重塑内部注意力和提高隐藏表示的质量来增强安全性。



## **23. TRAIL: Transferable Robust Adversarial Images via Latent diffusion**

TRAIL：通过潜在扩散的可转移鲁棒对抗图像 cs.CV

**SubmitDate**: 2025-05-22    [abs](http://arxiv.org/abs/2505.16166v1) [paper-pdf](http://arxiv.org/pdf/2505.16166v1)

**Authors**: Yuhao Xue, Zhifei Zhang, Xinyang Jiang, Yifei Shen, Junyao Gao, Wentao Gu, Jiale Zhao, Miaojing Shi, Cairong Zhao

**Abstract**: Adversarial attacks exploiting unrestricted natural perturbations present severe security risks to deep learning systems, yet their transferability across models remains limited due to distribution mismatches between generated adversarial features and real-world data. While recent works utilize pre-trained diffusion models as adversarial priors, they still encounter challenges due to the distribution shift between the distribution of ideal adversarial samples and the natural image distribution learned by the diffusion model. To address the challenge, we propose Transferable Robust Adversarial Images via Latent Diffusion (TRAIL), a test-time adaptation framework that enables the model to generate images from a distribution of images with adversarial features and closely resembles the target images. To mitigate the distribution shift, during attacks, TRAIL updates the diffusion U-Net's weights by combining adversarial objectives (to mislead victim models) and perceptual constraints (to preserve image realism). The adapted model then generates adversarial samples through iterative noise injection and denoising guided by these objectives. Experiments demonstrate that TRAIL significantly outperforms state-of-the-art methods in cross-model attack transferability, validating that distribution-aligned adversarial feature synthesis is critical for practical black-box attacks.

摘要: 利用不受限制的自然扰动的对抗攻击给深度学习系统带来了严重的安全风险，但由于生成的对抗特征与现实世界数据之间的分布不匹配，它们在模型之间的可移植性仍然有限。虽然最近的作品利用预先训练的扩散模型作为对抗性先验，但由于理想对抗性样本的分布和扩散模型学习的自然图像分布之间的分布变化，它们仍然面临挑战。为了应对这一挑战，我们提出了通过潜在扩散传输的鲁棒对抗图像（TRAIL），这是一种测试时自适应框架，使模型能够从具有对抗特征且与目标图像非常相似的图像分布中生成图像。为了减轻分布偏移，在攻击过程中，TRAIL通过结合对抗目标（误导受害者模型）和感知约束（保持图像真实性）来更新扩散U-Net的权重。然后，适应模型通过迭代噪声注入和这些目标指导的去噪生成对抗样本。实验表明，TRAIL在跨模型攻击可转移性方面明显优于最先进的方法，验证了分布对齐的对抗性特征合成对于实际黑盒攻击至关重要。



## **24. Adaptive Honeypot Allocation in Multi-Attacker Networks via Bayesian Stackelberg Games**

基于Bayesian Stackelberg博弈的多攻击者网络中自适应蜜罐分配 cs.GT

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.16043v1) [paper-pdf](http://arxiv.org/pdf/2505.16043v1)

**Authors**: Dongyoung Park, Gaby G. Dagher

**Abstract**: Defending against sophisticated cyber threats demands strategic allocation of limited security resources across complex network infrastructures. When the defender has limited defensive resources, the complexity of coordinating honeypot placements across hundreds of nodes grows exponentially. In this paper, we present a multi-attacker Bayesian Stackelberg framework modeling concurrent adversaries attempting to breach a directed network of system components. Our approach uniquely characterizes each adversary through distinct target preferences, exploit capabilities, and associated costs, while enabling defenders to strategically deploy honeypots at critical network positions. By integrating a multi-follower Stackelberg formulation with dynamic Bayesian belief updates, our framework allows defenders to continuously refine their understanding of attacker intentions based on actions detected through Intrusion Detection Systems (IDS). Experimental results show that the proposed method prevents attack success within a few rounds and scales well up to networks of 500 nodes with more than 1,500 edges, maintaining tractable run times.

摘要: 防御复杂的网络威胁需要在复杂的网络基础设施中战略性地分配有限的安全资源。当防御者的防御资源有限时，协调数百个节点上的蜜罐放置的复杂性就会呈指数级增长。本文中，我们提出了一个多攻击者Bayesian Stackelberg框架，对试图破坏系统组件有向网络的并发对手进行建模。我们的方法通过不同的目标偏好、利用能力和相关成本来独特地描述每个对手，同时使防御者能够在关键网络位置战略性地部署蜜罐。通过将多跟随者Stackelberg公式与动态Bayesian信念更新集成，我们的框架允许防御者根据入侵检测系统（IDS）检测到的动作不断完善对攻击者意图的理解。实验结果表明，所提出的方法可以在几轮内阻止攻击成功，并可以扩展到500个节点、超过1，500条边的网络，保持易于处理的运行时间。



## **25. SAeUron: Interpretable Concept Unlearning in Diffusion Models with Sparse Autoencoders**

SAeUron：使用稀疏自动编码器的扩散模型中的可解释概念消除 cs.LG

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2501.18052v3) [paper-pdf](http://arxiv.org/pdf/2501.18052v3)

**Authors**: Bartosz Cywiński, Kamil Deja

**Abstract**: Diffusion models, while powerful, can inadvertently generate harmful or undesirable content, raising significant ethical and safety concerns. Recent machine unlearning approaches offer potential solutions but often lack transparency, making it difficult to understand the changes they introduce to the base model. In this work, we introduce SAeUron, a novel method leveraging features learned by sparse autoencoders (SAEs) to remove unwanted concepts in text-to-image diffusion models. First, we demonstrate that SAEs, trained in an unsupervised manner on activations from multiple denoising timesteps of the diffusion model, capture sparse and interpretable features corresponding to specific concepts. Building on this, we propose a feature selection method that enables precise interventions on model activations to block targeted content while preserving overall performance. Our evaluation shows that SAeUron outperforms existing approaches on the UnlearnCanvas benchmark for concepts and style unlearning, and effectively eliminates nudity when evaluated with I2P. Moreover, we show that with a single SAE, we can remove multiple concepts simultaneously and that in contrast to other methods, SAeUron mitigates the possibility of generating unwanted content under adversarial attack. Code and checkpoints are available at https://github.com/cywinski/SAeUron.

摘要: 传播模型虽然强大，但可能会无意中产生有害或不良内容，从而引发严重的道德和安全问题。最近的机器去学习方法提供了潜在的解决方案，但通常缺乏透明度，因此很难理解它们对基本模型带来的变化。在这项工作中，我们引入了SAeUron，这是一种利用稀疏自动编码器（SAEs）学习的特征来删除文本到图像扩散模型中不需要的概念的新型方法。首先，我们证明，以无监督的方式对扩散模型的多个去噪时步的激活进行训练的SAEs可以捕获与特定概念相对应的稀疏且可解释的特征。在此基础上，我们提出了一种功能选择方法，该方法能够对模型激活进行精确干预，以阻止目标内容，同时保持整体性能。我们的评估表明，SAeUron在概念和风格消除方面优于UnlearnCanvas基准上的现有方法，并且在使用I2P评估时有效地消除了裸体。此外，我们表明，通过单个严重不良事件，我们可以同时删除多个概念，并且与其他方法相比，SAeUron降低了在对抗性攻击下生成不想要内容的可能性。代码和检查点可访问https://github.com/cywinski/SAeUron。



## **26. Ranking Free RAG: Replacing Re-ranking with Selection in RAG for Sensitive Domains**

免费RAG排名：在RAG中用敏感领域的选择取代重新排名 cs.CL

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.16014v1) [paper-pdf](http://arxiv.org/pdf/2505.16014v1)

**Authors**: Yash Saxena, Anpur Padia, Mandar S Chaudhary, Kalpa Gunaratna, Srinivasan Parthasarathy, Manas Gaur

**Abstract**: Traditional Retrieval-Augmented Generation (RAG) pipelines rely on similarity-based retrieval and re-ranking, which depend on heuristics such as top-k, and lack explainability, interpretability, and robustness against adversarial content. To address this gap, we propose a novel method METEORA that replaces re-ranking in RAG with a rationale-driven selection approach. METEORA operates in two stages. First, a general-purpose LLM is preference-tuned to generate rationales conditioned on the input query using direct preference optimization. These rationales guide the evidence chunk selection engine, which selects relevant chunks in three stages: pairing individual rationales with corresponding retrieved chunks for local relevance, global selection with elbow detection for adaptive cutoff, and context expansion via neighboring chunks. This process eliminates the need for top-k heuristics. The rationales are also used for consistency check using a Verifier LLM to detect and filter poisoned or misleading content for safe generation. The framework provides explainable and interpretable evidence flow by using rationales consistently across both selection and verification. Our evaluation across six datasets spanning legal, financial, and academic research domains shows that METEORA improves generation accuracy by 33.34% while using approximately 50% fewer chunks than state-of-the-art re-ranking methods. In adversarial settings, METEORA significantly improves the F1 score from 0.10 to 0.44 over the state-of-the-art perplexity-based defense baseline, demonstrating strong resilience to poisoning attacks. Code available at: https://anonymous.4open.science/r/METEORA-DC46/README.md

摘要: 传统的检索增强生成（RAG）管道依赖于基于相似性的检索和重新排序，这依赖于top-k等启发式方法，并且缺乏可解释性、可解释性和针对对抗性内容的鲁棒性。为了解决这一差距，我们提出了一种新颖的方法METEORA，用理性驱动的选择方法取代RAG中的重新排名。METEORA分两个阶段运营。首先，使用直接偏好优化，对通用LLM进行偏好调整，以生成以输入查询为条件的基本原理。这些原理指导证据块选择引擎，该引擎分三个阶段选择相关块：将个体原理与相应的检索到的块配对以获得局部相关性、通过肘部检测进行全局选择以获得自适应截止，以及通过邻近块进行上下文扩展。此过程消除了对top-k启发式的需要。这些原理还用于使用Verification LLM进行一致性检查，以检测和过滤有毒或误导性内容，以安全生成。该框架通过在选择和验证中一致使用原理来提供可解释和可解释的证据流。我们对涵盖法律、金融和学术研究领域的六个数据集的评估显示，METEORA将生成准确性提高了33.34%，同时比最先进的重新排序方法少使用约50%的块。在对抗环境中，METEORA将F1评分从0.10显着提高到0.44，比最先进的基于困惑的防御基线，表现出对中毒攻击的强大韧性。代码可访问：https://anonymous.4open.science/r/METEORA-DC46/README.md



## **27. Reverse Engineering Human Preferences with Reinforcement Learning**

利用强化学习反向工程人类偏好 cs.CL

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15795v1) [paper-pdf](http://arxiv.org/pdf/2505.15795v1)

**Authors**: Lisa Alazraki, Tan Yi-Chern, Jon Ander Campos, Maximilian Mozes, Marek Rei, Max Bartolo

**Abstract**: The capabilities of Large Language Models (LLMs) are routinely evaluated by other LLMs trained to predict human preferences. This framework--known as LLM-as-a-judge--is highly scalable and relatively low cost. However, it is also vulnerable to malicious exploitation, as LLM responses can be tuned to overfit the preferences of the judge. Previous work shows that the answers generated by a candidate-LLM can be edited post hoc to maximise the score assigned to them by a judge-LLM. In this study, we adopt a different approach and use the signal provided by judge-LLMs as a reward to adversarially tune models that generate text preambles designed to boost downstream performance. We find that frozen LLMs pipelined with these models attain higher LLM-evaluation scores than existing frameworks. Crucially, unlike other frameworks which intervene directly on the model's response, our method is virtually undetectable. We also demonstrate that the effectiveness of the tuned preamble generator transfers when the candidate-LLM and the judge-LLM are replaced with models that are not used during training. These findings raise important questions about the design of more reliable LLM-as-a-judge evaluation settings. They also demonstrate that human preferences can be reverse engineered effectively, by pipelining LLMs to optimise upstream preambles via reinforcement learning--an approach that could find future applications in diverse tasks and domains beyond adversarial attacks.

摘要: 大型语言模型（LLM）的能力通常由其他经过训练以预测人类偏好的LLM进行评估。这个框架-被称为LLM作为法官-具有高度可扩展性和相对较低的成本。然而，它也容易受到恶意利用，因为LLM响应可以被调整以过度适应法官的偏好。以前的工作表明，候选人LLM生成的答案可以事后编辑，以最大限度地提高法官LLM分配给他们的分数。在这项研究中，我们采用了一种不同的方法，并使用judge-LLM提供的信号作为奖励，以对抗性地调整模型，这些模型生成旨在提高下游性能的文本前置码。我们发现，使用这些模型流水线化的冻结LLM比现有框架获得更高的LLM评估分数。至关重要的是，与直接干预模型响应的其他框架不同，我们的方法几乎无法检测。我们还证明，当候选LLM和判断LLM被训练期间未使用的模型替换时，调整后的前同步码生成器的有效性会转移。这些发现提出了更可靠的法学硕士作为一个法官的评价设置的设计的重要问题。他们还证明，人类的偏好可以有效地进行逆向工程，通过流水线LLM来优化上游的优化，这种方法可以在对抗性攻击之外的各种任务和领域中找到未来的应用。



## **28. Scalable Defense against In-the-wild Jailbreaking Attacks with Safety Context Retrieval**

通过安全上下文检索针对野外越狱攻击的可扩展防御 cs.CR

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15753v1) [paper-pdf](http://arxiv.org/pdf/2505.15753v1)

**Authors**: Taiye Chen, Zeming Wei, Ang Li, Yisen Wang

**Abstract**: Large Language Models (LLMs) are known to be vulnerable to jailbreaking attacks, wherein adversaries exploit carefully engineered prompts to induce harmful or unethical responses. Such threats have raised critical concerns about the safety and reliability of LLMs in real-world deployment. While existing defense mechanisms partially mitigate such risks, subsequent advancements in adversarial techniques have enabled novel jailbreaking methods to circumvent these protections, exposing the limitations of static defense frameworks. In this work, we explore defending against evolving jailbreaking threats through the lens of context retrieval. First, we conduct a preliminary study demonstrating that even a minimal set of safety-aligned examples against a particular jailbreak can significantly enhance robustness against this attack pattern. Building on this insight, we further leverage the retrieval-augmented generation (RAG) techniques and propose Safety Context Retrieval (SCR), a scalable and robust safeguarding paradigm for LLMs against jailbreaking. Our comprehensive experiments demonstrate how SCR achieves superior defensive performance against both established and emerging jailbreaking tactics, contributing a new paradigm to LLM safety. Our code will be available upon publication.

摘要: 众所周知，大型语言模型（LLM）很容易受到越狱攻击，其中对手利用精心设计的提示来引发有害或不道德的反应。此类威胁引发了人们对LLM在现实世界部署中的安全性和可靠性的严重担忧。虽然现有的防御机制部分减轻了此类风险，但对抗技术的后续进步使新型越狱方法能够规避这些保护，暴露了静态防御框架的局限性。在这项工作中，我们探索通过上下文检索的视角抵御不断变化的越狱威胁。首先，我们进行了一项初步研究，证明即使是针对特定越狱的最少一组安全一致的示例也可以显着增强针对这种攻击模式的鲁棒性。在这一见解的基础上，我们进一步利用检索增强生成（RAG）技术并提出安全上下文检索（SR），这是一种针对LLM越狱的可扩展且强大的保护范式。我们全面的实验展示了可控硅如何在针对既定和新兴越狱策略的情况下实现卓越的防御性能，为LLM安全性贡献了新的范式。我们的代码将在发布后提供。



## **29. Alignment Under Pressure: The Case for Informed Adversaries When Evaluating LLM Defenses**

压力下的一致：评估LLM防御时知情对手的理由 cs.CR

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15738v1) [paper-pdf](http://arxiv.org/pdf/2505.15738v1)

**Authors**: Xiaoxue Yang, Bozhidar Stevanoski, Matthieu Meeus, Yves-Alexandre de Montjoye

**Abstract**: Large language models (LLMs) are rapidly deployed in real-world applications ranging from chatbots to agentic systems. Alignment is one of the main approaches used to defend against attacks such as prompt injection and jailbreaks. Recent defenses report near-zero Attack Success Rates (ASR) even against Greedy Coordinate Gradient (GCG), a white-box attack that generates adversarial suffixes to induce attacker-desired outputs. However, this search space over discrete tokens is extremely large, making the task of finding successful attacks difficult. GCG has, for instance, been shown to converge to local minima, making it sensitive to initialization choices. In this paper, we assess the future-proof robustness of these defenses using a more informed threat model: attackers who have access to some information about the alignment process. Specifically, we propose an informed white-box attack leveraging the intermediate model checkpoints to initialize GCG, with each checkpoint acting as a stepping stone for the next one. We show this approach to be highly effective across state-of-the-art (SOTA) defenses and models. We further show our informed initialization to outperform other initialization methods and show a gradient-informed checkpoint selection strategy to greatly improve attack performance and efficiency. Importantly, we also show our method to successfully find universal adversarial suffixes -- single suffixes effective across diverse inputs. Our results show that, contrary to previous beliefs, effective adversarial suffixes do exist against SOTA alignment-based defenses, that these can be found by existing attack methods when adversaries exploit alignment knowledge, and that even universal suffixes exist. Taken together, our results highlight the brittleness of current alignment-based methods and the need to consider stronger threat models when testing the safety of LLMs.

摘要: 大型语言模型（LLM）被快速部署在从聊天机器人到代理系统的实际应用中。对齐是用于防御诸如即时注入和越狱等攻击的主要方法之一。最近的防御报告甚至对贪婪坐标梯度（GCG）的攻击成功率（ASR）接近于零，GCG是一种白盒攻击，生成对抗性后缀以诱导攻击者期望的输出。然而，这种在离散令牌上的搜索空间非常大，使得找到成功攻击的任务变得困难。例如，GCG已被证明收敛到局部极小值，使其对初始化选择敏感。在本文中，我们使用一个更明智的威胁模型来评估这些防御系统的面向未来的鲁棒性：可以访问有关对齐过程的一些信息的攻击者。具体来说，我们提出了一种知情白盒攻击，利用中间模型检查点来初始化GCG，每个检查点都充当下一个检查点的垫脚石。我们证明这种方法在最先进的（SOTA）防御和模型中非常有效。我们进一步展示了我们的知情初始化，以优于其他初始化方法，并展示了一种基于梯度的检查点选择策略，以极大地提高攻击性能和效率。重要的是，我们还展示了成功找到通用对抗后缀的方法--在不同输入中有效的单个后缀。我们的结果表明，与之前的观点相反，针对基于SOTA匹配的防御，确实存在有效的对抗性后缀，当对手利用对齐知识时，这些后缀可以通过现有的攻击方法找到，甚至存在通用后缀。总而言之，我们的结果凸显了当前基于环境的方法的脆弱性，以及在测试LLM的安全性时需要考虑更强的威胁模型。



## **30. Beyond Classification: Evaluating Diffusion Denoised Smoothing for Security-Utility Trade off**

超越分类：评估安全-效用权衡的扩散去噪平滑 cs.LG

Paper accepted at the 33rd European Signal Processing Conference  (EUSIPCO 2025)

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15594v1) [paper-pdf](http://arxiv.org/pdf/2505.15594v1)

**Authors**: Yury Belousov, Brian Pulfer, Vitaliy Kinakh, Slava Voloshynovskiy

**Abstract**: While foundation models demonstrate impressive performance across various tasks, they remain vulnerable to adversarial inputs. Current research explores various approaches to enhance model robustness, with Diffusion Denoised Smoothing emerging as a particularly promising technique. This method employs a pretrained diffusion model to preprocess inputs before model inference. Yet, its effectiveness remains largely unexplored beyond classification. We aim to address this gap by analyzing three datasets with four distinct downstream tasks under three different adversarial attack algorithms. Our findings reveal that while foundation models maintain resilience against conventional transformations, applying high-noise diffusion denoising to clean images without any distortions significantly degrades performance by as high as 57%. Low-noise diffusion settings preserve performance but fail to provide adequate protection across all attack types. Moreover, we introduce a novel attack strategy specifically targeting the diffusion process itself, capable of circumventing defenses in the low-noise regime. Our results suggest that the trade-off between adversarial robustness and performance remains a challenge to be addressed.

摘要: 虽然基金会模型在各种任务中表现出令人印象深刻的性能，但它们仍然容易受到对抗性输入的影响。当前的研究探索了增强模型稳健性的各种方法，其中扩散去噪平滑成为一种特别有前途的技术。该方法使用预训练的扩散模型来在模型推理之前预处理输入。然而，其有效性在很大程度上仍然未经分类之外的探索。我们的目标是通过在三种不同的对抗攻击算法下分析具有四个不同下游任务的三个数据集来解决这一差距。我们的研究结果表明，虽然基础模型保持了对传统转换的弹性，但对没有任何失真的干净图像应用高噪音扩散去噪会显着降低性能高达57%。低噪音扩散设置可以保留性能，但无法为所有攻击类型提供足够的保护。此外，我们引入了一种专门针对扩散过程本身的新型攻击策略，能够绕过低噪音区域中的防御。我们的结果表明，对抗稳健性和性能之间的权衡仍然是一个需要解决的挑战。



## **31. Red-Teaming for Inducing Societal Bias in Large Language Models**

红色团队在大型语言模型中引入社会偏见 cs.CL

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2405.04756v2) [paper-pdf](http://arxiv.org/pdf/2405.04756v2)

**Authors**: Chu Fei Luo, Ahmad Ghawanmeh, Bharat Bhimshetty, Kashyap Murali, Murli Jadhav, Xiaodan Zhu, Faiza Khan Khattak

**Abstract**: Ensuring the safe deployment of AI systems is critical in industry settings where biased outputs can lead to significant operational, reputational, and regulatory risks. Thorough evaluation before deployment is essential to prevent these hazards. Red-teaming addresses this need by employing adversarial attacks to develop guardrails that detect and reject biased or harmful queries, enabling models to be retrained or steered away from harmful outputs. However, most red-teaming efforts focus on harmful or unethical instructions rather than addressing social bias, leaving this critical area under-explored despite its significant real-world impact, especially in customer-facing systems. We propose two bias-specific red-teaming methods, Emotional Bias Probe (EBP) and BiasKG, to evaluate how standard safety measures for harmful content affect bias. For BiasKG, we refactor natural language stereotypes into a knowledge graph. We use these attacking strategies to induce biased responses from several open- and closed-source language models. Unlike prior work, these methods specifically target social bias. We find our method increases bias in all models, even those trained with safety guardrails. Our work emphasizes uncovering societal bias in LLMs through rigorous evaluation, and recommends measures ensure AI safety in high-stakes industry deployments.

摘要: 在行业环境中，确保人工智能系统的安全部署至关重要，在这些环境中，有偏见的输出可能导致重大的运营、声誉和监管风险。部署前的彻底评估对于防止这些危险至关重要。红色团队通过使用对抗攻击来开发检测和拒绝有偏见或有害查询的护栏来满足这一需求，使模型能够被重新训练或引导远离有害输出。然而，大多数红色团队工作都集中在有害或不道德的指令上，而不是解决社会偏见，这使得这个关键领域的探索不足，尽管它对现实世界产生了重大影响，尤其是在面向客户的系统中。我们提出了两种针对偏见的红色团队方法：情感偏见探测器（EBP）和偏见KG，以评估有害内容的标准安全措施如何影响偏见。对于BiasKG，我们将自然语言刻板印象重新构建到知识图谱中。我们使用这些攻击策略来诱导几种开源和闭源语言模型的偏见反应。与之前的工作不同，这些方法专门针对社会偏见。我们发现我们的方法增加了所有模型的偏差，即使是那些接受过安全护栏训练的模型。我们的工作强调通过严格的评估来揭示LLM中的社会偏见，并建议采取措施确保高风险行业部署中的人工智能安全。



## **32. ModSec-AdvLearn: Countering Adversarial SQL Injections with Robust Machine Learning**

ModSec-AdvLearn：利用稳健的机器学习对抗敌对SQL注入 cs.LG

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2308.04964v4) [paper-pdf](http://arxiv.org/pdf/2308.04964v4)

**Authors**: Giuseppe Floris, Christian Scano, Biagio Montaruli, Luca Demetrio, Andrea Valenza, Luca Compagna, Davide Ariu, Luca Piras, Davide Balzarotti, Battista Biggio

**Abstract**: Many Web Application Firewalls (WAFs) leverage the OWASP CRS to block incoming malicious requests. The CRS consists of different sets of rules designed by domain experts to detect well-known web attack patterns. Both the set of rules and the weights used to combine them are manually defined, yielding four different default configurations of the CRS. In this work, we focus on the detection of SQLi attacks, and show that the manual configurations of the CRS typically yield a suboptimal trade-off between detection and false alarm rates. Furthermore, we show that these configurations are not robust to adversarial SQLi attacks, i.e., carefully-crafted attacks that iteratively refine the malicious SQLi payload by querying the target WAF to bypass detection. To overcome these limitations, we propose (i) using machine learning to automate the selection of the set of rules to be combined along with their weights, i.e., customizing the CRS configuration based on the monitored web services; and (ii) leveraging adversarial training to significantly improve its robustness to adversarial SQLi manipulations. Our experiments, conducted using the well-known open-source ModSecurity WAF equipped with the CRS rules, show that our approach, named ModSec-AdvLearn, can (i) increase the detection rate up to 30%, while retaining negligible false alarm rates and discarding up to 50% of the CRS rules; and (ii) improve robustness against adversarial SQLi attacks up to 85%, marking a significant stride toward designing more effective and robust WAFs. We release our open-source code at https://github.com/pralab/modsec-advlearn.

摘要: 许多Web应用程序防火墙（WAF）利用OWISP CRS来阻止进入的恶意请求。CRS由领域专家设计的不同规则集组成，用于检测众所周知的网络攻击模式。规则集和用于组合它们的权重都是手动定义的，从而产生四种不同的CRS默认配置。在这项工作中，我们重点关注SQLi攻击的检测，并表明CRS的手动配置通常会在检测和误报率之间产生次优的权衡。此外，我们证明了这些配置对对抗性SQLi攻击不鲁棒，即，精心设计的攻击，通过查询目标WAF来绕过检测，反复改进恶意SQLi有效负载。为了克服这些限制，我们提出（i）使用机器学习来自动选择要与它们的权重一起组合的规则集，即，基于所监视的web服务定制CRS配置;以及（ii）利用对抗性训练来显著提高其对对抗性SQLi操纵的鲁棒性。我们使用配备了CRS规则的知名开源ModSecurity WAF进行的实验表明，我们的方法（称为ModSec-AdvLearn）可以（i）将检测率提高到30%，同时保留可忽略的误报率并丢弃高达50%的CRS规则;以及（ii）将对抗SQLi攻击的稳健性提高至85%，标志着朝着设计更有效和更稳健的WAF迈出了重大一步。我们在https://github.com/pralab/modsec-advlearn上发布我们的开源代码。



## **33. Audio Jailbreak: An Open Comprehensive Benchmark for Jailbreaking Large Audio-Language Models**

Audio Jailbreak：一个用于越狱大型音频语言模型的开放综合基准测试 cs.SD

We release AJailBench, including both static and optimized  adversarial data, to facilitate future research:  https://github.com/mbzuai-nlp/AudioJailbreak

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15406v1) [paper-pdf](http://arxiv.org/pdf/2505.15406v1)

**Authors**: Zirui Song, Qian Jiang, Mingxuan Cui, Mingzhe Li, Lang Gao, Zeyu Zhang, Zixiang Xu, Yanbo Wang, Chenxi Wang, Guangxian Ouyang, Zhenhao Chen, Xiuying Chen

**Abstract**: The rise of Large Audio Language Models (LAMs) brings both potential and risks, as their audio outputs may contain harmful or unethical content. However, current research lacks a systematic, quantitative evaluation of LAM safety especially against jailbreak attacks, which are challenging due to the temporal and semantic nature of speech. To bridge this gap, we introduce AJailBench, the first benchmark specifically designed to evaluate jailbreak vulnerabilities in LAMs. We begin by constructing AJailBench-Base, a dataset of 1,495 adversarial audio prompts spanning 10 policy-violating categories, converted from textual jailbreak attacks using realistic text to speech synthesis. Using this dataset, we evaluate several state-of-the-art LAMs and reveal that none exhibit consistent robustness across attacks. To further strengthen jailbreak testing and simulate more realistic attack conditions, we propose a method to generate dynamic adversarial variants. Our Audio Perturbation Toolkit (APT) applies targeted distortions across time, frequency, and amplitude domains. To preserve the original jailbreak intent, we enforce a semantic consistency constraint and employ Bayesian optimization to efficiently search for perturbations that are both subtle and highly effective. This results in AJailBench-APT, an extended dataset of optimized adversarial audio samples. Our findings demonstrate that even small, semantically preserved perturbations can significantly reduce the safety performance of leading LAMs, underscoring the need for more robust and semantically aware defense mechanisms.

摘要: 大型音频语言模型（LAMs）的兴起带来了潜在的风险，因为它们的音频输出可能包含有害或不道德的内容。然而，目前的研究缺乏一个系统的，定量的评估LAM的安全性，特别是对越狱攻击，这是具有挑战性的，由于语音的时间和语义的性质。为了弥补这一差距，我们引入AJailBench，这是第一个专门用于评估LAM中越狱漏洞的基准测试。我们首先构建AJailBench-Base，这是一个包含1，495个对抗性音频提示的数据集，涵盖10个违反策略的类别，从使用真实文本的文本越狱攻击转换为语音合成。使用该数据集，我们评估了几种最先进的LAM，并发现没有一种在攻击中表现出一致的鲁棒性。为了进一步加强越狱测试并模拟更真实的攻击条件，我们提出了一种生成动态对抗变体的方法。我们的音频微扰工具包（APT）在时间、频率和幅度域中应用有针对性的失真。为了保留最初的越狱意图，我们强制执行语义一致性约束并采用Bayesian优化来有效地搜索微妙且高效的扰动。这会产生AJailBench-APT，这是一个优化的对抗性音频样本的扩展数据集。我们的研究结果表明，即使是很小的、在语义上保留的扰动也会显着降低领先LAM的安全性能，这凸显了对更强大和语义感知的防御机制的需求。



## **34. My Face Is Mine, Not Yours: Facial Protection Against Diffusion Model Face Swapping**

我的脸是我的，不是你的：防止扩散模型换脸的面部保护 cs.CV

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15336v1) [paper-pdf](http://arxiv.org/pdf/2505.15336v1)

**Authors**: Hon Ming Yam, Zhongliang Guo, Chun Pong Lau

**Abstract**: The proliferation of diffusion-based deepfake technologies poses significant risks for unauthorized and unethical facial image manipulation. While traditional countermeasures have primarily focused on passive detection methods, this paper introduces a novel proactive defense strategy through adversarial attacks that preemptively protect facial images from being exploited by diffusion-based deepfake systems. Existing adversarial protection methods predominantly target conventional generative architectures (GANs, AEs, VAEs) and fail to address the unique challenges presented by diffusion models, which have become the predominant framework for high-quality facial deepfakes. Current diffusion-specific adversarial approaches are limited by their reliance on specific model architectures and weights, rendering them ineffective against the diverse landscape of diffusion-based deepfake implementations. Additionally, they typically employ global perturbation strategies that inadequately address the region-specific nature of facial manipulation in deepfakes.

摘要: 基于扩散的深度伪造技术的激增给未经授权和不道德的面部图像操纵带来了巨大风险。虽然传统的对抗措施主要集中在被动检测方法上，但本文通过对抗攻击引入了一种新颖的主动防御策略，该策略先发制人地保护面部图像不被基于扩散的Deepfake系统利用。现有的对抗保护方法主要针对传统的生成架构（GAN、AE、VAE），无法解决扩散模型提出的独特挑战，而扩散模型已成为高质量面部深度造假的主要框架。当前特定于扩散的对抗方法因其对特定模型架构和权重的依赖而受到限制，导致它们对基于扩散的Deepfake实现的多样化环境无效。此外，他们通常采用全局扰动策略，无法充分解决深度造假中面部操纵的区域特定性质。



## **35. BadSR: Stealthy Label Backdoor Attacks on Image Super-Resolution**

BadSR：对图像超分辨率的隐形标签后门攻击 cs.CV

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15308v1) [paper-pdf](http://arxiv.org/pdf/2505.15308v1)

**Authors**: Ji Guo, Xiaolei Wen, Wenbo Jiang, Cheng Huang, Jinjin Li, Hongwei Li

**Abstract**: With the widespread application of super-resolution (SR) in various fields, researchers have begun to investigate its security. Previous studies have demonstrated that SR models can also be subjected to backdoor attacks through data poisoning, affecting downstream tasks. A backdoor SR model generates an attacker-predefined target image when given a triggered image while producing a normal high-resolution (HR) output for clean images. However, prior backdoor attacks on SR models have primarily focused on the stealthiness of poisoned low-resolution (LR) images while ignoring the stealthiness of poisoned HR images, making it easy for users to detect anomalous data. To address this problem, we propose BadSR, which improves the stealthiness of poisoned HR images. The key idea of BadSR is to approximate the clean HR image and the pre-defined target image in the feature space while ensuring that modifications to the clean HR image remain within a constrained range. The poisoned HR images generated by BadSR can be integrated with existing triggers. To further improve the effectiveness of BadSR, we design an adversarially optimized trigger and a backdoor gradient-driven poisoned sample selection method based on a genetic algorithm. The experimental results show that BadSR achieves a high attack success rate in various models and data sets, significantly affecting downstream tasks.

摘要: 随着超分辨率（SR）在各个领域的广泛应用，研究人员开始研究其安全性。之前的研究表明，SR模型也可能通过数据中毒而受到后门攻击，从而影响下游任务。后门SR模型在给定触发图像时生成攻击者预定义的目标图像，同时为干净图像生成正常的高分辨率（HR）输出。然而，之前对SR模型的后门攻击主要集中在有毒低分辨率（LR）图像的隐蔽性上，而忽略了有毒HR图像的隐蔽性，使用户很容易检测到异常数据。为了解决这个问题，我们提出了BadSR，它可以改善有毒HR图像的隐蔽性。BadSR的关键思想是在特征空间中逼近干净HR图像和预定义的目标图像，同时确保对干净HR图像的修改保持在约束范围内。BadSR生成的中毒HR图像可以与现有触发器集成。为了进一步提高BadSR的有效性，我们设计了一种对抗优化触发器和基于遗传算法的后门梯度驱动中毒样本选择方法。实验结果表明，BadSR在各种模型和数据集中都达到了很高的攻击成功率，对下游任务产生了显着影响。



## **36. Robustness Evaluation of Graph-based News Detection Using Network Structural Information**

利用网络结构信息进行基于图的新闻检测的鲁棒性评估 cs.SI

Accepted to Proceedings of the ACM SIGKDD Conference on Knowledge  Discovery and Data Mining 2025 (KDD 2025). 14 pages, 7 figures, 10 tables

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.14453v2) [paper-pdf](http://arxiv.org/pdf/2505.14453v2)

**Authors**: Xianghua Zeng, Hao Peng, Angsheng Li

**Abstract**: Although Graph Neural Networks (GNNs) have shown promising potential in fake news detection, they remain highly vulnerable to adversarial manipulations within social networks. Existing methods primarily establish connections between malicious accounts and individual target news to investigate the vulnerability of graph-based detectors, while they neglect the structural relationships surrounding targets, limiting their effectiveness in robustness evaluation. In this work, we propose a novel Structural Information principles-guided Adversarial Attack Framework, namely SI2AF, which effectively challenges graph-based detectors and further probes their detection robustness. Specifically, structural entropy is introduced to quantify the dynamic uncertainty in social engagements and identify hierarchical communities that encompass all user accounts and news posts. An influence metric is presented to measure each account's probability of engaging in random interactions, facilitating the design of multiple agents that manage distinct malicious accounts. For each target news, three attack strategies are developed through multi-agent collaboration within the associated subgraph to optimize evasion against black-box detectors. By incorporating the adversarial manipulations generated by SI2AF, we enrich the original network structure and refine graph-based detectors to improve their robustness against adversarial attacks. Extensive evaluations demonstrate that SI2AF significantly outperforms state-of-the-art baselines in attack effectiveness with an average improvement of 16.71%, and enhances GNN-based detection robustness by 41.54% on average.

摘要: 尽管图形神经网络（GNN）在假新闻检测方面表现出了广阔的潜力，但它们仍然极易受到社交网络内的对抗操纵的影响。现有的方法主要在恶意帐户和单个目标新闻之间建立联系，以调查基于图形的检测器的脆弱性，而它们忽视了目标周围的结构关系，限制了其稳健性评估的有效性。在这项工作中，我们提出了一种新型的结构信息原则指导的对抗攻击框架，即SI 2AF，它有效地挑战了基于图的检测器，并进一步探讨了其检测鲁棒性。具体来说，引入结构性信息来量化社交参与中的动态不确定性，并识别涵盖所有用户帐户和新闻帖子的分层社区。提供了一个影响力指标来衡量每个帐户参与随机交互的可能性，以促进管理不同恶意帐户的多个代理的设计。对于每个目标新闻，通过相关子图内的多代理协作开发三种攻击策略，以优化针对黑匣子检测器的规避。通过结合SI 2AF生成的对抗性操纵，我们丰富了原始的网络结构并改进了基于图的检测器，以提高其对对抗性攻击的鲁棒性。广泛的评估表明，SI 2AF在攻击有效性方面显着优于最先进的基线，平均提高了16.71%，并将基于GNN的检测鲁棒性平均提高了41.54%。



## **37. Blind Spot Navigation: Evolutionary Discovery of Sensitive Semantic Concepts for LVLMs**

盲点导航：LVLM敏感语义概念的进化发现 cs.CV

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15265v1) [paper-pdf](http://arxiv.org/pdf/2505.15265v1)

**Authors**: Zihao Pan, Yu Tong, Weibin Wu, Jingyi Wang, Lifeng Chen, Zhe Zhao, Jiajia Wei, Yitong Qiao, Zibin Zheng

**Abstract**: Adversarial attacks aim to generate malicious inputs that mislead deep models, but beyond causing model failure, they cannot provide certain interpretable information such as ``\textit{What content in inputs make models more likely to fail?}'' However, this information is crucial for researchers to specifically improve model robustness. Recent research suggests that models may be particularly sensitive to certain semantics in visual inputs (such as ``wet,'' ``foggy''), making them prone to errors. Inspired by this, in this paper we conducted the first exploration on large vision-language models (LVLMs) and found that LVLMs indeed are susceptible to hallucinations and various errors when facing specific semantic concepts in images. To efficiently search for these sensitive concepts, we integrated large language models (LLMs) and text-to-image (T2I) models to propose a novel semantic evolution framework. Randomly initialized semantic concepts undergo LLM-based crossover and mutation operations to form image descriptions, which are then converted by T2I models into visual inputs for LVLMs. The task-specific performance of LVLMs on each input is quantified as fitness scores for the involved semantics and serves as reward signals to further guide LLMs in exploring concepts that induce LVLMs. Extensive experiments on seven mainstream LVLMs and two multimodal tasks demonstrate the effectiveness of our method. Additionally, we provide interesting findings about the sensitive semantics of LVLMs, aiming to inspire further in-depth research.

摘要: 对抗性攻击旨在生成误导深度模型的恶意输入，但除了导致模型失败之外，它们无法提供某些可解释的信息，例如'\textit{输入中的哪些内容使模型更有可能失败？}”“然而，这些信息对于研究人员专门提高模型稳健性至关重要。最近的研究表明，模型可能对视觉输入中的某些语义（例如“湿”、“雾”）特别敏感，这使得它们容易出错。受此启发，本文对大型视觉语言模型（LVLM）进行了首次探索，发现LVLM在面对图像中的特定语义概念时确实容易产生幻觉和各种错误。为了有效地搜索这些敏感概念，我们集成了大型语言模型（LLM）和文本到图像（T2 I）模型，提出了一种新颖的语义进化框架。随机初始化的语义概念经过基于LLM的交叉和变异操作以形成图像描述，然后由T2 I模型将其转换为LVLM的视觉输入。LVLM在每个输入上的特定任务性能被量化为所涉及语义的适应度分数，并作为奖励信号，以进一步指导LLM探索引发LVLM的概念。对七种主流LVLM和两种多模式任务的广泛实验证明了我们方法的有效性。此外，我们还提供了有关LVLM敏感语义的有趣发现，旨在激发进一步的深入研究。



## **38. From Words to Collisions: LLM-Guided Evaluation and Adversarial Generation of Safety-Critical Driving Scenarios**

从言语到碰撞：法学硕士指导的评估和安全关键驾驶场景的对抗生成 cs.AI

New version of the paper

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2502.02145v3) [paper-pdf](http://arxiv.org/pdf/2502.02145v3)

**Authors**: Yuan Gao, Mattia Piccinini, Korbinian Moller, Amr Alanwar, Johannes Betz

**Abstract**: Ensuring the safety of autonomous vehicles requires virtual scenario-based testing, which depends on the robust evaluation and generation of safety-critical scenarios. So far, researchers have used scenario-based testing frameworks that rely heavily on handcrafted scenarios as safety metrics. To reduce the effort of human interpretation and overcome the limited scalability of these approaches, we combine Large Language Models (LLMs) with structured scenario parsing and prompt engineering to automatically evaluate and generate safety-critical driving scenarios. We introduce Cartesian and Ego-centric prompt strategies for scenario evaluation, and an adversarial generation module that modifies trajectories of risk-inducing vehicles (ego-attackers) to create critical scenarios. We validate our approach using a 2D simulation framework and multiple pre-trained LLMs. The results show that the evaluation module effectively detects collision scenarios and infers scenario safety. Meanwhile, the new generation module identifies high-risk agents and synthesizes realistic, safety-critical scenarios. We conclude that an LLM equipped with domain-informed prompting techniques can effectively evaluate and generate safety-critical driving scenarios, reducing dependence on handcrafted metrics. We release our open-source code and scenarios at: https://github.com/TUM-AVS/From-Words-to-Collisions.

摘要: 确保自动驾驶汽车的安全需要基于虚拟环境的测试，这取决于安全关键场景的稳健评估和生成。到目前为止，研究人员已经使用基于情景的测试框架，这些框架严重依赖手工制作的场景作为安全指标。为了减少人类解释的工作量并克服这些方法的有限可扩展性，我们将大型语言模型（LLM）与结构化场景解析相结合，并提示工程技术自动评估和生成对安全至关重要的驾驶场景。我们引入了用于场景评估的Cartesian和以自我为中心的提示策略，以及一个对抗生成模块，该模块修改风险诱导车辆（自我攻击者）的轨迹以创建关键场景。我们使用2D仿真框架和多个预先训练的LLM来验证我们的方法。结果表明，该评估模块能够有效地检测碰撞场景，并推断出场景安全性.与此同时，新一代模块识别高风险代理并综合现实的安全关键场景。我们的结论是，LLM配备域知情的提示技术可以有效地评估和生成安全关键的驾驶场景，减少依赖手工制作的指标。我们在https://github.com/TUM-AVS/From-Words-to-Collisions上发布我们的开源代码和场景。



## **39. Reliable Disentanglement Multi-view Learning Against View Adversarial Attacks**

可靠的解纠缠多视图学习对抗视图对抗攻击 cs.LG

11 pages, 11 figures, accepted by IJCAI 2025

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.04046v2) [paper-pdf](http://arxiv.org/pdf/2505.04046v2)

**Authors**: Xuyang Wang, Siyuan Duan, Qizhi Li, Guiduo Duan, Yuan Sun, Dezhong Peng

**Abstract**: Trustworthy multi-view learning has attracted extensive attention because evidence learning can provide reliable uncertainty estimation to enhance the credibility of multi-view predictions. Existing trusted multi-view learning methods implicitly assume that multi-view data is secure. However, in safety-sensitive applications such as autonomous driving and security monitoring, multi-view data often faces threats from adversarial perturbations, thereby deceiving or disrupting multi-view models. This inevitably leads to the adversarial unreliability problem (AUP) in trusted multi-view learning. To overcome this tricky problem, we propose a novel multi-view learning framework, namely Reliable Disentanglement Multi-view Learning (RDML). Specifically, we first propose evidential disentanglement learning to decompose each view into clean and adversarial parts under the guidance of corresponding evidences, which is extracted by a pretrained evidence extractor. Then, we employ the feature recalibration module to mitigate the negative impact of adversarial perturbations and extract potential informative features from them. Finally, to further ignore the irreparable adversarial interferences, a view-level evidential attention mechanism is designed. Extensive experiments on multi-view classification tasks with adversarial attacks show that RDML outperforms the state-of-the-art methods by a relatively large margin. Our code is available at https://github.com/Willy1005/2025-IJCAI-RDML.

摘要: 值得信赖的多视图学习引起了广泛关注，因为证据学习可以提供可靠的不确定性估计，以增强多视图预测的可信度。现有的可信多视图学习方法隐含地假设多视图数据是安全的。然而，在自动驾驶和安全监控等安全敏感应用中，多视图数据经常面临来自敌对扰动的威胁，从而欺骗或破坏多视图模型。这不可避免地会导致可信多视图学习中的对抗不可靠性问题（AUP）。为了克服这个棘手的问题，我们提出了一种新颖的多视图学习框架，即可靠解纠缠多视图学习（RDML）。具体来说，我们首先提出证据解纠缠学习，在相应证据的指导下将每个视图分解为干净且对抗的部分，这些证据由预先训练的证据提取器提取。然后，我们使用特征重新校准模块来减轻对抗性扰动的负面影响，并从中提取潜在的信息特征。最后，为了进一步忽略不可挽回的对抗干扰，设计了视角级证据关注机制。针对具有对抗性攻击的多视图分类任务的大量实验表明，RDML的性能优于最先进的方法，相对较大。我们的代码可在https://github.com/Willy1005/2025-IJCAI-RDML上获取。



## **40. Few-Shot Adversarial Low-Rank Fine-Tuning of Vision-Language Models**

视觉语言模型的少镜头对抗低级微调 cs.LG

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15130v1) [paper-pdf](http://arxiv.org/pdf/2505.15130v1)

**Authors**: Sajjad Ghiasvand, Haniyeh Ehsani Oskouie, Mahnoosh Alizadeh, Ramtin Pedarsani

**Abstract**: Vision-Language Models (VLMs) such as CLIP have shown remarkable performance in cross-modal tasks through large-scale contrastive pre-training. To adapt these large transformer-based models efficiently for downstream tasks, Parameter-Efficient Fine-Tuning (PEFT) techniques like LoRA have emerged as scalable alternatives to full fine-tuning, especially in few-shot scenarios. However, like traditional deep neural networks, VLMs are highly vulnerable to adversarial attacks, where imperceptible perturbations can significantly degrade model performance. Adversarial training remains the most effective strategy for improving model robustness in PEFT. In this work, we propose AdvCLIP-LoRA, the first algorithm designed to enhance the adversarial robustness of CLIP models fine-tuned with LoRA in few-shot settings. Our method formulates adversarial fine-tuning as a minimax optimization problem and provides theoretical guarantees for convergence under smoothness and nonconvex-strong-concavity assumptions. Empirical results across eight datasets using ViT-B/16 and ViT-B/32 models show that AdvCLIP-LoRA significantly improves robustness against common adversarial attacks (e.g., FGSM, PGD), without sacrificing much clean accuracy. These findings highlight AdvCLIP-LoRA as a practical and theoretically grounded approach for robust adaptation of VLMs in resource-constrained settings.

摘要: 通过大规模对比预训练，CLIP等视觉语言模型（VLM）在跨模式任务中表现出了出色的表现。为了有效地调整这些基于变压器的大型模型以适应下游任务，LoRA等参数高效微调（PEFT）技术已成为完全微调的可扩展替代方案，尤其是在少量场景中。然而，与传统的深度神经网络一样，VLM非常容易受到对抗攻击，其中不可感知的扰动可能会显着降低模型性能。对抗训练仍然是提高PEFT模型稳健性的最有效策略。在这项工作中，我们提出了AdvCLIP-LoRA，这是第一个旨在增强在少数镜头设置中使用LoRA微调的CLIP模型的对抗鲁棒性的算法。我们的方法将对抗性微调表述为极小极大优化问题，并为光滑性和非凸强插值假设下的收敛提供理论保证。使用ViT-B/16和ViT-B/32模型的八个数据集的经验结果表明，AdvCLIP-LoRA显着提高了针对常见对抗攻击（例如，FGSM、PVD），而不会牺牲太多干净的准确性。这些发现凸显了AdvCLIP-LoRA是一种实用且具有理论依据的方法，用于在资源有限的环境中稳健地适应VLM。



## **41. A Survey On Secure Machine Learning**

关于安全机器学习的调查 cs.CR

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.15124v1) [paper-pdf](http://arxiv.org/pdf/2505.15124v1)

**Authors**: Taobo Liao, Taoran Li, Prathamesh Nadkarni

**Abstract**: In this survey, we will explore the interaction between secure multiparty computation and the area of machine learning. Recent advances in secure multiparty computation (MPC) have significantly improved its applicability in the realm of machine learning (ML), offering robust solutions for privacy-preserving collaborative learning. This review explores key contributions that leverage MPC to enable multiple parties to engage in ML tasks without compromising the privacy of their data. The integration of MPC with ML frameworks facilitates the training and evaluation of models on combined datasets from various sources, ensuring that sensitive information remains encrypted throughout the process. Innovations such as specialized software frameworks and domain-specific languages streamline the adoption of MPC in ML, optimizing performance and broadening its usage. These frameworks address both semi-honest and malicious threat models, incorporating features such as automated optimizations and cryptographic auditing to ensure compliance and data integrity. The collective insights from these studies highlight MPC's potential in fostering collaborative yet confidential data analysis, marking a significant stride towards the realization of secure and efficient computational solutions in privacy-sensitive industries. This paper investigates a spectrum of SecureML libraries that includes cryptographic protocols, federated learning frameworks, and privacy-preserving algorithms. By surveying the existing literature, this paper aims to examine the efficacy of these libraries in preserving data privacy, ensuring model confidentiality, and fortifying ML systems against adversarial attacks. Additionally, the study explores an innovative application domain for SecureML techniques: the integration of these methodologies in gaming environments utilizing ML.

摘要: 在本次调查中，我们将探索安全多方计算与机器学习领域之间的互动。安全多方计算（MPC）的最新进展显着提高了其在机器学习（ML）领域的适用性，为保护隐私的协作学习提供了稳健的解决方案。本评论探讨了利用MPC使多方能够在不损害其数据隐私的情况下参与ML任务的关键贡献。MPC与ML框架的集成促进了对来自不同来源的组合数据集进行模型的训练和评估，确保敏感信息在整个过程中保持加密。专业软件框架和特定领域语言等创新简化了ML中MPC的采用，优化了性能并扩大了其用途。这些框架可解决半诚实和恶意威胁模型，并结合自动优化和加密审计等功能，以确保合规性和数据完整性。这些研究的集体见解凸显了MPC在促进协作但保密的数据分析方面的潜力，标志着在隐私敏感行业中实现安全高效的计算解决方案方面迈出了重要一步。本文研究了一系列SecureML库，包括加密协议、联邦学习框架和隐私保护算法。通过调查现有文献，本文旨在检查这些库在保护数据隐私、确保模型机密性以及加强ML系统抵御对抗攻击方面的功效。此外，该研究还探索了SecureML技术的创新应用领域：在利用ML的游戏环境中集成这些方法。



## **42. AudioJailbreak: Jailbreak Attacks against End-to-End Large Audio-Language Models**

AudioJailbreak：针对端到端大型音频语言模型的越狱攻击 cs.CR

**SubmitDate**: 2025-05-21    [abs](http://arxiv.org/abs/2505.14103v2) [paper-pdf](http://arxiv.org/pdf/2505.14103v2)

**Authors**: Guangke Chen, Fu Song, Zhe Zhao, Xiaojun Jia, Yang Liu, Yanchen Qiao, Weizhe Zhang

**Abstract**: Jailbreak attacks to Large audio-language models (LALMs) are studied recently, but they achieve suboptimal effectiveness, applicability, and practicability, particularly, assuming that the adversary can fully manipulate user prompts. In this work, we first conduct an extensive experiment showing that advanced text jailbreak attacks cannot be easily ported to end-to-end LALMs via text-to speech (TTS) techniques. We then propose AudioJailbreak, a novel audio jailbreak attack, featuring (1) asynchrony: the jailbreak audio does not need to align with user prompts in the time axis by crafting suffixal jailbreak audios; (2) universality: a single jailbreak perturbation is effective for different prompts by incorporating multiple prompts into perturbation generation; (3) stealthiness: the malicious intent of jailbreak audios will not raise the awareness of victims by proposing various intent concealment strategies; and (4) over-the-air robustness: the jailbreak audios remain effective when being played over the air by incorporating the reverberation distortion effect with room impulse response into the generation of the perturbations. In contrast, all prior audio jailbreak attacks cannot offer asynchrony, universality, stealthiness, or over-the-air robustness. Moreover, AudioJailbreak is also applicable to the adversary who cannot fully manipulate user prompts, thus has a much broader attack scenario. Extensive experiments with thus far the most LALMs demonstrate the high effectiveness of AudioJailbreak. We highlight that our work peeks into the security implications of audio jailbreak attacks against LALMs, and realistically fosters improving their security robustness. The implementation and audio samples are available at our website https://audiojailbreak.github.io/AudioJailbreak.

摘要: 最近研究了对大型音频语言模型（LALM）的越狱攻击，但它们的有效性、适用性和实用性达到了次优，特别是假设对手可以完全操纵用户提示。在这项工作中，我们首先进行了一项广泛的实验，表明高级文本越狱攻击无法通过文本转语音（TTC）技术轻松移植到端到端LALM。然后，我们提出AudioJailbreak，一种新颖的音频越狱攻击，其特点是：（1）狡猾：越狱音频不需要通过制作后缀的越狱音频在时间轴上与用户提示对齐;（2）通用性：通过将多个提示合并到扰动生成中，单个越狱扰动对不同的提示有效;（3）隐蔽性：越狱音频的恶意意图不会通过提出各种意图隐藏策略来提高受害者的意识;以及（4）空中鲁棒性：越狱音频在空中播放时仍然有效，通过将回响失真效应与房间脉冲响应结合起来扰动的产生。相比之下，所有先前的音频越狱攻击都无法提供灵活性、普遍性、隐蔽性或空中鲁棒性。此外，AudioJailbreak还适用于无法完全操纵用户提示的对手，因此具有更广泛的攻击场景。迄今为止，对大多数LALM的广泛实验证明了AudioJailbreak的高有效性。我们强调，我们的工作探讨了针对LALM的音频越狱攻击的安全影响，并切实促进了其安全稳健性的提高。实现和音频示例可在我们的网站https://audiojailbreak.github.io/AudioJailbreak上获取。



## **43. MrGuard: A Multilingual Reasoning Guardrail for Universal LLM Safety**

MrGuard：通用LLM安全的多语言推理保障 cs.CL

Preprint

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2504.15241v2) [paper-pdf](http://arxiv.org/pdf/2504.15241v2)

**Authors**: Yahan Yang, Soham Dan, Shuo Li, Dan Roth, Insup Lee

**Abstract**: Large Language Models (LLMs) are susceptible to adversarial attacks such as jailbreaking, which can elicit harmful or unsafe behaviors. This vulnerability is exacerbated in multilingual settings, where multilingual safety-aligned data is often limited. Thus, developing a guardrail capable of detecting and filtering unsafe content across diverse languages is critical for deploying LLMs in real-world applications. In this work, we introduce a multilingual guardrail with reasoning for prompt classification. Our method consists of: (1) synthetic multilingual data generation incorporating culturally and linguistically nuanced variants, (2) supervised fine-tuning, and (3) a curriculum-based Group Relative Policy Optimization (GRPO) framework that further improves performance. Experimental results demonstrate that our multilingual guardrail, MrGuard, consistently outperforms recent baselines across both in-domain and out-of-domain languages by more than 15%. We also evaluate MrGuard's robustness to multilingual variations, such as code-switching and low-resource language distractors in the prompt, and demonstrate that it preserves safety judgments under these challenging conditions. The multilingual reasoning capability of our guardrail enables it to generate explanations, which are particularly useful for understanding language-specific risks and ambiguities in multilingual content moderation.

摘要: 大型语言模型（LLM）容易受到诸如越狱之类的对抗性攻击，这可能会引发有害或不安全的行为。这种漏洞在多语言环境中会加剧，其中多语言安全一致的数据通常是有限的。因此，开发一个能够检测和过滤不同语言的不安全内容的护栏对于在现实世界的应用程序中部署LLM至关重要。在这项工作中，我们介绍了一种多语言护栏，具有快速分类的推理。我们的方法包括：（1）综合多语言数据生成，融合了文化和语言上的细微差别，（2）监督式微调，以及（3）进一步提高性能的基于课程的组相对政策优化（GRPO）框架。实验结果表明，我们的多语言护栏MrGuard在域内和域外语言中的表现始终优于最近的基线15%以上。我们还评估了MrGuard对多语言变体（例如提示中的代码切换和低资源语言干扰因素）的稳健性，并证明它在这些具有挑战性的条件下保留了安全判断。我们护栏的多语言推理能力使其能够生成解释，这对于理解多语言内容审核中的特定语言风险和歧义特别有用。



## **44. Lessons from Defending Gemini Against Indirect Prompt Injections**

预防双子座间接即时注射的教训 cs.CR

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2505.14534v1) [paper-pdf](http://arxiv.org/pdf/2505.14534v1)

**Authors**: Chongyang Shi, Sharon Lin, Shuang Song, Jamie Hayes, Ilia Shumailov, Itay Yona, Juliette Pluto, Aneesh Pappu, Christopher A. Choquette-Choo, Milad Nasr, Chawin Sitawarin, Gena Gibson, Andreas Terzis, John "Four" Flynn

**Abstract**: Gemini is increasingly used to perform tasks on behalf of users, where function-calling and tool-use capabilities enable the model to access user data. Some tools, however, require access to untrusted data introducing risk. Adversaries can embed malicious instructions in untrusted data which cause the model to deviate from the user's expectations and mishandle their data or permissions. In this report, we set out Google DeepMind's approach to evaluating the adversarial robustness of Gemini models and describe the main lessons learned from the process. We test how Gemini performs against a sophisticated adversary through an adversarial evaluation framework, which deploys a suite of adaptive attack techniques to run continuously against past, current, and future versions of Gemini. We describe how these ongoing evaluations directly help make Gemini more resilient against manipulation.

摘要: Gemini越来越多地用于代表用户执行任务，其中功能调用和工具使用功能使模型能够访问用户数据。然而，有些工具需要访问不受信任的数据，从而带来风险。对手可能会在不受信任的数据中嵌入恶意指令，导致模型偏离用户的期望并不当处理他们的数据或权限。在本报告中，我们阐述了Google DeepMind评估Gemini模型对抗稳健性的方法，并描述了从该过程中吸取的主要教训。我们通过对抗性评估框架测试Gemini如何对抗复杂对手，该框架部署了一套自适应攻击技术，以针对Gemini的过去、当前和未来版本持续运行。我们描述了这些正在进行的评估如何直接帮助双子座对操纵更具弹性。



## **45. Adverseness vs. Equilibrium: Exploring Graph Adversarial Resilience through Dynamic Equilibrium**

紧张与平衡：通过动态平衡探索图的对抗弹性 cs.LG

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2505.14463v1) [paper-pdf](http://arxiv.org/pdf/2505.14463v1)

**Authors**: Xinxin Fan, Wenxiong Chen, Mengfan Li, Wenqi Wei, Ling Liu

**Abstract**: Adversarial attacks to graph analytics are gaining increased attention. To date, two lines of countermeasures have been proposed to resist various graph adversarial attacks from the perspectives of either graph per se or graph neural networks. Nevertheless, a fundamental question lies in whether there exists an intrinsic adversarial resilience state within a graph regime and how to find out such a critical state if exists. This paper contributes to tackle the above research questions from three unique perspectives: i) we regard the process of adversarial learning on graph as a complex multi-object dynamic system, and model the behavior of adversarial attack; ii) we propose a generalized theoretical framework to show the existence of critical adversarial resilience state; and iii) we develop a condensed one-dimensional function to capture the dynamic variation of graph regime under perturbations, and pinpoint the critical state through solving the equilibrium point of dynamic system. Multi-facet experiments are conducted to show our proposed approach can significantly outperform the state-of-the-art defense methods under five commonly-used real-world datasets and three representative attacks.

摘要: 对图形分析的对抗攻击越来越受到关注。迄今为止，已经从图本身或图神经网络的角度提出了两系列对策来抵抗各种图对抗攻击。然而，一个根本问题在于，图机制内是否存在固有的对抗弹性状态，以及如何找出这样的临界状态（如果存在）。本文从三个独特的角度探讨了上述研究问题：i）我们将图上的对抗性学习过程视为一个复杂的多对象动态系统，并对对抗性攻击的行为进行建模; ii）我们提出了一个广义的理论框架来表明关键对抗弹性状态的存在;以及iii）我们开发了一个压缩的一维函数来捕捉图形区域在扰动下的动态变化，并通过求解动力系统的平衡点来确定临界状态。进行了多方面实验，表明我们提出的方法在五个常用的现实世界数据集和三种代表性攻击下可以显着优于最先进的防御方法。



## **46. Vulnerability of Transfer-Learned Neural Networks to Data Reconstruction Attacks in Small-Data Regime**

小数据环境下迁移学习神经网络的数据重构脆弱性 cs.CR

27 pages

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2505.14323v1) [paper-pdf](http://arxiv.org/pdf/2505.14323v1)

**Authors**: Tomasz Maciążek, Robert Allison

**Abstract**: Training data reconstruction attacks enable adversaries to recover portions of a released model's training data. We consider the attacks where a reconstructor neural network learns to invert the (random) mapping between training data and model weights. Prior work has shown that an informed adversary with access to released model's weights and all but one training data point can achieve high-quality reconstructions in this way. However, differential privacy can defend against such an attack with little to no loss in model's utility when the amount of training data is sufficiently large. In this work we consider a more realistic adversary who only knows the distribution from which a small training dataset has been sampled and who attacks a transfer-learned neural network classifier that has been trained on this dataset. We exhibit an attack that works in this realistic threat model and demonstrate that in the small-data regime it cannot be defended against by DP-SGD without severely damaging the classifier accuracy. This raises significant concerns about the use of such transfer-learned classifiers when protection of training-data is paramount. We demonstrate the effectiveness and robustness of our attack on VGG, EfficientNet and ResNet image classifiers transfer-learned on MNIST, CIFAR-10 and CelebA respectively. Additionally, we point out that the commonly used (true-positive) reconstruction success rate metric fails to reliably quantify the actual reconstruction effectiveness. Instead, we make use of the Neyman-Pearson lemma to construct the receiver operating characteristic curve and consider the associated true-positive reconstruction rate at a fixed level of the false-positive reconstruction rate.

摘要: 训练数据重建攻击使对手能够恢复已发布模型的部分训练数据。我们考虑了重建器神经网络学习倒置训练数据和模型权重之间的（随机）映射的攻击。之前的工作表明，能够访问已发布模型的权重和除一个训练数据点之外的所有训练数据点的知情对手可以通过这种方式实现高质量的重建。然而，当训练数据量足够大时，差异隐私可以抵御此类攻击，而模型的效用几乎没有损失。在这项工作中，我们考虑了一个更现实的对手，他只知道对小训练数据集进行抽样的分布，并攻击在此数据集上训练的转移学习神经网络分类器。我们展示了一种适用于这个现实威胁模型的攻击，并证明在小数据机制中，DP-BCD无法在不严重损害分类器准确性的情况下抵御它。当训练数据的保护至关重要时，这引发了人们对此类转移学习分类器的使用的严重担忧。我们证明了我们对分别在MNIST、CIFAR-10和CelebA上转移学习的VGG、EfficientNet和ResNet图像分类器的攻击的有效性和鲁棒性。此外，我们指出，常用的（真阳性）重建成功率指标无法可靠地量化实际重建的有效性。相反，我们利用内曼-皮尔逊引理来构建接收器工作特征曲线，并考虑在假阳性重建率的固定水平下的相关真阳性重建率。



## **47. Exploring Jailbreak Attacks on LLMs through Intent Concealment and Diversion**

探索通过意图隐瞒和转移对LLM的越狱攻击 cs.CR

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2505.14316v1) [paper-pdf](http://arxiv.org/pdf/2505.14316v1)

**Authors**: Tiehan Cui, Yanxu Mao, Peipei Liu, Congying Liu, Datao You

**Abstract**: Although large language models (LLMs) have achieved remarkable advancements, their security remains a pressing concern. One major threat is jailbreak attacks, where adversarial prompts bypass model safeguards to generate harmful or objectionable content. Researchers study jailbreak attacks to understand security and robustness of LLMs. However, existing jailbreak attack methods face two main challenges: (1) an excessive number of iterative queries, and (2) poor generalization across models. In addition, recent jailbreak evaluation datasets focus primarily on question-answering scenarios, lacking attention to text generation tasks that require accurate regeneration of toxic content. To tackle these challenges, we propose two contributions: (1) ICE, a novel black-box jailbreak method that employs Intent Concealment and divErsion to effectively circumvent security constraints. ICE achieves high attack success rates (ASR) with a single query, significantly improving efficiency and transferability across different models. (2) BiSceneEval, a comprehensive dataset designed for assessing LLM robustness in question-answering and text-generation tasks. Experimental results demonstrate that ICE outperforms existing jailbreak techniques, revealing critical vulnerabilities in current defense mechanisms. Our findings underscore the necessity of a hybrid security strategy that integrates predefined security mechanisms with real-time semantic decomposition to enhance the security of LLMs.

摘要: 尽管大型语言模型（LLM）取得了显着的进步，但其安全性仍然是一个紧迫的问题。一个主要威胁是越狱攻击，其中敌对性会促使绕过模型保护措施来生成有害或令人反感的内容。研究人员研究越狱攻击以了解LLM的安全性和稳健性。然而，现有的越狱攻击方法面临两个主要挑战：（1）迭代查询数量过多，（2）模型之间的概括性较差。此外，最近的越狱评估数据集主要关注问答场景，缺乏对需要准确再生有毒内容的文本生成任务的关注。为了应对这些挑战，我们提出了两个贡献：（1）ICE，一种新的黑盒越狱方法，采用意图隐藏和分裂，以有效地规避安全约束。ICE通过单个查询实现了高攻击成功率（ASR），显著提高了效率和跨不同模型的可移植性。(2)BiSceneEval是一个综合数据集，旨在评估LLM在问答和文本生成任务中的鲁棒性。实验结果表明，ICE优于现有的越狱技术，揭示了当前防御机制中的关键漏洞。我们的研究结果强调了混合安全策略的必要性，该策略将预定义的安全机制与实时语义分解集成在一起，以增强LLM的安全性。



## **48. EVA: Red-Teaming GUI Agents via Evolving Indirect Prompt Injection**

伊娃：通过不断发展的间接提示注入进行红色团队化图形用户界面代理 cs.AI

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2505.14289v1) [paper-pdf](http://arxiv.org/pdf/2505.14289v1)

**Authors**: Yijie Lu, Tianjie Ju, Manman Zhao, Xinbei Ma, Yuan Guo, ZhuoSheng Zhang

**Abstract**: As multimodal agents are increasingly trained to operate graphical user interfaces (GUIs) to complete user tasks, they face a growing threat from indirect prompt injection, attacks in which misleading instructions are embedded into the agent's visual environment, such as popups or chat messages, and misinterpreted as part of the intended task. A typical example is environmental injection, in which GUI elements are manipulated to influence agent behavior without directly modifying the user prompt. To address these emerging attacks, we propose EVA, a red teaming framework for indirect prompt injection which transforms the attack into a closed loop optimization by continuously monitoring an agent's attention distribution over the GUI and updating adversarial cues, keywords, phrasing, and layout, in response. Compared with prior one shot methods that generate fixed prompts without regard for how the model allocates visual attention, EVA dynamically adapts to emerging attention hotspots, yielding substantially higher attack success rates and far greater transferability across diverse GUI scenarios. We evaluate EVA on six widely used generalist and specialist GUI agents in realistic settings such as popup manipulation, chat based phishing, payments, and email composition. Experimental results show that EVA substantially improves success rates over static baselines. Under goal agnostic constraints, where the attacker does not know the agent's task intent, EVA still discovers effective patterns. Notably, we find that injection styles transfer well across models, revealing shared behavioral biases in GUI agents. These results suggest that evolving indirect prompt injection is a powerful tool not only for red teaming agents, but also for uncovering common vulnerabilities in their multimodal decision making.

摘要: 随着多模式代理越来越多地接受操作图形用户界面（GUIs）来完成用户任务的培训，它们面临着来自间接提示注入的越来越大的威胁，即将误导性指令嵌入到代理的视觉环境中的攻击，例如弹出窗口或聊天消息，并被误解为预期任务的一部分。一个典型的例子是环境注入，其中操纵图形用户界面元素来影响代理行为，而无需直接修改用户提示。为了解决这些新出现的攻击，我们提出了伊娃，这是一个用于间接提示注入的红色团队框架，它通过持续监视代理在图形用户界面上的注意力分布并更新对抗线索、关键词、措辞和布局来将攻击转化为闭环优化。作为响应。与之前的一次性方法（无需考虑模型如何分配视觉注意力）相比，伊娃动态适应新出现的注意力热点，从而产生更高的攻击成功率和更大的跨不同图形用户界面场景的可移植性。我们在现实环境中对六个广泛使用的通才和专业图形用户界面代理进行了评估，例如弹出窗口操作、基于聊天的网络钓鱼、支付和电子邮件合成。实验结果表明，与静态基线相比，伊娃大幅提高了成功率。在目标不可知约束下，攻击者不知道代理的任务意图，伊娃仍然会发现有效的模式。值得注意的是，我们发现注入风格在模型之间很好地转移，揭示了图形用户界面代理中的共同行为偏差。这些结果表明，不断发展的间接提示注入不仅是红色团队代理的强大工具，而且还可以发现其多模式决策中的常见漏洞。



## **49. Universal Acoustic Adversarial Attacks for Flexible Control of Speech-LLMs**

语音灵活控制的通用声学对抗攻击-LLM cs.CL

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2505.14286v1) [paper-pdf](http://arxiv.org/pdf/2505.14286v1)

**Authors**: Rao Ma, Mengjie Qian, Vyas Raina, Mark Gales, Kate Knill

**Abstract**: The combination of pre-trained speech encoders with large language models has enabled the development of speech LLMs that can handle a wide range of spoken language processing tasks. While these models are powerful and flexible, this very flexibility may make them more vulnerable to adversarial attacks. To examine the extent of this problem, in this work we investigate universal acoustic adversarial attacks on speech LLMs. Here a fixed, universal, adversarial audio segment is prepended to the original input audio. We initially investigate attacks that cause the model to either produce no output or to perform a modified task overriding the original prompt. We then extend the nature of the attack to be selective so that it activates only when specific input attributes, such as a speaker gender or spoken language, are present. Inputs without the targeted attribute should be unaffected, allowing fine-grained control over the model outputs. Our findings reveal critical vulnerabilities in Qwen2-Audio and Granite-Speech and suggest that similar speech LLMs may be susceptible to universal adversarial attacks. This highlights the need for more robust training strategies and improved resistance to adversarial attacks.

摘要: 预训练的语音编码器与大型语言模型的结合使得能够开发出可以处理广泛口语处理任务的语音LLM。虽然这些模型强大且灵活，但这种灵活性可能使它们更容易受到对抗攻击。为了研究这个问题的严重程度，在这项工作中，我们研究了对语音LLM的普遍声学对抗攻击。这里，固定的、通用的、对抗性的音频段被预先添加到原始输入音频上。我们最初调查导致模型不产生输出或执行覆盖原始提示的修改任务的攻击。然后，我们将攻击的性质扩展为选择性，以便只有在特定输入属性（例如说话者性别或口语）存在时，它才会激活。没有目标属性的输入应该不受影响，允许对模型输出进行细粒度控制。我们的研究结果揭示了Qwen 2-Audio和Granite-Speech中的关键漏洞，并表明类似的语音LLM可能容易受到普遍对抗攻击。这凸显了需要更强大的训练策略和提高对对抗性攻击的抵抗力。



## **50. IP Leakage Attacks Targeting LLM-Based Multi-Agent Systems**

针对基于LLM的多代理系统的IP泄露攻击 cs.CR

**SubmitDate**: 2025-05-20    [abs](http://arxiv.org/abs/2505.12442v2) [paper-pdf](http://arxiv.org/pdf/2505.12442v2)

**Authors**: Liwen Wang, Wenxuan Wang, Shuai Wang, Zongjie Li, Zhenlan Ji, Zongyi Lyu, Daoyuan Wu, Shing-Chi Cheung

**Abstract**: The rapid advancement of Large Language Models (LLMs) has led to the emergence of Multi-Agent Systems (MAS) to perform complex tasks through collaboration. However, the intricate nature of MAS, including their architecture and agent interactions, raises significant concerns regarding intellectual property (IP) protection. In this paper, we introduce MASLEAK, a novel attack framework designed to extract sensitive information from MAS applications. MASLEAK targets a practical, black-box setting, where the adversary has no prior knowledge of the MAS architecture or agent configurations. The adversary can only interact with the MAS through its public API, submitting attack query $q$ and observing outputs from the final agent. Inspired by how computer worms propagate and infect vulnerable network hosts, MASLEAK carefully crafts adversarial query $q$ to elicit, propagate, and retain responses from each MAS agent that reveal a full set of proprietary components, including the number of agents, system topology, system prompts, task instructions, and tool usages. We construct the first synthetic dataset of MAS applications with 810 applications and also evaluate MASLEAK against real-world MAS applications, including Coze and CrewAI. MASLEAK achieves high accuracy in extracting MAS IP, with an average attack success rate of 87% for system prompts and task instructions, and 92% for system architecture in most cases. We conclude by discussing the implications of our findings and the potential defenses.

摘要: 大型语言模型（LLM）的快速发展导致了通过协作执行复杂任务的多智能体系统（MAS）的出现。然而，MAS的复杂性质，包括其架构和代理交互，引发了有关知识产权（IP）保护的严重担忧。本文介绍MASLEAK，这是一种新型攻击框架，旨在从MAS应用程序中提取敏感信息。MASLEAK针对的是实用的黑匣子设置，其中对手不了解MAS架构或代理配置。对手只能通过其公共API与MAS交互，提交攻击查询$q$并观察最终代理的输出。受计算机蠕虫传播和感染脆弱网络主机的方式的启发，MASLEAK精心设计了对抗性查询$q$，以引发、传播和保留每个MAS代理的响应，这些响应揭示了全套专有组件，包括代理数量、系统布局、系统提示、任务指令和工具使用。我们构建了包含810个应用程序的第一个MAS应用程序合成数据集，并根据现实世界的MAS应用程序（包括Coze和CrewAI）评估MASLEAK。MASLEAK在提取MAS IP方面实现了高准确性，系统提示和任务指令的平均攻击成功率为87%，大多数情况下系统架构的平均攻击成功率为92%。最后，我们讨论了我们发现的影响和潜在的防御措施。



