# Latest Adversarial Attack Papers
**update at 2025-07-01 11:02:59**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. SQUASH: A SWAP-Based Quantum Attack to Sabotage Hybrid Quantum Neural Networks**

SQUASH：一种基于交换的量子攻击，旨在破坏混合量子神经网络 quant-ph

Keywords: Quantum Machine Learning, Hybrid Quantum Neural Networks,  SWAP Test, Fidelity, Circuit-level Attack

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24081v1) [paper-pdf](http://arxiv.org/pdf/2506.24081v1)

**Authors**: Rahul Kumar, Wenqi Wei, Ying Mao, Junaid Farooq, Ying Wang, Juntao Chen

**Abstract**: We propose a circuit-level attack, SQUASH, a SWAP-Based Quantum Attack to sabotage Hybrid Quantum Neural Networks (HQNNs) for classification tasks. SQUASH is executed by inserting SWAP gate(s) into the variational quantum circuit of the victim HQNN. Unlike conventional noise-based or adversarial input attacks, SQUASH directly manipulates the circuit structure, leading to qubit misalignment and disrupting quantum state evolution. This attack is highly stealthy, as it does not require access to training data or introduce detectable perturbations in input states. Our results demonstrate that SQUASH significantly degrades classification performance, with untargeted SWAP attacks reducing accuracy by up to 74.08\% and targeted SWAP attacks reducing target class accuracy by up to 79.78\%. These findings reveal a critical vulnerability in HQNN implementations, underscoring the need for more resilient architectures against circuit-level adversarial interventions.

摘要: 我们提出了一种电路级攻击SQUASH，一种基于SWAP的量子攻击，以破坏用于分类任务的混合量子神经网络（HQNN）。SQUASH通过将SWAP门插入到受害者HQNN的变分量子电路中来执行。与传统的基于噪声或对抗性输入攻击不同，SQUASH直接操纵电路结构，导致量子比特错位并破坏量子态演化。这种攻击是高度隐蔽的，因为它不需要访问训练数据或在输入状态中引入可检测的扰动。我们的结果表明，SQUASH会显着降低分类性能，非目标SWAP攻击将准确性降低高达74.08%，而目标SWAP攻击将目标类准确性降低高达79.78%。这些发现揭示了HQNN实现中的一个关键漏洞，强调了对电路级对抗干预的更具弹性的架构的需要。



## **2. STACK: Adversarial Attacks on LLM Safeguard Pipelines**

STACK：对LLM Safeguard Pipelines的对抗性攻击 cs.CL

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24068v1) [paper-pdf](http://arxiv.org/pdf/2506.24068v1)

**Authors**: Ian R. McKenzie, Oskar J. Hollinsworth, Tom Tseng, Xander Davies, Stephen Casper, Aaron D. Tucker, Robert Kirk, Adam Gleave

**Abstract**: Frontier AI developers are relying on layers of safeguards to protect against catastrophic misuse of AI systems. Anthropic guards their latest Claude 4 Opus model using one such defense pipeline, and other frontier developers including Google DeepMind and OpenAI pledge to soon deploy similar defenses. However, the security of such pipelines is unclear, with limited prior work evaluating or attacking these pipelines. We address this gap by developing and red-teaming an open-source defense pipeline. First, we find that a novel few-shot-prompted input and output classifier outperforms state-of-the-art open-weight safeguard model ShieldGemma across three attacks and two datasets, reducing the attack success rate (ASR) to 0% on the catastrophic misuse dataset ClearHarm. Second, we introduce a STaged AttaCK (STACK) procedure that achieves 71% ASR on ClearHarm in a black-box attack against the few-shot-prompted classifier pipeline. Finally, we also evaluate STACK in a transfer setting, achieving 33% ASR, providing initial evidence that it is feasible to design attacks with no access to the target pipeline. We conclude by suggesting specific mitigations that developers could use to thwart staged attacks.

摘要: 前沿人工智能开发人员依靠多层保障措施来防止人工智能系统的灾难性滥用。Anthropic使用这样的防御管道来保护他们最新的Claude 4 Opus模型，包括Google DeepMind和OpenAI在内的其他前沿开发商承诺很快部署类似的防御。然而，此类管道的安全性尚不清楚，之前评估或攻击这些管道的工作有限。我们通过开发和组建开源防御管道来解决这一差距。首先，我们发现一种新型的几次激发输入和输出分类器在三次攻击和两个数据集中优于最先进的开权保护模型ShieldGemma，将灾难性滥用数据集ClearHarm的攻击成功率（ASO）降低至0%。其次，我们引入了一个STaged AttaCK（STACK）过程，该过程在ClearHarm上实现了71%的ASB，针对少量镜头提示的分类器管道进行黑匣子攻击。最后，我们还在传输环境中评估了STACK，实现了33%的ASB，提供了初步证据，证明在不访问目标管道的情况下设计攻击是可行的。最后，我们建议开发人员可以用来阻止分阶段攻击的具体缓解措施。



## **3. Consensus-based optimization for closed-box adversarial attacks and a connection to evolution strategies**

针对闭箱对抗攻击的基于假设的优化以及与进化策略的联系 math.OC

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24048v1) [paper-pdf](http://arxiv.org/pdf/2506.24048v1)

**Authors**: Tim Roith, Leon Bungert, Philipp Wacker

**Abstract**: Consensus-based optimization (CBO) has established itself as an efficient gradient-free optimization scheme, with attractive mathematical properties, such as mean-field convergence results for non-convex loss functions. In this work, we study CBO in the context of closed-box adversarial attacks, which are imperceptible input perturbations that aim to fool a classifier, without accessing its gradient. Our contribution is to establish a connection between the so-called consensus hopping as introduced by Riedl et al. and natural evolution strategies (NES) commonly applied in the context of adversarial attacks and to rigorously relate both methods to gradient-based optimization schemes. Beyond that, we provide a comprehensive experimental study that shows that despite the conceptual similarities, CBO can outperform NES and other evolutionary strategies in certain scenarios.

摘要: 基于边界的优化（CBO）已经成为一种高效的无梯度优化方案，具有吸引人的数学性质，例如非凸损失函数的平均场收敛结果。在这项工作中，我们在闭箱对抗攻击的背景下研究CBO，这是一种难以察觉的输入扰动，旨在欺骗分类器，而无需访问其梯度。我们的贡献是在Riedl等人提出的所谓共识跳跃与通常在对抗性攻击背景下应用的自然进化策略（NES）之间建立联系，并将这两种方法与基于梯度的优化方案严格联系起来。除此之外，我们还提供了一项全面的实验研究，表明尽管概念相似，但CBO在某些情况下可以优于NES和其他进化策略。



## **4. Quickest Detection of Adversarial Attacks Against Correlated Equilibria**

最快检测针对相关均衡的对抗攻击 cs.GT

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.24040v1) [paper-pdf](http://arxiv.org/pdf/2506.24040v1)

**Authors**: Kiarash Kazari, Aris Kanellopoulos, György Dán

**Abstract**: We consider correlated equilibria in strategic games in an adversarial environment, where an adversary can compromise the public signal used by the players for choosing their strategies, while players aim at detecting a potential attack as soon as possible to avoid loss of utility. We model the interaction between the adversary and the players as a zero-sum game and we derive the maxmin strategies for both the defender and the attacker using the framework of quickest change detection. We define a class of adversarial strategies that achieve the optimal trade-off between attack impact and attack detectability and show that a generalized CUSUM scheme is asymptotically optimal for the detection of the attacks. Our numerical results on the Sioux-Falls benchmark traffic routing game show that the proposed detection scheme can effectively limit the utility loss by a potential adversary.

摘要: 我们考虑了在对抗环境中的策略博弈中的相关均衡，在这种环境中，对手可以损害玩家选择策略所使用的公共信号，而玩家的目标是尽快检测到潜在的攻击，以避免效用损失。我们建模的对手和球员之间的相互作用作为一个零和博弈，我们推导出最大最小的防御者和攻击者使用的框架，最快的变化检测的策略。我们定义了一类对抗策略，实现了攻击影响和攻击可检测性之间的最佳权衡，并证明了广义的CRAMUM方案对于检测攻击是渐近最优的。在Sioux-Falls基准流量路由博弈上的数值结果表明，该检测方案可以有效地限制潜在对手的效用损失。



## **5. Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation**

谜语我！检索增强一代的隐形会员推断 cs.CR

This is the full version (27 pages) of the paper 'Riddle Me This!  Stealthy Membership Inference for Retrieval-Augmented Generation' published  at CCS 2025

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2502.00306v2) [paper-pdf](http://arxiv.org/pdf/2502.00306v2)

**Authors**: Ali Naseh, Yuefeng Peng, Anshuman Suri, Harsh Chaudhari, Alina Oprea, Amir Houmansadr

**Abstract**: Retrieval-Augmented Generation (RAG) enables Large Language Models (LLMs) to generate grounded responses by leveraging external knowledge databases without altering model parameters. Although the absence of weight tuning prevents leakage via model parameters, it introduces the risk of inference adversaries exploiting retrieved documents in the model's context. Existing methods for membership inference and data extraction often rely on jailbreaking or carefully crafted unnatural queries, which can be easily detected or thwarted with query rewriting techniques common in RAG systems. In this work, we present Interrogation Attack (IA), a membership inference technique targeting documents in the RAG datastore. By crafting natural-text queries that are answerable only with the target document's presence, our approach demonstrates successful inference with just 30 queries while remaining stealthy; straightforward detectors identify adversarial prompts from existing methods up to ~76x more frequently than those generated by our attack. We observe a 2x improvement in TPR@1%FPR over prior inference attacks across diverse RAG configurations, all while costing less than $0.02 per document inference.

摘要: 检索增强生成（RAG）使大型语言模型（LLM）能够通过利用外部知识数据库来生成接地响应，而无需更改模型参数。尽管缺乏权重调整可以防止模型参数泄露，但它引入了推理对手利用模型上下文中检索到的文档的风险。现有的隶属关系推断和数据提取方法通常依赖于越狱或精心制作的非自然查询，这些查询可以通过RAG系统中常见的查询重写技术轻松检测或阻止。在这项工作中，我们介绍了审讯攻击（IA），这是一种针对RAG收件箱中文档的成员资格推断技术。通过制作仅在目标文档存在的情况下才能回答的自然文本查询，我们的方法仅用30个查询就能证明成功推理，同时保持隐蔽性;简单的检测器识别来自现有方法的对抗性提示的频率高达约76倍，比我们的攻击产生的提示。我们观察到，在各种RAG配置中，TPR@1%FPR比之前的推理攻击提高了2倍，同时每个文档推理的成本不到0.02美元。



## **6. Benchmarking Spiking Neural Network Learning Methods with Varying Locality**

对具有不同局部性的尖峰神经网络学习方法进行基准测试 cs.NE

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2402.01782v2) [paper-pdf](http://arxiv.org/pdf/2402.01782v2)

**Authors**: Jiaqi Lin, Sen Lu, Malyaban Bal, Abhronil Sengupta

**Abstract**: Spiking Neural Networks (SNNs), providing more realistic neuronal dynamics, have been shown to achieve performance comparable to Artificial Neural Networks (ANNs) in several machine learning tasks. Information is processed as spikes within SNNs in an event-based mechanism that significantly reduces energy consumption. However, training SNNs is challenging due to the non-differentiable nature of the spiking mechanism. Traditional approaches, such as Backpropagation Through Time (BPTT), have shown effectiveness but come with additional computational and memory costs and are biologically implausible. In contrast, recent works propose alternative learning methods with varying degrees of locality, demonstrating success in classification tasks. In this work, we show that these methods share similarities during the training process, while they present a trade-off between biological plausibility and performance. Further, given the implicitly recurrent nature of SNNs, this research investigates the influence of the addition of explicit recurrence to SNNs. We experimentally prove that the addition of explicit recurrent weights enhances the robustness of SNNs. We also investigate the performance of local learning methods under gradient and non-gradient-based adversarial attacks.

摘要: 尖峰神经网络（SNN）提供了更真实的神经元动力学，已被证明在几项机器学习任务中可以实现与人工神经网络（ANN）相当的性能。信息在基于事件的机制中作为SNN内的尖峰进行处理，从而显着降低了能源消耗。然而，由于尖峰机制的不可微性质，训练SNN具有挑战性。传统方法，例如时间反向传播（BPTT），已经显示出有效性，但会带来额外的计算和存储成本，并且在生物学上是不可信的。相比之下，最近的作品提出了具有不同局部性的替代学习方法，证明了分类任务的成功。在这项工作中，我们表明这些方法在训练过程中有相似之处，同时它们在生物相似性和性能之间进行了权衡。此外，鉴于SNN的隐式回归性质，本研究调查了SNN添加显式回归的影响。我们通过实验证明，添加显式循环权重增强了SNN的鲁棒性。我们还研究了本地学习方法在梯度和非基于梯度的对抗攻击下的性能。



## **7. A Unified Framework for Stealthy Adversarial Generation via Latent Optimization and Transferability Enhancement**

通过潜在优化和可移植性增强实现隐形对抗生成的统一框架 cs.CV

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.23676v1) [paper-pdf](http://arxiv.org/pdf/2506.23676v1)

**Authors**: Gaozheng Pei, Ke Ma, Dongpeng Zhang, Chengzhi Sun, Qianqian Xu, Qingming Huang

**Abstract**: Due to their powerful image generation capabilities, diffusion-based adversarial example generation methods through image editing are rapidly gaining popularity. However, due to reliance on the discriminative capability of the diffusion model, these diffusion-based methods often struggle to generalize beyond conventional image classification tasks, such as in Deepfake detection. Moreover, traditional strategies for enhancing adversarial example transferability are challenging to adapt to these methods. To address these challenges, we propose a unified framework that seamlessly incorporates traditional transferability enhancement strategies into diffusion model-based adversarial example generation via image editing, enabling their application across a wider range of downstream tasks. Our method won first place in the "1st Adversarial Attacks on Deepfake Detectors: A Challenge in the Era of AI-Generated Media" competition at ACM MM25, which validates the effectiveness of our approach.

摘要: 由于其强大的图像生成能力，通过图像编辑的基于扩散的对抗性示例生成方法正在迅速流行。然而，由于依赖于扩散模型的辨别能力，这些基于扩散的方法通常很难概括超出传统图像分类任务，例如Deepfake检测。此外，增强对抗性示例可移植性的传统策略在适应这些方法方面具有挑战性。为了应对这些挑战，我们提出了一个统一的框架，通过图像编辑将传统的可移植性增强策略无缝地整合到基于扩散模型的对抗性示例生成中，使其能够在更广泛的下游任务中应用。我们的方法在ACN MM25举行的“Deepfake Detector的第一次对抗攻击：人工智能生成媒体时代的挑战”竞赛中获得了第一名，这验证了我们方法的有效性。



## **8. Robustness of Misinformation Classification Systems to Adversarial Examples Through BeamAttack**

错误信息分类系统对BeamAttack对抗示例的鲁棒性 cs.CL

12 pages main text, 27 pages total including references and  appendices. 13 figures, 10 tables. Accepted for publication in the LNCS  proceedings of CLEF 2025 (Best-of-Labs track)

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.23661v1) [paper-pdf](http://arxiv.org/pdf/2506.23661v1)

**Authors**: Arnisa Fazla, Lucas Krauter, David Guzman Piedrahita, Andrianos Michail

**Abstract**: We extend BeamAttack, an adversarial attack algorithm designed to evaluate the robustness of text classification systems through word-level modifications guided by beam search. Our extensions include support for word deletions and the option to skip substitutions, enabling the discovery of minimal modifications that alter model predictions. We also integrate LIME to better prioritize word replacements. Evaluated across multiple datasets and victim models (BiLSTM, BERT, and adversarially trained RoBERTa) within the BODEGA framework, our approach achieves over a 99\% attack success rate while preserving the semantic and lexical similarity of the original texts. Through both quantitative and qualitative analysis, we highlight BeamAttack's effectiveness and its limitations. Our implementation is available at https://github.com/LucK1Y/BeamAttack

摘要: 我们扩展了BeamAttack，这是一种对抗攻击算法，旨在通过束搜索指导的词级修改来评估文本分类系统的稳健性。我们的扩展包括对字词删除和跳过替换的选项，从而能够发现改变模型预测的最小修改。我们还集成了LIME，以更好地优先考虑单词替换。在BODEGA框架内对多个数据集和受害者模型（BiLSTM、BERT和对抗训练的RoBERTa）进行评估，我们的方法实现了超过99%的攻击成功率，同时保留了原始文本的语义和词汇相似性。通过定量和定性分析，我们强调了BeamAttack的有效性及其局限性。我们的实施可在https://github.com/LucK1Y/BeamAttack上获取



## **9. PBCAT: Patch-based composite adversarial training against physically realizable attacks on object detection**

PBCAT：针对对象检测物理上可实现的攻击的基于补丁的复合对抗训练 cs.CV

Accepted by ICCV 2025

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.23581v1) [paper-pdf](http://arxiv.org/pdf/2506.23581v1)

**Authors**: Xiao Li, Yiming Zhu, Yifan Huang, Wei Zhang, Yingzhe He, Jie Shi, Xiaolin Hu

**Abstract**: Object detection plays a crucial role in many security-sensitive applications. However, several recent studies have shown that object detectors can be easily fooled by physically realizable attacks, \eg, adversarial patches and recent adversarial textures, which pose realistic and urgent threats. Adversarial Training (AT) has been recognized as the most effective defense against adversarial attacks. While AT has been extensively studied in the $l_\infty$ attack settings on classification models, AT against physically realizable attacks on object detectors has received limited exploration. Early attempts are only performed to defend against adversarial patches, leaving AT against a wider range of physically realizable attacks under-explored. In this work, we consider defending against various physically realizable attacks with a unified AT method. We propose PBCAT, a novel Patch-Based Composite Adversarial Training strategy. PBCAT optimizes the model by incorporating the combination of small-area gradient-guided adversarial patches and imperceptible global adversarial perturbations covering the entire image. With these designs, PBCAT has the potential to defend against not only adversarial patches but also unseen physically realizable attacks such as adversarial textures. Extensive experiments in multiple settings demonstrated that PBCAT significantly improved robustness against various physically realizable attacks over state-of-the-art defense methods. Notably, it improved the detection accuracy by 29.7\% over previous defense methods under one recent adversarial texture attack.

摘要: 对象检测在许多安全敏感应用程序中发挥着至关重要的作用。然而，最近的几项研究表明，对象检测器很容易被物理上可实现的攻击所愚弄，例如对抗补丁和最近的对抗纹理，这些攻击构成了现实而紧迫的威胁。对抗训练（AT）被认为是对抗攻击的最有效防御。虽然AT在分类模型的$l_\infty$攻击设置中得到了广泛研究，但针对对象检测器物理上可实现的攻击的AT的探索有限。早期的尝试只是为了防御对抗补丁，这使得AT能够对抗更广泛的物理可实现的攻击，但尚未得到充分的探索。在这项工作中，我们考虑使用统一的AT方法来防御各种物理上可实现的攻击。我们提出了PBCAT，这是一种新型的基于补丁的复合对抗训练策略。PBCAT通过结合小区域梯度引导的对抗补丁和覆盖整个图像的不可感知的全局对抗扰动来优化模型。通过这些设计，PBCAT不仅有潜力防御对抗补丁，还有潜力防御不可见的物理可实现的攻击，例如对抗纹理。在多个环境中进行的大量实验表明，与最先进的防御方法相比，PBCAT显着提高了针对各种物理可实现攻击的鲁棒性。值得注意的是，在最近的一次对抗性纹理攻击下，它比之前的防御方法提高了29.7%。



## **10. Efficient Resource Allocation under Adversary Attacks: A Decomposition-Based Approach**

敌对攻击下的高效资源分配：基于分解的方法 cs.DS

**SubmitDate**: 2025-06-30    [abs](http://arxiv.org/abs/2506.23442v1) [paper-pdf](http://arxiv.org/pdf/2506.23442v1)

**Authors**: Mansoor Davoodi, Setareh Maghsudi

**Abstract**: We address the problem of allocating limited resources in a network under persistent yet statistically unknown adversarial attacks. Each node in the network may be degraded, but not fully disabled, depending on its available defensive resources. The objective is twofold: to minimize total system damage and to reduce cumulative resource allocation and transfer costs over time. We model this challenge as a bi-objective optimization problem and propose a decomposition-based solution that integrates chance-constrained programming with network flow optimization. The framework separates the problem into two interrelated subproblems: determining optimal node-level allocations across time slots, and computing efficient inter-node resource transfers. We theoretically prove the convergence of our method to the optimal solution that would be obtained with full statistical knowledge of the adversary. Extensive simulations demonstrate that our method efficiently learns the adversarial patterns and achieves substantial gains in minimizing both damage and operational costs, comparing three benchmark strategies under various parameter settings.

摘要: 我们解决了在持续但统计上未知的对抗性攻击下在网络中分配有限资源的问题。网络中的每个节点可能会降级，但不会完全禁用，具体取决于其可用的防御资源。目标是双重的：最大限度地减少系统的总体损害，并随着时间的推移减少累积资源分配和转移成本。我们将这一挑战建模为双目标优化问题，并提出一种基于分解的解决方案，将机会约束规划与网络流优化集成在一起。该框架将问题分为两个相互关联的子问题：确定跨时段的最佳节点级分配，以及计算高效的节点间资源传输。我们从理论上证明了我们的方法收敛于最优解，该最优解将通过对手的充分统计知识获得。广泛的模拟表明，我们的方法可以有效地学习对抗模式，并在最大限度地减少损害和运营成本方面取得了实质性进展，通过比较不同参数设置下的三种基准策略。



## **11. TuCo: Measuring the Contribution of Fine-Tuning to Individual Responses of LLMs**

TuCo：衡量微调对LLM个人响应的贡献 cs.CL

ICML 2025

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2506.23423v1) [paper-pdf](http://arxiv.org/pdf/2506.23423v1)

**Authors**: Felipe Nuti, Tim Franzmeyer, João Henriques

**Abstract**: Past work has studied the effects of fine-tuning on large language models' (LLMs) overall performance on certain tasks. However, a quantitative and systematic method for analyzing its effect on individual outputs is still lacking. Here, we propose a new method for measuring the contribution that fine-tuning makes to individual LLM responses, assuming access to the original pre-trained model. Our method tracks the model's intermediate hidden states, providing a more fine-grained insight into the effects of fine-tuning than a simple comparison of final outputs from pre-trained and fine-tuned models. We introduce and theoretically analyze an exact decomposition of any fine-tuned LLM into a pre-training component and a fine-tuning component. Empirically, we find that model behavior and performance can be steered by up- or down-scaling the fine-tuning component during the forward pass. Motivated by this finding and our theoretical analysis, we define the Tuning Contribution (TuCo) as the ratio of the magnitudes of the fine-tuning component to the pre-training component. We observe that three prominent adversarial attacks on LLMs circumvent safety measures in a way that reduces TuCo, and that TuCo is consistently lower on prompts where these attacks succeed compared to those where they do not. This suggests that attenuating the effect of fine-tuning on model outputs plays a role in the success of such attacks. In summary, TuCo enables the quantitative study of how fine-tuning influences model behavior and safety, and vice versa.

摘要: 过去的工作研究了微调对大型语言模型（LLM）在某些任务上整体性能的影响。然而，仍然缺乏一种定量、系统的方法来分析其对单个产出的影响。在这里，我们提出了一种新的方法来衡量微调对个体LLM响应的贡献，假设可以访问原始的预训练模型。我们的方法跟踪模型的中间隐藏状态，与预训练和微调模型的最终输出的简单比较相比，提供了对微调效果的更细粒度的见解。我们引入并从理论上分析将任何微调LLM精确分解为预训练组件和微调组件。从经验上看，我们发现模型行为和性能可以通过在前向传递期间放大或缩小微调组件来引导。受这一发现和理论分析的启发，我们将调整贡献（TuCo）定义为微调分量与预训练分量的幅度之比。我们观察到，针对LLM的三种突出的对抗性攻击以某种程度上减少了TuCo的方式规避了安全措施，并且与失败的情况相比，TuCo在这些攻击成功的提示上始终较低。这表明减弱微调对模型输出的影响在此类攻击的成功中发挥了作用。总之，TuCo能够定量研究微调如何影响模型行为和安全性，反之亦然。



## **12. Enhancing Adversarial Robustness through Multi-Objective Representation Learning**

通过多目标表示学习增强对抗鲁棒性 cs.LG

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2410.01697v4) [paper-pdf](http://arxiv.org/pdf/2410.01697v4)

**Authors**: Sedjro Salomon Hotegni, Sebastian Peitz

**Abstract**: Deep neural networks (DNNs) are vulnerable to small adversarial perturbations, which are tiny changes to the input data that appear insignificant but cause the model to produce drastically different outputs. Many defense methods require modifying model architectures during evaluation or performing test-time data purification. This not only introduces additional complexity but is often architecture-dependent. We show, however, that robust feature learning during training can significantly enhance DNN robustness. We propose MOREL, a multi-objective approach that aligns natural and adversarial features using cosine similarity and multi-positive contrastive losses to encourage similar features for same-class inputs. Extensive experiments demonstrate that MOREL significantly improves robustness against both white-box and black-box attacks. Our code is available at https://github.com/salomonhotegni/MOREL

摘要: 深度神经网络（DNN）容易受到微小的对抗性扰动的影响，这些扰动是对输入数据的微小变化，这些变化看起来微不足道，但会导致模型产生截然不同的输出。许多防御方法需要在评估或执行测试时数据净化期间修改模型架构。这不仅带来了额外的复杂性，而且通常取决于体系结构。然而，我们表明，训练期间的稳健特征学习可以显着增强DNN稳健性。我们提出了MOREL，这是一种多目标方法，它使用余弦相似性和多正对比损失来对齐自然和对抗特征，以鼓励同类输入的相似特征。大量的实验表明，MOREL显着提高了对白盒和黑盒攻击的鲁棒性。我们的代码可在https://github.com/salomonhotegni/MOREL上获取



## **13. Adversarial Robustness Unhardening via Backdoor Attacks in Federated Learning**

联邦学习中通过后门攻击解除对抗鲁棒性 cs.LG

15 pages, 8 main pages of text, 13 figures, 5 tables. Made for a  Neurips workshop on backdoor attacks - extended version

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2310.11594v3) [paper-pdf](http://arxiv.org/pdf/2310.11594v3)

**Authors**: Taejin Kim, Jiarui Li, Shubhranshu Singh, Nikhil Madaan, Carlee Joe-Wong

**Abstract**: The delicate equilibrium between user privacy and the ability to unleash the potential of distributed data is an important concern. Federated learning, which enables the training of collaborative models without sharing of data, has emerged as a privacy-centric solution. This approach brings forth security challenges, notably poisoning and backdoor attacks where malicious entities inject corrupted data into the training process, as well as evasion attacks that aim to induce misclassifications at test time. Our research investigates the intersection of adversarial training, a common defense method against evasion attacks, and backdoor attacks within federated learning. We introduce Adversarial Robustness Unhardening (ARU), which is employed by a subset of adversarial clients to intentionally undermine model robustness during federated training, rendering models susceptible to a broader range of evasion attacks. We present extensive experiments evaluating ARU's impact on adversarial training and existing robust aggregation defenses against poisoning and backdoor attacks. Our results show that ARU can substantially undermine adversarial training's ability to harden models against test-time evasion attacks, and that adversaries employing ARU can even evade robust aggregation defenses that often neutralize poisoning or backdoor attacks.

摘要: 用户隐私与释放分布式数据潜力的能力之间的微妙平衡是一个重要问题。联合学习可以在不共享数据的情况下训练协作模型，已成为一种以隐私为中心的解决方案。这种方法带来了安全挑战，特别是恶意实体将损坏的数据注入到训练过程中的中毒和后门攻击，以及旨在在测试时引发错误分类的规避攻击。我们的研究调查了联邦学习中对抗训练、针对规避攻击的常见防御方法和后门攻击的交叉点。我们引入了对抗稳健性解除硬化（ARU），一部分对抗客户端使用它来在联邦训练期间故意破坏模型稳健性，使模型容易受到更广泛的规避攻击。我们进行了广泛的实验，评估ARU对对抗训练以及针对中毒和后门攻击的现有强大聚集防御的影响。我们的结果表明，ARU可以极大地削弱对抗训练强化模型抵御测试时规避攻击的能力，而且使用ARU的对手甚至可以规避通常可以中和中毒或后门攻击的强大聚集防御。



## **14. Scaling Laws for Black box Adversarial Attacks**

黑匣子对抗攻击的缩放定律 cs.LG

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2411.16782v3) [paper-pdf](http://arxiv.org/pdf/2411.16782v3)

**Authors**: Chuan Liu, Huanran Chen, Yichi Zhang, Yinpeng Dong, Jun Zhu

**Abstract**: Adversarial examples usually exhibit good cross-model transferability, enabling attacks on black-box models with limited information about their architectures and parameters, which are highly threatening in commercial black-box scenarios. Model ensembling is an effective strategy to improve the transferability of adversarial examples by attacking multiple surrogate models. However, since prior studies usually adopt few models in the ensemble, there remains an open question of whether scaling the number of models can further improve black-box attacks. Inspired by the scaling law of large foundation models, we investigate the scaling laws of black-box adversarial attacks in this work. Through theoretical analysis and empirical evaluations, we conclude with clear scaling laws that using more surrogate models enhances adversarial transferability. Comprehensive experiments verify the claims on standard image classifiers, diverse defended models and multimodal large language models using various adversarial attack methods. Specifically, by scaling law, we achieve 90%+ transfer attack success rate on even proprietary models like GPT-4o. Further visualization indicates that there is also a scaling law on the interpretability and semantics of adversarial perturbations.

摘要: 对抗性示例通常表现出良好的跨模型可移植性，从而能够在有关其架构和参数的有限信息的情况下对黑匣子模型进行攻击，这在商业黑匣子场景中具有高度威胁性。模型集成是通过攻击多个代理模型来提高对抗性示例可移植性的有效策略。然而，由于之前的研究通常在整体中采用很少的模型，因此扩大模型数量是否可以进一步改善黑匣子攻击仍然是一个悬而未决的问题。受大型基金会模型缩放定律的启发，我们在这项工作中研究了黑匣子对抗攻击的缩放定律。通过理论分析和实证评估，我们得出了明确的缩放定律，即使用更多的代理模型增强了对抗性可转让性。全面的实验验证了标准图像分类器、多样化防御模型和使用各种对抗攻击方法的多模式大型语言模型的主张。具体来说，通过缩放定律，即使是GPT-4 o等专有模型，我们也能实现90%以上的传输攻击成功率。进一步的可视化表明，对抗性扰动的可解释性和语义也存在缩放定律。



## **15. MedLeak: Multimodal Medical Data Leakage in Secure Federated Learning with Crafted Models**

MedLeak：使用精心设计的模型的安全联邦学习中的多模式医疗数据泄露 cs.LG

Accepted by the IEEE/ACM conference on Connected Health:  Applications, Systems and Engineering Technologies 2025 (CHASE'25)

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2407.09972v2) [paper-pdf](http://arxiv.org/pdf/2407.09972v2)

**Authors**: Shanghao Shi, Md Shahedul Haque, Abhijeet Parida, Chaoyu Zhang, Marius George Linguraru, Y. Thomas Hou, Syed Muhammad Anwar, Wenjing Lou

**Abstract**: Federated learning (FL) allows participants to collaboratively train machine learning models while keeping their data local, making it ideal for collaborations among healthcare institutions on sensitive data. However, in this paper, we propose a novel privacy attack called MedLeak, which allows a malicious FL server to recover high-quality site-specific private medical data from the client model updates. MedLeak works by introducing an adversarially crafted model during the FL training process. Honest clients, unaware of the insidious changes in the published models, continue to send back their updates as per the standard FL protocol. Leveraging a novel analytical method, MedLeak can efficiently recover private client data from the aggregated parameter updates, eliminating costly optimization. In addition, the scheme relies solely on the aggregated updates, thus rendering secure aggregation protocols ineffective, as they depend on the randomization of intermediate results for security while leaving the final aggregated results unaltered.   We implement MedLeak on medical image datasets (MedMNIST, COVIDx CXR-4, and Kaggle Brain Tumor MRI), as well as a medical text dataset (MedAbstract). The results demonstrate that our attack achieves high recovery rates and strong quantitative scores on both image and text datasets. We also thoroughly evaluate MedLeak across different attack parameters, providing insights into key factors that influence attack performance and potential defenses. Furthermore, we demonstrate that the recovered data can support downstream tasks such as disease classification with minimal performance loss. Our findings validate the need for enhanced privacy measures in FL systems, particularly for safeguarding sensitive medical data against powerful model inversion attacks.

摘要: 联合学习（FL）允许参与者协作训练机器学习模型，同时将其数据保持在本地，因此非常适合医疗机构之间就敏感数据进行合作。然而，在本文中，我们提出了一种名为MedLeak的新型隐私攻击，它允许恶意FL服务器从客户端模型更新中恢复高质量的特定站点私人医疗数据。MedLeak的工作原理是在FL培训过程中引入一个对抗性制作的模型。诚实的客户没有意识到已发布模型中的潜在变化，会继续根据标准FL协议发送更新。利用新颖的分析方法，MedLeak可以从聚合参数更新中有效地恢复私人客户数据，从而消除昂贵的优化。此外，该方案仅依赖于聚合更新，从而导致安全聚合协议无效，因为它们依赖于中间结果的随机化以实现安全，而最终的聚合结果保持不变。   我们在医学图像数据集（MedMNIST、COVIDx CXR-4和Kaggle脑肿瘤MRI）以及医学文本数据集（MedAbstract）上实施MedLeak。结果表明，我们的攻击在图像和文本数据集上都实现了高恢复率和强大的量化分数。我们还对不同的攻击参数进行了彻底评估，从而深入了解影响攻击性能和潜在防御的关键因素。此外，我们证明恢复的数据可以以最小的性能损失支持疾病分类等下游任务。我们的研究结果证实了FL系统中增强隐私措施的必要性，特别是为了保护敏感医疗数据免受强大的模型倒置攻击。



## **16. Securing AI Systems: A Guide to Known Attacks and Impacts**

保护人工智能系统：已知攻击和影响指南 cs.CR

34 pages, 16 figures

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2506.23296v1) [paper-pdf](http://arxiv.org/pdf/2506.23296v1)

**Authors**: Naoto Kiribuchi, Kengo Zenitani, Takayuki Semitsu

**Abstract**: Embedded into information systems, artificial intelligence (AI) faces security threats that exploit AI-specific vulnerabilities. This paper provides an accessible overview of adversarial attacks unique to predictive and generative AI systems. We identify eleven major attack types and explicitly link attack techniques to their impacts -- including information leakage, system compromise, and resource exhaustion -- mapped to the confidentiality, integrity, and availability (CIA) security triad. We aim to equip researchers, developers, security practitioners, and policymakers, even those without specialized AI security expertise, with foundational knowledge to recognize AI-specific risks and implement effective defenses, thereby enhancing the overall security posture of AI systems.

摘要: 人工智能（AI）嵌入到信息系统中，面临着利用人工智能特定漏洞的安全威胁。本文提供了预测和生成人工智能系统特有的对抗性攻击的易于理解的概述。我们确定了十一种主要攻击类型，并将攻击技术与其影响明确联系起来（包括信息泄露、系统泄露和资源耗尽），映射到机密性、完整性和可用性（CIA）安全三重位。我们的目标是为研究人员、开发人员、安全从业者和政策制定者（即使是那些没有专业人工智能安全专业知识的人）提供识别人工智能特定风险并实施有效防御的基础知识，从而增强人工智能系统的整体安全态势。



## **17. From Prompt Injections to Protocol Exploits: Threats in LLM-Powered AI Agents Workflows**

从即时注入到协议漏洞：LLM支持的人工智能代理工作流程中的威胁 cs.CR

29 pages, 15 figures, 6 tables

**SubmitDate**: 2025-06-29    [abs](http://arxiv.org/abs/2506.23260v1) [paper-pdf](http://arxiv.org/pdf/2506.23260v1)

**Authors**: Mohamed Amine Ferrag, Norbert Tihanyi, Djallel Hamouda, Leandros Maglaras, Merouane Debbah

**Abstract**: Autonomous AI agents powered by large language models (LLMs) with structured function-calling interfaces have dramatically expanded capabilities for real-time data retrieval, complex computation, and multi-step orchestration. Yet, the explosive proliferation of plugins, connectors, and inter-agent protocols has outpaced discovery mechanisms and security practices, resulting in brittle integrations vulnerable to diverse threats. In this survey, we introduce the first unified, end-to-end threat model for LLM-agent ecosystems, spanning host-to-tool and agent-to-agent communications, formalize adversary capabilities and attacker objectives, and catalog over thirty attack techniques. Specifically, we organized the threat model into four domains: Input Manipulation (e.g., prompt injections, long-context hijacks, multimodal adversarial inputs), Model Compromise (e.g., prompt- and parameter-level backdoors, composite and encrypted multi-backdoors, poisoning strategies), System and Privacy Attacks (e.g., speculative side-channels, membership inference, retrieval poisoning, social-engineering simulations), and Protocol Vulnerabilities (e.g., exploits in Model Context Protocol (MCP), Agent Communication Protocol (ACP), Agent Network Protocol (ANP), and Agent-to-Agent (A2A) protocol). For each category, we review representative scenarios, assess real-world feasibility, and evaluate existing defenses. Building on our threat taxonomy, we identify key open challenges and future research directions, such as securing MCP deployments through dynamic trust management and cryptographic provenance tracking; designing and hardening Agentic Web Interfaces; and achieving resilience in multi-agent and federated environments. Our work provides a comprehensive reference to guide the design of robust defense mechanisms and establish best practices for resilient LLM-agent workflows.

摘要: 由具有结构化功能调用接口的大型语言模型（LLM）支持的自主人工智能代理极大地扩展了实时数据检索、复杂计算和多步骤编排的能力。然而，插件、连接器和代理间协议的爆炸性激增已经超过了发现机制和安全实践的速度，导致集成脆弱，容易受到各种威胁的影响。在本调查中，我们为LLM代理生态系统引入了第一个统一的端到端威胁模型，涵盖主机到工具和代理到代理的通信，正式化对手能力和攻击者目标，并对三十多种攻击技术进行了分类。具体来说，我们将威胁模型组织为四个领域：输入操纵（例如，提示注入、长上下文劫持、多模式对抗输入）、模型妥协（例如，提示和参数级后门、复合和加密的多后门、中毒策略）、系统和隐私攻击（例如，推测性侧通道、成员资格推断、检索中毒、社会工程模拟）和协议漏洞（例如，模型上下文协议（HCP）、代理通信协议（ACP）、代理网络协议（ANP）和代理对代理（A2 A）协议中的漏洞利用）。对于每个类别，我们都会审查代表性场景、评估现实世界的可行性并评估现有的防御措施。基于我们的威胁分类法，我们确定了关键的开放挑战和未来的研究方向，例如通过动态信任管理和加密来源跟踪来保护LCP部署;设计和强化统计Web界面;以及在多代理和联邦环境中实现弹性。我们的工作提供了全面的参考，以指导稳健的防御机制的设计并为弹性LLM代理工作流程建立最佳实践。



## **18. Fragile, Robust, and Antifragile: A Perspective from Parameter Responses in Reinforcement Learning Under Stress**

脆弱性、稳健性和反脆弱性：从压力下强化学习中的参数响应的角度来看 cs.LG

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.23036v1) [paper-pdf](http://arxiv.org/pdf/2506.23036v1)

**Authors**: Zain ul Abdeen, Ming Jin

**Abstract**: This paper explores Reinforcement learning (RL) policy robustness by systematically analyzing network parameters under internal and external stresses. Inspired by synaptic plasticity in neuroscience, synaptic filtering introduces internal stress by selectively perturbing parameters, while adversarial attacks apply external stress through modified agent observations. This dual approach enables the classification of parameters as fragile, robust, or antifragile, based on their influence on policy performance in clean and adversarial settings. Parameter scores are defined to quantify these characteristics, and the framework is validated on PPO-trained agents in Mujoco continuous control environments. The results highlight the presence of antifragile parameters that enhance policy performance under stress, demonstrating the potential of targeted filtering techniques to improve RL policy adaptability. These insights provide a foundation for future advancements in the design of robust and antifragile RL systems.

摘要: 本文通过系统分析内部和外部压力下的网络参数来探索强化学习（RL）策略的稳健性。受神经科学中突触可塑性的启发，突触过滤通过选择性地扰乱参数来引入内部压力，而对抗攻击则通过修改的主体观察来施加外部压力。这种双重方法可以根据参数在干净和敌对环境中对政策绩效的影响将参数分类为脆弱、稳健或反脆弱。定义参数分数来量化这些特征，并在Mujoco连续控制环境中的PPA训练代理上验证了该框架。结果强调了反脆弱参数的存在，可以增强压力下的政策性能，证明了有针对性的过滤技术提高RL政策适应性的潜力。这些见解为未来鲁棒和抗脆弱RL系统设计的进步提供了基础。



## **19. Revisiting CroPA: A Reproducibility Study and Enhancements for Cross-Prompt Adversarial Transferability in Vision-Language Models**

重温CroPA：视觉语言模型中交叉提示对抗可移植性的再现性研究和增强 cs.CV

Accepted to MLRC 2025

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.22982v1) [paper-pdf](http://arxiv.org/pdf/2506.22982v1)

**Authors**: Atharv Mittal, Agam Pandey, Amritanshu Tiwari, Sukrit Jindal, Swadesh Swain

**Abstract**: Large Vision-Language Models (VLMs) have revolutionized computer vision, enabling tasks such as image classification, captioning, and visual question answering. However, they remain highly vulnerable to adversarial attacks, particularly in scenarios where both visual and textual modalities can be manipulated. In this study, we conduct a comprehensive reproducibility study of "An Image is Worth 1000 Lies: Adversarial Transferability Across Prompts on Vision-Language Models" validating the Cross-Prompt Attack (CroPA) and confirming its superior cross-prompt transferability compared to existing baselines. Beyond replication we propose several key improvements: (1) A novel initialization strategy that significantly improves Attack Success Rate (ASR). (2) Investigate cross-image transferability by learning universal perturbations. (3) A novel loss function targeting vision encoder attention mechanisms to improve generalization. Our evaluation across prominent VLMs -- including Flamingo, BLIP-2, and InstructBLIP as well as extended experiments on LLaVA validates the original results and demonstrates that our improvements consistently boost adversarial effectiveness. Our work reinforces the importance of studying adversarial vulnerabilities in VLMs and provides a more robust framework for generating transferable adversarial examples, with significant implications for understanding the security of VLMs in real-world applications.

摘要: 大型视觉语言模型（VLM）彻底改变了计算机视觉，实现了图像分类、字幕和视觉问答等任务。然而，它们仍然非常容易受到对抗攻击，特别是在视觉和文本模式都可以被操纵的场景中。在这项研究中，我们对“一个图像值得1000个谎言：视觉语言模型上的冲突可移植性”进行了全面的重复性研究，验证了交叉提示攻击（CroPA），并确认了与现有基线相比其优越的交叉提示可移植性。除了复制之外，我们还提出了几项关键改进：（1）一种新颖的初始化策略，可以显着提高攻击成功率（ASB）。(2)通过学习普适扰动来研究跨图像的可移植性。(3)一种针对视觉编码器注意力机制的新型损失函数，以提高概括性。我们对著名VLM（包括Flamingo、BLIP-2和INSTBLIP）的评估以及LLaVA的扩展实验验证了原始结果，并证明我们的改进持续提高了对抗有效性。我们的工作强调了研究VLM中对抗性漏洞的重要性，并为生成可转移的对抗性示例提供了一个更强大的框架，这对于理解现实世界应用程序中的VLM的安全性具有重要意义。



## **20. VFEFL: Privacy-Preserving Federated Learning against Malicious Clients via Verifiable Functional Encryption**

VFEFL：通过可验证的功能加密针对恶意客户端保护隐私的联邦学习 cs.CR

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.12846v2) [paper-pdf](http://arxiv.org/pdf/2506.12846v2)

**Authors**: Nina Cai, Jinguang Han, Weizhi Meng

**Abstract**: Federated learning is a promising distributed learning paradigm that enables collaborative model training without exposing local client data, thereby protect data privacy. However, it also brings new threats and challenges. The advancement of model inversion attacks has rendered the plaintext transmission of local models insecure, while the distributed nature of federated learning makes it particularly vulnerable to attacks raised by malicious clients. To protect data privacy and prevent malicious client attacks, this paper proposes a privacy-preserving federated learning framework based on verifiable functional encryption, without a non-colluding dual-server setup or additional trusted third-party. Specifically, we propose a novel decentralized verifiable functional encryption (DVFE) scheme that enables the verification of specific relationships over multi-dimensional ciphertexts. This scheme is formally treated, in terms of definition, security model and security proof. Furthermore, based on the proposed DVFE scheme, we design a privacy-preserving federated learning framework VFEFL that incorporates a novel robust aggregation rule to detect malicious clients, enabling the effective training of high-accuracy models under adversarial settings. Finally, we provide formal analysis and empirical evaluation of the proposed schemes. The results demonstrate that our approach achieves the desired privacy protection, robustness, verifiability and fidelity, while eliminating the reliance on non-colluding dual-server settings or trusted third parties required by existing methods.

摘要: 联邦学习是一种有前途的分布式学习范式，可以在不暴露本地客户数据的情况下实现协作模型训练，从而保护数据隐私。然而，它也带来了新的威胁和挑战。模型倒置攻击的发展使本地模型的明文传输变得不安全，而联邦学习的分布式性质使其特别容易受到恶意客户端发起的攻击。为了保护数据隐私并防止恶意客户端攻击，本文提出了一种基于可验证功能加密的隐私保护联邦学习框架，无需非勾结双服务器设置或额外的受信任第三方。具体来说，我们提出了一种新型的去中心化可验证功能加密（DVFE）方案，该方案能够验证多维密文上的特定关系。从定义、安全模型和安全证明方面对该方案进行了正式处理。此外，基于提出的DVFE方案，我们设计了一个保护隐私的联邦学习框架VFEFL，该框架结合了一种新颖的鲁棒聚合规则来检测恶意客户端，从而能够在对抗环境下有效训练高准确度模型。最后，我们对所提出的方案进行了形式分析和实证评估。结果表明，我们的方法实现了预期的隐私保护、稳健性、可验证性和保真度，同时消除了对现有方法所需的非勾结双服务器设置或受信任第三方的依赖。



## **21. Concept Pinpoint Eraser for Text-to-image Diffusion Models via Residual Attention Gate**

通过剩余注意力门的文本到图像扩散模型的概念精确擦除器 cs.CV

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.22806v1) [paper-pdf](http://arxiv.org/pdf/2506.22806v1)

**Authors**: Byung Hyun Lee, Sungjin Lim, Seunggyu Lee, Dong Un Kang, Se Young Chun

**Abstract**: Remarkable progress in text-to-image diffusion models has brought a major concern about potentially generating images on inappropriate or trademarked concepts. Concept erasing has been investigated with the goals of deleting target concepts in diffusion models while preserving other concepts with minimal distortion. To achieve these goals, recent concept erasing methods usually fine-tune the cross-attention layers of diffusion models. In this work, we first show that merely updating the cross-attention layers in diffusion models, which is mathematically equivalent to adding \emph{linear} modules to weights, may not be able to preserve diverse remaining concepts. Then, we propose a novel framework, dubbed Concept Pinpoint Eraser (CPE), by adding \emph{nonlinear} Residual Attention Gates (ResAGs) that selectively erase (or cut) target concepts while safeguarding remaining concepts from broad distributions by employing an attention anchoring loss to prevent the forgetting. Moreover, we adversarially train CPE with ResAG and learnable text embeddings in an iterative manner to maximize erasing performance and enhance robustness against adversarial attacks. Extensive experiments on the erasure of celebrities, artistic styles, and explicit contents demonstrated that the proposed CPE outperforms prior arts by keeping diverse remaining concepts while deleting the target concepts with robustness against attack prompts. Code is available at https://github.com/Hyun1A/CPE

摘要: 文本到图像扩散模型的显着进展引发了人们对可能基于不恰当或商标概念生成图像的重大担忧。人们对概念擦除进行了研究，目标是删除扩散模型中的目标概念，同时以最小的失真保留其他概念。为了实现这些目标，最近的概念删除方法通常会微调扩散模型的交叉注意层。在这项工作中，我们首先表明，仅仅更新扩散模型中的交叉注意层（在数学上相当于将\{线性}模块添加到权重中）可能无法保留多样化的剩余概念。然后，我们提出了一个名为概念精确擦除器（CPD）的新颖框架，通过添加\{非线性}剩余注意力门（ResAG），该门选择性地擦除（或切割）目标概念，同时通过采用注意力锚定损失来保护剩余概念免受广泛分布的影响来防止遗忘。此外，我们以迭代的方式使用ResAG和可学习文本嵌入来对抗性训练CPD，以最大限度地提高擦除性能并增强对抗性攻击的鲁棒性。关于删除名人、艺术风格和明确内容的广泛实验表明，拟议的CPD通过保留多样的剩余概念，同时删除目标概念来对攻击提示具有鲁棒性，从而优于现有技术。代码可访问https://github.com/Hyun1A/CPE



## **22. Enhancing the Capability and Robustness of Large Language Models through Reinforcement Learning-Driven Query Refinement**

通过强化学习驱动的查询细化增强大型语言模型的能力和鲁棒性 cs.CL

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2407.01461v3) [paper-pdf](http://arxiv.org/pdf/2407.01461v3)

**Authors**: Xiaohua Wang, Zisu Huang, Feiran Zhang, Zhibo Xu, Cenyuan Zhang, Qi Qian, Xiaoqing Zheng, Xuanjing Huang

**Abstract**: The capacity of large language models (LLMs) to generate honest, harmless, and helpful responses heavily relies on the quality of user prompts. However, these prompts often tend to be brief and vague, thereby significantly limiting the full potential of LLMs. Moreover, harmful prompts can be meticulously crafted and manipulated by adversaries to jailbreak LLMs, inducing them to produce potentially toxic content. To enhance the capabilities of LLMs while maintaining strong robustness against harmful jailbreak inputs, this study proposes a transferable and pluggable framework that refines user prompts before they are input into LLMs. This strategy improves the quality of the queries, empowering LLMs to generate more truthful, benign and useful responses. Specifically, a lightweight query refinement model is introduced and trained using a specially designed reinforcement learning approach that incorporates multiple objectives to enhance particular capabilities of LLMs. Extensive experiments demonstrate that the refinement model not only improves the quality of responses but also strengthens their robustness against jailbreak attacks. Code is available at: https://github.com/Huangzisu/query-refinement .

摘要: 大型语言模型（LLM）生成诚实、无害且有帮助的响应的能力严重依赖于用户提示的质量。然而，这些提示往往简短且模糊，从而严重限制了法学硕士的全部潜力。此外，对手可能会精心设计和操纵有害提示来越狱LLM，诱导它们产生潜在的有毒内容。为了增强LLM的能力，同时保持针对有害越狱输入的强大鲁棒性，本研究提出了一种可转移且可插入的框架，该框架在用户提示被输入LLM之前对其进行完善。该策略提高了查询的质量，使LLM能够生成更真实、良性和有用的响应。具体来说，使用专门设计的强化学习方法引入并训练轻量级查询细化模型，该方法结合了多个目标以增强LLM的特定能力。大量实验表明，细化模型不仅提高了响应的质量，而且增强了响应对越狱攻击的鲁棒性。代码可访问：https://github.com/Huangzisu/query-refinement。



## **23. Smaller = Weaker? Benchmarking Robustness of Quantized LLMs in Code Generation**

更小=更弱？量化LLM在代码生成中的基准测试鲁棒性 cs.SE

13 pages, 6 figures

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.22776v1) [paper-pdf](http://arxiv.org/pdf/2506.22776v1)

**Authors**: Sen Fang, Weiyuan Ding, Antonio Mastropaolo, Bowen Xu

**Abstract**: Quantization has emerged as a mainstream method for compressing Large Language Models (LLMs), reducing memory requirements and accelerating inference without architectural modifications. While existing research primarily focuses on evaluating the effectiveness of quantized LLMs compared to their original counterparts, the impact on robustness remains largely unexplored.In this paper, we present the first systematic investigation of how quantization affects the robustness of LLMs in code generation tasks. Through extensive experiments across four prominent LLM families (LLaMA, DeepSeek, CodeGen, and StarCoder) with parameter scales ranging from 350M to 33B, we evaluate robustness from dual perspectives: adversarial attacks on input prompts and noise perturbations on model architecture. Our findings challenge conventional wisdom by demonstrating that quantized LLMs often exhibit superior robustness compared to their full-precision counterparts, with 51.59% versus 42.86% of our adversarial experiments showing better resilience in quantized LLMs. Similarly, our noise perturbation experiments also confirm that LLMs after quantitation generally withstand higher levels of weight disturbances. These results suggest that quantization not only reduces computational requirements but can actually enhance LLMs' reliability in code generation tasks, providing valuable insights for developing more robust and efficient LLM deployment strategies.

摘要: 量化已成为压缩大型语言模型（LLM）、减少内存需求并加速推理的主流方法，无需修改架构。虽然现有的研究主要集中在评估量化LLM与原始同类相比的有效性，但对稳健性的影响在很大程度上尚未探索。在本文中，我们首次系统地研究量化如何影响LLM在代码生成任务中的稳健性。通过对四个著名的LLM家族（LLaMA、DeepSeek、CodeGen和StarCoder）进行广泛实验，参数范围从350 M到33 B，我们从双重角度评估稳健性：对输入提示的对抗攻击和模型架构的噪音扰动。我们的研究结果挑战了传统智慧，证明量化LLM通常表现出比全精度同行更出色的鲁棒性，我们的对抗实验中分别有51.59%和42.86%表现出量化LLM更好的弹性。同样，我们的噪音扰动实验也证实，定量后的LLM通常可以承受更高水平的体重扰动。这些结果表明，量化不仅降低了计算要求，而且实际上可以增强LLM在代码生成任务中的可靠性，为开发更稳健、更高效的LLM部署策略提供有价值的见解。



## **24. Kill Two Birds with One Stone! Trajectory enabled Unified Online Detection of Adversarial Examples and Backdoor Attacks**

一举两得！轨迹支持对抗性示例和后门攻击的统一在线检测 cs.CR

**SubmitDate**: 2025-06-28    [abs](http://arxiv.org/abs/2506.22722v1) [paper-pdf](http://arxiv.org/pdf/2506.22722v1)

**Authors**: Anmin Fu, Fanyu Meng, Huaibing Peng, Hua Ma, Zhi Zhang, Yifeng Zheng, Willy Susilo, Yansong Gao

**Abstract**: The proposed UniGuard is the first unified online detection framework capable of simultaneously addressing adversarial examples and backdoor attacks. UniGuard builds upon two key insights: first, both AE and backdoor attacks have to compromise the inference phase, making it possible to tackle them simultaneously during run-time via online detection. Second, an adversarial input, whether a perturbed sample in AE attacks or a trigger-carrying sample in backdoor attacks, exhibits distinctive trajectory signatures from a benign sample as it propagates through the layers of a DL model in forward inference. The propagation trajectory of the adversarial sample must deviate from that of its benign counterpart; otherwise, the adversarial objective cannot be fulfilled. Detecting these trajectory signatures is inherently challenging due to their subtlety; UniGuard overcomes this by treating the propagation trajectory as a time-series signal, leveraging LSTM and spectrum transformation to amplify differences between adversarial and benign trajectories that are subtle in the time domain. UniGuard exceptional efficiency and effectiveness have been extensively validated across various modalities (image, text, and audio) and tasks (classification and regression), ranging from diverse model architectures against a wide range of AE attacks and backdoor attacks, including challenging partial backdoors and dynamic triggers. When compared to SOTA methods, including ContraNet (NDSS 22) specific for AE detection and TED (IEEE SP 24) specific for backdoor detection, UniGuard consistently demonstrates superior performance, even when matched against each method's strengths in addressing their respective threats-each SOTA fails to parts of attack strategies while UniGuard succeeds for all.

摘要: 拟议的UniGuard是第一个能够同时解决对抗性示例和后门攻击的统一在线检测框架。UniGuard基于两个关键见解：首先，AE和后门攻击都必须损害推理阶段，以便可以通过在线检测在运行时同时解决它们。其次，对抗性输入，无论是AE攻击中的受干扰样本还是后门攻击中的携带样本，在良性样本在前向推理中传播时，都会表现出良性样本的独特轨迹特征。对抗性样本的传播轨迹必须偏离其良性样本的传播轨迹;否则，对抗性目标就无法实现。由于这些轨迹特征的微妙性，检测这些轨迹特征本质上具有挑战性; UniGuard通过将传播轨迹视为时间序列信号，利用LSTM和频谱变换来放大在时间域中微妙的对抗轨迹和良性轨迹之间的差异来克服了这一点。UniGuard卓越的效率和有效性已在各种形式（图像、文本和音频）和任务（分类和回归）中得到了广泛验证，范围从针对广泛的AE攻击和后门攻击的不同模型架构，包括具有挑战性的部分后门和动态触发器。与SOTA方法（包括专用于AE检测的ContrNet（NDSS 22）和专用于后门检测的TED（IEEE SP 24））相比，UniGuard始终表现出卓越的性能，即使与每种方法在解决各自威胁方面的优势相匹配-每个SOTA都失败了部分攻击策略，而UniGuard则成功了所有人。



## **25. VERA: Variational Inference Framework for Jailbreaking Large Language Models**

VERA：越狱大型语言模型的变分推理框架 cs.CR

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.22666v1) [paper-pdf](http://arxiv.org/pdf/2506.22666v1)

**Authors**: Anamika Lochab, Lu Yan, Patrick Pynadath, Xiangyu Zhang, Ruqi Zhang

**Abstract**: The rise of API-only access to state-of-the-art LLMs highlights the need for effective black-box jailbreak methods to identify model vulnerabilities in real-world settings. Without a principled objective for gradient-based optimization, most existing approaches rely on genetic algorithms, which are limited by their initialization and dependence on manually curated prompt pools. Furthermore, these methods require individual optimization for each prompt, failing to provide a comprehensive characterization of model vulnerabilities. To address this gap, we introduce VERA: Variational infErence fRamework for jAilbreaking. VERA casts black-box jailbreak prompting as a variational inference problem, training a small attacker LLM to approximate the target LLM's posterior over adversarial prompts. Once trained, the attacker can generate diverse, fluent jailbreak prompts for a target query without re-optimization. Experimental results show that VERA achieves strong performance across a range of target LLMs, highlighting the value of probabilistic inference for adversarial prompt generation.

摘要: 仅限API访问最先进的LLM的兴起凸显了有效的黑匣子越狱方法来识别现实世界环境中的模型漏洞的必要性。由于没有基于梯度的优化的原则目标，大多数现有方法都依赖于遗传算法，而遗传算法受到初始化和对手动策划提示池的依赖的限制。此外，这些方法需要对每个提示进行单独优化，无法提供模型漏洞的全面描述。为了解决这个差距，我们引入了VERA：变分影响Erence fRamework for jAilbreaking。VERA将黑匣子越狱提示视为变分推理问题，训练小型攻击者LLM在对抗性提示上逼近目标LLM的后验。经过训练后，攻击者可以为目标查询生成多样化、流畅的越狱提示，而无需重新优化。实验结果表明，VERA在一系列目标LLM中实现了强劲的性能，凸显了概率推理对对抗提示生成的价值。



## **26. MetaCipher: A General and Extensible Reinforcement Learning Framework for Obfuscation-Based Jailbreak Attacks on Black-Box LLMs**

MetaCipher：一个通用且可扩展的强化学习框架，用于对黑匣子LLM进行基于模糊的越狱攻击 cs.CR

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.22557v1) [paper-pdf](http://arxiv.org/pdf/2506.22557v1)

**Authors**: Boyuan Chen, Minghao Shao, Abdul Basit, Siddharth Garg, Muhammad Shafique

**Abstract**: The growing capabilities of large language models (LLMs) have exposed them to increasingly sophisticated jailbreak attacks. Among these, obfuscation-based attacks -- which encrypt malicious content to evade detection -- remain highly effective. By leveraging the reasoning ability of advanced LLMs to interpret encrypted prompts, such attacks circumvent conventional defenses that rely on keyword detection or context filtering. These methods are very difficult to defend against, as existing safety mechanisms are not designed to interpret or decode ciphered content. In this work, we propose \textbf{MetaCipher}, a novel obfuscation-based jailbreak framework, along with a reinforcement learning-based dynamic cipher selection mechanism that adaptively chooses optimal encryption strategies from a cipher pool. This approach enhances jailbreak effectiveness and generalizability across diverse task types, victim LLMs, and safety guardrails. Our framework is modular and extensible by design, supporting arbitrary cipher families and accommodating evolving adversarial strategies. We complement our method with a large-scale empirical analysis of cipher performance across multiple victim LLMs. Within as few as 10 queries, MetaCipher achieves over 92\% attack success rate (ASR) on most recent standard malicious prompt benchmarks against state-of-the-art non-reasoning LLMs, and over 74\% ASR against reasoning-capable LLMs, outperforming all existing obfuscation-based jailbreak methods. These results highlight the long-term robustness and adaptability of our approach, making it more resilient than prior methods in the face of advancing safety measures.

摘要: 大型语言模型（LLM）不断增长的能力使它们面临越来越复杂的越狱攻击。其中，基于模糊的攻击（对恶意内容进行加密以逃避检测）仍然非常有效。通过利用高级LLM的推理能力来解释加密提示，此类攻击绕过了依赖关键字检测或上下文过滤的传统防御措施。这些方法非常难以防御，因为现有的安全机制不是为了解释或解码加密内容而设计的。在这项工作中，我们提出了\textBF{MetaCipher}，这是一种新型的基于模糊的越狱框架，以及一种基于强化学习的动态密码选择机制，该机制从密码池中自适应地选择最佳加密策略。这种方法增强了不同任务类型、受害者LLM和安全护栏的越狱有效性和普遍性。我们的框架是模块化的，可通过设计扩展，支持任意密码族并适应不断发展的对抗策略。我们通过对多个受害LLM的密码性能进行大规模实证分析来补充我们的方法。在短短10个查询内，MetaCipher针对最新的非推理LLM在最新标准恶意提示基准上就达到了超过92%的攻击成功率（ASB），针对具有推理能力的LLM达到了超过74%的攻击成功率，优于所有现有的基于模糊的越狱方法。这些结果凸显了我们方法的长期稳健性和适应性，使其在面对先进的安全措施时比以前的方法更具弹性。



## **27. ARMOR: Robust Reinforcement Learning-based Control for UAVs under Physical Attacks**

ARMOR：物理攻击下无人机的鲁棒强化学习控制 cs.LG

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.22423v1) [paper-pdf](http://arxiv.org/pdf/2506.22423v1)

**Authors**: Pritam Dash, Ethan Chan, Nathan P. Lawrence, Karthik Pattabiraman

**Abstract**: Unmanned Aerial Vehicles (UAVs) depend on onboard sensors for perception, navigation, and control. However, these sensors are susceptible to physical attacks, such as GPS spoofing, that can corrupt state estimates and lead to unsafe behavior. While reinforcement learning (RL) offers adaptive control capabilities, existing safe RL methods are ineffective against such attacks. We present ARMOR (Adaptive Robust Manipulation-Optimized State Representations), an attack-resilient, model-free RL controller that enables robust UAV operation under adversarial sensor manipulation. Instead of relying on raw sensor observations, ARMOR learns a robust latent representation of the UAV's physical state via a two-stage training framework. In the first stage, a teacher encoder, trained with privileged attack information, generates attack-aware latent states for RL policy training. In the second stage, a student encoder is trained via supervised learning to approximate the teacher's latent states using only historical sensor data, enabling real-world deployment without privileged information. Our experiments show that ARMOR outperforms conventional methods, ensuring UAV safety. Additionally, ARMOR improves generalization to unseen attacks and reduces training cost by eliminating the need for iterative adversarial training.

摘要: 无人机（UAV）依赖机载传感器来进行感知、导航和控制。然而，这些传感器很容易受到物理攻击，例如GPS欺骗，这可能会破坏状态估计并导致不安全行为。虽然强化学习（RL）提供了自适应控制能力，但现有的安全强化学习方法对此类攻击无效。我们提出了ARMOR（自适应鲁棒操纵优化状态表示），这是一种具有攻击弹性、无模型的RL控制器，可在对抗性传感器操纵下实现鲁棒的无人机操作。ARMOR不是依赖原始传感器观察，而是通过两阶段训练框架学习无人机物理状态的稳健潜在表示。在第一阶段，用特权攻击信息训练的教师编码器生成攻击感知潜在状态用于RL策略训练。在第二阶段，通过监督学习训练学生编码器，仅使用历史传感器数据来逼近教师的潜在状态，从而实现在没有特权信息的情况下的现实世界部署。我们的实验表明，ARMOR优于传统方法，确保了无人机的安全。此外，ARMOR提高了对不可见攻击的概括性，并通过消除迭代对抗训练的需要来降低训练成本。



## **28. Secure Video Quality Assessment Resisting Adversarial Attacks**

安全的视频质量评估抵抗对抗攻击 cs.CV

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2410.06866v2) [paper-pdf](http://arxiv.org/pdf/2410.06866v2)

**Authors**: Ao-Xiang Zhang, Yuan-Gen Wang, Yu Ran, Weixuan Tang, Qingxiao Guan, Chunsheng Yang

**Abstract**: The exponential surge in video traffic has intensified the imperative for Video Quality Assessment (VQA). Leveraging cutting-edge architectures, current VQA models have achieved human-comparable accuracy. However, recent studies have revealed the vulnerability of existing VQA models against adversarial attacks. To establish a reliable and practical assessment system, a secure VQA model capable of resisting such malicious attacks is urgently demanded. Unfortunately, no attempt has been made to explore this issue. This paper first attempts to investigate general adversarial defense principles, aiming at endowing existing VQA models with security. Specifically, we first introduce random spatial grid sampling on the video frame for intra-frame defense. Then, we design pixel-wise randomization through a guardian map, globally neutralizing adversarial perturbations. Meanwhile, we extract temporal information from the video sequence as compensation for inter-frame defense. Building upon these principles, we present a novel VQA framework from the security-oriented perspective, termed SecureVQA. Extensive experiments indicate that SecureVQA sets a new benchmark in security while achieving competitive VQA performance compared with state-of-the-art models. Ablation studies delve deeper into analyzing the principles of SecureVQA, demonstrating their generalization and contributions to the security of leading VQA models.

摘要: 视频流量的指数级激增加剧了视频质量评估（VQA）的紧迫性。利用尖端架构，当前的VQA模型已实现人类可比的准确性。然而，最近的研究揭示了现有VQA模型抵御对抗攻击的脆弱性。为了建立可靠、实用的评估系统，迫切需要一种能够抵抗此类恶意攻击的安全VQA模型。不幸的是，尚未尝试探讨这个问题。本文首先尝试研究一般的对抗性防御原则，旨在赋予现有的VQA模型安全性。具体来说，我们首先在视频帧上引入随机空间网格采样以进行帧内防御。然后，我们通过守护者地图设计像素级随机化，在全球范围内中和对抗性扰动。同时，我们从视频序列中提取时间信息作为帧间防御的补偿。在这些原则的基础上，我们从面向安全的角度提出了一种新型VQA框架，称为SecureVQA。大量实验表明，SecureVQA在安全方面树立了新的基准，同时与最先进的模型相比实现了有竞争力的VQA性能。消融研究更深入地分析SecureVQA的原则，展示其普遍性和对领先VQA模型安全性的贡献。



## **29. A Self-scaled Approximate $\ell_0$ Regularization Robust Model for Outlier Detection**

用于离群点检测的自缩放近似$\ell_0$正规化鲁棒模型 eess.SP

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.22277v1) [paper-pdf](http://arxiv.org/pdf/2506.22277v1)

**Authors**: Pengyang Song, Jue Wang

**Abstract**: Robust regression models in the presence of outliers have significant practical relevance in areas such as signal processing, financial econometrics, and energy management. Many existing robust regression methods, either grounded in statistical theory or sparse signal recovery, typically rely on the explicit or implicit assumption of outlier sparsity to filter anomalies and recover the underlying signal or data. However, these methods often suffer from limited robustness or high computational complexity, rendering them inefficient for large-scale problems. In this work, we propose a novel robust regression model based on a Self-scaled Approximate l0 Regularization Model (SARM) scheme. By introducing a self-scaling mechanism into the regularization term, the proposed model mitigates the negative impact of uneven or excessively large outlier magnitudes on robustness. We also develop an alternating minimization algorithm grounded in Proximal Operators and Block Coordinate Descent. We rigorously prove the algorithm convergence. Empirical comparisons with several state-of-the-art robust regression methods demonstrate that SARM not only achieves superior robustness but also significantly improves computational efficiency. Motivated by both the theoretical error bound and empirical observations, we further design a Two-Stage SARM (TSSARM) framework, which better utilizes sample information when the singular values of the design matrix are widely spread, thereby enhancing robustness under certain conditions. Finally, we validate our approach on a real-world load forecasting task. The experimental results show that our method substantially enhances the robustness of load forecasting against adversarial data attacks, which is increasingly critical in the era of heightened data security concerns.

摘要: 存在异常值的稳健回归模型在信号处理、金融计量经济学和能源管理等领域具有重要的实际意义。许多现有的稳健回归方法，无论是基于统计理论还是基于稀疏信号恢复，通常依赖于离群点稀疏性的显式或隐式假设来过滤异常并恢复基础信号或数据。然而，这些方法往往鲁棒性有限或计算复杂性高，导致它们对于大规模问题效率低下。在这项工作中，我们提出了一种基于自缩放近似10正规化模型（SARM）方案的新型鲁棒回归模型。通过在正则化项中引入自缩放机制，该模型减轻了不均匀或过大的离群值幅度对鲁棒性的负面影响。我们还开发了一个交替最小化算法接地在邻近算子和块坐标下降。我们严格证明了算法的收敛性。与几种最先进的鲁棒回归方法的实证比较表明，SARM不仅实现了优越的鲁棒性，而且显着提高了计算效率。在理论误差界和经验观察的激励下，我们进一步设计了一个两阶段SARM（TSARM）框架，当设计矩阵的奇异值广泛传播时，该框架可以更好地利用样本信息，从而增强某些条件下的鲁棒性。最后，我们在现实世界的负荷预测任务中验证了我们的方法。实验结果表明，我们的方法大大增强了负载预测针对对抗性数据攻击的鲁棒性，这在数据安全问题加剧的时代变得越来越重要。



## **30. Enhancing Object Detection Robustness: Detecting and Restoring Confidence in the Presence of Adversarial Patch Attacks**

增强对象检测稳健性：在存在对抗性补丁攻击时检测和恢复信心 cs.CV

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2403.12988v2) [paper-pdf](http://arxiv.org/pdf/2403.12988v2)

**Authors**: Roie Kazoom, Raz Birman, Ofer Hadar

**Abstract**: The widespread adoption of computer vision systems has underscored their susceptibility to adversarial attacks, particularly adversarial patch attacks on object detectors. This study evaluates defense mechanisms for the YOLOv5 model against such attacks. Optimized adversarial patches were generated and placed in sensitive image regions, by applying EigenCAM and grid search to determine optimal placement. We tested several defenses, including Segment and Complete (SAC), Inpainting, and Latent Diffusion Models. Our pipeline comprises three main stages: patch application, object detection, and defense analysis. Results indicate that adversarial patches reduce average detection confidence by 22.06\%. Defenses restored confidence levels by 3.45\% (SAC), 5.05\% (Inpainting), and significantly improved them by 26.61\%, which even exceeds the original accuracy levels, when using the Latent Diffusion Model, highlighting its superior effectiveness in mitigating the effects of adversarial patches.

摘要: 计算机视觉系统的广泛采用强调了它们对对抗性攻击的敏感性，特别是对对象检测器的对抗性补丁攻击。本研究评估了针对此类攻击的YOLOv5模型的防御机制。通过应用EigenCAM和网格搜索来确定最佳位置，生成优化的对抗补丁并将其放置在敏感图像区域中。我们测试了几种防御方法，包括分段和完全（SAC），修复和潜在扩散模型。我们的管道包括三个主要阶段：补丁应用、对象检测和防御分析。结果表明，对抗补丁使平均检测置信度降低了22.06%。使用潜伏扩散模型时，防御使信心水平恢复了3.45%（SAC）、5.05%（修补），并显着提高了26.61%，甚至超过了最初的准确性水平，凸显了其在减轻对抗性斑块影响方面的卓越有效性。



## **31. Advancing Jailbreak Strategies: A Hybrid Approach to Exploiting LLM Vulnerabilities and Bypassing Modern Defenses**

推进越狱策略：利用LLM漏洞和扩展现代防御的混合方法 cs.CL

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.21972v1) [paper-pdf](http://arxiv.org/pdf/2506.21972v1)

**Authors**: Mohamed Ahmed, Mohamed Abdelmouty, Mingyu Kim, Gunvanth Kandula, Alex Park, James C. Davis

**Abstract**: The advancement of Pre-Trained Language Models (PTLMs) and Large Language Models (LLMs) has led to their widespread adoption across diverse applications. Despite their success, these models remain vulnerable to attacks that exploit their inherent weaknesses to bypass safety measures. Two primary inference-phase threats are token-level and prompt-level jailbreaks. Token-level attacks embed adversarial sequences that transfer well to black-box models like GPT but leave detectable patterns and rely on gradient-based token optimization, whereas prompt-level attacks use semantically structured inputs to elicit harmful responses yet depend on iterative feedback that can be unreliable. To address the complementary limitations of these methods, we propose two hybrid approaches that integrate token- and prompt-level techniques to enhance jailbreak effectiveness across diverse PTLMs. GCG + PAIR and the newly explored GCG + WordGame hybrids were evaluated across multiple Vicuna and Llama models. GCG + PAIR consistently raised attack-success rates over its constituent techniques on undefended models; for instance, on Llama-3, its Attack Success Rate (ASR) reached 91.6%, a substantial increase from PAIR's 58.4% baseline. Meanwhile, GCG + WordGame matched the raw performance of WordGame maintaining a high ASR of over 80% even under stricter evaluators like Mistral-Sorry-Bench. Crucially, both hybrids retained transferability and reliably pierced advanced defenses such as Gradient Cuff and JBShield, which fully blocked single-mode attacks. These findings expose previously unreported vulnerabilities in current safety stacks, highlight trade-offs between raw success and defensive robustness, and underscore the need for holistic safeguards against adaptive adversaries.

摘要: 预训练语言模型（PTLM）和大型语言模型（LLM）的进步导致它们在不同的应用程序中广泛采用。尽管取得了成功，但这些模型仍然容易受到利用其固有弱点绕过安全措施的攻击。两种主要的推理阶段威胁是代币级和预算级越狱。令牌级攻击嵌入对抗序列，这些序列可以很好地传输到GPT等黑匣子模型，但留下可检测的模式并依赖于基于梯度的令牌优化，而预算级攻击使用语义结构化的输入来引发有害响应，但依赖于可能不可靠的迭代反馈。为了解决这些方法的互补局限性，我们提出了两种混合方法，集成代币和预算级技术，以增强不同PTLM之间的越狱有效性。GCG + PAIR和新探索的GCG + WordGame混合体在多个Vicuna和Lama模型中进行了评估。GCG + PAIR在无防御模型上始终提高了其组成技术的攻击成功率;例如，在Lama-3上，其攻击成功率（ASB）达到91.6%，比PAIR的58.4%基线大幅提高。与此同时，GCG + WordGame与WordGame的原始表现相媲美，即使在Mistral-Sorry-Bench等更严格的评估者下，也保持了超过80%的高ASB。至关重要的是，这两种混合体都保留了可转移性，并可靠地突破了Gradient Cuff和JB Shield等先进防御，从而完全阻止了单一模式攻击。这些发现暴露了当前安全堆栈中以前未报告的漏洞，强调了原始成功和防御稳健性之间的权衡，并强调了针对适应性对手的全面保障措施的必要性。



## **32. Releasing Inequality Phenomenon in $\ell_{\infty}$-norm Adversarial Training via Input Gradient Distillation**

通过输入梯度蒸馏释放$\ell_{\infty}$-norm对抗训练中的不平等现象 cs.CV

16 pages. Accepted by IEEE TIFS

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2305.09305v3) [paper-pdf](http://arxiv.org/pdf/2305.09305v3)

**Authors**: Junxi Chen, Junhao Dong, Xiaohua Xie, Jianhuang Lai

**Abstract**: Adversarial training (AT) is considered the most effective defense against adversarial attacks. However, a recent study revealed that \(\ell_{\infty}\)-norm adversarial training (\(\ell_{\infty}\)-AT) will also induce unevenly distributed input gradients, which is called the inequality phenomenon. This phenomenon makes the \(\ell_{\infty}\)-norm adversarially trained model more vulnerable than the standard-trained model when high-attribution or randomly selected pixels are perturbed, enabling robust and practical black-box attacks against \(\ell_{\infty}\)-adversarially trained models. In this paper, we propose a simple yet effective method called Input Gradient Distillation (IGD) to release the inequality phenomenon in $\ell_{\infty}$-AT. IGD distills the standard-trained teacher model's equal decision pattern into the $\ell_{\infty}$-adversarially trained student model by aligning input gradients of the student model and the standard-trained model with the Cosine Similarity. Experiments show that IGD can mitigate the inequality phenomenon and its threats while preserving adversarial robustness. Compared to vanilla $\ell_{\infty}$-AT, IGD reduces error rates against inductive noise, inductive occlusion, random noise, and noisy images in ImageNet-C by up to 60\%, 16\%, 50\%, and 21\%, respectively. Other than empirical experiments, we also conduct a theoretical analysis to explain why releasing the inequality phenomenon can improve such robustness and discuss why the severity of the inequality phenomenon varies according to the dataset's image resolution. Our code is available at https://github.com/fhdnskfbeuv/Inuput-Gradient-Distillation

摘要: 对抗训练（AT）被认为是对抗攻击最有效的防御。然而，最近的一项研究表明，\（\ell_{\infty}\）-规范对抗训练（\（\ell_{\infty}\）-AT）也会引起不均匀分布的输入梯度，这被称为不平等现象。当高属性或随机选择的像素受到干扰时，这种现象使得\（\ell_{\infty}\）-规范对抗训练模型比标准训练模型更容易受到攻击，从而能够对\（\ell_{\infty}\）-对抗训练模型进行鲁棒且实用的黑匣子攻击。在本文中，我们提出了一种简单而有效的方法，称为输入梯度蒸馏（IGD），以消除$\ell_{\infty}$-AT中的不等式现象。IGD通过将学生模型和标准训练模型的输入梯度与Cosine相似性对齐，将标准训练教师模型的平等决策模式提炼为$\ell_{\infty}$-对抗训练学生模型。实验表明，IGD可以缓解不平等现象及其威胁，同时保持对抗稳健性。与vanilla $\ell_{\infty}$-AT相比，IGD将ImageNet-C中感应性噪音、感应性遮挡、随机噪音和含噪图像的错误率分别降低了高达60%、16%、50%和21%。除了实证实验之外，我们还进行理论分析来解释为什么释放不平等现象可以提高这种稳健性，并讨论为什么不平等现象的严重程度会根据数据集的图像分辨率而变化。我们的代码可在https://github.com/fhdnskfbeuv/Inuput-Gradient-Distillation上获取



## **33. One Video to Steal Them All: 3D-Printing IP Theft through Optical Side-Channels**

一个视频窃取全部内容：通过光学侧通道的3D打印IP盗窃 cs.CR

17 pages [Extended Version]

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.21897v1) [paper-pdf](http://arxiv.org/pdf/2506.21897v1)

**Authors**: Twisha Chattopadhyay, Fabricio Ceschin, Marco E. Garza, Dymytriy Zyunkin, Animesh Chhotaray, Aaron P. Stebner, Saman Zonouz, Raheem Beyah

**Abstract**: The 3D printing industry is rapidly growing and increasingly adopted across various sectors including manufacturing, healthcare, and defense. However, the operational setup often involves hazardous environments, necessitating remote monitoring through cameras and other sensors, which opens the door to cyber-based attacks. In this paper, we show that an adversary with access to video recordings of the 3D printing process can reverse engineer the underlying 3D print instructions. Our model tracks the printer nozzle movements during the printing process and maps the corresponding trajectory into G-code instructions. Further, it identifies the correct parameters such as feed rate and extrusion rate, enabling successful intellectual property theft. To validate this, we design an equivalence checker that quantitatively compares two sets of 3D print instructions, evaluating their similarity in producing objects alike in shape, external appearance, and internal structure. Unlike simple distance-based metrics such as normalized mean square error, our equivalence checker is both rotationally and translationally invariant, accounting for shifts in the base position of the reverse engineered instructions caused by different camera positions. Our model achieves an average accuracy of 90.87 percent and generates 30.20 percent fewer instructions compared to existing methods, which often produce faulty or inaccurate prints. Finally, we demonstrate a fully functional counterfeit object generated by reverse engineering 3D print instructions from video.

摘要: 3D打印行业正在迅速发展，并在制造业、医疗保健和国防等各个行业越来越多地采用。然而，操作设置通常涉及危险的环境，需要通过摄像头和其他传感器进行远程监控，这为基于网络的攻击打开了大门。在本文中，我们表明，能够访问3D打印过程视频记录的对手可以对底层3D打印指令进行反向工程。我们的模型在打印过程中跟踪打印机喷嘴的移动，并将相应的轨迹映射到G代码指令中。此外，它还可以识别正确的参数，例如给料率和挤出率，从而实现成功的知识产权盗窃。为了验证这一点，我们设计了一个等效性检查器，该检查器定量比较两组3D打印指令，评估它们在生成形状、外观和内部结构相似的对象时的相似性。与简单的基于距离的指标（例如标准化均方误差）不同，我们的等效检查器在旋转和平移上都是不变的，可以考虑不同摄像机位置引起的反向工程指令基本位置的变化。与经常产生有缺陷或不准确的印刷品的现有方法相比，我们的模型实现了90.87%的平均准确性，生成的指令减少了30.20%。最后，我们演示了通过从视频中进行反向工程3D打印指令生成的功能齐全的伪造对象。



## **34. On the Feasibility of Poisoning Text-to-Image AI Models via Adversarial Mislabeling**

论通过对抗性错误标签毒害文本到图像人工智能模型的可行性 cs.CR

ACM Conference on Computer and Communications Security 2025

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.21874v1) [paper-pdf](http://arxiv.org/pdf/2506.21874v1)

**Authors**: Stanley Wu, Ronik Bhaskar, Anna Yoo Jeong Ha, Shawn Shan, Haitao Zheng, Ben Y. Zhao

**Abstract**: Today's text-to-image generative models are trained on millions of images sourced from the Internet, each paired with a detailed caption produced by Vision-Language Models (VLMs). This part of the training pipeline is critical for supplying the models with large volumes of high-quality image-caption pairs during training. However, recent work suggests that VLMs are vulnerable to stealthy adversarial attacks, where adversarial perturbations are added to images to mislead the VLMs into producing incorrect captions.   In this paper, we explore the feasibility of adversarial mislabeling attacks on VLMs as a mechanism to poisoning training pipelines for text-to-image models. Our experiments demonstrate that VLMs are highly vulnerable to adversarial perturbations, allowing attackers to produce benign-looking images that are consistently miscaptioned by the VLM models. This has the effect of injecting strong "dirty-label" poison samples into the training pipeline for text-to-image models, successfully altering their behavior with a small number of poisoned samples. We find that while potential defenses can be effective, they can be targeted and circumvented by adaptive attackers. This suggests a cat-and-mouse game that is likely to reduce the quality of training data and increase the cost of text-to-image model development. Finally, we demonstrate the real-world effectiveness of these attacks, achieving high attack success (over 73%) even in black-box scenarios against commercial VLMs (Google Vertex AI and Microsoft Azure).

摘要: 当今的文本到图像生成模型是在来自互联网的数百万张图像上训练的，每个图像都与视觉语言模型（VLM）生成的详细标题配对。训练管道的这一部分对于在训练期间为模型提供大量高质量图像字幕对至关重要。然而，最近的研究表明，VLM很容易受到隐蔽的对抗攻击，对抗性扰动被添加到图像中以误导VLM产生错误的字幕。   本文中，我们探讨了对VLM的对抗性错误标签攻击作为毒害文本到图像模型训练管道的机制的可行性。我们的实验表明，VLM非常容易受到对抗性扰动的影响，这使得攻击者能够生成看似友善的图像，而这些图像始终被VLM模型字幕错误。这的效果是将强“肮脏标签”毒物样本注入文本到图像模型的训练管道中，用少量毒物样本成功改变它们的行为。我们发现，虽然潜在的防御措施可能有效，但它们可能会被适应性攻击者瞄准和规避。这表明猫鼠游戏可能会降低训练数据的质量并增加文本到图像模型开发的成本。最后，我们展示了这些攻击在现实世界中的有效性，即使在针对商业VLM（Google Vertex AI和Microsoft Azure）的黑匣子场景中，也实现了很高的攻击成功率（超过73%）。



## **35. Adversarial Threats in Quantum Machine Learning: A Survey of Attacks and Defenses**

量子机器学习中的对抗威胁：攻击和防御调查 quant-ph

23 pages, 5 figures

**SubmitDate**: 2025-06-27    [abs](http://arxiv.org/abs/2506.21842v1) [paper-pdf](http://arxiv.org/pdf/2506.21842v1)

**Authors**: Archisman Ghosh, Satwik Kundu, Swaroop Ghosh

**Abstract**: Quantum Machine Learning (QML) integrates quantum computing with classical machine learning, primarily to solve classification, regression and generative tasks. However, its rapid development raises critical security challenges in the Noisy Intermediate-Scale Quantum (NISQ) era. This chapter examines adversarial threats unique to QML systems, focusing on vulnerabilities in cloud-based deployments, hybrid architectures, and quantum generative models. Key attack vectors include model stealing via transpilation or output extraction, data poisoning through quantum-specific perturbations, reverse engineering of proprietary variational quantum circuits, and backdoor attacks. Adversaries exploit noise-prone quantum hardware and insufficiently secured QML-as-a-Service (QMLaaS) workflows to compromise model integrity, ownership, and functionality. Defense mechanisms leverage quantum properties to counter these threats. Noise signatures from training hardware act as non-invasive watermarks, while hardware-aware obfuscation techniques and ensemble strategies disrupt cloning attempts. Emerging solutions also adapt classical adversarial training and differential privacy to quantum settings, addressing vulnerabilities in quantum neural networks and generative architectures. However, securing QML requires addressing open challenges such as balancing noise levels for reliability and security, mitigating cross-platform attacks, and developing quantum-classical trust frameworks. This chapter summarizes recent advances in attacks and defenses, offering a roadmap for researchers and practitioners to build robust, trustworthy QML systems resilient to evolving adversarial landscapes.

摘要: 量子机器学习（QML）将量子计算与经典机器学习相结合，主要用于解决分类，回归和生成任务。然而，它的快速发展在噪声中等规模量子（NISQ）时代提出了关键的安全挑战。本章研究QML系统特有的对抗性威胁，重点关注基于云的部署，混合架构和量子生成模型中的漏洞。关键攻击向量包括通过转译或输出提取的模型窃取、通过特定量子扰动的数据中毒、专有变分量子电路的逆向工程以及后门攻击。攻击者利用易受噪声影响的量子硬件和安全性不足的QML-as-a-Service（QMLaaS）工作流来破坏模型的完整性、所有权和功能。防御机制利用量子特性来对抗这些威胁。来自训练硬件的噪声签名充当非侵入性水印，而硬件感知的混淆技术和集成策略会破坏克隆尝试。新兴的解决方案还将经典的对抗训练和差分隐私适应于量子设置，解决了量子神经网络和生成架构中的漏洞。然而，保护QML需要解决开放性挑战，例如平衡可靠性和安全性的噪声水平，减轻跨平台攻击，以及开发量子经典信任框架。本章总结了攻击和防御方面的最新进展，为研究人员和从业者提供了一个路线图，以构建健壮、可靠的QML系统，适应不断变化的对抗环境。



## **36. Quantum Token Obfuscation via Superposition: A Post-Quantum Security Framework Using Multi-Basis Verification and Entropy-Driven Evolution**

通过叠加实现量子代币混淆：使用多基验证和熵驱动进化的后量子安全框架 quant-ph

16 pages, Significant revisions based on reviewer feedback

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2411.01252v3) [paper-pdf](http://arxiv.org/pdf/2411.01252v3)

**Authors**: S. M. Yousuf Iqbal Tomal, Abdullah Al Shafin

**Abstract**: Traditional cryptographic techniques, including token obfuscation, are increasingly vulnerable to quantum attacks due to advancements in quantum computing. Quantum algorithms such as Shor's and Grover's pose significant threats to classical security methods, necessitating quantum-resistant alternatives. This study proposes a quantum-based approach to token obfuscation that leverages superposition and multi-basis verification to enhance security against quantum adversaries. Tokens are encoded in quantum superposition states, ensuring probabilistic concealment until measured. A multi-basis verification protocol strengthens authentication by requiring validation across multiple quantum measurement bases. Additionally, a quantum decay protocol and token refresh mechanism dynamically manage the token lifecycle to prevent prolonged exposure and replay attacks. The model was tested through quantum simulations, evaluating entropy quality, adversarial robustness, and token verification reliability. Experimental validation demonstrates an entropy quality score of 0.9996, a 0% attack success rate across five adversarial models, and a 67% false positive rate, indicating strict security constraints. These findings confirm the effectiveness of quantum-based token obfuscation in preventing unauthorized reconstruction. The proposed approach provides a foundation for post-quantum cryptographic security by integrating entropy-driven state transformations, dynamic token evolution, and multi-basis verification. Future work will focus on optimizing computational efficiency and testing real-world implementations on quantum hardware.

摘要: 由于量子计算的进步，包括令牌混淆在内的传统加密技术越来越容易受到量子攻击。Shor和Grover等量子算法对经典安全方法构成了重大威胁，需要抗量子替代方案。这项研究提出了一种基于量子的代币混淆方法，利用叠加和多基验证来增强针对量子对手的安全性。代币以量子叠加状态编码，确保在测量之前的概率隐藏。多基验证协议通过要求跨多个量子测量基进行验证来加强身份验证。此外，量子衰变协议和令牌刷新机制动态管理令牌生命周期，以防止长时间暴露和重播攻击。该模型通过量子模拟进行了测试，评估了信息质量、对抗稳健性和令牌验证可靠性。实验验证表明，信息质量评分为0.9996，五个对抗模型的攻击成功率为0%，假阳性率为67%，表明安全约束严格。这些发现证实了基于量子的令牌混淆在防止未经授权重建方面的有效性。所提出的方法通过集成信息量驱动的状态转换、动态令牌进化和多基验证，为后量子密码安全提供了基础。未来的工作将重点关注优化计算效率和测试量子硬件上的现实实现。



## **37. PuriDefense: Randomized Local Implicit Adversarial Purification for Defending Black-box Query-based Attacks**

PuriDefense：用于防御基于黑匣子查询的攻击的随机本地隐式对抗净化 cs.CR

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2401.10586v2) [paper-pdf](http://arxiv.org/pdf/2401.10586v2)

**Authors**: Ping Guo, Xiang Li, Zhiyuan Yang, Xi Lin, Qingchuan Zhao, Qingfu Zhang

**Abstract**: Black-box query-based attacks constitute significant threats to Machine Learning as a Service (MLaaS) systems since they can generate adversarial examples without accessing the target model's architecture and parameters. Traditional defense mechanisms, such as adversarial training, gradient masking, and input transformations, either impose substantial computational costs or compromise the test accuracy of non-adversarial inputs. To address these challenges, we propose an efficient defense mechanism, PuriDefense, that employs random patch-wise purifications with an ensemble of lightweight purification models at a low level of inference cost. These models leverage the local implicit function and rebuild the natural image manifold. Our theoretical analysis suggests that this approach slows down the convergence of query-based attacks by incorporating randomness into purifications. Extensive experiments on CIFAR-10 and ImageNet validate the effectiveness of our proposed purifier-based defense mechanism, demonstrating significant improvements in robustness against query-based attacks.

摘要: 基于黑匣子查询的攻击对机器学习即服务（MLaas）系统构成重大威胁，因为它们可以在不访问目标模型的架构和参数的情况下生成对抗性示例。传统的防御机制，例如对抗性训练、梯度掩蔽和输入转换，要么会带来巨大的计算成本，要么会损害非对抗性输入的测试准确性。为了应对这些挑战，我们提出了一种有效的防御机制PuriDefense，该机制采用随机逐块纯化和一系列轻量级纯化模型，以较低的推理成本进行。这些模型利用局部隐式函数并重建自然图像集。我们的理论分析表明，这种方法通过将随机性纳入净化中来减缓基于查询的攻击的收敛。CIFAR-10和ImageNet上的大量实验验证了我们提出的基于净化器的防御机制的有效性，证明了针对基于查询的攻击的鲁棒性有了显着改进。



## **38. A Troublemaker with Contagious Jailbreak Makes Chaos in Honest Towns**

具有传染性的越狱麻烦制造者扰乱诚实城镇 cs.CL

ACL 2025 Main

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2410.16155v2) [paper-pdf](http://arxiv.org/pdf/2410.16155v2)

**Authors**: Tianyi Men, Pengfei Cao, Zhuoran Jin, Yubo Chen, Kang Liu, Jun Zhao

**Abstract**: With the development of large language models, they are widely used as agents in various fields. A key component of agents is memory, which stores vital information but is susceptible to jailbreak attacks. Existing research mainly focuses on single-agent attacks and shared memory attacks. However, real-world scenarios often involve independent memory. In this paper, we propose the Troublemaker Makes Chaos in Honest Town (TMCHT) task, a large-scale, multi-agent, multi-topology text-based attack evaluation framework. TMCHT involves one attacker agent attempting to mislead an entire society of agents. We identify two major challenges in multi-agent attacks: (1) Non-complete graph structure, (2) Large-scale systems. We attribute these challenges to a phenomenon we term toxicity disappearing. To address these issues, we propose an Adversarial Replication Contagious Jailbreak (ARCJ) method, which optimizes the retrieval suffix to make poisoned samples more easily retrieved and optimizes the replication suffix to make poisoned samples have contagious ability. We demonstrate the superiority of our approach in TMCHT, with 23.51%, 18.95%, and 52.93% improvements in line topology, star topology, and 100-agent settings. Encourage community attention to the security of multi-agent systems.

摘要: 随着大型语言模型的发展，它们被广泛用作各个领域的代理。代理的一个关键组成部分是内存，它存储重要信息，但容易受到越狱攻击。现有的研究主要集中在单代理攻击和共享内存攻击上。然而，现实世界的场景通常涉及独立记忆。在本文中，我们提出了Troubblemaker Makes Chaos in Honest Town（TMCHT）任务，这是一个大规模、多代理、多基于文本的攻击评估框架。TMCHT涉及一名攻击者特工试图误导整个特工社会。我们确定了多代理攻击中的两个主要挑战：（1）不完整的图结构，（2）大规模系统。我们将这些挑战归因于我们称之为毒性消失的现象。为了解决这些问题，我们提出了一种对抗复制传染越狱（ASCJ）方法，该方法优化检索后缀以使中毒样本更容易检索，并优化复制后缀以使中毒样本具有传染能力。我们在TMCHT中证明了我们的方法的优越性，线路布局、星型布局和100个代理设置分别提高了23.51%、18.95%和52.93%。鼓励社区关注多代理系统的安全性。



## **39. Generative Adversarial Evasion and Out-of-Distribution Detection for UAV Cyber-Attacks**

无人机网络攻击的生成性对抗规避和分布外检测 cs.LG

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.21142v1) [paper-pdf](http://arxiv.org/pdf/2506.21142v1)

**Authors**: Deepak Kumar Panda, Weisi Guo

**Abstract**: The growing integration of UAVs into civilian airspace underscores the need for resilient and intelligent intrusion detection systems (IDS), as traditional anomaly detection methods often fail to identify novel threats. A common approach treats unfamiliar attacks as out-of-distribution (OOD) samples; however, this leaves systems vulnerable when mitigation is inadequate. Moreover, conventional OOD detectors struggle to distinguish stealthy adversarial attacks from genuine OOD events. This paper introduces a conditional generative adversarial network (cGAN)-based framework for crafting stealthy adversarial attacks that evade IDS mechanisms. We first design a robust multi-class IDS classifier trained on benign UAV telemetry and known cyber-attacks, including Denial of Service (DoS), false data injection (FDI), man-in-the-middle (MiTM), and replay attacks. Using this classifier, our cGAN perturbs known attacks to generate adversarial samples that misclassify as benign while retaining statistical resemblance to OOD distributions. These adversarial samples are iteratively refined to achieve high stealth and success rates. To detect such perturbations, we implement a conditional variational autoencoder (CVAE), leveraging negative log-likelihood to separate adversarial inputs from authentic OOD samples. Comparative evaluation shows that CVAE-based regret scores significantly outperform traditional Mahalanobis distance-based detectors in identifying stealthy adversarial threats. Our findings emphasize the importance of advanced probabilistic modeling to strengthen IDS capabilities against adaptive, generative-model-based cyber intrusions.

摘要: 无人机越来越多地融入民用空域凸显了对弹性和智能入侵检测系统（IDS）的需求，因为传统的异常检测方法往往无法识别新的威胁。一种常见的方法将不熟悉的攻击视为分发外（OOD）样本;但是，当缓解措施不充分时，这会使系统变得脆弱。此外，传统的OOD检测器很难区分隐形对抗攻击与真正的OOD事件。本文介绍了一个基于条件生成对抗网络（cGAN）的框架，用于设计逃避IDS机制的隐形对抗攻击。我们首先设计了一个强大的多类IDS分类器，该分类器基于良性无人机遥感和已知的网络攻击（包括拒绝服务（DPS）、虚假数据注入（Direct）、中间人（MiTM）和重播攻击）进行训练。使用这个分类器，我们的cGAN扰乱已知的攻击，以生成敌对样本，这些样本错误分类为良性，同时保留与OOD分布的统计相似性。这些对抗样本经过迭代细化，以实现高隐形率和成功率。为了检测此类扰动，我们实现了一个条件变分自动编码器（CVAE），利用负的log似然性将对抗输入与真实的OOD样本分开。比较评估表明，在识别隐形对抗威胁方面，基于CVAE的后悔分数显着优于传统的Mahalanobis距离检测器。我们的研究结果强调了高级概率建模对于加强IDS抵御自适应、基于生成模型的网络入侵的能力的重要性。



## **40. Curriculum-Guided Antifragile Reinforcement Learning for Secure UAV Deconfliction under Observation-Space Attacks**

基于课程引导的反脆弱强化学习的无人机观测空间攻击安全解冲突 cs.LG

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.21129v1) [paper-pdf](http://arxiv.org/pdf/2506.21129v1)

**Authors**: Deepak Kumar Panda, Adolfo Perrusquia, Weisi Guo

**Abstract**: Reinforcement learning (RL) policies deployed in safety-critical systems, such as unmanned aerial vehicle (UAV) navigation in dynamic airspace, are vulnerable to out-ofdistribution (OOD) adversarial attacks in the observation space. These attacks induce distributional shifts that significantly degrade value estimation, leading to unsafe or suboptimal decision making rendering the existing policy fragile. To address this vulnerability, we propose an antifragile RL framework designed to adapt against curriculum of incremental adversarial perturbations. The framework introduces a simulated attacker which incrementally increases the strength of observation-space perturbations which enables the RL agent to adapt and generalize across a wider range of OOD observations and anticipate previously unseen attacks. We begin with a theoretical characterization of fragility, formally defining catastrophic forgetting as a monotonic divergence in value function distributions with increasing perturbation strength. Building on this, we define antifragility as the boundedness of such value shifts and derive adaptation conditions under which forgetting is stabilized. Our method enforces these bounds through iterative expert-guided critic alignment using Wasserstein distance minimization across incrementally perturbed observations. We empirically evaluate the approach in a UAV deconfliction scenario involving dynamic 3D obstacles. Results show that the antifragile policy consistently outperforms standard and robust RL baselines when subjected to both projected gradient descent (PGD) and GPS spoofing attacks, achieving up to 15% higher cumulative reward and over 30% fewer conflict events. These findings demonstrate the practical and theoretical viability of antifragile reinforcement learning for secure and resilient decision-making in environments with evolving threat scenarios.

摘要: 部署在安全关键系统中的强化学习（RL）策略，例如动态空域中的无人机（UAV）导航，容易受到观测空间中的分布外（OOD）对抗性攻击。这些攻击会导致分布变化，从而显著降低价值估计，导致不安全或次优决策，使现有政策变得脆弱。为了解决这个漏洞，我们提出了一个反脆弱的强化学习框架，旨在适应课程的增量对抗扰动。该框架引入了一个模拟攻击者，该攻击者逐渐增加观测空间扰动的强度，使RL代理能够在更大范围的OOD观测中进行适应和概括，并预测以前看不见的攻击。我们首先从脆弱性的理论表征，正式定义灾难性遗忘作为一个单调的偏离值函数分布的扰动强度增加。在此基础上，我们将反脆弱性定义为这种价值转移的有界性，并推导出遗忘稳定的适应条件。我们的方法通过迭代专家指导的评论家对齐使用Wasserstein距离最小化增量扰动观测来强制执行这些界限。我们根据经验评估了涉及动态3D障碍物的无人机冲突场景中的方法。结果表明，当受到投影梯度下降（PVD）和GPS欺骗攻击时，反脆弱性政策始终优于标准和稳健的RL基线，实现了高达15%的累积奖励和超过30%的冲突事件。这些发现证明了反脆弱强化学习在威胁场景不断变化的环境中进行安全和弹性决策的实践和理论可行性。



## **41. Robust Policy Switching for Antifragile Reinforcement Learning for UAV Deconfliction in Adversarial Environments**

对抗环境中无人机去冲突的反脆弱强化学习的鲁棒策略切换 cs.LG

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.21127v1) [paper-pdf](http://arxiv.org/pdf/2506.21127v1)

**Authors**: Deepak Kumar Panda, Weisi Guo

**Abstract**: The increasing automation of navigation for unmanned aerial vehicles (UAVs) has exposed them to adversarial attacks that exploit vulnerabilities in reinforcement learning (RL) through sensor manipulation. Although existing robust RL methods aim to mitigate such threats, their effectiveness has limited generalization to out-of-distribution shifts from the optimal value distribution, as they are primarily designed to handle fixed perturbation. To address this limitation, this paper introduces an antifragile RL framework that enhances adaptability to broader distributional shifts by incorporating a switching mechanism based on discounted Thompson sampling (DTS). This mechanism dynamically selects among multiple robust policies to minimize adversarially induced state-action-value distribution shifts. The proposed approach first derives a diverse ensemble of action robust policies by accounting for a range of perturbations in the policy space. These policies are then modeled as a multiarmed bandit (MAB) problem, where DTS optimally selects policies in response to nonstationary Bernoulli rewards, effectively adapting to evolving adversarial strategies. Theoretical framework has also been provided where by optimizing the DTS to minimize the overall regrets due to distributional shift, results in effective adaptation against unseen adversarial attacks thus inducing antifragility. Extensive numerical simulations validate the effectiveness of the proposed framework in complex navigation environments with multiple dynamic three-dimensional obstacles and with stronger projected gradient descent (PGD) and spoofing attacks. Compared to conventional robust, non-adaptive RL methods, the antifragile approach achieves superior performance, demonstrating shorter navigation path lengths and a higher rate of conflict-free navigation trajectories compared to existing robust RL techniques

摘要: 无人机（UAV）导航自动化程度的不断提高，使它们面临对抗性攻击，这些攻击通过传感器操纵利用强化学习（RL）中的漏洞。虽然现有的鲁棒RL方法旨在减轻这种威胁，但它们的有效性限制了从最优值分布到分布外偏移的推广，因为它们主要是为了处理固定扰动而设计的。为了解决这个问题，本文介绍了一个反脆弱的RL框架，提高了适应性更广泛的分布变化，通过纳入一个开关机制的基础上折扣汤普森采样（Thompson sampling，缩写）。该机制在多个稳健策略中动态选择，以最大限度地减少不利引起的状态-动作-价值分布转变。所提出的方法首先通过考虑政策空间中的一系列扰动来推导出多样化的行动稳健政策。然后，这些政策被建模为多臂强盗（MAB）问题，其中NPS根据非平稳伯努里奖励最优选择政策，有效地适应不断发展的对抗策略。还提供了理论框架，通过优化时间表以最大限度地减少由于分布转移而引起的总体遗憾，从而有效适应不可见的对抗攻击，从而引发反脆弱性。大量的数值模拟验证了所提出的框架在具有多个动态三维障碍物以及更强的投影梯度下降（PVD）和欺骗攻击的复杂导航环境中的有效性。与传统的鲁棒性、非自适应RL方法相比，反脆弱方法实现了卓越的性能，与现有的鲁棒性RL技术相比，展示了更短的导航路径长度和更高的无冲突导航轨迹率



## **42. PhishKey: A Novel Centroid-Based Approach for Enhanced Phishing Detection Using Adaptive HTML Component Extraction**

PhishKey：一种新型的基于中心的方法，使用自适应HTML组件提取来增强网络钓鱼检测 cs.CR

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.21106v1) [paper-pdf](http://arxiv.org/pdf/2506.21106v1)

**Authors**: Felipe Castaño, Eduardo Fidalgo, Enrique Alegre, Rocio Alaiz-Rodríguez, Raul Orduna, Francesco Zola

**Abstract**: Phishing attacks pose a significant cybersecurity threat, evolving rapidly to bypass detection mechanisms and exploit human vulnerabilities. This paper introduces PhishKey to address the challenges of adaptability, robustness, and efficiency. PhishKey is a novel phishing detection method using automatic feature extraction from hybrid sources. PhishKey combines character-level processing with Convolutional Neural Networks (CNN) for URL classification, and a Centroid-Based Key Component Phishing Extractor (CAPE) for HTML content at the word level. CAPE reduces noise and ensures complete sample processing avoiding crop operations on the input data. The predictions from both modules are integrated using a soft-voting ensemble to achieve more accurate and reliable classifications. Experimental evaluations on four state-of-the-art datasets demonstrate the effectiveness of PhishKey. It achieves up to 98.70% F1 Score and shows strong resistance to adversarial manipulations such as injection attacks with minimal performance degradation.

摘要: 网络钓鱼攻击构成了重大的网络安全威胁，并迅速演变为绕过检测机制并利用人类脆弱性。本文引入PhishKey来解决适应性、稳健性和效率方面的挑战。PhishKey是一种新型的网络钓鱼检测方法，使用从混合源中自动提取特征。PhishKey将字符级处理与卷积神经网络（CNN）相结合，用于URL分类，并将基于Centroid的关键组件钓鱼提取器（CAPE）用于单词级的HTML内容。CAPE降低了噪音并确保完整的样本处理，避免对输入数据进行裁剪操作。两个模块的预测使用软投票集成进行集成，以实现更准确和可靠的分类。对四个最先进数据集的实验评估证明了PhishKey的有效性。它实现了高达98.70%的F1得分，并显示出对注入攻击等对抗性操作的强大抵抗力，性能下降最小。



## **43. Boosting Generative Adversarial Transferability with Self-supervised Vision Transformer Features**

利用自我监督视觉Transformer功能增强生成性对抗性可移植性 cs.CV

14 pages, 9 figures, to appear in ICCV 2025

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.21046v1) [paper-pdf](http://arxiv.org/pdf/2506.21046v1)

**Authors**: Shangbo Wu, Yu-an Tan, Ruinan Ma, Wencong Ma, Dehua Zhu, Yuanzhang Li

**Abstract**: The ability of deep neural networks (DNNs) come from extracting and interpreting features from the data provided. By exploiting intermediate features in DNNs instead of relying on hard labels, we craft adversarial perturbation that generalize more effectively, boosting black-box transferability. These features ubiquitously come from supervised learning in previous work. Inspired by the exceptional synergy between self-supervised learning and the Transformer architecture, this paper explores whether exploiting self-supervised Vision Transformer (ViT) representations can improve adversarial transferability. We present dSVA -- a generative dual self-supervised ViT features attack, that exploits both global structural features from contrastive learning (CL) and local textural features from masked image modeling (MIM), the self-supervised learning paradigm duo for ViTs. We design a novel generative training framework that incorporates a generator to create black-box adversarial examples, and strategies to train the generator by exploiting joint features and the attention mechanism of self-supervised ViTs. Our findings show that CL and MIM enable ViTs to attend to distinct feature tendencies, which, when exploited in tandem, boast great adversarial generalizability. By disrupting dual deep features distilled by self-supervised ViTs, we are rewarded with remarkable black-box transferability to models of various architectures that outperform state-of-the-arts. Code available at https://github.com/spencerwooo/dSVA.

摘要: 深度神经网络（DNN）的能力来自于从所提供的数据中提取和解释特征。通过利用DNN中的中间特征而不是依赖硬标签，我们设计了更有效地概括的对抗性扰动，从而提高了黑匣子的可移植性。这些功能普遍来自之前工作中的监督学习。受自监督学习和Transformer架构之间的特殊协同作用的启发，本文探讨了利用自监督Vision Transformer（ViT）表示是否可以提高对抗性可转移性。我们提出了dSVA --一种生成式双重自我监督ViT特征攻击，它利用来自对比学习（CL）的全局结构特征和来自掩蔽图像建模（TIM）的局部纹理特征，这是ViT的自我监督学习范式二人组。我们设计了一个新颖的生成式训练框架，其中包含一个生成器来创建黑匣子对抗示例，以及通过利用联合特征和自我监督ViT的注意力机制来训练生成器的策略。我们的研究结果表明，CL和TIM使ViT能够关注不同的特征趋势，当协同利用时，这些特征趋势具有很强的对抗概括性。通过破坏由自我监督ViT提炼的双重深度特征，我们获得了出色的黑匣子可移植性，以转移到性能优于最新技术水平的各种架构的模型。代码可在https://github.com/spencerwooo/dSVA获得。



## **44. Doppelganger Method: Breaking Role Consistency in LLM Agent via Prompt-based Transferable Adversarial Attack**

分身方法：通过基于预算的可转移对抗攻击打破LLM代理中的角色一致性 cs.AI

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.14539v2) [paper-pdf](http://arxiv.org/pdf/2506.14539v2)

**Authors**: Daewon Kang, YeongHwan Shin, Doyeon Kim, Kyu-Hwan Jung, Meong Hi Son

**Abstract**: Since the advent of large language models, prompt engineering now enables the rapid, low-effort creation of diverse autonomous agents that are already in widespread use. Yet this convenience raises urgent concerns about the safety, robustness, and behavioral consistency of the underlying prompts, along with the pressing challenge of preventing those prompts from being exposed to user's attempts. In this paper, we propose the ''Doppelganger method'' to demonstrate the risk of an agent being hijacked, thereby exposing system instructions and internal information. Next, we define the ''Prompt Alignment Collapse under Adversarial Transfer (PACAT)'' level to evaluate the vulnerability to this adversarial transfer attack. We also propose a ''Caution for Adversarial Transfer (CAT)'' prompt to counter the Doppelganger method. The experimental results demonstrate that the Doppelganger method can compromise the agent's consistency and expose its internal information. In contrast, CAT prompts enable effective defense against this adversarial attack.

摘要: 自从大型语言模型的出现以来，即时工程现在可以快速、低努力地创建已经广泛使用的各种自治代理。然而，这种便利性引发了人们对底层提示的安全性、稳健性和行为一致性的紧迫担忧，以及防止这些提示暴露于用户尝试的紧迫挑战。在本文中，我们提出了“Doppelganger方法”来演示代理被劫持从而暴露系统指令和内部信息的风险。接下来，我们定义“对抗性转移下的提示对齐崩溃（PACAT RST）”级别来评估这种对抗性转移攻击的脆弱性。我们还提出了“对抗性转移的警告（CAT）”提示来对抗Doppelganger方法。实验结果表明，Doppelganger方法会损害代理的一致性并暴露其内部信息。相比之下，CAT提示可以有效防御这种对抗性攻击。



## **45. GuardSet-X: Massive Multi-Domain Safety Policy-Grounded Guardrail Dataset**

GuardSet-X：基于安全策略的大规模多域Guardious数据集 cs.CR

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.19054v2) [paper-pdf](http://arxiv.org/pdf/2506.19054v2)

**Authors**: Mintong Kang, Zhaorun Chen, Chejian Xu, Jiawei Zhang, Chengquan Guo, Minzhou Pan, Ivan Revilla, Yu Sun, Bo Li

**Abstract**: As LLMs become widespread across diverse applications, concerns about the security and safety of LLM interactions have intensified. Numerous guardrail models and benchmarks have been developed to ensure LLM content safety. However, existing guardrail benchmarks are often built upon ad hoc risk taxonomies that lack a principled grounding in standardized safety policies, limiting their alignment with real-world operational requirements. Moreover, they tend to overlook domain-specific risks, while the same risk category can carry different implications across different domains. To bridge these gaps, we introduce GuardSet-X, the first massive multi-domain safety policy-grounded guardrail dataset. GuardSet-X offers: (1) broad domain coverage across eight safety-critical domains, such as finance, law, and codeGen; (2) policy-grounded risk construction based on authentic, domain-specific safety guidelines; (3) diverse interaction formats, encompassing declarative statements, questions, instructions, and multi-turn conversations; (4) advanced benign data curation via detoxification prompting to challenge over-refusal behaviors; and (5) \textbf{attack-enhanced instances} that simulate adversarial inputs designed to bypass guardrails. Based on GuardSet-X, we benchmark 19 advanced guardrail models and uncover a series of findings, such as: (1) All models achieve varied F1 scores, with many demonstrating high variance across risk categories, highlighting their limited domain coverage and insufficient handling of domain-specific safety concerns; (2) As models evolve, their coverage of safety risks broadens, but performance on common risk categories may decrease; (3) All models remain vulnerable to optimized adversarial attacks. We believe that \dataset and the unique insights derived from our evaluations will advance the development of policy-aligned and resilient guardrail systems.

摘要: 随着LLM在各种应用程序中的广泛应用，对LLM交互的安全性和安全性的担忧也在加剧。已经开发了许多护栏模型和基准，以确保LLM内容安全。然而，现有的护栏基准通常建立在临时风险分类的基础上，缺乏标准化安全政策的原则基础，限制了它们与现实世界运营要求的一致性。此外，它们往往忽视特定领域的风险，而同一风险类别在不同领域可能产生不同的影响。为了弥合这些差距，我们引入了GuardSet-X，这是第一个基于多领域安全政策的大型护栏数据集。GuardSet-X提供：（1）涵盖金融、法律和代码Gen等八个安全关键领域的广泛领域;（2）基于真实的、特定领域的安全指南的基于政策的风险构建;（3）多样化的互动形式，包括声明性声明、问题、说明和多轮对话;（4）通过排毒促进挑战过度拒绝行为的高级良性数据策展;和（5）\textBF{attack-enhanced instants}，模拟旨在绕过护栏的对抗输入。基于GuardSet-X，我们对19种先进护栏模型进行了基准测试，并发现了一系列发现，例如：（1）所有模型都获得了不同的F1评分，其中许多模型在风险类别中表现出较高的差异，凸显了其有限的领域覆盖范围和对特定领域安全问题的处理不足;（2）随着模型的发展，其安全风险覆盖范围扩大，但常见风险类别的性能可能会下降;（3）所有模型仍然容易受到优化的对抗攻击。我们相信，数据集和从我们的评估中得出的独特见解将推动符合政策且有弹性的护栏系统的开发。



## **46. AgentStealth: Reinforcing Large Language Model for Anonymizing User-generated Text**

AgentStealth：加强大型语言模型以简化用户生成的文本 cs.CL

This work has been submitted to NeurIPS 2025. Under review

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.22508v1) [paper-pdf](http://arxiv.org/pdf/2506.22508v1)

**Authors**: Chenyang Shao, Tianxing Li, Chenhao Pu, Fengli Xu, Yong Li

**Abstract**: In today's digital world, casual user-generated content often contains subtle cues that may inadvertently expose sensitive personal attributes. Such risks underscore the growing importance of effective text anonymization to safeguard individual privacy. However, existing methods either rely on rigid replacements that damage utility or cloud-based LLMs that are costly and pose privacy risks. To address these issues, we explore the use of locally deployed smaller-scale language models (SLMs) for anonymization. Yet training effective SLMs remains challenging due to limited high-quality supervision. To address the challenge, we propose AgentStealth, a self-reinforcing LLM anonymization framework.First, we introduce an adversarial anonymization workflow enhanced by In-context Contrastive Learning and Adaptive Utility-Aware Control. Second, we perform supervised adaptation of SLMs using high-quality data collected from the workflow, which includes both anonymization and attack signals. Finally, we apply online reinforcement learning where the model leverages its internal adversarial feedback to iteratively improve anonymization performance. Experiments on two datasets show that our method outperforms baselines in both anonymization effectiveness (+12.3%) and utility (+6.8%). Our lightweight design supports direct deployment on edge devices, avoiding cloud reliance and communication-based privacy risks. Our code is open-source at https://github.com/tsinghua-fib-lab/AgentStealth.

摘要: 在当今的数字世界中，用户生成的随意内容通常包含微妙的线索，这些线索可能会无意中暴露敏感的个人属性。此类风险凸显了有效的文本匿名化对于保护个人隐私的重要性。然而，现有方法要么依赖于损害公用事业的严格替代品，要么依赖于成本高昂并构成隐私风险的基于云的LLM。为了解决这些问题，我们探索使用本地部署的较小规模语言模型（SLC）进行匿名化。然而，由于高质量的监督有限，培训有效的CRM仍然具有挑战性。为了应对这一挑战，我们提出了AgentStealth，这是一个自我增强的LLM匿名化框架。首先，我们引入了一种由上下文内对比学习和自适应实用性感知控制增强的对抗性匿名化工作流程。其次，我们使用从工作流程中收集的高质量数据（包括匿名化和攻击信号）对CRM进行监督调整。最后，我们应用在线强化学习，其中模型利用其内部对抗反馈来迭代改进匿名化性能。对两个数据集的实验表明，我们的方法在匿名有效性（+12.3%）和实用性（+6.8%）方面都优于基线。我们的轻量级设计支持在边缘设备上直接部署，避免云依赖和基于通信的隐私风险。我们的代码是开源的，网址是https://github.com/tsinghua-fib-lab/AgentStealth。



## **47. E-FreeM2: Efficient Training-Free Multi-Scale and Cross-Modal News Verification via MLLMs**

E-FreeM 2：通过MLLM进行高效的免培训多规模和跨模式新闻验证 cs.MM

Accepted to AsiaCCS 2025 @ SCID

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.20944v1) [paper-pdf](http://arxiv.org/pdf/2506.20944v1)

**Authors**: Van-Hoang Phan, Long-Khanh Pham, Dang Vu, Anh-Duy Tran, Minh-Son Dao

**Abstract**: The rapid spread of misinformation in mobile and wireless networks presents critical security challenges. This study introduces a training-free, retrieval-based multimodal fact verification system that leverages pretrained vision-language models and large language models for credibility assessment. By dynamically retrieving and cross-referencing trusted data sources, our approach mitigates vulnerabilities of traditional training-based models, such as adversarial attacks and data poisoning. Additionally, its lightweight design enables seamless edge device integration without extensive on-device processing. Experiments on two fact-checking benchmarks achieve SOTA results, confirming its effectiveness in misinformation detection and its robustness against various attack vectors, highlighting its potential to enhance security in mobile and wireless communication environments.

摘要: 移动和无线网络中错误信息的迅速传播带来了严峻的安全挑战。本研究引入了一种免训练、基于检索的多模式事实验证系统，该系统利用预先训练的视觉语言模型和大型语言模型进行可信度评估。通过动态检索和交叉引用可信数据源，我们的方法减轻了传统基于训练的模型的漏洞，例如对抗性攻击和数据中毒。此外，其轻量级设计可以实现无缝边缘设备集成，无需进行大量的设备上处理。在两个事实检查基准上进行的实验获得了SOTA结果，证实了其在错误信息检测方面的有效性及其对各种攻击载体的鲁棒性，凸显了其增强移动和无线通信环境安全性的潜力。



## **48. SPA: Towards More Stealth and Persistent Backdoor Attacks in Federated Learning**

SPA：在联邦学习中实现更多的隐形和持续的后门攻击 cs.CR

18 pages

**SubmitDate**: 2025-06-26    [abs](http://arxiv.org/abs/2506.20931v1) [paper-pdf](http://arxiv.org/pdf/2506.20931v1)

**Authors**: Chengcheng Zhu, Ye Li, Bosen Rao, Jiale Zhang, Yunlong Mao, Sheng Zhong

**Abstract**: Federated Learning (FL) has emerged as a leading paradigm for privacy-preserving distributed machine learning, yet the distributed nature of FL introduces unique security challenges, notably the threat of backdoor attacks. Existing backdoor strategies predominantly rely on end-to-end label supervision, which, despite their efficacy, often results in detectable feature disentanglement and limited persistence. In this work, we propose a novel and stealthy backdoor attack framework, named SPA, which fundamentally departs from traditional approaches by leveraging feature-space alignment rather than direct trigger-label association. Specifically, SPA reduces representational distances between backdoor trigger features and target class features, enabling the global model to misclassify trigger-embedded inputs with high stealth and persistence. We further introduce an adaptive, adversarial trigger optimization mechanism, utilizing boundary-search in the feature space to enhance attack longevity and effectiveness, even against defensive FL scenarios and non-IID data distributions. Extensive experiments on various FL benchmarks demonstrate that SPA consistently achieves high attack success rates with minimal impact on model utility, maintains robustness under challenging participation and data heterogeneity conditions, and exhibits persistent backdoor effects far exceeding those of conventional techniques. Our results call urgent attention to the evolving sophistication of backdoor threats in FL and emphasize the pressing need for advanced, feature-level defense techniques.

摘要: 联邦学习（FL）已成为保护隐私的分布式机器学习的主要范式，但FL的分布式性质带来了独特的安全挑战，特别是后门攻击的威胁。现有的后门策略主要依赖于端到端的标签监督，尽管其有效，但通常会导致可检测的特征解开和有限的持久性。在这项工作中，我们提出了一种新颖且隐蔽的后门攻击框架，名为SPA，它通过利用特征空间对齐而不是直接的供应商标签关联从根本上背离了传统方法。具体来说，SPA缩小了后门触发功能和目标类功能之间的表示距离，使全局模型能够以高度隐蔽性和持久性的方式对嵌入服务器的输入进行错误分类。我们进一步引入了一种自适应、对抗性触发优化机制，利用特征空间中的边界搜索来增强攻击寿命和有效性，即使针对防御性FL场景和非IID数据分布也是如此。对各种FL基准的广泛实验表明，SPA始终实现高攻击成功率，对模型效用的影响最小，在具有挑战性的参与和数据异类条件下保持稳健性，并表现出远超传统技术的持久后门效应。我们的研究结果引起了人们对佛罗里达州后门威胁不断变化的复杂性的紧急关注，并强调了对先进的功能级防御技术的迫切需求。



## **49. Empowering Digital Agriculture: A Privacy-Preserving Framework for Data Sharing and Collaborative Research**

授权数字农业：数据共享和合作研究的隐私保护框架 cs.CR

arXiv admin note: text overlap with arXiv:2409.06069

**SubmitDate**: 2025-06-25    [abs](http://arxiv.org/abs/2506.20872v1) [paper-pdf](http://arxiv.org/pdf/2506.20872v1)

**Authors**: Osama Zafar, Rosemarie Santa González, Mina Namazi, Alfonso Morales, Erman Ayday

**Abstract**: Data-driven agriculture, which integrates technology and data into agricultural practices, has the potential to improve crop yield, disease resilience, and long-term soil health. However, privacy concerns, such as adverse pricing, discrimination, and resource manipulation, deter farmers from sharing data, as it can be used against them. To address this barrier, we propose a privacy-preserving framework that enables secure data sharing and collaboration for research and development while mitigating privacy risks. The framework combines dimensionality reduction techniques (like Principal Component Analysis (PCA)) and differential privacy by introducing Laplacian noise to protect sensitive information. The proposed framework allows researchers to identify potential collaborators for a target farmer and train personalized machine learning models either on the data of identified collaborators via federated learning or directly on the aggregated privacy-protected data. It also allows farmers to identify potential collaborators based on similarities. We have validated this on real-life datasets, demonstrating robust privacy protection against adversarial attacks and utility performance comparable to a centralized system. We demonstrate how this framework can facilitate collaboration among farmers and help researchers pursue broader research objectives. The adoption of the framework can empower researchers and policymakers to leverage agricultural data responsibly, paving the way for transformative advances in data-driven agriculture. By addressing critical privacy challenges, this work supports secure data integration, fostering innovation and sustainability in agricultural systems.

摘要: 数据驱动农业将技术和数据集成到农业实践中，有潜力提高作物产量、抗病能力和长期土壤健康。然而，不利定价、歧视和资源操纵等隐私问题阻碍了农民共享数据，因为数据可能会被用来对付他们。为了解决这一障碍，我们提出了一个隐私保护框架，该框架能够实现安全的数据共享和研发协作，同时降低隐私风险。该框架结合了降维技术（如主成分分析（PCA））和通过引入拉普拉斯噪音来保护敏感信息的差异隐私。拟议的框架允许研究人员识别目标农民的潜在合作者，并通过联邦学习根据已识别合作者的数据或直接根据聚合的隐私保护数据训练个性化机器学习模型。它还允许农民根据相似性来识别潜在的合作者。我们已经在现实生活中的数据集上验证了这一点，展示了针对对抗性攻击的强大隐私保护以及与集中式系统相当的实用性能。我们展示了这个框架如何促进农民之间的合作并帮助研究人员追求更广泛的研究目标。该框架的采用可以使研究人员和政策制定者能够负责任地利用农业数据，为数据驱动农业的变革性进步铺平道路。通过解决关键的隐私挑战，这项工作支持安全的数据集成，促进农业系统的创新和可持续性。



## **50. Universal and Efficient Detection of Adversarial Data through Nonuniform Impact on Network Layers**

通过对网络层的非均匀影响来普遍有效地检测对抗数据 cs.LG

arXiv admin note: substantial text overlap with arXiv:2410.17442

**SubmitDate**: 2025-06-25    [abs](http://arxiv.org/abs/2506.20816v1) [paper-pdf](http://arxiv.org/pdf/2506.20816v1)

**Authors**: Furkan Mumcu, Yasin Yilmaz

**Abstract**: Deep Neural Networks (DNNs) are notoriously vulnerable to adversarial input designs with limited noise budgets. While numerous successful attacks with subtle modifications to original input have been proposed, defense techniques against these attacks are relatively understudied. Existing defense approaches either focus on improving DNN robustness by negating the effects of perturbations or use a secondary model to detect adversarial data. Although equally important, the attack detection approach, which is studied in this work, provides a more practical defense compared to the robustness approach. We show that the existing detection methods are either ineffective against the state-of-the-art attack techniques or computationally inefficient for real-time processing. We propose a novel universal and efficient method to detect adversarial examples by analyzing the varying degrees of impact of attacks on different DNN layers. {Our method trains a lightweight regression model that predicts deeper-layer features from early-layer features, and uses the prediction error to detect adversarial samples.} Through theoretical arguments and extensive experiments, we demonstrate that our detection method is highly effective, computationally efficient for real-time processing, compatible with any DNN architecture, and applicable across different domains, such as image, video, and audio.

摘要: 众所周知，深度神经网络（DNN）很容易受到噪音预算有限的对抗输入设计的影响。虽然已经提出了许多对原始输入进行细微修改的成功攻击，但针对这些攻击的防御技术的研究相对不足。现有的防御方法要么专注于通过抵消扰动的影响来提高DNN稳健性，要么使用二级模型来检测对抗数据。尽管同样重要，但与鲁棒性方法相比，本文中研究的攻击检测方法提供了更实用的防御。我们表明，现有的检测方法要么对最先进的攻击技术无效，要么对于实时处理来说计算效率低下。我们提出了一种新颖的通用且有效的方法来通过分析攻击对不同DNN层的不同程度的影响来检测对抗性示例。{Our该方法训练一个轻量级回归模型，该模型从早期层特征预测较深层特征，并使用预测误差来检测对抗样本。通过理论论证和大量实验，我们证明我们的检测方法高效、实时处理计算高效、与任何DNN架构兼容，并且适用于不同领域，例如图像、视频和音频。



