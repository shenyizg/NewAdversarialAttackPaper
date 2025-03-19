# Latest Adversarial Attack Papers
**update at 2025-03-19 10:38:18**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. VGFL-SA: Vertical Graph Federated Learning Structure Attack Based on Contrastive Learning**

VGFL-SA：基于对比学习的垂直图联邦学习结构攻击 cs.LG

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2502.16793v2) [paper-pdf](http://arxiv.org/pdf/2502.16793v2)

**Authors**: Yang Chen, Bin Zhou

**Abstract**: Graph Neural Networks (GNNs) have gained attention for their ability to learn representations from graph data. Due to privacy concerns and conflicts of interest that prevent clients from directly sharing graph data with one another, Vertical Graph Federated Learning (VGFL) frameworks have been developed. Recent studies have shown that VGFL is vulnerable to adversarial attacks that degrade performance. However, it is a common problem that client nodes are often unlabeled in the realm of VGFL. Consequently, the existing attacks, which rely on the availability of labeling information to obtain gradients, are inherently constrained in their applicability. This limitation precludes their deployment in practical, real-world environments. To address the above problems, we propose a novel graph adversarial attack against VGFL, referred to as VGFL-SA, to degrade the performance of VGFL by modifying the local clients structure without using labels. Specifically, VGFL-SA uses a contrastive learning method to complete the attack before the local clients are trained. VGFL-SA first accesses the graph structure and node feature information of the poisoned clients, and generates the contrastive views by node-degree-based edge augmentation and feature shuffling augmentation. Then, VGFL-SA uses the shared graph encoder to get the embedding of each view, and the gradients of the adjacency matrices are obtained by the contrastive function. Finally, perturbed edges are generated using gradient modification rules. We validated the performance of VGFL-SA by performing a node classification task on real-world datasets, and the results show that VGFL-SA achieves good attack effectiveness and transferability.

摘要: 图形神经网络(GNN)因其从图形数据中学习表示的能力而受到关注。由于隐私问题和利益冲突阻碍了客户之间直接共享图形数据，垂直图形联合学习(VGFL)框架已经开发出来。最近的研究表明，VGFL很容易受到降低性能的对抗性攻击。然而，在VGFL领域中，一个常见的问题是客户端节点通常是未标记的。因此，现有的攻击依赖于标记信息的可用性来获得梯度，其适用性受到固有的限制。这一限制排除了它们在实际、真实环境中的部署。针对上述问题，我们提出了一种新的针对VGFL的图对抗攻击，称为VGFL-SA，通过修改本地客户端结构而不使用标签来降低VGFL的性能。具体地说，VGFL-SA使用对比学习方法在本地客户端训练之前完成攻击。VGFL-SA首先获取中毒客户端的图结构和节点特征信息，然后通过基于节点度的边增强和特征置乱增强生成对比视图。然后，VGFL-SA使用共享图编码器得到每个视点的嵌入，并通过对比函数得到邻接矩阵的梯度。最后，使用梯度修正规则生成扰动边缘。我们通过在真实数据集上执行节点分类任务来验证VGFL-SA的性能，结果表明VGFL-SA具有良好的攻击有效性和可转移性。



## **2. Unveiling the Role of Randomization in Multiclass Adversarial Classification: Insights from Graph Theory**

揭示随机化在多类对抗分类中的作用：来自图论的见解 cs.LG

9 pages (main), 30 in total. Camera-ready version, accepted at  AISTATS 2025

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.14299v1) [paper-pdf](http://arxiv.org/pdf/2503.14299v1)

**Authors**: Lucas Gnecco-Heredia, Matteo Sammut, Muni Sreenivas Pydi, Rafael Pinot, Benjamin Negrevergne, Yann Chevaleyre

**Abstract**: Randomization as a mean to improve the adversarial robustness of machine learning models has recently attracted significant attention. Unfortunately, much of the theoretical analysis so far has focused on binary classification, providing only limited insights into the more complex multiclass setting. In this paper, we take a step toward closing this gap by drawing inspiration from the field of graph theory. Our analysis focuses on discrete data distributions, allowing us to cast the adversarial risk minimization problems within the well-established framework of set packing problems. By doing so, we are able to identify three structural conditions on the support of the data distribution that are necessary for randomization to improve robustness. Furthermore, we are able to construct several data distributions where (contrarily to binary classification) switching from a deterministic to a randomized solution significantly reduces the optimal adversarial risk. These findings highlight the crucial role randomization can play in enhancing robustness to adversarial attacks in multiclass classification.

摘要: 随机化作为提高机器学习模型对抗性稳健性的一种手段，最近引起了人们的广泛关注。不幸的是，到目前为止，许多理论分析都集中在二进制分类上，对更复杂的多类设置只提供了有限的见解。在本文中，我们从图论领域汲取灵感，朝着缩小这一差距迈出了一步。我们的分析集中在离散数据分布上，允许我们在集合打包问题的良好框架内求解对抗性风险最小化问题。通过这样做，我们能够确定数据分布支持上的三个结构条件，这三个条件是随机化提高稳健性所必需的。此外，我们能够构建几个数据分布，其中(与二进制分类相反)从确定性解决方案切换到随机解决方案显著地降低了最优对抗风险。这些发现突显了随机化在增强多类分类中对抗攻击的稳健性方面所起的关键作用。



## **3. Multimodal Adversarial Defense for Vision-Language Models by Leveraging One-To-Many Relationships**

利用一对多关系对视觉语言模型进行多模式对抗防御 cs.CV

Under review

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2405.18770v2) [paper-pdf](http://arxiv.org/pdf/2405.18770v2)

**Authors**: Futa Waseda, Antonio Tejero-de-Pablos, Isao Echizen

**Abstract**: Pre-trained vision-language (VL) models are highly vulnerable to adversarial attacks. However, existing defense methods primarily focus on image classification, overlooking two key aspects of VL tasks: multimodal attacks, where both image and text can be perturbed, and the one-to-many relationship of images and texts, where a single image can correspond to multiple textual descriptions and vice versa (1:N and N:1). This work is the first to explore defense strategies against multimodal attacks in VL tasks, whereas prior VL defense methods focus on vision robustness. We propose multimodal adversarial training (MAT), which incorporates adversarial perturbations in both image and text modalities during training, significantly outperforming existing unimodal defenses. Furthermore, we discover that MAT is limited by deterministic one-to-one (1:1) image-text pairs in VL training data. To address this, we conduct a comprehensive study on leveraging one-to-many relationships to enhance robustness, investigating diverse augmentation techniques. Our analysis shows that, for a more effective defense, augmented image-text pairs should be well-aligned, diverse, yet avoid distribution shift -- conditions overlooked by prior research. Our experiments show that MAT can effectively be applied to different VL models and tasks to improve adversarial robustness, outperforming previous efforts. Our code will be made public upon acceptance.

摘要: 预先训练好的视觉语言(VL)模型极易受到敌意攻击。然而，现有的防御方法主要集中在图像分类上，忽略了VL任务的两个关键方面：多模式攻击，其中图像和文本都可以被干扰，以及图像和文本的一对多关系，其中一幅图像可以对应于多个文本描述，反之亦然(1：N和N：1)。该工作首次探索了VL任务中针对多模式攻击的防御策略，而以往的VL防御方法侧重于视觉稳健性。我们提出了多模式对抗训练(MAT)，它在训练过程中结合了图像和文本模式的对抗扰动，显著优于现有的单模式防御。此外，我们发现MAT受到VL训练数据中确定性的一对一(1：1)图文对的限制。为了解决这个问题，我们进行了一项关于利用一对多关系来增强健壮性的全面研究，调查了各种增强技术。我们的分析表明，为了更有效地防御，增强的图文对应该很好地对齐、多样化，但又要避免分布偏移--这是以前的研究忽视的条件。我们的实验表明，MAT可以有效地应用于不同的虚拟学习模型和任务，以提高对手的健壮性，表现出优于以往的努力。我们的代码将在接受后公开。



## **4. XOXO: Stealthy Cross-Origin Context Poisoning Attacks against AI Coding Assistants**

XOXO：针对人工智能编码助理的隐形跨源上下文中毒攻击 cs.CR

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.14281v1) [paper-pdf](http://arxiv.org/pdf/2503.14281v1)

**Authors**: Adam Štorek, Mukur Gupta, Noopur Bhatt, Aditya Gupta, Janie Kim, Prashast Srivastava, Suman Jana

**Abstract**: AI coding assistants are widely used for tasks like code generation, bug detection, and comprehension. These tools now require large and complex contexts, automatically sourced from various origins$\unicode{x2014}$across files, projects, and contributors$\unicode{x2014}$forming part of the prompt fed to underlying LLMs. This automatic context-gathering introduces new vulnerabilities, allowing attackers to subtly poison input to compromise the assistant's outputs, potentially generating vulnerable code, overlooking flaws, or introducing critical errors. We propose a novel attack, Cross-Origin Context Poisoning (XOXO), that is particularly challenging to detect as it relies on adversarial code modifications that are semantically equivalent. Traditional program analysis techniques struggle to identify these correlations since the semantics of the code remain correct, making it appear legitimate. This allows attackers to manipulate code assistants into producing incorrect outputs, including vulnerabilities or backdoors, while shifting the blame to the victim developer or tester. We introduce a novel, task-agnostic black-box attack algorithm GCGS that systematically searches the transformation space using a Cayley Graph, achieving an 83.09% attack success rate on average across five tasks and eleven models, including GPT-4o and Claude 3.5 Sonnet v2 used by many popular AI coding assistants. Furthermore, existing defenses, including adversarial fine-tuning, are ineffective against our attack, underscoring the need for new security measures in LLM-powered coding tools.

摘要: AI编码助手被广泛用于代码生成、错误检测和理解等任务。这些工具现在需要大型而复杂的上下文，这些上下文自动来自不同来源的$\unicode{x2014}$，跨文件、项目和贡献者$\unicode{x2014}$，构成馈送到底层LLM的提示的一部分。这种自动上下文收集引入了新的漏洞，允许攻击者巧妙地毒化输入以危害助手的输出，可能会生成易受攻击的代码、忽略缺陷或引入严重错误。我们提出了一种新的攻击，跨来源上下文中毒(XOXO)，由于它依赖于语义等价的敌意代码修改，因此检测起来特别具有挑战性。传统的程序分析技术很难识别这些相关性，因为代码的语义保持正确，使其看起来是合法的。这允许攻击者操纵代码助手产生不正确的输出，包括漏洞或后门，同时将责任转嫁给受害者开发人员或测试人员。提出了一种新的任务无关的黑盒攻击算法GCGS，该算法使用Cayley图系统地搜索变换空间，在5个任务和11个模型上获得了83.09%的平均攻击成功率，其中包括许多流行的AI编码助手使用的GPT-40和Claude 3.5 Sonnet v2。此外，现有的防御措施，包括对抗性微调，对我们的攻击是无效的，这突显了在LLM支持的编码工具中需要新的安全措施。



## **5. TAROT: Task-Oriented Authorship Obfuscation Using Policy Optimization Methods**

TAROT：使用策略优化方法的面向任务的作者混淆 cs.CL

Accepted to the NAACL PrivateNLP 2025 Workshop

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2407.21630v2) [paper-pdf](http://arxiv.org/pdf/2407.21630v2)

**Authors**: Gabriel Loiseau, Damien Sileo, Damien Riquet, Maxime Meyer, Marc Tommasi

**Abstract**: Authorship obfuscation aims to disguise the identity of an author within a text by altering the writing style, vocabulary, syntax, and other linguistic features associated with the text author. This alteration needs to balance privacy and utility. While strong obfuscation techniques can effectively hide the author's identity, they often degrade the quality and usefulness of the text for its intended purpose. Conversely, maintaining high utility tends to provide insufficient privacy, making it easier for an adversary to de-anonymize the author. Thus, achieving an optimal trade-off between these two conflicting objectives is crucial. In this paper, we propose TAROT: Task-Oriented Authorship Obfuscation Using Policy Optimization, a new unsupervised authorship obfuscation method whose goal is to optimize the privacy-utility trade-off by regenerating the entire text considering its downstream utility. Our approach leverages policy optimization as a fine-tuning paradigm over small language models in order to rewrite texts by preserving author identity and downstream task utility. We show that our approach largely reduces the accuracy of attackers while preserving utility. We make our code and models publicly available.

摘要: 作者身份混淆旨在通过改变与文本作者相关的写作风格、词汇、句法和其他语言特征来掩盖作者在文本中的身份。这一改变需要平衡隐私和效用。虽然强大的混淆技术可以有效地隐藏作者的身份，但它们往往会降低文本的质量和对预期目的的有用性。相反，保持高实用性往往会提供不充分的隐私，使对手更容易解除作者的匿名。因此，在这两个相互冲突的目标之间实现最佳权衡至关重要。在本文中，我们提出了一种新的无监督作者身份混淆方法--TAROT：基于策略优化的面向任务的作者身份混淆方法，其目标是通过重新生成考虑下游效用的整个文本来优化隐私和效用之间的权衡。我们的方法利用策略优化作为小语言模型上的微调范式，以便通过保留作者身份和下游任务效用来重写文本。我们表明，我们的方法在很大程度上降低了攻击者的准确性，同时保持了实用性。我们公开我们的代码和模型。



## **6. Adversarial Training for Multimodal Large Language Models against Jailbreak Attacks**

针对越狱攻击的多模式大型语言模型对抗训练 cs.CV

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.04833v2) [paper-pdf](http://arxiv.org/pdf/2503.04833v2)

**Authors**: Liming Lu, Shuchao Pang, Siyuan Liang, Haotian Zhu, Xiyu Zeng, Aishan Liu, Yunhuai Liu, Yongbin Zhou

**Abstract**: Multimodal large language models (MLLMs) have made remarkable strides in cross-modal comprehension and generation tasks. However, they remain vulnerable to jailbreak attacks, where crafted perturbations bypass security guardrails and elicit harmful outputs. In this paper, we present the first adversarial training (AT) paradigm tailored to defend against jailbreak attacks during the MLLM training phase. Extending traditional AT to this domain poses two critical challenges: efficiently tuning massive parameters and ensuring robustness against attacks across multiple modalities. To address these challenges, we introduce Projection Layer Against Adversarial Training (ProEAT), an end-to-end AT framework. ProEAT incorporates a projector-based adversarial training architecture that efficiently handles large-scale parameters while maintaining computational feasibility by focusing adversarial training on a lightweight projector layer instead of the entire model; additionally, we design a dynamic weight adjustment mechanism that optimizes the loss function's weight allocation based on task demands, streamlining the tuning process. To enhance defense performance, we propose a joint optimization strategy across visual and textual modalities, ensuring robust resistance to jailbreak attacks originating from either modality. Extensive experiments conducted on five major jailbreak attack methods across three mainstream MLLMs demonstrate the effectiveness of our approach. ProEAT achieves state-of-the-art defense performance, outperforming existing baselines by an average margin of +34% across text and image modalities, while incurring only a 1% reduction in clean accuracy. Furthermore, evaluations on real-world embodied intelligent systems highlight the practical applicability of our framework, paving the way for the development of more secure and reliable multimodal systems.

摘要: 多通道大语言模型在跨通道理解和生成任务方面取得了显著进展。然而，它们仍然容易受到越狱攻击，在越狱攻击中，精心设计的扰动绕过安全护栏，引发有害输出。在这篇文章中，我们提出了在MLLM训练阶段为防御越狱攻击而定制的第一个对抗性训练(AT)范例。将传统的AT扩展到这一领域会带来两个关键挑战：有效地调整大量参数和确保对跨多个通道的攻击的健壮性。为了应对这些挑战，我们引入了投影层对抗对手训练(ProEAT)，这是一个端到端的AT框架。ProEAT结合了基于投影仪的对抗性训练体系结构，通过将对抗性训练集中在轻量级投影器层而不是整个模型上，在保持计算可行性的同时有效地处理大规模参数；此外，我们设计了动态权重调整机制，基于任务需求优化损失函数的权重分配，从而简化了调整过程。为了提高防御性能，我们提出了一种跨视觉和文本模式的联合优化策略，确保对来自任何一种模式的越狱攻击具有强大的抵抗力。在三种主流MLLMS上对五种主要的越狱攻击方法进行了广泛的实验，证明了该方法的有效性。ProEAT实现了最先进的防御性能，在文本和图像模式中的表现比现有基线平均高出34%，而干净的准确性仅降低了1%。此外，对真实世界体现的智能系统的评估突出了我们框架的实用适用性，为开发更安全可靠的多式联运系统铺平了道路。



## **7. Survey of Adversarial Robustness in Multimodal Large Language Models**

多模式大型语言模型中的对抗鲁棒性研究 cs.CV

9 pages

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.13962v1) [paper-pdf](http://arxiv.org/pdf/2503.13962v1)

**Authors**: Chengze Jiang, Zhuangzhuang Wang, Minjing Dong, Jie Gui

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated exceptional performance in artificial intelligence by facilitating integrated understanding across diverse modalities, including text, images, video, audio, and speech. However, their deployment in real-world applications raises significant concerns about adversarial vulnerabilities that could compromise their safety and reliability. Unlike unimodal models, MLLMs face unique challenges due to the interdependencies among modalities, making them susceptible to modality-specific threats and cross-modal adversarial manipulations. This paper reviews the adversarial robustness of MLLMs, covering different modalities. We begin with an overview of MLLMs and a taxonomy of adversarial attacks tailored to each modality. Next, we review key datasets and evaluation metrics used to assess the robustness of MLLMs. After that, we provide an in-depth review of attacks targeting MLLMs across different modalities. Our survey also identifies critical challenges and suggests promising future research directions.

摘要: 多模式大语言模型(MLLM)通过促进跨不同模式的集成理解，包括文本、图像、视频、音频和语音，在人工智能中表现出出色的性能。然而，它们在实际应用程序中的部署引发了人们对可能危及其安全性和可靠性的对抗性漏洞的严重担忧。与单模模型不同，由于各通道之间的相互依赖关系，最大似然模型面临着独特的挑战，这使得它们容易受到特定通道的威胁和跨通道的对抗性操作。本文综述了MLLMS的对抗稳健性，涵盖了不同的模式。我们首先概述MLLMS和针对每种模式量身定做的对抗性攻击分类。接下来，我们回顾了用于评估MLLMS稳健性的关键数据集和评估指标。在此之后，我们提供了针对不同模式的MLLM的攻击的深入回顾。我们的调查还确定了关键挑战，并提出了前景看好的未来研究方向。



## **8. Make the Most of Everything: Further Considerations on Disrupting Diffusion-based Customization**

充分利用一切：关于破坏基于扩散的定制的进一步思考 cs.CV

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.13945v1) [paper-pdf](http://arxiv.org/pdf/2503.13945v1)

**Authors**: Long Tang, Dengpan Ye, Sirun Chen, Xiuwen Shi, Yunna Lv, Ziyi Liu

**Abstract**: The fine-tuning technique for text-to-image diffusion models facilitates image customization but risks privacy breaches and opinion manipulation. Current research focuses on prompt- or image-level adversarial attacks for anti-customization, yet it overlooks the correlation between these two levels and the relationship between internal modules and inputs. This hinders anti-customization performance in practical threat scenarios. We propose Dual Anti-Diffusion (DADiff), a two-stage adversarial attack targeting diffusion customization, which, for the first time, integrates the adversarial prompt-level attack into the generation process of image-level adversarial examples. In stage 1, we generate prompt-level adversarial vectors to guide the subsequent image-level attack. In stage 2, besides conducting the end-to-end attack on the UNet model, we disrupt its self- and cross-attention modules, aiming to break the correlations between image pixels and align the cross-attention results computed using instance prompts and adversarial prompt vectors within the images. Furthermore, we introduce a local random timestep gradient ensemble strategy, which updates adversarial perturbations by integrating random gradients from multiple segmented timesets. Experimental results on various mainstream facial datasets demonstrate 10%-30% improvements in cross-prompt, keyword mismatch, cross-model, and cross-mechanism anti-customization with DADiff compared to existing methods.

摘要: 文本到图像扩散模型的微调技术为图像定制提供了便利，但存在侵犯隐私和操纵意见的风险。目前的研究侧重于针对反定制的提示级或图像级对抗性攻击，而忽略了这两个级别之间的相关性以及内部模块和输入之间的关系。这阻碍了实际威胁场景中的反定制性能。本文提出了双重反扩散攻击(DADiff)，这是一种针对扩散定制的两阶段对抗性攻击，首次将对抗性提示级攻击融入到图像级对抗性实例的生成过程中。在第一阶段，我们生成提示级对抗向量来指导后续的图像级攻击。在第二阶段中，除了对UNET模型进行端到端攻击外，我们还扰乱了其自我注意和交叉注意模块，旨在打破图像像素之间的相关性，并对齐使用图像中的实例提示和对抗性提示向量计算的交叉注意结果。此外，我们还引入了一种局部随机时间步长梯度集成策略，该策略通过整合来自多个分段时间集的随机梯度来更新对抗性扰动。在各种主流人脸数据集上的实验结果表明，与现有方法相比，DADiff在交叉提示、关键词不匹配、跨模型和跨机制反定制等方面都有10%-30%的改进。



## **9. GSBA$^K$: $top$-$K$ Geometric Score-based Black-box Attack**

GSBA$^K$：$top$-$K$基于几何分数的黑匣子攻击 cs.CV

This article has been accepted for publication at ICLR 2025

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.12827v2) [paper-pdf](http://arxiv.org/pdf/2503.12827v2)

**Authors**: Md Farhamdur Reza, Richeng Jin, Tianfu Wu, Huaiyu Dai

**Abstract**: Existing score-based adversarial attacks mainly focus on crafting $top$-1 adversarial examples against classifiers with single-label classification. Their attack success rate and query efficiency are often less than satisfactory, particularly under small perturbation requirements; moreover, the vulnerability of classifiers with multi-label learning is yet to be studied. In this paper, we propose a comprehensive surrogate free score-based attack, named \b geometric \b score-based \b black-box \b attack (GSBA$^K$), to craft adversarial examples in an aggressive $top$-$K$ setting for both untargeted and targeted attacks, where the goal is to change the $top$-$K$ predictions of the target classifier. We introduce novel gradient-based methods to find a good initial boundary point to attack. Our iterative method employs novel gradient estimation techniques, particularly effective in $top$-$K$ setting, on the decision boundary to effectively exploit the geometry of the decision boundary. Additionally, GSBA$^K$ can be used to attack against classifiers with $top$-$K$ multi-label learning. Extensive experimental results on ImageNet and PASCAL VOC datasets validate the effectiveness of GSBA$^K$ in crafting $top$-$K$ adversarial examples.

摘要: 现有的基于分数的对抗性攻击主要集中在针对具有单标签分类的分类器构造$top$-1个对抗性实例。它们的攻击成功率和查询效率往往不尽如人意，特别是在小扰动要求下；此外，具有多标签学习的分类器的脆弱性还有待研究。在本文中，我们提出了一种全面的基于分数的无代理攻击，称为基于几何分数的黑盒攻击(GSBA$^K$)，以在攻击性的$top$-$K$环境中为非目标攻击和目标攻击创建对手示例，其中目标是改变目标分类器的$top$-$K$预测。我们引入了新的基于梯度的方法来寻找一个好的初始边界点进行攻击。我们的迭代方法使用了新的梯度估计技术，特别是在决策边界上的$top$-$K$设置上，以有效地利用决策边界的几何形状。此外，GSBA$^K$可用于攻击具有$TOP$-$K$多标签学习的分类器。在ImageNet和Pascal VOC数据集上的大量实验结果验证了GSBA$^K$在构造$TOP$-$K$对抗性实例方面的有效性。



## **10. Securing Virtual Reality Experiences: Unveiling and Tackling Cybersickness Attacks with Explainable AI**

保护虚拟现实体验：利用可解释人工智能揭露和应对网络疾病攻击 cs.CR

This work has been submitted to the IEEE for possible publication

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.13419v1) [paper-pdf](http://arxiv.org/pdf/2503.13419v1)

**Authors**: Ripan Kumar Kundu, Matthew Denton, Genova Mongalo, Prasad Calyam, Khaza Anuarul Hoque

**Abstract**: The synergy between virtual reality (VR) and artificial intelligence (AI), specifically deep learning (DL)-based cybersickness detection models, has ushered in unprecedented advancements in immersive experiences by automatically detecting cybersickness severity and adaptively various mitigation techniques, offering a smooth and comfortable VR experience. While this DL-enabled cybersickness detection method provides promising solutions for enhancing user experiences, it also introduces new risks since these models are vulnerable to adversarial attacks; a small perturbation of the input data that is visually undetectable to human observers can fool the cybersickness detection model and trigger unexpected mitigation, thus disrupting user immersive experiences (UIX) and even posing safety risks. In this paper, we present a new type of VR attack, i.e., a cybersickness attack, which successfully stops the triggering of cybersickness mitigation by fooling DL-based cybersickness detection models and dramatically hinders the UIX. Next, we propose a novel explainable artificial intelligence (XAI)-guided cybersickness attack detection framework to detect such attacks in VR to ensure UIX and a comfortable VR experience. We evaluate the proposed attack and the detection framework using two state-of-the-art open-source VR cybersickness datasets: Simulation 2021 and Gameplay dataset. Finally, to verify the effectiveness of our proposed method, we implement the attack and the XAI-based detection using a testbed with a custom-built VR roller coaster simulation with an HTC Vive Pro Eye headset and perform a user study. Our study shows that such an attack can dramatically hinder the UIX. However, our proposed XAI-guided cybersickness attack detection can successfully detect cybersickness attacks and trigger the proper mitigation, effectively reducing VR cybersickness.

摘要: 虚拟现实(VR)和人工智能(AI)之间的协同，特别是基于深度学习(DL)的晕屏检测模型，通过自动检测晕屏的严重程度和自适应的各种缓解技术，带来了沉浸式体验的前所未有的进步，提供了流畅舒适的VR体验。虽然这种支持DL的晕车检测方法为增强用户体验提供了有希望的解决方案，但它也引入了新的风险，因为这些模型容易受到对手攻击；输入数据的微小扰动在视觉上无法被人类观察者检测到，就可以欺骗晕车检测模型并引发意外的缓解，从而扰乱用户沉浸式体验(UIX)，甚至构成安全风险。在本文中，我们提出了一种新型的虚拟现实攻击，即晕屏攻击，它通过愚弄基于DL的晕屏检测模型成功地阻止了晕屏缓解的触发，并显著阻碍了UIX。接下来，我们提出了一种新颖的可解释人工智能(XAI)引导的网络病攻击检测框架来检测VR中的此类攻击，以确保UIX和舒适的VR体验。我们使用两个最先进的开源虚拟现实晕屏数据集：模拟2021和游戏玩法数据集来评估所提出的攻击和检测框架。最后，为了验证我们提出的方法的有效性，我们使用HTC Vive Pro Eye耳机通过定制的VR过山车模拟实验床实现了攻击和基于XAI的检测，并进行了用户研究。我们的研究表明，这样的攻击可以极大地阻碍UIX。然而，我们提出的XAI引导的晕屏攻击检测可以成功检测到晕屏攻击并触发适当的缓解，有效地减少VR晕屏。



## **11. On the Byzantine-Resilience of Distillation-Based Federated Learning**

基于蒸馏的联邦学习的拜占庭弹性 cs.LG

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2402.12265v3) [paper-pdf](http://arxiv.org/pdf/2402.12265v3)

**Authors**: Christophe Roux, Max Zimmer, Sebastian Pokutta

**Abstract**: Federated Learning (FL) algorithms using Knowledge Distillation (KD) have received increasing attention due to their favorable properties with respect to privacy, non-i.i.d. data and communication cost. These methods depart from transmitting model parameters and instead communicate information about a learning task by sharing predictions on a public dataset. In this work, we study the performance of such approaches in the byzantine setting, where a subset of the clients act in an adversarial manner aiming to disrupt the learning process. We show that KD-based FL algorithms are remarkably resilient and analyze how byzantine clients can influence the learning process. Based on these insights, we introduce two new byzantine attacks and demonstrate their ability to break existing byzantine-resilient methods. Additionally, we propose a novel defence method which enhances the byzantine resilience of KD-based FL algorithms. Finally, we provide a general framework to obfuscate attacks, making them significantly harder to detect, thereby improving their effectiveness. Our findings serve as an important building block in the analysis of byzantine FL, contributing through the development of new attacks and new defence mechanisms, further advancing the robustness of KD-based FL algorithms.

摘要: 基于知识蒸馏(KD)的联合学习(FL)算法因其在隐私、非I.I.D.等方面的良好特性而受到越来越多的关注。数据和通信成本。这些方法不同于传输模型参数，而是通过共享对公共数据集的预测来传递关于学习任务的信息。在这项工作中，我们研究了这些方法在拜占庭环境下的性能，在拜占庭环境中，客户的子集以对抗性的方式行动，旨在扰乱学习过程。我们证明了基于KD的FL算法具有显著的弹性，并分析了拜占庭客户端如何影响学习过程。基于这些见解，我们引入了两个新的拜占庭攻击，并展示了它们打破现有拜占庭弹性方法的能力。此外，我们还提出了一种新的防御方法，增强了基于KD的FL算法的拜占庭抗攻击能力。最后，我们提供了一个通用框架来混淆攻击，使它们更难被检测到，从而提高了它们的有效性。我们的发现是拜占庭FL分析的重要组成部分，通过开发新的攻击和新的防御机制，进一步提高了基于KD的FL算法的健壮性。



## **12. How Good is my Histopathology Vision-Language Foundation Model? A Holistic Benchmark**

我的组织学视觉语言基础模型有多好？整体基准 eess.IV

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.12990v1) [paper-pdf](http://arxiv.org/pdf/2503.12990v1)

**Authors**: Roba Al Majzoub, Hashmat Malik, Muzammal Naseer, Zaigham Zaheer, Tariq Mahmood, Salman Khan, Fahad Khan

**Abstract**: Recently, histopathology vision-language foundation models (VLMs) have gained popularity due to their enhanced performance and generalizability across different downstream tasks. However, most existing histopathology benchmarks are either unimodal or limited in terms of diversity of clinical tasks, organs, and acquisition instruments, as well as their partial availability to the public due to patient data privacy. As a consequence, there is a lack of comprehensive evaluation of existing histopathology VLMs on a unified benchmark setting that better reflects a wide range of clinical scenarios. To address this gap, we introduce HistoVL, a fully open-source comprehensive benchmark comprising images acquired using up to 11 various acquisition tools that are paired with specifically crafted captions by incorporating class names and diverse pathology descriptions. Our Histo-VL includes 26 organs, 31 cancer types, and a wide variety of tissue obtained from 14 heterogeneous patient cohorts, totaling more than 5 million patches obtained from over 41K WSIs viewed under various magnification levels. We systematically evaluate existing histopathology VLMs on Histo-VL to simulate diverse tasks performed by experts in real-world clinical scenarios. Our analysis reveals interesting findings, including large sensitivity of most existing histopathology VLMs to textual changes with a drop in balanced accuracy of up to 25% in tasks such as Metastasis detection, low robustness to adversarial attacks, as well as improper calibration of models evident through high ECE values and low model prediction confidence, all of which can affect their clinical implementation.

摘要: 最近，组织病理学视觉-语言基础模型(VLM)因其在不同下游任务中增强的性能和普适性而广受欢迎。然而，现有的大多数组织病理学基准要么是单一的，要么在临床任务、器官和采集工具的多样性方面受到限制，而且由于患者数据的隐私，它们对公众的部分可用性。因此，在一个统一的基准设置上缺乏对现有组织病理学VLM的全面评估，以更好地反映广泛的临床情景。为了弥补这一差距，我们引入了HistoVL，这是一个完全开源的综合基准，包括使用多达11种不同的采集工具获取的图像，这些工具通过结合类名和不同的病理描述与专门制作的说明相匹配。我们的HISTO-VL包括26个器官，31种癌症类型，以及从14个不同的患者队列中获得的各种组织，总计超过500万个斑块，这些斑块是在不同放大水平下从超过41K的WSIS中获得的。我们系统地评估HISTO-VL上现有的组织病理学VLM，以模拟真实世界临床场景中专家执行的各种任务。我们的分析揭示了有趣的发现，包括大多数现有的组织病理学VLM对文本变化的高度敏感性，在转移检测等任务中平衡准确率下降高达25%，对对抗性攻击的稳健性低，以及通过高EC值和低模型预测置信度明显地对模型进行不正确的校准，所有这些都可能影响其临床实施。



## **13. Distributed Black-box Attack: Do Not Overestimate Black-box Attacks**

分布式黑匣子攻击：不要高估黑匣子攻击 cs.LG

Accepted by ICLR Workshop, 2025

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2210.16371v5) [paper-pdf](http://arxiv.org/pdf/2210.16371v5)

**Authors**: Han Wu, Sareh Rowlands, Johan Wahlstrom

**Abstract**: As cloud computing becomes pervasive, deep learning models are deployed on cloud servers and then provided as APIs to end users. However, black-box adversarial attacks can fool image classification models without access to model structure and weights. Recent studies have reported attack success rates of over 95% with fewer than 1,000 queries. Then the question arises: whether black-box attacks have become a real threat against cloud APIs? To shed some light on this, our research indicates that black-box attacks are not as effective against cloud APIs as proposed in research papers due to several common mistakes that overestimate the efficiency of black-box attacks. To avoid similar mistakes, we conduct black-box attacks directly on cloud APIs rather than local models.

摘要: 随着云计算变得普遍，深度学习模型被部署在云服务器上，然后作为API提供给最终用户。然而，黑匣子对抗攻击可以在不访问模型结构和权重的情况下欺骗图像分类模型。最近的研究报告称，查询少于1，000个，攻击成功率超过95%。那么问题就来了：黑匣子攻击是否已成为对云API的真正威胁？为了阐明这一点，我们的研究表明，由于存在几个高估黑匣子攻击效率的常见错误，黑匣子攻击对云API的效果并不像研究论文中提出的那样有效。为了避免类似的错误，我们直接对云API而不是本地模型进行黑匣子攻击。



## **14. Improving Generalization of Universal Adversarial Perturbation via Dynamic Maximin Optimization**

通过动态最大化优化改进普遍对抗扰动的推广 cs.LG

Accepted in AAAI 2025

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.12793v1) [paper-pdf](http://arxiv.org/pdf/2503.12793v1)

**Authors**: Yechao Zhang, Yingzhe Xu, Junyu Shi, Leo Yu Zhang, Shengshan Hu, Minghui Li, Yanjun Zhang

**Abstract**: Deep neural networks (DNNs) are susceptible to universal adversarial perturbations (UAPs). These perturbations are meticulously designed to fool the target model universally across all sample classes. Unlike instance-specific adversarial examples (AEs), generating UAPs is more complex because they must be generalized across a wide range of data samples and models. Our research reveals that existing universal attack methods, which optimize UAPs using DNNs with static model parameter snapshots, do not fully leverage the potential of DNNs to generate more effective UAPs. Rather than optimizing UAPs against static DNN models with a fixed training set, we suggest using dynamic model-data pairs to generate UAPs. In particular, we introduce a dynamic maximin optimization strategy, aiming to optimize the UAP across a variety of optimal model-data pairs. We term this approach DM-UAP. DM-UAP utilizes an iterative max-min-min optimization framework that refines the model-data pairs, coupled with a curriculum UAP learning algorithm to examine the combined space of model parameters and data thoroughly. Comprehensive experiments on the ImageNet dataset demonstrate that the proposed DM-UAP markedly enhances both cross-sample universality and cross-model transferability of UAPs. Using only 500 samples for UAP generation, DM-UAP outperforms the state-of-the-art approach with an average increase in fooling ratio of 12.108%.

摘要: 深度神经网络(DNN)容易受到普遍的对抗性扰动(UAP)的影响。这些扰动是精心设计的，目的是在所有样本类中普遍欺骗目标模型。与实例特定的对抗性示例(AE)不同，生成UAP更加复杂，因为它们必须在广泛的数据样本和模型中推广。我们的研究表明，现有的通用攻击方法使用带有静态模型参数快照的DNN来优化UAP，没有充分利用DNN的潜力来生成更有效的UAP。我们建议使用动态模型-数据对来生成UAP，而不是针对具有固定训练集的静态DNN模型来优化UAP。特别是，我们引入了动态最大优化策略，旨在对UAP进行各种最优模型-数据对的优化。我们称这种方法为DM-UAP。DM-UAP使用迭代的最大-最小-最小优化框架来细化模型-数据对，并结合课程UAP学习算法来彻底检查模型参数和数据的组合空间。在ImageNet数据集上的综合实验表明，DM-UAP显著增强了UAP的跨样本普适性和跨模型可转移性。仅使用500个样本生成UAP，DM-UAP的性能优于最先进的方法，平均傻瓜率提高了12.108%。



## **15. Algebraic Adversarial Attacks on Explainability Models**

对可解释性模型的代数对抗攻击 cs.LG

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2503.12683v1) [paper-pdf](http://arxiv.org/pdf/2503.12683v1)

**Authors**: Lachlan Simpson, Federico Costanza, Kyle Millar, Adriel Cheng, Cheng-Chew Lim, Hong Gunn Chew

**Abstract**: Classical adversarial attacks are phrased as a constrained optimisation problem. Despite the efficacy of a constrained optimisation approach to adversarial attacks, one cannot trace how an adversarial point was generated. In this work, we propose an algebraic approach to adversarial attacks and study the conditions under which one can generate adversarial examples for post-hoc explainability models. Phrasing neural networks in the framework of geometric deep learning, algebraic adversarial attacks are constructed through analysis of the symmetry groups of neural networks. Algebraic adversarial examples provide a mathematically tractable approach to adversarial examples. We validate our approach of algebraic adversarial examples on two well-known and one real-world dataset.

摘要: 经典的对抗攻击被描述为一个约束优化问题。尽管约束优化方法对对抗攻击有效，但人们无法追踪对抗点是如何生成的。在这项工作中，我们提出了一种对抗性攻击的代数方法，并研究了为事后可解释性模型生成对抗性示例的条件。在几何深度学习的框架下对神经网络进行分段，通过分析神经网络的对称群来构建代数对抗攻击。代数对抗性例子为对抗性例子提供了一种数学上易于处理的方法。我们在两个知名数据集和一个现实世界数据集上验证了我们的代数对抗示例方法。



## **16. Provably Reliable Conformal Prediction Sets in the Presence of Data Poisoning**

数据中毒情况下可证明可靠的保形预测集 cs.LG

Accepted at ICLR 2025 (Spotlight)

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2410.09878v4) [paper-pdf](http://arxiv.org/pdf/2410.09878v4)

**Authors**: Yan Scholten, Stephan Günnemann

**Abstract**: Conformal prediction provides model-agnostic and distribution-free uncertainty quantification through prediction sets that are guaranteed to include the ground truth with any user-specified probability. Yet, conformal prediction is not reliable under poisoning attacks where adversaries manipulate both training and calibration data, which can significantly alter prediction sets in practice. As a solution, we propose reliable prediction sets (RPS): the first efficient method for constructing conformal prediction sets with provable reliability guarantees under poisoning. To ensure reliability under training poisoning, we introduce smoothed score functions that reliably aggregate predictions of classifiers trained on distinct partitions of the training data. To ensure reliability under calibration poisoning, we construct multiple prediction sets, each calibrated on distinct subsets of the calibration data. We then aggregate them into a majority prediction set, which includes a class only if it appears in a majority of the individual sets. Both proposed aggregations mitigate the influence of datapoints in the training and calibration data on the final prediction set. We experimentally validate our approach on image classification tasks, achieving strong reliability while maintaining utility and preserving coverage on clean data. Overall, our approach represents an important step towards more trustworthy uncertainty quantification in the presence of data poisoning.

摘要: 保角预测通过预测集提供与模型无关和无分布的不确定性量化，这些预测集保证以任何用户指定的概率包括基本事实。然而，在中毒攻击下，保角预测是不可靠的，其中对手同时操纵训练和校准数据，这在实践中可能会显著改变预测集。作为解决方案，我们提出了可靠预测集(RPS)：在中毒情况下构造具有可证明可靠性保证的共形预测集的第一种有效方法。为了确保在训练中毒情况下的可靠性，我们引入了平滑得分函数，它可靠地聚合了在不同的训练数据分区上训练的分类器的预测。为了确保在校准中毒情况下的可靠性，我们构造了多个预测集，每个预测集都在校准数据的不同子集上进行校准。然后我们将它们聚集到一个多数预测集合中，该集合只包括一个类，当它出现在大多数单独的集合中时。这两种建议的聚合都减轻了训练和校准数据中的数据点对最终预测集的影响。我们在实验上验证了我们的方法在图像分类任务上的有效性，在保持实用性和对干净数据的覆盖率的同时实现了很强的可靠性。总体而言，我们的方法代表着在存在数据中毒的情况下朝着更可信的不确定性量化迈出的重要一步。



## **17. Mind the Gap: Detecting Black-box Adversarial Attacks in the Making through Query Update Analysis**

注意差距：通过查询更新分析检测正在形成的黑匣子对抗攻击 cs.CR

14 pages

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2503.02986v3) [paper-pdf](http://arxiv.org/pdf/2503.02986v3)

**Authors**: Jeonghwan Park, Niall McLaughlin, Ihsen Alouani

**Abstract**: Adversarial attacks remain a significant threat that can jeopardize the integrity of Machine Learning (ML) models. In particular, query-based black-box attacks can generate malicious noise without having access to the victim model's architecture, making them practical in real-world contexts. The community has proposed several defenses against adversarial attacks, only to be broken by more advanced and adaptive attack strategies. In this paper, we propose a framework that detects if an adversarial noise instance is being generated. Unlike existing stateful defenses that detect adversarial noise generation by monitoring the input space, our approach learns adversarial patterns in the input update similarity space. In fact, we propose to observe a new metric called Delta Similarity (DS), which we show it captures more efficiently the adversarial behavior. We evaluate our approach against 8 state-of-the-art attacks, including adaptive attacks, where the adversary is aware of the defense and tries to evade detection. We find that our approach is significantly more robust than existing defenses both in terms of specificity and sensitivity.

摘要: 对抗性攻击仍然是一个严重的威胁，可能会危及机器学习(ML)模型的完整性。特别是，基于查询的黑盒攻击可以在不访问受害者模型的体系结构的情况下生成恶意噪声，使它们在真实世界的上下文中具有实用性。社区已经提出了几种针对对抗性攻击的防御措施，但都被更先进和适应性更强的攻击策略打破了。在这篇文章中，我们提出了一个框架，它检测是否正在生成对抗性噪声实例。与现有的通过监测输入空间来检测对抗性噪声产生的状态防御方法不同，我们的方法在输入更新相似性空间中学习对抗性模式。事实上，我们提出了一种新的度量，称为Delta相似度(DS)，我们表明它更有效地捕获了对手的行为。我们评估了我们的方法针对8种最先进的攻击，包括自适应攻击，在这些攻击中，对手知道防御并试图逃避检测。我们发现，我们的方法在特异性和敏感性方面都明显比现有的防御方法更稳健。



## **18. GAN-Based Single-Stage Defense for Traffic Sign Classification Under Adversarial Patch Attack**

对抗补丁攻击下基于GAN的交通标志分类单级防御 cs.CV

This work has been submitted to the IEEE Transactions on Intelligent  Transportation Systems (T-ITS) for possible publication

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2503.12567v1) [paper-pdf](http://arxiv.org/pdf/2503.12567v1)

**Authors**: Abyad Enan, Mashrur Chowdhury

**Abstract**: Computer Vision plays a critical role in ensuring the safe navigation of autonomous vehicles (AVs). An AV perception module is responsible for capturing and interpreting the surrounding environment to facilitate safe navigation. This module enables AVs to recognize traffic signs, traffic lights, and various road users. However, the perception module is vulnerable to adversarial attacks, which can compromise their accuracy and reliability. One such attack is the adversarial patch attack (APA), a physical attack in which an adversary strategically places a specially crafted sticker on an object to deceive object classifiers. In APA, an adversarial patch is positioned on a target object, leading the classifier to misidentify it. Such an APA can cause AVs to misclassify traffic signs, leading to catastrophic incidents. To enhance the security of an AV perception system against APAs, this study develops a Generative Adversarial Network (GAN)-based single-stage defense strategy for traffic sign classification. This approach is tailored to defend against APAs on different classes of traffic signs without prior knowledge of a patch's design. This study found this approach to be effective against patches of varying sizes. Our experimental analysis demonstrates that the defense strategy presented in this paper improves the classifier's accuracy under APA conditions by up to 80.8% and enhances overall classification accuracy for all the traffic signs considered in this study by 58%, compared to a classifier without any defense mechanism. Our defense strategy is model-agnostic, making it applicable to any traffic sign classifier, regardless of the underlying classification model.

摘要: 计算机视觉在保证自动驾驶车辆的安全导航中起着至关重要的作用。一个AV感知模块负责捕捉和解释周围环境，以促进安全导航。该模块使自动驾驶系统能够识别交通标志、红绿灯和各种道路使用者。然而，感知模块容易受到敌意攻击，这可能会损害其准确性和可靠性。其中一种攻击是对抗性补丁攻击(APA)，这是一种物理攻击，对手在对象上策略性地放置专门制作的贴纸，以欺骗对象分类器。在APA中，敌意补丁被定位在目标对象上，导致分类器错误地识别它。这样的APA可能会导致AVS对交通标志进行错误分类，从而导致灾难性事件。为了提高反病毒感知系统对抗自动识别系统的安全性，提出了一种基于生成性对抗网络的交通标志分类单级防御策略。这种方法是为防御不同类别交通标志上的APA而量身定做的，而不需要事先知道补丁的设计。这项研究发现，这种方法对不同大小的补丁有效。我们的实验分析表明，与没有任何防御机制的分类器相比，本文提出的防御策略在APA条件下将分类器的准确率提高了80.8%，对本研究中考虑的所有交通标志的整体分类准确率提高了58%。我们的防御策略是模型不可知的，使其适用于任何交通标志分类器，而不考虑底层的分类模型。



## **19. On the Privacy Risks of Spiking Neural Networks: A Membership Inference Analysis**

尖峰神经网络的隐私风险：成员推断分析 cs.LG

13 pages, 6 figures

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2502.13191v2) [paper-pdf](http://arxiv.org/pdf/2502.13191v2)

**Authors**: Junyi Guan, Abhijith Sharma, Chong Tian, Salem Lahlou

**Abstract**: Spiking Neural Networks (SNNs) are increasingly explored for their energy efficiency and robustness in real-world applications, yet their privacy risks remain largely unexamined. In this work, we investigate the susceptibility of SNNs to Membership Inference Attacks (MIAs) -- a major privacy threat where an adversary attempts to determine whether a given sample was part of the training dataset. While prior work suggests that SNNs may offer inherent robustness due to their discrete, event-driven nature, we find that its resilience diminishes as latency (T) increases. Furthermore, we introduce an input dropout strategy under black box setting, that significantly enhances membership inference in SNNs. Our findings challenge the assumption that SNNs are inherently more secure, and even though they are expected to be better, our results reveal that SNNs exhibit privacy vulnerabilities that are equally comparable to Artificial Neural Networks (ANNs). Our code is available at https://anonymous.4open.science/r/MIA_SNN-3610.

摘要: 尖峰神经网络(SNN)在实际应用中因其能量效率和稳健性而受到越来越多的研究，但其隐私风险在很大程度上仍未得到检查。在这项工作中，我们调查了SNN对成员关系推断攻击(MIA)的敏感性--MIA是一种主要的隐私威胁，攻击者试图确定给定样本是否属于训练数据集。虽然以前的工作表明，由于SNN的离散、事件驱动的性质，它可能提供固有的健壮性，但我们发现，它的弹性随着延迟(T)的增加而减弱。此外，我们在黑箱设置下引入了一种输入丢弃策略，显著增强了SNN中的成员关系推理。我们的发现挑战了SNN天生更安全的假设，尽管预计SNN会更好，但我们的结果显示，SNN表现出与人工神经网络(ANN)相当的隐私漏洞。我们的代码可以在https://anonymous.4open.science/r/MIA_SNN-3610.上找到



## **20. Towards Privacy-Preserving Data-Driven Education: The Potential of Federated Learning**

迈向保护隐私的数据驱动教育：联邦学习的潜力 cs.LG

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2503.13550v1) [paper-pdf](http://arxiv.org/pdf/2503.13550v1)

**Authors**: Mohammad Khalil, Ronas Shakya, Qinyi Liu

**Abstract**: The increasing adoption of data-driven applications in education such as in learning analytics and AI in education has raised significant privacy and data protection concerns. While these challenges have been widely discussed in previous works, there are still limited practical solutions. Federated learning has recently been discoursed as a promising privacy-preserving technique, yet its application in education remains scarce. This paper presents an experimental evaluation of federated learning for educational data prediction, comparing its performance to traditional non-federated approaches. Our findings indicate that federated learning achieves comparable predictive accuracy. Furthermore, under adversarial attacks, federated learning demonstrates greater resilience compared to non-federated settings. We summarise that our results reinforce the value of federated learning as a potential approach for balancing predictive performance and privacy in educational contexts.

摘要: 教育领域越来越多地采用数据驱动应用程序，例如学习分析和教育领域的人工智能，引发了严重的隐私和数据保护问题。虽然这些挑战在之前的作品中得到了广泛讨论，但实际的解决方案仍然有限。联邦学习最近被认为是一种有前途的隐私保护技术，但其在教育中的应用仍然很少。本文对教育数据预测的联邦学习进行了实验评估，并将其性能与传统非联邦方法进行了比较。我们的研究结果表明，联邦学习可以实现相当的预测准确性。此外，在对抗性攻击下，与非联邦环境相比，联邦学习表现出更大的弹性。我们总结说，我们的结果强化了联邦学习作为在教育环境中平衡预测性能和隐私的潜在方法的价值。



## **21. CARNet: Collaborative Adversarial Resilience for Robust Underwater Image Enhancement and Perception**

CARNet：协作对抗弹性，实现稳健的水下图像增强和感知 cs.CV

13 pages, 13 figures

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2309.01102v2) [paper-pdf](http://arxiv.org/pdf/2309.01102v2)

**Authors**: Zengxi Zhang, Zeru Shi, Zhiying Jiang, Jinyuan Liu

**Abstract**: Due to the uneven absorption of different light wavelengths in aquatic environments, underwater images suffer from low visibility and clear color deviations. With the advancement of autonomous underwater vehicles, extensive research has been conducted on learning-based underwater enhancement algorithms. These works can generate visually pleasing enhanced images and mitigate the adverse effects of degraded images on subsequent perception tasks. However, learning-based methods are susceptible to the inherent fragility of adversarial attacks, causing significant disruption in enhanced results. In this work, we introduce a collaborative adversarial resilience network, dubbed CARNet, for underwater image enhancement and subsequent detection tasks. Concretely, we first introduce an invertible network with strong perturbation-perceptual abilities to isolate attacks from underwater images, preventing interference with visual quality enhancement and perceptual tasks. Furthermore, an attack pattern discriminator is introduced to adaptively identify and eliminate various types of attacks. Additionally, we propose a bilevel attack optimization strategy to heighten the robustness of the network against different types of attacks under the collaborative adversarial training of vision-driven and perception-driven attacks. Extensive experiments demonstrate that the proposed method outputs visually appealing enhancement images and performs an average 6.71% higher detection mAP than state-of-the-art methods.

摘要: 由于水环境对不同波长光的吸收不均匀，水下图像的能见度较低，颜色偏差较明显。随着自主水下机器人的发展，基于学习的水下增强算法得到了广泛的研究。这些工作可以产生视觉上令人愉悦的增强图像，并缓解退化图像对后续感知任务的不利影响。然而，基于学习的方法容易受到对抗性攻击固有的脆弱性的影响，从而对增强的结果造成重大破坏。在这项工作中，我们介绍了一个协作的对抗性网络，称为CARNET，用于水下图像增强和后续检测任务。具体地说，我们首先引入了一个具有很强扰动感知能力的可逆网络，将攻击从水下图像中分离出来，防止了对视觉质量增强和感知任务的干扰。此外，还引入了攻击模式鉴别器来自适应地识别和消除各种类型的攻击。此外，在视觉驱动攻击和感知驱动攻击的协同对抗训练下，我们提出了一种双层攻击优化策略，以提高网络对不同类型攻击的鲁棒性。大量实验表明，该方法输出的增强图像视觉效果良好，检测结果比现有方法平均提高6.71%。



## **22. Augmented Adversarial Trigger Learning**

增强对抗触发学习 cs.LG

**SubmitDate**: 2025-03-16    [abs](http://arxiv.org/abs/2503.12339v1) [paper-pdf](http://arxiv.org/pdf/2503.12339v1)

**Authors**: Zhe Wang, Yanjun Qi

**Abstract**: Gradient optimization-based adversarial attack methods automate the learning of adversarial triggers to generate jailbreak prompts or leak system prompts. In this work, we take a closer look at the optimization objective of adversarial trigger learning and propose ATLA: Adversarial Trigger Learning with Augmented objectives. ATLA improves the negative log-likelihood loss used by previous studies into a weighted loss formulation that encourages the learned adversarial triggers to optimize more towards response format tokens. This enables ATLA to learn an adversarial trigger from just one query-response pair and the learned trigger generalizes well to other similar queries. We further design a variation to augment trigger optimization with an auxiliary loss that suppresses evasive responses. We showcase how to use ATLA to learn adversarial suffixes jailbreaking LLMs and to extract hidden system prompts. Empirically we demonstrate that ATLA consistently outperforms current state-of-the-art techniques, achieving nearly 100% success in attacking while requiring 80% fewer queries. ATLA learned jailbreak suffixes demonstrate high generalization to unseen queries and transfer well to new LLMs.

摘要: 基于梯度优化的对抗性攻击方法自动学习对抗性触发器，以生成越狱提示或泄漏系统提示。在这项工作中，我们更深入地研究了对抗性触发学习的优化目标，并提出了ATLA：带有扩充目标的对抗性触发学习。ATLA将先前研究使用的负对数似然损失改进为加权损失公式，该公式鼓励学习的对抗性触发器更多地针对响应格式令牌进行优化。这使ATLA能够仅从一个查询-响应对学习敌意触发器，并且学习的触发器很好地推广到其他类似的查询。我们进一步设计了一种变体，以增加辅助损耗来增强触发优化，从而抑制回避响应。我们展示了如何使用ATLA来学习敌意后缀，越狱LLM和提取隐藏的系统提示。经验证明，ATLA始终优于当前最先进的技术，在攻击中取得了近100%的成功，所需查询减少了80%。ATLA学习到的越狱后缀对看不见的查询表现出高度的概括性，并很好地转移到新的LLM。



## **23. Training-Free Mitigation of Adversarial Attacks on Deep Learning-Based MRI Reconstruction**

基于深度学习的MRI重建的对抗攻击的免培训缓解 cs.CV

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2501.01908v2) [paper-pdf](http://arxiv.org/pdf/2501.01908v2)

**Authors**: Mahdi Saberi, Chi Zhang, Mehmet Akcakaya

**Abstract**: Deep learning (DL) methods, especially those based on physics-driven DL, have become the state-of-the-art for reconstructing sub-sampled magnetic resonance imaging (MRI) data. However, studies have shown that these methods are susceptible to small adversarial input perturbations, or attacks, resulting in major distortions in the output images. Various strategies have been proposed to reduce the effects of these attacks, but they require retraining and may lower reconstruction quality for non-perturbed/clean inputs. In this work, we propose a novel approach for mitigating adversarial attacks on MRI reconstruction models without any retraining. Our framework is based on the idea of cyclic measurement consistency. The output of the model is mapped to another set of MRI measurements for a different sub-sampling pattern, and this synthesized data is reconstructed with the same model. Intuitively, without an attack, the second reconstruction is expected to be consistent with the first, while with an attack, disruptions are present. A novel objective function is devised based on this idea, which is minimized within a small ball around the attack input for mitigation. Experimental results show that our method substantially reduces the impact of adversarial perturbations across different datasets, attack types/strengths and PD-DL networks, and qualitatively and quantitatively outperforms conventional mitigation methods that involve retraining. Finally, we extend our mitigation method to two important practical scenarios: a blind setup, where the attack strength or algorithm is not known to the end user; and an adaptive attack setup, where the attacker has full knowledge of the defense strategy. Our approach remains effective in both cases.

摘要: 深度学习方法，特别是基于物理驱动的深度学习方法，已经成为重建亚采样磁共振成像(MRI)数据的最新方法。然而，研究表明，这些方法容易受到小的对抗性输入扰动或攻击，导致输出图像的严重失真。已经提出了各种策略来减少这些攻击的影响，但它们需要重新培训，并且可能会降低非扰动/干净输入的重建质量。在这项工作中，我们提出了一种新的方法来缓解对MRI重建模型的敌意攻击，而不需要任何重新训练。我们的框架是基于循环测量一致性的思想。该模型的输出被映射到用于不同子采样模式的另一组MRI测量，并且利用相同的模型重建该合成数据。直观地说，在没有攻击的情况下，第二次重建预计与第一次一致，而在攻击时，会出现中断。基于这一思想，设计了一种新的目标函数，该函数在攻击输入周围的小球内最小化以减少攻击。实验结果表明，我们的方法大大降低了不同数据集、攻击类型/强度和PD-DL网络之间的对抗性扰动的影响，并且在定性和定量上都优于传统的涉及再训练的缓解方法。最后，我们将我们的缓解方法扩展到两个重要的实际场景：盲目设置，其中最终用户不知道攻击强度或算法；以及自适应攻击设置，其中攻击者完全了解防御策略。我们的方法在这两种情况下仍然有效。



## **24. Multi-Agent Systems Execute Arbitrary Malicious Code**

多代理系统执行任意恶意代码 cs.CR

30 pages, 5 figures, 8 tables

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2503.12188v1) [paper-pdf](http://arxiv.org/pdf/2503.12188v1)

**Authors**: Harold Triedman, Rishi Jha, Vitaly Shmatikov

**Abstract**: Multi-agent systems coordinate LLM-based agents to perform tasks on users' behalf. In real-world applications, multi-agent systems will inevitably interact with untrusted inputs, such as malicious Web content, files, email attachments, etc.   Using several recently proposed multi-agent frameworks as concrete examples, we demonstrate that adversarial content can hijack control and communication within the system to invoke unsafe agents and functionalities. This results in a complete security breach, up to execution of arbitrary malicious code on the user's device and/or exfiltration of sensitive data from the user's containerized environment. We show that control-flow hijacking attacks succeed even if the individual agents are not susceptible to direct or indirect prompt injection, and even if they refuse to perform harmful actions.

摘要: 多代理系统协调基于LLM的代理代表用户执行任务。在现实世界的应用程序中，多代理系统将不可避免地与不受信任的输入进行交互，例如恶意Web内容、文件、电子邮件附件等。   使用最近提出的几个多代理框架作为具体示例，我们证明对抗性内容可以劫持系统内的控制和通信，以调用不安全的代理和功能。这会导致完全的安全漏洞，甚至在用户设备上执行任意恶意代码和/或从用户的容器化环境中泄露敏感数据。我们表明，即使各个代理不容易受到直接或间接的即时注入，并且即使他们拒绝执行有害动作，控制流劫持攻击也会成功。



## **25. From ML to LLM: Evaluating the Robustness of Phishing Webpage Detection Models against Adversarial Attacks**

从ML到LLM：评估网络钓鱼网页检测模型对抗对抗攻击的稳健性 cs.CR

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2407.20361v3) [paper-pdf](http://arxiv.org/pdf/2407.20361v3)

**Authors**: Aditya Kulkarni, Vivek Balachandran, Dinil Mon Divakaran, Tamal Das

**Abstract**: Phishing attacks attempt to deceive users into stealing sensitive information, posing a significant cybersecurity threat. Advances in machine learning (ML) and deep learning (DL) have led to the development of numerous phishing webpage detection solutions, but these models remain vulnerable to adversarial attacks. Evaluating their robustness against adversarial phishing webpages is essential. Existing tools contain datasets of pre-designed phishing webpages for a limited number of brands, and lack diversity in phishing features.   To address these challenges, we develop PhishOracle, a tool that generates adversarial phishing webpages by embedding diverse phishing features into legitimate webpages. We evaluate the robustness of three existing task-specific models -- Stack model, VisualPhishNet, and Phishpedia -- against PhishOracle-generated adversarial phishing webpages and observe a significant drop in their detection rates. In contrast, a multimodal large language model (MLLM)-based phishing detector demonstrates stronger robustness against these adversarial attacks but still is prone to evasion. Our findings highlight the vulnerability of phishing detection models to adversarial attacks, emphasizing the need for more robust detection approaches. Furthermore, we conduct a user study to evaluate whether PhishOracle-generated adversarial phishing webpages can deceive users. The results show that many of these phishing webpages evade not only existing detection models but also users. We also develop the PhishOracle web app, allowing users to input a legitimate URL, select relevant phishing features and generate a corresponding phishing webpage. All resources will be made publicly available on GitHub.

摘要: 网络钓鱼攻击试图欺骗用户窃取敏感信息，构成重大的网络安全威胁。机器学习(ML)和深度学习(DL)的进步导致了许多钓鱼网页检测解决方案的发展，但这些模型仍然容易受到对手攻击。评估它们对敌意网络钓鱼网页的健壮性是至关重要的。现有工具包含为有限数量的品牌预先设计的钓鱼网页的数据集，并且在钓鱼功能方面缺乏多样性。为了应对这些挑战，我们开发了PhishOracle，这是一个通过在合法网页中嵌入不同的钓鱼功能来生成敌意钓鱼网页的工具。我们评估了现有的三种特定任务的模型--Stack模型、VisualPhishNet和Phishpedia--对PhishOracle生成的敌意钓鱼网页的稳健性，并观察到它们的检测率显著下降。相比之下，基于多模式大语言模型(MLLM)的网络钓鱼检测器对这些敌意攻击表现出更强的稳健性，但仍然容易被规避。我们的发现突出了网络钓鱼检测模型对对手攻击的脆弱性，强调了需要更强大的检测方法。此外，我们还进行了一项用户研究，以评估PhishOracle生成的敌意钓鱼网页是否可以欺骗用户。结果表明，许多钓鱼网页不仅规避了现有的检测模型，而且还规避了用户。我们还开发了PhishOracle Web应用程序，允许用户输入合法的URL，选择相关的网络钓鱼功能并生成相应的网络钓鱼网页。所有资源都将在GitHub上公开提供。



## **26. Robust Dataset Distillation by Matching Adversarial Trajectories**

通过匹配对抗轨迹进行稳健的数据集提取 cs.CV

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2503.12069v1) [paper-pdf](http://arxiv.org/pdf/2503.12069v1)

**Authors**: Wei Lai, Tianyu Ding, ren dongdong, Lei Wang, Jing Huo, Yang Gao, Wenbin Li

**Abstract**: Dataset distillation synthesizes compact datasets that enable models to achieve performance comparable to training on the original large-scale datasets. However, existing distillation methods overlook the robustness of the model, resulting in models that are vulnerable to adversarial attacks when trained on distilled data. To address this limitation, we introduce the task of ``robust dataset distillation", a novel paradigm that embeds adversarial robustness into the synthetic datasets during the distillation process. We propose Matching Adversarial Trajectories (MAT), a method that integrates adversarial training into trajectory-based dataset distillation. MAT incorporates adversarial samples during trajectory generation to obtain robust training trajectories, which are then used to guide the distillation process. As experimentally demonstrated, even through natural training on our distilled dataset, models can achieve enhanced adversarial robustness while maintaining competitive accuracy compared to existing distillation methods. Our work highlights robust dataset distillation as a new and important research direction and provides a strong baseline for future research to bridge the gap between efficient training and adversarial robustness.

摘要: 数据集精馏合成了紧凑的数据集，使模型能够获得与原始大规模数据集上的训练相当的性能。然而，现有的蒸馏方法忽略了模型的稳健性，导致在对蒸馏数据进行训练时，模型容易受到敌意攻击。为了解决这一局限性，我们引入了“稳健数据集精馏”的任务，这是一种在精馏过程中将对抗性健壮性嵌入到合成数据集中的新范例。我们提出了匹配对抗性轨迹(MAT)，一种将对抗性训练与基于轨迹的数据集提取相结合的方法。MAT在轨迹生成过程中加入对抗性样本，以获得稳健的训练轨迹，然后使用这些轨迹来指导蒸馏过程。正如实验所证明的，即使通过在我们的蒸馏数据集上的自然训练，模型也可以实现增强的对抗性稳健性，同时与现有的蒸馏方法相比保持竞争的准确性。我们的工作突出了稳健数据集蒸馏作为一个新的重要研究方向，并为未来的研究提供了一个强有力的基线，以弥合有效训练和对手稳健性之间的差距。



## **27. On Minimizing Adversarial Counterfactual Error in Adversarial RL**

对抗性RL中对抗性反事实错误的最小化 cs.LG

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2406.04724v3) [paper-pdf](http://arxiv.org/pdf/2406.04724v3)

**Authors**: Roman Belaire, Arunesh Sinha, Pradeep Varakantham

**Abstract**: Deep Reinforcement Learning (DRL) policies are highly susceptible to adversarial noise in observations, which poses significant risks in safety-critical scenarios. The challenge inherent to adversarial perturbations is that by altering the information observed by the agent, the state becomes only partially observable. Existing approaches address this by either enforcing consistent actions across nearby states or maximizing the worst-case value within adversarially perturbed observations. However, the former suffers from performance degradation when attacks succeed, while the latter tends to be overly conservative, leading to suboptimal performance in benign settings. We hypothesize that these limitations stem from their failing to account for partial observability directly. To this end, we introduce a novel objective called Adversarial Counterfactual Error (ACoE), defined on the beliefs about the true state and balancing value optimization with robustness. To make ACoE scalable in model-free settings, we propose the theoretically-grounded surrogate objective Cumulative-ACoE (C-ACoE). Our empirical evaluations on standard benchmarks (MuJoCo, Atari, and Highway) demonstrate that our method significantly outperforms current state-of-the-art approaches for addressing adversarial RL challenges, offering a promising direction for improving robustness in DRL under adversarial conditions. Our code is available at https://github.com/romanbelaire/acoe-robust-rl.

摘要: 深度强化学习(DRL)策略很容易受到观测中的对抗性噪声的影响，这在安全关键的场景中会带来巨大的风险。对抗性扰动固有的挑战是，通过改变代理人观察到的信息，状态只变得部分可观察到。现有的方法要么在附近的州强制执行一致的行动，要么在相反的扰动观测中最大化最坏情况的值来解决这个问题。然而，当攻击成功时，前者的性能会下降，而后者往往过于保守，导致在良性设置下的性能不佳。我们假设，这些限制源于它们未能直接解释部分可观测性。为此，我们引入了一个新的目标，称为对抗性反事实错误(ACoE)，它定义在关于真实状态的信念和价值优化与稳健性之间的平衡。为了使ACoE在无模型环境下具有可扩展性，我们提出了基于理论的代理目标累积ACoE(C-ACoE)。我们在标准基准(MuJoCo、Atari和Road)上的经验评估表明，我们的方法显著优于当前最先进的方法来应对对抗性的RL挑战，为提高对抗性条件下的DRL的健壮性提供了一个有希望的方向。我们的代码可以在https://github.com/romanbelaire/acoe-robust-rl.上找到



## **28. Robust and Efficient Adversarial Defense in SNNs via Image Purification and Joint Detection**

通过图像净化和联合检测在SNN中实现稳健有效的对抗防御 cs.CV

**SubmitDate**: 2025-03-15    [abs](http://arxiv.org/abs/2404.17092v2) [paper-pdf](http://arxiv.org/pdf/2404.17092v2)

**Authors**: Weiran Chen, Qi Xu

**Abstract**: Spiking Neural Networks (SNNs) aim to bridge the gap between neuroscience and machine learning by emulating the structure of the human nervous system. However, like convolutional neural networks, SNNs are vulnerable to adversarial attacks. To tackle the challenge, we propose a biologically inspired methodology to enhance the robustness of SNNs, drawing insights from the visual masking effect and filtering theory. First, an end-to-end SNN-based image purification model is proposed to defend against adversarial attacks, including a noise extraction network and a non-blind denoising network. The former network extracts noise features from noisy images, while the latter component employs a residual U-Net structure to reconstruct high-quality noisy images and generate clean images. Simultaneously, a multi-level firing SNN based on Squeeze-and-Excitation Network is introduced to improve the robustness of the classifier. Crucially, the proposed image purification network serves as a pre-processing module, avoiding modifications to classifiers. Unlike adversarial training, our method is highly flexible and can be seamlessly integrated with other defense strategies. Experimental results on various datasets demonstrate that the proposed methodology outperforms state-of-the-art baselines in terms of defense effectiveness, training time, and resource consumption.

摘要: 尖峰神经网络(SNN)旨在通过模拟人类神经系统的结构来弥合神经科学和机器学习之间的差距。然而，与卷积神经网络一样，SNN也容易受到敌意攻击。为了应对这一挑战，我们借鉴视觉掩蔽效应和过滤理论，提出了一种受生物启发的方法来增强SNN的稳健性。首先，提出了一种端到端的SNN图像净化模型，该模型包括噪声提取网络和非盲去噪网络。前者从含噪图像中提取噪声特征，后者利用残差U网结构重建高质量的含噪图像并生成清晰的图像。同时，为了提高分类器的鲁棒性，引入了一种基于挤压激励网络的多级激发SNN。最重要的是，提出的图像净化网络作为一个预处理模块，避免了对分类器的修改。与对抗性训练不同，我们的方法高度灵活，可以与其他防御策略无缝集成。在不同数据集上的实验结果表明，该方法在防御效果、训练时间和资源消耗方面都优于最新的基线。



## **29. A Framework for Evaluating Emerging Cyberattack Capabilities of AI**

评估人工智能新兴网络攻击能力的框架 cs.CR

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11917v1) [paper-pdf](http://arxiv.org/pdf/2503.11917v1)

**Authors**: Mikel Rodriguez, Raluca Ada Popa, Four Flynn, Lihao Liang, Allan Dafoe, Anna Wang

**Abstract**: As frontier models become more capable, the community has attempted to evaluate their ability to enable cyberattacks. Performing a comprehensive evaluation and prioritizing defenses are crucial tasks in preparing for AGI safely. However, current cyber evaluation efforts are ad-hoc, with no systematic reasoning about the various phases of attacks, and do not provide a steer on how to use targeted defenses. In this work, we propose a novel approach to AI cyber capability evaluation that (1) examines the end-to-end attack chain, (2) helps to identify gaps in the evaluation of AI threats, and (3) helps defenders prioritize targeted mitigations and conduct AI-enabled adversary emulation to support red teaming. To achieve these goals, we propose adapting existing cyberattack chain frameworks to AI systems. We analyze over 12,000 instances of real-world attempts to use AI in cyberattacks catalogued by Google's Threat Intelligence Group. Using this analysis, we curate a representative collection of seven cyberattack chain archetypes and conduct a bottleneck analysis to identify areas of potential AI-driven cost disruption. Our evaluation benchmark consists of 50 new challenges spanning different phases of cyberattacks. Based on this, we devise targeted cybersecurity model evaluations, report on the potential for AI to amplify offensive cyber capabilities across specific attack phases, and conclude with recommendations on prioritizing defenses. In all, we consider this to be the most comprehensive AI cyber risk evaluation framework published so far.

摘要: 随着前沿模型变得更有能力，该社区试图评估他们启用网络攻击的能力。进行全面的评估和确定防御的优先顺序是安全准备AGI的关键任务。然而，目前的网络评估工作是临时的，没有关于攻击的各个阶段的系统推理，也没有为如何使用定向防御提供指导。在这项工作中，我们提出了一种新的人工智能网络能力评估方法，该方法(1)检查端到端攻击链，(2)帮助识别人工智能威胁评估中的差距，(3)帮助防御者确定目标缓解的优先顺序，并进行支持红色团队的人工智能启用的对手仿真。为了实现这些目标，我们建议将现有的网络攻击链框架应用于人工智能系统。我们分析了谷歌威胁情报集团编目的超过12,000起试图在网络攻击中使用人工智能的真实世界实例。利用这一分析，我们收集了具有代表性的七个网络攻击链原型，并进行了瓶颈分析，以确定潜在的人工智能驱动的成本中断领域。我们的评估基准包括50个新挑战，涵盖网络攻击的不同阶段。在此基础上，我们设计了有针对性的网络安全模型评估，报告了人工智能在特定攻击阶段放大进攻性网络能力的潜力，并对防御的优先顺序提出了建议。总而言之，我们认为这是迄今为止发布的最全面的人工智能网络风险评估框架。



## **30. Order Fairness Evaluation of DAG-based ledgers**

基于DAB的分类帐的订单公平性评估 cs.CR

17 double-column pages with 9 pages dedicated to references and  appendices, 22 figures, 13 of which are in the appendices

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2502.17270v2) [paper-pdf](http://arxiv.org/pdf/2502.17270v2)

**Authors**: Erwan Mahe, Sara Tucci-Piergiovanni

**Abstract**: Order fairness in distributed ledgers refers to properties that relate the order in which transactions are sent or received to the order in which they are eventually finalized, i.e., totally ordered. The study of such properties is relatively new and has been especially stimulated by the rise of Maximal Extractable Value (MEV) attacks in blockchain environments. Indeed, in many classical blockchain protocols, leaders are responsible for selecting the transactions to be included in blocks, which creates a clear vulnerability and opportunity for transaction order manipulation.   Unlike blockchains, DAG-based ledgers allow participants in the network to independently propose blocks, which are then arranged as vertices of a directed acyclic graph. Interestingly, leaders in DAG-based ledgers are elected only after the fact, once transactions are already part of the graph, to determine their total order. In other words, transactions are not chosen by single leaders; instead, they are collectively validated by the nodes, and leaders are only elected to establish an ordering. This approach intuitively reduces the risk of transaction manipulation and enhances fairness.   In this paper, we aim to quantify the capability of DAG-based ledgers to achieve order fairness. To this end, we define new variants of order fairness adapted to DAG-based ledgers and evaluate the impact of an adversary capable of compromising a limited number of nodes (below the one-third threshold) to reorder transactions. We analyze how often our order fairness properties are violated under different network conditions and parameterizations of the DAG algorithm, depending on the adversary's power.   Our study shows that DAG-based ledgers are still vulnerable to reordering attacks, as an adversary can coordinate a minority of Byzantine nodes to manipulate the DAG's structure.

摘要: 分布式分类账中的顺序公平性是指将发送或接收交易的顺序与最终确定的顺序(即完全有序)联系起来的属性。对这类属性的研究相对较新，尤其是区块链环境中最大可提取价值(MEV)攻击的兴起。事实上，在许多经典的区块链协议中，领导者负责选择要包含在区块中的交易，这为交易顺序操纵创造了明显的漏洞和机会。与区块链不同，基于DAG的分类账允许网络中的参与者独立提出区块，然后将这些区块排列为有向无环图的顶点。有趣的是，只有在交易已经成为图表的一部分后，才会在基于DAG的分类账中选出领导人，以确定其总顺序。换句话说，事务不是由单个领导者选择的；相反，它们由节点集体验证，而领导者只被选举来建立顺序。这种方法直观地降低了交易操纵的风险，提高了公平性。在本文中，我们的目标是量化基于DAG的分类帐实现顺序公平的能力。为此，我们定义了适用于基于DAG的分类账的顺序公平性的新变体，并评估了攻击者能够危害有限数量的节点(低于三分之一的阈值)来重新排序事务的影响。我们分析了在不同的网络条件和DAG算法的参数设置下，我们的顺序公平性被违反的频率，这取决于对手的力量。我们的研究表明，基于DAG的分类账仍然容易受到重新排序攻击，因为对手可以协调少数拜占庭节点来操纵DAG的结构。



## **31. Enhancing Resiliency of Sketch-based Security via LSB Sharing-based Dynamic Late Merging**

通过基于TSB共享的动态后期合并增强基于草图的安全性 cs.CR

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11777v1) [paper-pdf](http://arxiv.org/pdf/2503.11777v1)

**Authors**: Seungsam Yang, Seyed Mohammad Mehdi Mirnajafizadeh, Sian Kim, Rhongho Jang, DaeHun Nyang

**Abstract**: With the exponentially growing Internet traffic, sketch data structure with a probabilistic algorithm has been expected to be an alternative solution for non-compromised (non-selective) security monitoring. While facilitating counting within a confined memory space, the sketch's memory efficiency and accuracy were further pushed to their limit through finer-grained and dynamic control of constrained memory space to adapt to the data stream's inherent skewness (i.e., Zipf distribution), namely small counters with extensions. In this paper, we unveil a vulnerable factor of the small counter design by introducing a new sketch-oriented attack, which threatens a stream of state-of-the-art sketches and their security applications. With the root cause analyses, we propose Siamese Counter with enhanced adversarial resiliency and verified feasibility with extensive experimental and theoretical analyses. Under a sketch pollution attack, Siamese Counter delivers 47% accurate results than a state-of-the-art scheme, and demonstrates up to 82% more accurate estimation under normal measurement scenarios.

摘要: 随着互联网流量的指数级增长，使用概率算法的草图数据结构有望成为非妥协(非选择性)安全监控的替代方案。在便于在有限的内存空间内计数的同时，通过对受限的内存空间进行更细粒度的动态控制来适应数据流固有的偏斜性(即Zipf分布)，即带有扩展的小计数器，进一步将草图的内存效率和精度推到了极限。在本文中，我们通过引入一种新的面向草图的攻击来揭示小计数器设计的一个易受攻击的因素，该攻击威胁到了一系列最新的草图及其安全应用。通过对根本原因的分析，我们提出了增强对抗能力的暹罗对抗，并通过大量的实验和理论分析验证了其可行性。在素描污染攻击下，暹罗计数器比最先进的方案提供47%的准确结果，在正常测量场景下的精确度高达82%。



## **32. Making Every Step Effective: Jailbreaking Large Vision-Language Models Through Hierarchical KV Equalization**

让每一步都有效：通过分层KN均衡化破解大型视觉语言模型 cs.CV

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11750v1) [paper-pdf](http://arxiv.org/pdf/2503.11750v1)

**Authors**: Shuyang Hao, Yiwei Wang, Bryan Hooi, Jun Liu, Muhao Chen, Zi Huang, Yujun Cai

**Abstract**: In the realm of large vision-language models (LVLMs), adversarial jailbreak attacks serve as a red-teaming approach to identify safety vulnerabilities of these models and their associated defense mechanisms. However, we identify a critical limitation: not every adversarial optimization step leads to a positive outcome, and indiscriminately accepting optimization results at each step may reduce the overall attack success rate. To address this challenge, we introduce HKVE (Hierarchical Key-Value Equalization), an innovative jailbreaking framework that selectively accepts gradient optimization results based on the distribution of attention scores across different layers, ensuring that every optimization step positively contributes to the attack. Extensive experiments demonstrate HKVE's significant effectiveness, achieving attack success rates of 75.08% on MiniGPT4, 85.84% on LLaVA and 81.00% on Qwen-VL, substantially outperforming existing methods by margins of 20.43\%, 21.01\% and 26.43\% respectively. Furthermore, making every step effective not only leads to an increase in attack success rate but also allows for a reduction in the number of iterations, thereby lowering computational costs. Warning: This paper contains potentially harmful example data.

摘要: 在大型视觉语言模型领域，对抗性越狱攻击是一种识别这些模型及其相关防御机制的安全漏洞的红团队方法。然而，我们发现了一个关键的限制：并不是每个对抗性的优化步骤都会带来积极的结果，并且不分青红皂白地接受每个步骤的优化结果可能会降低总体攻击成功率。为了应对这一挑战，我们引入了HKVE(分层键值均衡)，这是一个创新的越狱框架，它根据注意力分数在不同层的分布来选择性地接受梯度优化结果，确保每一个优化步骤都对攻击有积极的贡献。大量实验表明，HKVE具有显著的攻击效果，在MiniGPT4、LLaVA和Qwen-VL上的攻击成功率分别达到75.08%、85.84%和81.00%，分别比现有方法高出20.43、21.01和26.43%。此外，使每一步都有效，不仅可以提高攻击成功率，还可以减少迭代次数，从而降低计算成本。警告：本文包含可能有害的示例数据。



## **33. Are Deep Speech Denoising Models Robust to Adversarial Noise?**

深度语音去噪模型对对抗噪音是否稳健？ cs.SD

13 pages, 5 figures

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11627v1) [paper-pdf](http://arxiv.org/pdf/2503.11627v1)

**Authors**: Will Schwarzer, Philip S. Thomas, Andrea Fanelli, Xiaoyu Liu

**Abstract**: Deep noise suppression (DNS) models enjoy widespread use throughout a variety of high-stakes speech applications. However, in this paper, we show that four recent DNS models can each be reduced to outputting unintelligible gibberish through the addition of imperceptible adversarial noise. Furthermore, our results show the near-term plausibility of targeted attacks, which could induce models to output arbitrary utterances, and over-the-air attacks. While the success of these attacks varies by model and setting, and attacks appear to be strongest when model-specific (i.e., white-box and non-transferable), our results highlight a pressing need for practical countermeasures in DNS systems.

摘要: 深度噪音抑制（DNS）模型在各种高风险语音应用中广泛使用。然而，在本文中，我们表明，最近的四个DNS模型都可以通过添加难以感知的对抗性噪音而简化为输出难以理解的胡言乱语。此外，我们的结果还表明了有针对性的攻击的近期可能性，这可能会导致模型输出任意言论，以及空中攻击。虽然这些攻击的成功因模型和设置而异，但攻击在特定于模型时似乎最强（即，白盒和不可转让），我们的结果凸显了DNS系统中对实用对策的迫切需求。



## **34. Tit-for-Tat: Safeguarding Large Vision-Language Models Against Jailbreak Attacks via Adversarial Defense**

针锋相对：通过对抗性防御保护大型视觉语言模型免受越狱攻击 cs.CR

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11619v1) [paper-pdf](http://arxiv.org/pdf/2503.11619v1)

**Authors**: Shuyang Hao, Yiwei Wang, Bryan Hooi, Ming-Hsuan Yang, Jun Liu, Chengcheng Tang, Zi Huang, Yujun Cai

**Abstract**: Deploying large vision-language models (LVLMs) introduces a unique vulnerability: susceptibility to malicious attacks via visual inputs. However, existing defense methods suffer from two key limitations: (1) They solely focus on textual defenses, fail to directly address threats in the visual domain where attacks originate, and (2) the additional processing steps often incur significant computational overhead or compromise model performance on benign tasks. Building on these insights, we propose ESIII (Embedding Security Instructions Into Images), a novel methodology for transforming the visual space from a source of vulnerability into an active defense mechanism. Initially, we embed security instructions into defensive images through gradient-based optimization, obtaining security instructions in the visual dimension. Subsequently, we integrate security instructions from visual and textual dimensions with the input query. The collaboration between security instructions from different dimensions ensures comprehensive security protection. Extensive experiments demonstrate that our approach effectively fortifies the robustness of LVLMs against such attacks while preserving their performance on standard benign tasks and incurring an imperceptible increase in time costs.

摘要: 部署大型视觉语言模型(LVLM)引入了一个独特的漏洞：通过视觉输入易受恶意攻击。然而，现有的防御方法有两个关键的局限性：(1)它们只关注文本防御，不能直接应对发起攻击的视域中的威胁；(2)额外的处理步骤往往会导致巨大的计算开销或对良性任务的模型性能造成影响。基于这些见解，我们提出了ESIII(Embedding Security Instructions To Images)，这是一种将视觉空间从易受攻击源转变为主动防御机制的新方法。首先，通过基于梯度的优化将安全指令嵌入到防御图像中，得到视觉维度的安全指令。随后，我们将来自视觉和文本维度的安全指令与输入查询集成在一起。来自不同维度的安全指令之间的协作确保了全面的安全防护。大量的实验表明，我们的方法有效地增强了LVLM对此类攻击的健壮性，同时保持了它们在标准良性任务上的性能，并导致了不可察觉的时间开销的增加。



## **35. Feasibility of Randomized Detector Tuning for Attack Impact Mitigation**

随机检测器调整以缓解攻击影响的可行性 eess.SY

8 pages, 4 figures, submitted to CDC 2025

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11417v1) [paper-pdf](http://arxiv.org/pdf/2503.11417v1)

**Authors**: Sribalaji C. Anand, Kamil Hassan, Henrik Sandberg

**Abstract**: This paper considers the problem of detector tuning against false data injection attacks. In particular, we consider an adversary injecting false sensor data to maximize the state deviation of the plant, referred to as impact, whilst being stealthy. To minimize the impact of stealthy attacks, inspired by moving target defense, the operator randomly switches the detector thresholds. In this paper, we theoretically derive the sufficient (and in some cases necessary) conditions under which the impact of stealthy attacks can be made smaller with randomized switching of detector thresholds compared to static thresholds. We establish the conditions for the stateless ($\chi^2$) and the stateful (CUSUM) detectors. The results are illustrated through numerical examples.

摘要: 本文考虑了检测器针对错误数据注入攻击进行调整的问题。特别是，我们考虑对手注入错误的传感器数据，以最大化工厂的状态偏差（称为影响），同时保持隐蔽。为了最大限度地减少隐形攻击的影响，受移动目标防御的启发，操作员随机切换检测器阈值。在本文中，我们从理论上推导出了充分（在某些情况下是必要的）条件，在此条件下，与静态阈值相比，通过随机切换检测器阈值来降低隐形攻击的影响。我们为无状态（$\chi ' 2 $）和有状态（CLARUM）检测器建立条件。通过算例说明了结果。



## **36. Hiding Local Manipulations on SAR Images: a Counter-Forensic Attack**

隐藏SAR图像上的局部操纵：反法医攻击 cs.CV

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2407.07041v2) [paper-pdf](http://arxiv.org/pdf/2407.07041v2)

**Authors**: Sara Mandelli, Edoardo Daniele Cannas, Paolo Bestagini, Stefano Tebaldini, Stefano Tubaro

**Abstract**: The vast accessibility of Synthetic Aperture Radar (SAR) images through online portals has propelled the research across various fields. This widespread use and easy availability have unfortunately made SAR data susceptible to malicious alterations, such as local editing applied to the images for inserting or covering the presence of sensitive targets. Vulnerability is further emphasized by the fact that most SAR products, despite their original complex nature, are often released as amplitude-only information, allowing even inexperienced attackers to edit and easily alter the pixel content. To contrast malicious manipulations, in the last years the forensic community has begun to dig into the SAR manipulation issue, proposing detectors that effectively localize the tampering traces in amplitude images. Nonetheless, in this paper we demonstrate that an expert practitioner can exploit the complex nature of SAR data to obscure any signs of manipulation within a locally altered amplitude image. We refer to this approach as a counter-forensic attack. To achieve the concealment of manipulation traces, the attacker can simulate a re-acquisition of the manipulated scene by the SAR system that initially generated the pristine image. In doing so, the attacker can obscure any evidence of manipulation, making it appear as if the image was legitimately produced by the system. This attack has unique features that make it both highly generalizable and relatively easy to apply. First, it is a black-box attack, meaning it is not designed to deceive a specific forensic detector. Furthermore, it does not require a training phase and is not based on adversarial operations. We assess the effectiveness of the proposed counter-forensic approach across diverse scenarios, examining various manipulation operations.

摘要: 通过在线门户网站可以广泛访问合成孔径雷达(SAR)图像，推动了这项研究在各个领域的开展。不幸的是，这种广泛的使用和易得性使得SAR数据容易被恶意更改，例如对图像进行本地编辑，以插入或覆盖敏感目标的存在。尽管大多数合成孔径雷达产品最初的性质很复杂，但往往以仅幅度信息的形式发布，即使是经验不足的攻击者也可以编辑和轻松更改像素内容，这进一步加剧了漏洞。为了对比恶意篡改，在过去的几年里，法医界已经开始深入研究SAR篡改问题，提出了有效定位幅度图像中篡改痕迹的检测器。然而，在这篇文章中，我们证明了专家从业者可以利用SAR数据的复杂性来模糊局部改变的幅度图像中的任何操纵迹象。我们将这种做法称为反法医攻击。为了实现操纵痕迹的隐藏，攻击者可以模拟最初生成原始图像的SAR系统重新获取操纵场景。在这样做的过程中，攻击者可以掩盖任何操纵的证据，使其看起来像是系统合法生成的图像。这种攻击具有独特的功能，使其既具有高度的通用性，又相对容易应用。首先，这是一个黑匣子攻击，这意味着它不是为了欺骗特定的法医检测器而设计的。此外，它不需要训练阶段，也不以对抗行动为基础。我们评估了所提出的反取证方法在不同情况下的有效性，检查了各种操纵操作。



## **37. On the Impact of Uncertainty and Calibration on Likelihood-Ratio Membership Inference Attacks**

不确定性和校准对可能性比隶属推理攻击的影响 cs.IT

16 pages, 23 figures

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2402.10686v3) [paper-pdf](http://arxiv.org/pdf/2402.10686v3)

**Authors**: Meiyi Zhu, Caili Guo, Chunyan Feng, Osvaldo Simeone

**Abstract**: In a membership inference attack (MIA), an attacker exploits the overconfidence exhibited by typical machine learning models to determine whether a specific data point was used to train a target model. In this paper, we analyze the performance of the likelihood ratio attack (LiRA) within an information-theoretical framework that allows the investigation of the impact of the aleatoric uncertainty in the true data generation process, of the epistemic uncertainty caused by a limited training data set, and of the calibration level of the target model. We compare three different settings, in which the attacker receives decreasingly informative feedback from the target model: confidence vector (CV) disclosure, in which the output probability vector is released; true label confidence (TLC) disclosure, in which only the probability assigned to the true label is made available by the model; and decision set (DS) disclosure, in which an adaptive prediction set is produced as in conformal prediction. We derive bounds on the advantage of an MIA adversary with the aim of offering insights into the impact of uncertainty and calibration on the effectiveness of MIAs. Simulation results demonstrate that the derived analytical bounds predict well the effectiveness of MIAs.

摘要: 在成员关系推理攻击(MIA)中，攻击者利用典型机器学习模型表现出的过度自信来确定是否使用特定数据点来训练目标模型。在本文中，我们在信息论框架内分析了似然比攻击(LIRA)的性能，该框架允许调查真实数据生成过程中任意不确定性的影响、有限训练数据集引起的认知不确定性的影响以及目标模型的校准水平。我们比较了三种不同的设置，其中攻击者从目标模型接收信息递减的反馈：置信度向量(CV)披露，其中输出概率向量被释放；真实标签置信度(TLC)披露，其中模型仅提供分配给真实标签的概率；以及决策集(DS)披露，其中产生与保形预测相同的自适应预测集。我们得出了MIA对手的优势界限，目的是对不确定性和校准对MIA有效性的影响提供见解。仿真结果表明，推导出的解析界很好地预测了MIA的有效性。



## **38. SmartShards: Churn-Tolerant Continuously Available Distributed Ledger**

SmartShards：耐流失的持续可用分布式分类帐 cs.DC

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2503.11077v1) [paper-pdf](http://arxiv.org/pdf/2503.11077v1)

**Authors**: Joseph Oglio, Mikhail Nesterenko, Gokarna Sharma

**Abstract**: We present SmartShards: a new sharding algorithm for improving Byzantine tolerance and churn resistance in blockchains. Our algorithm places a peer in multiple shards to create an overlap. This simplifies cross-shard communication and shard membership management. We describe SmartShards, prove it correct and evaluate its performance.   We propose several SmartShards extensions: defense against a slowly adaptive adversary, combining transactions into blocks, fortification against the join/leave attack.

摘要: 我们介绍了SmartShards：一种新的分片算法，用于提高区块链中的拜占庭容忍度和抗流失性。我们的算法将对等点放置在多个碎片中以创建重叠。这简化了跨碎片通信和碎片成员管理。我们描述SmartShards，证明其正确性并评估其性能。   我们提出了几个SmartShards扩展：防御适应缓慢的对手、将事务组合到块中、防御加入/离开攻击。



## **39. AnywhereDoor: Multi-Target Backdoor Attacks on Object Detection**

AnywhereDoor：对象检测的多目标后门攻击 cs.CR

15 pages; This update was mistakenly uploaded as a new manuscript on  arXiv (2503.06529). The wrong submission has now been withdrawn, and we  replace the old one here

**SubmitDate**: 2025-03-14    [abs](http://arxiv.org/abs/2411.14243v2) [paper-pdf](http://arxiv.org/pdf/2411.14243v2)

**Authors**: Jialin Lu, Junjie Shan, Ziqi Zhao, Ka-Ho Chow

**Abstract**: As object detection becomes integral to many safety-critical applications, understanding its vulnerabilities is essential. Backdoor attacks, in particular, pose a serious threat by implanting hidden triggers in victim models, which adversaries can later exploit to induce malicious behaviors during inference. However, current understanding is limited to single-target attacks, where adversaries must define a fixed malicious behavior (target) before training, making inference-time adaptability impossible. Given the large output space of object detection (including object existence prediction, bounding box estimation, and classification), the feasibility of flexible, inference-time model control remains unexplored. This paper introduces AnywhereDoor, a multi-target backdoor attack for object detection. Once implanted, AnywhereDoor allows adversaries to make objects disappear, fabricate new ones, or mislabel them, either across all object classes or specific ones, offering an unprecedented degree of control. This flexibility is enabled by three key innovations: (i) objective disentanglement to scale the number of supported targets; (ii) trigger mosaicking to ensure robustness even against region-based detectors; and (iii) strategic batching to address object-level data imbalances that hinder manipulation. Extensive experiments demonstrate that AnywhereDoor grants attackers a high degree of control, improving attack success rates by 26% compared to adaptations of existing methods for such flexible control.

摘要: 随着对象检测成为许多安全关键型应用程序不可或缺的一部分，了解其漏洞至关重要。尤其是后门攻击，通过在受害者模型中植入隐藏的触发器，构成了严重的威胁，攻击者稍后可以利用这些触发器在推理过程中诱导恶意行为。然而，目前的理解仅限于单目标攻击，即攻击者必须在训练前定义一个固定的恶意行为(目标)，这使得推理时间适应性变得不可能。考虑到目标检测(包括目标存在预测、包围盒估计和分类)的大输出空间，灵活的推理时间模型控制的可行性仍未被探索。本文介绍了Anywhere Door，一种用于目标检测的多目标后门攻击。一旦被植入，Anywhere Door允许对手在所有对象类或特定对象类中让对象消失、捏造新对象或错误标记对象，提供前所未有的控制程度。这种灵活性是由三项关键创新实现的：(I)客观解缠，以扩大受支持目标的数量；(Ii)触发马赛克，以确保即使针对基于区域的探测器也具有稳健性；以及(Iii)战略批处理，以解决阻碍操纵的对象级数据失衡。广泛的实验表明，Anywhere Door给予攻击者高度的控制，与采用现有方法进行这种灵活的控制相比，攻击成功率提高了26%。



## **40. Efficient Reachability Analysis for Convolutional Neural Networks Using Hybrid Zonotopes**

使用混合区位的卷积神经网络有效可达性分析 math.OC

Accepted by 2025 American Control Conference (ACC). 8 pages, 1 figure

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10840v1) [paper-pdf](http://arxiv.org/pdf/2503.10840v1)

**Authors**: Yuhao Zhang, Xiangru Xu

**Abstract**: Feedforward neural networks are widely used in autonomous systems, particularly for control and perception tasks within the system loop. However, their vulnerability to adversarial attacks necessitates formal verification before deployment in safety-critical applications. Existing set propagation-based reachability analysis methods for feedforward neural networks often struggle to achieve both scalability and accuracy. This work presents a novel set-based approach for computing the reachable sets of convolutional neural networks. The proposed method leverages a hybrid zonotope representation and an efficient neural network reduction technique, providing a flexible trade-off between computational complexity and approximation accuracy. Numerical examples are presented to demonstrate the effectiveness of the proposed approach.

摘要: 前向神经网络广泛用于自治系统中，特别是用于系统循环内的控制和感知任务。然而，它们容易受到对抗攻击，因此在部署到安全关键应用程序之前需要进行正式验证。现有的用于前向神经网络的基于集合传播的可达性分析方法常常难以实现可扩展性和准确性。这项工作提出了一种新颖的基于集合的方法来计算卷积神经网络的可达集合。所提出的方法利用混合区位表示和高效的神经网络约简技术，在计算复杂性和逼近准确性之间提供了灵活的权衡。通过数值算例来证明所提出方法的有效性。



## **41. Attacking Multimodal OS Agents with Malicious Image Patches**

使用恶意图像补丁攻击多模式操作系统代理 cs.CR

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10809v1) [paper-pdf](http://arxiv.org/pdf/2503.10809v1)

**Authors**: Lukas Aichberger, Alasdair Paren, Yarin Gal, Philip Torr, Adel Bibi

**Abstract**: Recent advances in operating system (OS) agents enable vision-language models to interact directly with the graphical user interface of an OS. These multimodal OS agents autonomously perform computer-based tasks in response to a single prompt via application programming interfaces (APIs). Such APIs typically support low-level operations, including mouse clicks, keyboard inputs, and screenshot captures. We introduce a novel attack vector: malicious image patches (MIPs) that have been adversarially perturbed so that, when captured in a screenshot, they cause an OS agent to perform harmful actions by exploiting specific APIs. For instance, MIPs embedded in desktop backgrounds or shared on social media can redirect an agent to a malicious website, enabling further exploitation. These MIPs generalise across different user requests and screen layouts, and remain effective for multiple OS agents. The existence of such attacks highlights critical security vulnerabilities in OS agents, which should be carefully addressed before their widespread adoption.

摘要: 操作系统(OS)代理的最新进展使视觉语言模型能够直接与OS的图形用户界面交互。这些多模式OS代理通过应用编程接口(API)响应单个提示自主地执行基于计算机的任务。这类API通常支持低级操作，包括鼠标点击、键盘输入和截图。我们引入了一种新的攻击载体：恶意图像补丁(MIP)，这些补丁已经被恶意干扰，因此，当它们被捕获到屏幕截图时，它们会导致操作系统代理通过利用特定API来执行有害操作。例如，嵌入桌面背景或在社交媒体上共享的MIP可以将代理重定向至恶意网站，从而实现进一步攻击。这些MIP概括了不同的用户请求和屏幕布局，并对多个操作系统代理保持有效。此类攻击的存在突显了操作系统代理中的严重安全漏洞，在广泛采用之前应该仔细解决这些漏洞。



## **42. Byzantine-Resilient Federated Learning via Distributed Optimization**

通过分布式优化实现拜占庭弹性联邦学习 cs.LG

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10792v1) [paper-pdf](http://arxiv.org/pdf/2503.10792v1)

**Authors**: Yufei Xia, Wenrui Yu, Qiongxiu Li

**Abstract**: Byzantine attacks present a critical challenge to Federated Learning (FL), where malicious participants can disrupt the training process, degrade model accuracy, and compromise system reliability. Traditional FL frameworks typically rely on aggregation-based protocols for model updates, leaving them vulnerable to sophisticated adversarial strategies. In this paper, we demonstrate that distributed optimization offers a principled and robust alternative to aggregation-centric methods. Specifically, we show that the Primal-Dual Method of Multipliers (PDMM) inherently mitigates Byzantine impacts by leveraging its fault-tolerant consensus mechanism. Through extensive experiments on three datasets (MNIST, FashionMNIST, and Olivetti), under various attack scenarios including bit-flipping and Gaussian noise injection, we validate the superior resilience of distributed optimization protocols. Compared to traditional aggregation-centric approaches, PDMM achieves higher model utility, faster convergence, and improved stability. Our results highlight the effectiveness of distributed optimization in defending against Byzantine threats, paving the way for more secure and resilient federated learning systems.

摘要: 拜占庭攻击对联邦学习(FL)提出了严峻的挑战，恶意参与者可以扰乱训练过程，降低模型精度，并危及系统可靠性。传统的FL框架通常依赖于基于聚合的协议来进行模型更新，这使得它们容易受到复杂的对手策略的影响。在本文中，我们论证了分布式优化为以聚合为中心的方法提供了一种原则性和健壮性的替代方法。具体地说，我们证明了原始-对偶乘子方法(PDMM)通过利用其容错共识机制内在地缓解了拜占庭的影响。通过在三个数据集(MNIST、FashionMNIST和Olivetti)上的大量实验，我们验证了分布式优化协议在比特翻转和高斯噪声注入等不同攻击场景下的优越抗攻击能力。与传统的以聚合为中心的方法相比，PDMM实现了更高的模型效用、更快的收敛速度和更好的稳定性。我们的结果突出了分布式优化在防御拜占庭威胁方面的有效性，为更安全和更具弹性的联合学习系统铺平了道路。



## **43. A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1**

令人沮丧的简单但高效的攻击基线：针对GPT-4.5/4 o/o 1的强黑匣子模型的成功率超过90% cs.CV

Code at: https://github.com/VILA-Lab/M-Attack

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10635v1) [paper-pdf](http://arxiv.org/pdf/2503.10635v1)

**Authors**: Zhaoyi Li, Xiaohan Zhao, Dong-Dong Wu, Jiacheng Cui, Zhiqiang Shen

**Abstract**: Despite promising performance on open-source large vision-language models (LVLMs), transfer-based targeted attacks often fail against black-box commercial LVLMs. Analyzing failed adversarial perturbations reveals that the learned perturbations typically originate from a uniform distribution and lack clear semantic details, resulting in unintended responses. This critical absence of semantic information leads commercial LVLMs to either ignore the perturbation entirely or misinterpret its embedded semantics, thereby causing the attack to fail. To overcome these issues, we notice that identifying core semantic objects is a key objective for models trained with various datasets and methodologies. This insight motivates our approach that refines semantic clarity by encoding explicit semantic details within local regions, thus ensuring interoperability and capturing finer-grained features, and by concentrating modifications on semantically rich areas rather than applying them uniformly. To achieve this, we propose a simple yet highly effective solution: at each optimization step, the adversarial image is cropped randomly by a controlled aspect ratio and scale, resized, and then aligned with the target image in the embedding space. Experimental results confirm our hypothesis. Our adversarial examples crafted with local-aggregated perturbations focused on crucial regions exhibit surprisingly good transferability to commercial LVLMs, including GPT-4.5, GPT-4o, Gemini-2.0-flash, Claude-3.5-sonnet, Claude-3.7-sonnet, and even reasoning models like o1, Claude-3.7-thinking and Gemini-2.0-flash-thinking. Our approach achieves success rates exceeding 90% on GPT-4.5, 4o, and o1, significantly outperforming all prior state-of-the-art attack methods. Our optimized adversarial examples under different configurations and training code are available at https://github.com/VILA-Lab/M-Attack.

摘要: 尽管在开源的大型视觉语言模型(LVLM)上性能很好，但基于传输的定向攻击对黑盒商业LVLM往往失败。分析失败的对抗性扰动发现，学习的扰动通常源于均匀分布，缺乏明确的语义细节，导致意外响应。这种严重的语义信息缺失导致商业LVLM要么完全忽略该扰动，要么曲解其嵌入的语义，从而导致攻击失败。为了克服这些问题，我们注意到，识别核心语义对象是用各种数据集和方法训练的模型的关键目标。这种洞察力激发了我们的方法，通过在局部区域内编码显式的语义细节来细化语义清晰度，从而确保互操作性和捕获更细粒度的特征，并通过将修改集中在语义丰富的区域而不是统一地应用它们。为了实现这一点，我们提出了一种简单而高效的解决方案：在每个优化步骤中，通过控制纵横比和比例来随机裁剪敌对图像，调整大小，然后在嵌入空间中与目标图像对齐。实验结果证实了我们的假设。我们的对抗性例子使用聚焦于关键区域的局部聚集扰动来制作，表现出出奇地好的可转换性，可以用于商业LVLM，包括GPT-4.5、GPT-40、Gemini-2.0-Flash、Claude-3.5-十四行诗、Claude-3.7-十四行诗，甚至像o1、Claude-3.7-Think和Gemini-2.0-Flash-Think这样的推理模型。我们的方法在GPT-4.5、40o和o1上的成功率超过90%，大大超过了之前所有最先进的攻击方法。我们在不同配置和训练代码下的优化对抗性示例可在https://github.com/VILA-Lab/M-Attack.获得



## **44. Hierarchical Self-Supervised Adversarial Training for Robust Vision Models in Histopathology**

组织病理学中鲁棒视觉模型的分层自监督对抗训练 cs.CV

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10629v1) [paper-pdf](http://arxiv.org/pdf/2503.10629v1)

**Authors**: Hashmat Shadab Malik, Shahina Kunhimon, Muzammal Naseer, Fahad Shahbaz Khan, Salman Khan

**Abstract**: Adversarial attacks pose significant challenges for vision models in critical fields like healthcare, where reliability is essential. Although adversarial training has been well studied in natural images, its application to biomedical and microscopy data remains limited. Existing self-supervised adversarial training methods overlook the hierarchical structure of histopathology images, where patient-slide-patch relationships provide valuable discriminative signals. To address this, we propose Hierarchical Self-Supervised Adversarial Training (HSAT), which exploits these properties to craft adversarial examples using multi-level contrastive learning and integrate it into adversarial training for enhanced robustness. We evaluate HSAT on multiclass histopathology dataset OpenSRH and the results show that HSAT outperforms existing methods from both biomedical and natural image domains. HSAT enhances robustness, achieving an average gain of 54.31% in the white-box setting and reducing performance drops to 3-4% in the black-box setting, compared to 25-30% for the baseline. These results set a new benchmark for adversarial training in this domain, paving the way for more robust models. Our Code for training and evaluation is available at https://github.com/HashmatShadab/HSAT.

摘要: 对抗性攻击对医疗保健等关键领域的视觉模型构成了重大挑战，在这些领域，可靠性至关重要。虽然对抗性训练已经在自然图像中得到了很好的研究，但它在生物医学和显微数据中的应用仍然有限。现有的自我监督的对抗性训练方法忽略了组织病理学图像的层次结构，其中患者-切片-补丁关系提供了有价值的区分信号。为了解决这一问题，我们提出了分层自监督对抗训练(HSAT)，它利用这些性质利用多级对比学习来构造对抗实例，并将其整合到对抗训练中以增强稳健性。我们在多类组织病理学数据集OpenSRH上对HSAT进行了评估，结果表明HSAT的性能优于生物医学和自然图像领域的现有方法。HSAT增强了健壮性，在白盒设置中实现了54.31%的平均增益，并将黑盒设置中的性能下降到了3-4%，而基准设置为25-30%。这些结果为这一领域的对抗性训练设定了新的基准，为更稳健的模型铺平了道路。我们的培训和评估准则可在https://github.com/HashmatShadab/HSAT.上获得



## **45. Towards Class-wise Robustness Analysis**

走向班级稳健性分析 cs.LG

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2411.19853v2) [paper-pdf](http://arxiv.org/pdf/2411.19853v2)

**Authors**: Tejaswini Medi, Julia Grabinski, Margret Keuper

**Abstract**: While being very successful in solving many downstream tasks, the application of deep neural networks is limited in real-life scenarios because of their susceptibility to domain shifts such as common corruptions, and adversarial attacks. The existence of adversarial examples and data corruption significantly reduces the performance of deep classification models. Researchers have made strides in developing robust neural architectures to bolster decisions of deep classifiers. However, most of these works rely on effective adversarial training methods, and predominantly focus on overall model robustness, disregarding class-wise differences in robustness, which are critical. Exploiting weakly robust classes is a potential avenue for attackers to fool the image recognition models. Therefore, this study investigates class-to-class biases across adversarially trained robust classification models to understand their latent space structures and analyze their strong and weak class-wise properties. We further assess the robustness of classes against common corruptions and adversarial attacks, recognizing that class vulnerability extends beyond the number of correct classifications for a specific class. We find that the number of false positives of classes as specific target classes significantly impacts their vulnerability to attacks. Through our analysis on the Class False Positive Score, we assess a fair evaluation of how susceptible each class is to misclassification.

摘要: 虽然深度神经网络在解决许多下游任务方面非常成功，但由于其对域转移的敏感性，如常见的腐败和敌对攻击，其在现实生活场景中的应用受到限制。对抗性例子和数据破坏的存在大大降低了深度分类模型的性能。研究人员在开发稳健的神经体系结构以支持深度分类器的决策方面取得了很大进展。然而，这些工作大多依赖于有效的对抗性训练方法，并且主要关注整体模型的稳健性，而忽略了类之间的稳健性差异，这是至关重要的。利用健壮性较弱的类是攻击者愚弄图像识别模型的潜在途径。因此，本研究通过研究反向训练的稳健分类模型的类对类偏差，以了解它们的潜在空间结构，并分析它们的强弱类性质。我们进一步评估了类对常见的腐败和敌意攻击的健壮性，认识到类的脆弱性超出了特定类的正确分类的数量。我们发现，作为特定目标类的类的误报数量显著影响其易受攻击的程度。通过我们对班级假阳性分数的分析，我们评估了每个班级对错误分类的易感性的公平评估。



## **46. TH-Bench: Evaluating Evading Attacks via Humanizing AI Text on Machine-Generated Text Detectors**

TH-Bench：通过机器生成文本检测器上人性化人工智能文本来评估逃避攻击 cs.CR

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.08708v2) [paper-pdf](http://arxiv.org/pdf/2503.08708v2)

**Authors**: Jingyi Zheng, Junfeng Wang, Zhen Sun, Wenhan Dong, Yule Liu, Xinlei He

**Abstract**: As Large Language Models (LLMs) advance, Machine-Generated Texts (MGTs) have become increasingly fluent, high-quality, and informative. Existing wide-range MGT detectors are designed to identify MGTs to prevent the spread of plagiarism and misinformation. However, adversaries attempt to humanize MGTs to evade detection (named evading attacks), which requires only minor modifications to bypass MGT detectors. Unfortunately, existing attacks generally lack a unified and comprehensive evaluation framework, as they are assessed using different experimental settings, model architectures, and datasets. To fill this gap, we introduce the Text-Humanization Benchmark (TH-Bench), the first comprehensive benchmark to evaluate evading attacks against MGT detectors. TH-Bench evaluates attacks across three key dimensions: evading effectiveness, text quality, and computational overhead. Our extensive experiments evaluate 6 state-of-the-art attacks against 13 MGT detectors across 6 datasets, spanning 19 domains and generated by 11 widely used LLMs. Our findings reveal that no single evading attack excels across all three dimensions. Through in-depth analysis, we highlight the strengths and limitations of different attacks. More importantly, we identify a trade-off among three dimensions and propose two optimization insights. Through preliminary experiments, we validate their correctness and effectiveness, offering potential directions for future research.

摘要: 随着大型语言模型(LLM)的发展，机器生成文本(MGTS)变得越来越流畅、高质量和信息丰富。现有的大范围MGT探测器旨在识别MGT，以防止抄袭和错误信息的传播。然而，攻击者试图使MGTS人性化以躲避检测(称为躲避攻击)，这只需要进行少量修改即可绕过MGT检测器。遗憾的是，现有的攻击通常缺乏统一和全面的评估框架，因为它们是使用不同的实验设置、模型架构和数据集进行评估的。为了填补这一空白，我们引入了文本人性化基准(TH-BENCH)，这是第一个评估针对MGT检测器的躲避攻击的综合基准。TH-BENCH从三个关键维度对攻击进行评估：规避效率、文本质量和计算开销。我们的广泛实验评估了6种针对6个数据集的13个MGT检测器的最先进攻击，这些攻击跨越19个域，由11个广泛使用的LLM生成。我们的发现表明，没有一次逃避攻击在所有三个维度上都表现出色。通过深入的分析，我们突出了不同攻击的优势和局限性。更重要的是，我们确定了三个维度之间的权衡，并提出了两个优化见解。通过初步实验，验证了它们的正确性和有效性，为以后的研究提供了潜在的方向。



## **47. I Can Tell Your Secrets: Inferring Privacy Attributes from Mini-app Interaction History in Super-apps**

我可以告诉你的秘密：从超级应用中的迷你应用交互历史记录推断隐私属性 cs.CR

Accepted by USENIX Security 2025

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10239v1) [paper-pdf](http://arxiv.org/pdf/2503.10239v1)

**Authors**: Yifeng Cai, Ziqi Zhang, Mengyu Yao, Junlin Liu, Xiaoke Zhao, Xinyi Fu, Ruoyu Li, Zhe Li, Xiangqun Chen, Yao Guo, Ding Li

**Abstract**: Super-apps have emerged as comprehensive platforms integrating various mini-apps to provide diverse services. While super-apps offer convenience and enriched functionality, they can introduce new privacy risks. This paper reveals a new privacy leakage source in super-apps: mini-app interaction history, including mini-app usage history (Mini-H) and operation history (Op-H). Mini-H refers to the history of mini-apps accessed by users, such as their frequency and categories. Op-H captures user interactions within mini-apps, including button clicks, bar drags, and image views. Super-apps can naturally collect these data without instrumentation due to the web-based feature of mini-apps. We identify these data types as novel and unexplored privacy risks through a literature review of 30 papers and an empirical analysis of 31 super-apps. We design a mini-app interaction history-oriented inference attack (THEFT), to exploit this new vulnerability. Using THEFT, the insider threats within the low-privilege business department of the super-app vendor acting as the adversary can achieve more than 95.5% accuracy in inferring privacy attributes of over 16.1% of users. THEFT only requires a small training dataset of 200 users from public breached databases on the Internet. We also engage with super-app vendors and a standards association to increase industry awareness and commitment to protect this data. Our contributions are significant in identifying overlooked privacy risks, demonstrating the effectiveness of a new attack, and influencing industry practices toward better privacy protection in the super-app ecosystem.

摘要: 超级应用已经成为整合各种小应用以提供多样化服务的综合平台。虽然超级应用程序提供了便利和丰富的功能，但它们也可能带来新的隐私风险。揭示了超级应用中一个新的隐私泄漏源：小应用交互历史，包括小应用使用历史(Mini-H)和操作历史(Op-H)。Mini-H指的是用户访问小应用的历史，比如它们的频率和类别。OP-H捕获迷你应用程序中的用户交互，包括按钮点击、栏拖动和图像查看。由于小应用程序的基于网络的功能，超级应用程序可以自然地收集这些数据，而不需要工具。我们通过对30篇论文的文献回顾和对31个超级应用程序的实证分析，将这些数据类型确定为新的和未探索的隐私风险。我们设计了一个面向交互历史的迷你APP推理攻击(盗窃)，来利用这个新的漏洞。使用盗窃，作为对手的超级应用供应商低权限业务部门的内部威胁可以达到95.5%以上的准确率，推断出16.1%以上的用户的隐私属性。盗窃只需要一个由200名用户组成的小型培训数据集，这些用户来自互联网上被入侵的公共数据库。我们还与超级应用程序供应商和标准协会接触，以提高行业意识和承诺保护这些数据。我们的贡献在识别被忽视的隐私风险、展示新攻击的有效性以及影响行业做法以在超级应用生态系统中更好地保护隐私方面具有重要意义。



## **48. Robustness Tokens: Towards Adversarial Robustness of Transformers**

稳健代币：走向变形金刚的对抗稳健 cs.LG

This paper has been accepted for publication at the European  Conference on Computer Vision (ECCV), 2024

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2503.10191v1) [paper-pdf](http://arxiv.org/pdf/2503.10191v1)

**Authors**: Brian Pulfer, Yury Belousov, Slava Voloshynovskiy

**Abstract**: Recently, large pre-trained foundation models have become widely adopted by machine learning practitioners for a multitude of tasks. Given that such models are publicly available, relying on their use as backbone models for downstream tasks might result in high vulnerability to adversarial attacks crafted with the same public model. In this work, we propose Robustness Tokens, a novel approach specific to the transformer architecture that fine-tunes a few additional private tokens with low computational requirements instead of tuning model parameters as done in traditional adversarial training. We show that Robustness Tokens make Vision Transformer models significantly more robust to white-box adversarial attacks while also retaining the original downstream performances.

摘要: 最近，大型预训练基础模型已被机器学习从业者广泛采用，用于多种任务。鉴于此类模型是公开的，依赖于它们作为下游任务的主干模型可能会导致对使用相同公共模型设计的对抗攻击的高度脆弱性。在这项工作中，我们提出了鲁棒性令牌，这是一种特定于Transformer架构的新颖方法，可以以低计算要求微调一些额外的私有令牌，而不是像传统对抗训练中那样调整模型参数。我们表明，鲁棒性令牌使Vision Transformer模型对白盒对抗攻击的鲁棒性明显更强，同时还保留了原始的下游性能。



## **49. Can't Slow me Down: Learning Robust and Hardware-Adaptive Object Detectors against Latency Attacks for Edge Devices**

无法让我慢下来：学习强大且硬件自适应的对象检测器来对抗边缘设备的延迟攻击 cs.CV

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2412.02171v2) [paper-pdf](http://arxiv.org/pdf/2412.02171v2)

**Authors**: Tianyi Wang, Zichen Wang, Cong Wang, Yuanchao Shu, Ruilong Deng, Peng Cheng, Jiming Chen

**Abstract**: Object detection is a fundamental enabler for many real-time downstream applications such as autonomous driving, augmented reality and supply chain management. However, the algorithmic backbone of neural networks is brittle to imperceptible perturbations in the system inputs, which were generally known as misclassifying attacks. By targeting the real-time processing capability, a new class of latency attacks are reported recently. They exploit new attack surfaces in object detectors by creating a computational bottleneck in the post-processing module, that leads to cascading failure and puts the real-time downstream tasks at risks. In this work, we take an initial attempt to defend against this attack via background-attentive adversarial training that is also cognizant of the underlying hardware capabilities. We first draw system-level connections between latency attack and hardware capacity across heterogeneous GPU devices. Based on the particular adversarial behaviors, we utilize objectness loss as a proxy and build background attention into the adversarial training pipeline, and achieve a reasonable balance between clean and robust accuracy. The extensive experiments demonstrate the defense effectiveness of restoring real-time processing capability from $13$ FPS to $43$ FPS on Jetson Orin NX, with a better trade-off between the clean and robust accuracy.

摘要: 目标检测是自动驾驶、增强现实和供应链管理等许多实时下游应用的基本使能。然而，神经网络的算法主干对系统输入中的不可察觉的扰动是脆弱的，这种扰动通常被称为误分类攻击。以实时处理能力为目标，最近报道了一类新的延迟攻击。它们通过在后处理模块中创建计算瓶颈来利用对象检测器中的新攻击面，从而导致级联故障并使实时下游任务处于危险之中。在这项工作中，我们初步尝试通过背景专注的对手训练来防御这种攻击，该训练也认识到潜在的硬件能力。我们首先在系统级将延迟攻击与跨不同类型的GPU设备的硬件容量联系起来。基于特定的对抗性行为，我们利用客观性损失作为代理，在对抗性训练流水线中加入背景注意，在干净和健壮的准确率之间取得合理的平衡。广泛的实验证明了在Jetson Orin NX上将实时处理能力从13美元FPS恢复到43美元FPS的防御效果，并在干净和健壮的准确性之间进行了更好的权衡。



## **50. Prompt-Driven Contrastive Learning for Transferable Adversarial Attacks**

可转移对抗攻击的预算驱动对比学习 cs.CV

Accepted to ECCV 2024 (Oral), Project Page:  https://PDCL-Attack.github.io

**SubmitDate**: 2025-03-13    [abs](http://arxiv.org/abs/2407.20657v2) [paper-pdf](http://arxiv.org/pdf/2407.20657v2)

**Authors**: Hunmin Yang, Jongoh Jeong, Kuk-Jin Yoon

**Abstract**: Recent vision-language foundation models, such as CLIP, have demonstrated superior capabilities in learning representations that can be transferable across diverse range of downstream tasks and domains. With the emergence of such powerful models, it has become crucial to effectively leverage their capabilities in tackling challenging vision tasks. On the other hand, only a few works have focused on devising adversarial examples that transfer well to both unknown domains and model architectures. In this paper, we propose a novel transfer attack method called PDCL-Attack, which leverages the CLIP model to enhance the transferability of adversarial perturbations generated by a generative model-based attack framework. Specifically, we formulate an effective prompt-driven feature guidance by harnessing the semantic representation power of text, particularly from the ground-truth class labels of input images. To the best of our knowledge, we are the first to introduce prompt learning to enhance the transferable generative attacks. Extensive experiments conducted across various cross-domain and cross-model settings empirically validate our approach, demonstrating its superiority over state-of-the-art methods.

摘要: 最近的视觉语言基础模型，如CLIP，在学习表征方面表现出了优越的能力，这些表征可以跨不同范围的下游任务和领域转移。随着如此强大的模型的出现，有效地利用它们的能力来处理具有挑战性的愿景任务变得至关重要。另一方面，只有少数工作专注于设计能够很好地移植到未知领域和模型体系结构的对抗性例子。本文提出了一种新的转移攻击方法PDCL-Attack，该方法利用CLIP模型来增强基于产生式模型的攻击框架产生的敌意扰动的可转移性。具体地说，我们通过利用文本的语义表征能力，特别是从输入图像的基本事实类标签来制定有效的提示驱动的特征指导。据我们所知，我们是第一个引入快速学习来增强可转移的生成性攻击的。在各种跨域和跨模型环境下进行的广泛实验验证了我们的方法，证明了它比最先进的方法更优越。



