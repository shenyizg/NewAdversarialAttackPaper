# Latest Adversarial Attack Papers
**update at 2025-06-16 10:31:09**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Self-interpreting Adversarial Images**

自我解释对抗图像 cs.CR

in USENIX Security 2025

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2407.08970v4) [paper-pdf](http://arxiv.org/pdf/2407.08970v4)

**Authors**: Tingwei Zhang, Collin Zhang, John X. Morris, Eugene Bagdasarian, Vitaly Shmatikov

**Abstract**: We introduce a new type of indirect, cross-modal injection attacks against visual language models that enable creation of self-interpreting images. These images contain hidden "meta-instructions" that control how models answer users' questions about the image and steer models' outputs to express an adversary-chosen style, sentiment, or point of view.   Self-interpreting images act as soft prompts, conditioning the model to satisfy the adversary's (meta-)objective while still producing answers based on the image's visual content. Meta-instructions are thus a stronger form of prompt injection. Adversarial images look natural and the model's answers are coherent and plausible, yet they also follow the adversary-chosen interpretation, e.g., political spin, or even objectives that are not achievable with explicit text instructions.   We evaluate the efficacy of self-interpreting images for a variety of models, interpretations, and user prompts. We describe how these attacks could cause harm by enabling creation of self-interpreting content that carries spam, misinformation, or spin. Finally, we discuss defenses.

摘要: 我们针对视觉语言模型引入了一种新型的间接、跨模式注入攻击，可以创建自我解释的图像。这些图像包含隐藏的“元指令”，这些指令控制模型如何回答用户有关图像的问题，并引导模型的输出来表达对手选择的风格、情感或观点。   自我解释图像充当软提示，调节模型以满足对手的（Meta）目标，同时仍然根据图像的视觉内容生成答案。因此，元指令是一种更强的提示注入形式。对抗图像看起来很自然，模型的答案连贯且可信，但它们也遵循对抗选择的解释，例如，政治旋转，甚至是通过明确的文本指令无法实现的目标。   我们评估各种模型、解释和用户提示的自我解释图像的有效性。我们描述了这些攻击如何通过创建携带垃圾邮件、错误信息或旋转的自我解释内容来造成伤害。最后，我们谈谈防御。



## **2. Improving Large Language Model Safety with Contrastive Representation Learning**

通过对比表示学习提高大型语言模型安全性 cs.CL

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11938v1) [paper-pdf](http://arxiv.org/pdf/2506.11938v1)

**Authors**: Samuel Simko, Mrinmaya Sachan, Bernhard Schölkopf, Zhijing Jin

**Abstract**: Large Language Models (LLMs) are powerful tools with profound societal impacts, yet their ability to generate responses to diverse and uncontrolled inputs leaves them vulnerable to adversarial attacks. While existing defenses often struggle to generalize across varying attack types, recent advancements in representation engineering offer promising alternatives. In this work, we propose a defense framework that formulates model defense as a contrastive representation learning (CRL) problem. Our method finetunes a model using a triplet-based loss combined with adversarial hard negative mining to encourage separation between benign and harmful representations. Our experimental results across multiple models demonstrate that our approach outperforms prior representation engineering-based defenses, improving robustness against both input-level and embedding-space attacks without compromising standard performance. Our code is available at https://github.com/samuelsimko/crl-llm-defense

摘要: 大型语言模型（LLM）是具有深远社会影响的强大工具，但它们对多样化且不受控制的输入产生响应的能力使它们容易受到对抗性攻击。虽然现有的防御通常很难在不同的攻击类型中进行概括，但表示工程的最新进展提供了有希望的替代方案。在这项工作中，我们提出了一个防御框架，将模型防御制定为对比表示学习（RTL）问题。我们的方法使用基于三重组的损失结合对抗性硬负挖掘来微调模型，以鼓励良性和有害表示之间的分离。我们跨多个模型的实验结果表明，我们的方法优于基于先验表示工程的防御，在不损害标准性能的情况下提高了针对输入级和嵌入空间攻击的鲁棒性。我们的代码可在https://github.com/samuelsimko/crl-llm-defense上获取



## **3. Graph of Attacks with Pruning: Optimizing Stealthy Jailbreak Prompt Generation for Enhanced LLM Content Moderation**

带有修剪的攻击图：优化隐形越狱提示生成以增强的LLM内容审核 cs.CR

14 pages, 5 figures

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2501.18638v2) [paper-pdf](http://arxiv.org/pdf/2501.18638v2)

**Authors**: Daniel Schwartz, Dmitriy Bespalov, Zhe Wang, Ninad Kulkarni, Yanjun Qi

**Abstract**: As large language models (LLMs) become increasingly prevalent, ensuring their robustness against adversarial misuse is crucial. This paper introduces the GAP (Graph of Attacks with Pruning) framework, an advanced approach for generating stealthy jailbreak prompts to evaluate and enhance LLM safeguards. GAP addresses limitations in existing tree-based LLM jailbreak methods by implementing an interconnected graph structure that enables knowledge sharing across attack paths. Our experimental evaluation demonstrates GAP's superiority over existing techniques, achieving a 20.8% increase in attack success rates while reducing query costs by 62.7%. GAP consistently outperforms state-of-the-art methods for attacking both open and closed LLMs, with attack success rates of >96%. Additionally, we present specialized variants like GAP-Auto for automated seed generation and GAP-VLM for multimodal attacks. GAP-generated prompts prove highly effective in improving content moderation systems, increasing true positive detection rates by 108.5% and accuracy by 183.6% when used for fine-tuning. Our implementation is available at https://github.com/dsbuddy/GAP-LLM-Safety.

摘要: 随着大型语言模型（LLM）变得越来越普遍，确保其针对对抗性滥用的鲁棒性至关重要。本文介绍了GAP（带有修剪的攻击图）框架，这是一种生成隐形越狱提示以评估和增强LLM保障措施的高级方法。GAP通过实现互连的图结构来解决现有基于树的LLM越狱方法的局限性，该结构能够实现跨攻击路径的知识共享。我们的实验评估证明了GAP相对于现有技术的优越性，攻击成功率提高了20.8%，同时将查询成本降低了62.7%。对于攻击开放式和封闭式LLM，RAP始终优于最先进的方法，攻击成功率> 96%。此外，我们还提供了专门的变体，例如用于自动种子生成的GAP-Auto和用于多模式攻击的GAP-VLM。事实证明，由间隙生成的提示在改进内容审核系统方面非常有效，用于微调时，真阳性检测率可提高108.5%，准确率可提高183.6%。我们的实施可在https://github.com/dsbuddy/GAP-LLM-Safety上获取。



## **4. Attention-based Adversarial Robust Distillation in Radio Signal Classifications for Low-Power IoT Devices**

低功耗物联网设备无线信号分类中基于注意力的对抗鲁棒蒸馏 cs.LG

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11892v1) [paper-pdf](http://arxiv.org/pdf/2506.11892v1)

**Authors**: Lu Zhang, Sangarapillai Lambotharan, Gan Zheng, Guisheng Liao, Basil AsSadhan, Fabio Roli

**Abstract**: Due to great success of transformers in many applications such as natural language processing and computer vision, transformers have been successfully applied in automatic modulation classification. We have shown that transformer-based radio signal classification is vulnerable to imperceptible and carefully crafted attacks called adversarial examples. Therefore, we propose a defense system against adversarial examples in transformer-based modulation classifications. Considering the need for computationally efficient architecture particularly for Internet of Things (IoT)-based applications or operation of devices in environment where power supply is limited, we propose a compact transformer for modulation classification. The advantages of robust training such as adversarial training in transformers may not be attainable in compact transformers. By demonstrating this, we propose a novel compact transformer that can enhance robustness in the presence of adversarial attacks. The new method is aimed at transferring the adversarial attention map from the robustly trained large transformer to a compact transformer. The proposed method outperforms the state-of-the-art techniques for the considered white-box scenarios including fast gradient method and projected gradient descent attacks. We have provided reasoning of the underlying working mechanisms and investigated the transferability of the adversarial examples between different architectures. The proposed method has the potential to protect the transformer from the transferability of adversarial examples.

摘要: 由于转换器在自然语言处理和计算机视觉等许多应用中取得了巨大成功，转换器已成功应用于自动调制分类。我们已经表明，基于变压器的无线电信号分类很容易受到难以察觉且精心设计的攻击，称为对抗性示例。因此，我们提出了一种针对基于变压器的调制分类中对抗性示例的防御系统。考虑到特别是对于基于物联网（IoT）的应用或在电源受限的环境中的设备的操作的计算高效架构的需要，我们提出了用于调制分类的紧凑型Transformer。强大的训练，如变压器中的对抗训练的优点可能无法在紧凑的变压器。通过证明这一点，我们提出了一种新型的紧凑型Transformer，可以增强对抗攻击的鲁棒性。新方法的目的是将对抗性注意力地图从经过稳健训练的大型Transformer转移到紧凑Transformer。对于所考虑的白盒场景，包括快速梯度方法和投影梯度下降攻击，所提出的方法优于最新技术。我们提供了底层工作机制的推理，并研究了不同架构之间对抗性示例的可移植性。所提出的方法有可能保护Transformer免受对抗性示例的可移植性的影响。



## **5. Black-Box Adversarial Attacks on LLM-Based Code Completion**

基于LLM的代码补全黑盒对抗攻击 cs.CR

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2408.02509v2) [paper-pdf](http://arxiv.org/pdf/2408.02509v2)

**Authors**: Slobodan Jenko, Niels Mündler, Jingxuan He, Mark Vero, Martin Vechev

**Abstract**: Modern code completion engines, powered by large language models (LLMs), assist millions of developers with their strong capabilities to generate functionally correct code. Due to this popularity, it is crucial to investigate the security implications of relying on LLM-based code completion. In this work, we demonstrate that state-of-the-art black-box LLM-based code completion engines can be stealthily biased by adversaries to significantly increase their rate of insecure code generation. We present the first attack, named INSEC, that achieves this goal. INSEC works by injecting an attack string as a short comment in the completion input. The attack string is crafted through a query-based optimization procedure starting from a set of carefully designed initialization schemes. We demonstrate INSEC's broad applicability and effectiveness by evaluating it on various state-of-the-art open-source models and black-box commercial services (e.g., OpenAI API and GitHub Copilot). On a diverse set of security-critical test cases, covering 16 CWEs across 5 programming languages, INSEC increases the rate of generated insecure code by more than 50%, while maintaining the functional correctness of generated code. We consider INSEC practical -- it requires low resources and costs less than 10 US dollars to develop on commodity hardware. Moreover, we showcase the attack's real-world deployability, by developing an IDE plug-in that stealthily injects INSEC into the GitHub Copilot extension.

摘要: 由大型语言模型（LLM）支持的现代代码完成引擎可以帮助数百万开发人员以其强大的能力生成功能正确的代码。由于这种受欢迎程度，研究依赖基于LLM的代码完成的安全影响至关重要。在这项工作中，我们证明了最先进的基于LLM的黑匣子代码完成引擎可能会受到对手的悄悄偏见，以显着提高其不安全代码生成率。我们介绍了第一个攻击，名为INSEC，可以实现这一目标。INSEC的工作原理是在完成输入中注入攻击字符串作为简短注释。攻击字符串是通过基于查询的优化过程从一组精心设计的初始化方案开始精心设计的。我们通过在各种最先进的开源模型和黑匣子商业服务上进行评估来展示INSEC的广泛适用性和有效性（例如，OpenAI API和GitHub Copilot）。在一组多样化的安全关键测试用例中，涵盖5种编程语言的16个CWE，INSEC将生成的不安全代码的比率提高了50%以上，同时保持生成代码的功能正确性。我们认为INSEC是可行的--在商品硬件上进行开发所需的资源较少，成本不到10美元。此外，我们通过开发一个将INSEC秘密注入GitHub Copilot扩展的IDE插件，展示了攻击在现实世界中的可部署性。



## **6. TrustGLM: Evaluating the Robustness of GraphLLMs Against Prompt, Text, and Structure Attacks**

TrustGLM：评估GraphLLM针对提示、文本和结构攻击的稳健性 cs.LG

12 pages, 5 figures, in KDD 2025

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11844v1) [paper-pdf](http://arxiv.org/pdf/2506.11844v1)

**Authors**: Qihai Zhang, Xinyue Sheng, Yuanfu Sun, Qiaoyu Tan

**Abstract**: Inspired by the success of large language models (LLMs), there is a significant research shift from traditional graph learning methods to LLM-based graph frameworks, formally known as GraphLLMs. GraphLLMs leverage the reasoning power of LLMs by integrating three key components: the textual attributes of input nodes, the structural information of node neighborhoods, and task-specific prompts that guide decision-making. Despite their promise, the robustness of GraphLLMs against adversarial perturbations remains largely unexplored-a critical concern for deploying these models in high-stakes scenarios. To bridge the gap, we introduce TrustGLM, a comprehensive study evaluating the vulnerability of GraphLLMs to adversarial attacks across three dimensions: text, graph structure, and prompt manipulations. We implement state-of-the-art attack algorithms from each perspective to rigorously assess model resilience. Through extensive experiments on six benchmark datasets from diverse domains, our findings reveal that GraphLLMs are highly susceptible to text attacks that merely replace a few semantically similar words in a node's textual attribute. We also find that standard graph structure attack methods can significantly degrade model performance, while random shuffling of the candidate label set in prompt templates leads to substantial performance drops. Beyond characterizing these vulnerabilities, we investigate defense techniques tailored to each attack vector through data-augmented training and adversarial training, which show promising potential to enhance the robustness of GraphLLMs. We hope that our open-sourced library will facilitate rapid, equitable evaluation and inspire further innovative research in this field.

摘要: 受大型语言模型（LLM）成功的启发，研究从传统的图学习方法发生了重大转变，转向基于LLM的图框架（正式称为GraphLLM）。GraphLLM通过集成三个关键组件来利用LLM的推理能力：输入节点的文本属性、节点邻居的结构信息以及指导决策的特定任务提示。尽管它们有希望，但GraphLLM对对抗性扰动的稳健性在很大程度上仍然没有被开发--这是在高风险场景中部署这些模型的一个关键问题。为了弥合这一差距，我们引入了TrustGLM，这是一项综合研究，评估了GraphLLM在三个维度（文本、图形结构和提示操作）中对对抗攻击的脆弱性。我们从各个角度实施最先进的攻击算法，以严格评估模型弹性。通过对来自不同领域的六个基准数据集的广泛实验，我们的研究结果表明，GraphLLM非常容易受到文本攻击，这些攻击只是替换节点文本属性中的一些语义相似的单词。我们还发现，标准图结构攻击方法会显着降低模型性能，而提示模板中候选标签集的随机洗牌会导致性能大幅下降。除了描述这些漏洞之外，我们还通过数据增强训练和对抗训练研究了针对每个攻击载体量身定制的防御技术，这些技术在增强GraphLLM稳健性方面表现出了广阔的潜力。我们希望我们的开源图书馆能够促进快速、公平的评估，并激发该领域的进一步创新研究。



## **7. Differential Privacy in Machine Learning: From Symbolic AI to LLMs**

机器学习中的差异隐私：从符号人工智能到LLM cs.CR

arXiv admin note: text overlap with arXiv:2303.00654 by other authors

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11687v1) [paper-pdf](http://arxiv.org/pdf/2506.11687v1)

**Authors**: Francisco Aguilera-Martínez, Fernando Berzal

**Abstract**: Machine learning models should not reveal particular information that is not otherwise accessible. Differential privacy provides a formal framework to mitigate privacy risks by ensuring that the inclusion or exclusion of any single data point does not significantly alter the output of an algorithm, thus limiting the exposure of private information. This survey paper explores the foundational definitions of differential privacy, reviews its original formulations and tracing its evolution through key research contributions. It then provides an in-depth examination of how DP has been integrated into machine learning models, analyzing existing proposals and methods to preserve privacy when training ML models. Finally, it describes how DP-based ML techniques can be evaluated in practice. %Finally, it discusses the broader implications of DP, highlighting its potential for public benefit, its real-world applications, and the challenges it faces, including vulnerabilities to adversarial attacks. By offering a comprehensive overview of differential privacy in machine learning, this work aims to contribute to the ongoing development of secure and responsible AI systems.

摘要: 机器学习模型不应透露以其他方式无法访问的特定信息。差异隐私提供了一个正式的框架，可以通过确保任何单个数据点的包含或排除不会显着改变算法的输出来降低隐私风险，从而限制私人信息的暴露。这篇调查论文探讨了差异隐私的基本定义，回顾了其原始公式，并通过关键研究贡献追踪其演变。然后，它深入研究了DP如何集成到机器学习模型中，分析了在训练ML模型时保护隐私的现有提案和方法。最后，它描述了如何在实践中评估基于DP的ML技术。%最后，它讨论了DP的更广泛影响，强调了其对公共利益的潜力、其在现实世界中的应用以及它面临的挑战，包括对抗性攻击的脆弱性。通过全面概述机器学习中的差异隐私，这项工作旨在为安全且负责任的人工智能系统的持续开发做出贡献。



## **8. KCES: Training-Free Defense for Robust Graph Neural Networks via Kernel Complexity**

KCES：通过核复杂性为鲁棒图神经网络提供免训练防御 cs.LG

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11611v1) [paper-pdf](http://arxiv.org/pdf/2506.11611v1)

**Authors**: Yaning Jia, Shenyang Deng, Chiyu Ma, Yaoqing Yang, Soroush Vosoughi

**Abstract**: Graph Neural Networks (GNNs) have achieved impressive success across a wide range of graph-based tasks, yet they remain highly vulnerable to small, imperceptible perturbations and adversarial attacks. Although numerous defense methods have been proposed to address these vulnerabilities, many rely on heuristic metrics, overfit to specific attack patterns, and suffer from high computational complexity. In this paper, we propose Kernel Complexity-Based Edge Sanitization (KCES), a training-free, model-agnostic defense framework. KCES leverages Graph Kernel Complexity (GKC), a novel metric derived from the graph's Gram matrix that characterizes GNN generalization via its test error bound. Building on GKC, we define a KC score for each edge, measuring the change in GKC when the edge is removed. Edges with high KC scores, typically introduced by adversarial perturbations, are pruned to mitigate their harmful effects, thereby enhancing GNNs' robustness. KCES can also be seamlessly integrated with existing defense strategies as a plug-and-play module without requiring training. Theoretical analysis and extensive experiments demonstrate that KCES consistently enhances GNN robustness, outperforms state-of-the-art baselines, and amplifies the effectiveness of existing defenses, offering a principled and efficient solution for securing GNNs.

摘要: 图神经网络（GNN）在广泛的基于图的任务中取得了令人印象深刻的成功，但它们仍然极易受到小的、难以察觉的扰动和对抗性攻击的影响。尽管已经提出了许多防御方法来解决这些漏洞，但许多方法依赖于启发式指标，过度适合特定的攻击模式，并且计算复杂性很高。在本文中，我们提出了基于核心复杂性的边缘清理（KCES），这是一种免训练、模型不可知的防御框架。KCES利用图核复杂性（GKC），这是一种从图的Gram矩阵派生的新型指标，通过其测试误差界限来描述GNN一般化的特征。在GKC的基础上，我们为每条边定义了KC分数，测量删除边时GKC的变化。通常由对抗性扰动引入的具有高KC分数的边被修剪以减轻其有害影响，从而增强GNN的鲁棒性。KCES还可以作为即插即用模块与现有防御策略无缝集成，无需培训。理论分析和大量实验表明，KCES持续增强GNN稳健性，优于最先进的基线，并放大了现有防御的有效性，为保护GNN提供了原则性且高效的解决方案。



## **9. Investigating Vulnerabilities and Defenses Against Audio-Visual Attacks: A Comprehensive Survey Emphasizing Multimodal Models**

调查漏洞和针对视听攻击的防御：强调多模式模型的全面调查 cs.CR

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11521v1) [paper-pdf](http://arxiv.org/pdf/2506.11521v1)

**Authors**: Jinming Wen, Xinyi Wu, Shuai Zhao, Yanhao Jia, Yuwen Li

**Abstract**: Multimodal large language models (MLLMs), which bridge the gap between audio-visual and natural language processing, achieve state-of-the-art performance on several audio-visual tasks. Despite the superior performance of MLLMs, the scarcity of high-quality audio-visual training data and computational resources necessitates the utilization of third-party data and open-source MLLMs, a trend that is increasingly observed in contemporary research. This prosperity masks significant security risks. Empirical studies demonstrate that the latest MLLMs can be manipulated to produce malicious or harmful content. This manipulation is facilitated exclusively through instructions or inputs, including adversarial perturbations and malevolent queries, effectively bypassing the internal security mechanisms embedded within the models. To gain a deeper comprehension of the inherent security vulnerabilities associated with audio-visual-based multimodal models, a series of surveys investigates various types of attacks, including adversarial and backdoor attacks. While existing surveys on audio-visual attacks provide a comprehensive overview, they are limited to specific types of attacks, which lack a unified review of various types of attacks. To address this issue and gain insights into the latest trends in the field, this paper presents a comprehensive and systematic review of audio-visual attacks, which include adversarial attacks, backdoor attacks, and jailbreak attacks. Furthermore, this paper also reviews various types of attacks in the latest audio-visual-based MLLMs, a dimension notably absent in existing surveys. Drawing upon comprehensive insights from a substantial review, this paper delineates both challenges and emergent trends for future research on audio-visual attacks and defense.

摘要: 多模式大型语言模型（MLLM）弥合了视听和自然语言处理之间的差距，在多项视听任务上实现了最先进的性能。尽管MLLM性能优越，但高质量视听训练数据和计算资源的稀缺使得需要利用第三方数据和开源MLLM，这是当代研究中越来越多地观察到的趋势。这种繁荣掩盖了巨大的安全风险。实证研究表明，最新的MLLM可能会被操纵以产生恶意或有害内容。这种操纵完全通过指令或输入（包括对抗性扰动和恶意查询）来促进，有效地绕过了模型中嵌入的内部安全机制。为了更深入地了解与基于视听的多模式模型相关的固有安全漏洞，一系列调查调查了各种类型的攻击，包括对抗性攻击和后门攻击。虽然现有的关于视听攻击的调查提供了全面的概述，但仅限于特定类型的攻击，缺乏对各种类型的攻击的统一审查。为了解决这个问题并深入了解该领域的最新趋势，本文对视听攻击进行了全面、系统的回顾，其中包括对抗性攻击、后门攻击和越狱攻击。此外，本文还审查了最新的基于视听的MLLM中的各种类型的攻击，这是现有调查中明显缺乏的一个方面。本文从大量的评论中得出了全面的见解，描绘了未来视听攻击和防御研究的挑战和新兴趋势。



## **10. On the Natural Robustness of Vision-Language Models Against Visual Perception Attacks in Autonomous Driving**

自动驾驶中视觉语言模型对视觉感知攻击的自然鲁棒性 cs.CV

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11472v1) [paper-pdf](http://arxiv.org/pdf/2506.11472v1)

**Authors**: Pedram MohajerAnsari, Amir Salarpour, Michael Kühr, Siyu Huang, Mohammad Hamad, Sebastian Steinhorst, Habeeb Olufowobi, Mert D. Pesé

**Abstract**: Autonomous vehicles (AVs) rely on deep neural networks (DNNs) for critical tasks such as traffic sign recognition (TSR), automated lane centering (ALC), and vehicle detection (VD). However, these models are vulnerable to attacks that can cause misclassifications and compromise safety. Traditional defense mechanisms, including adversarial training, often degrade benign accuracy and fail to generalize against unseen attacks. In this work, we introduce Vehicle Vision Language Models (V2LMs), fine-tuned vision-language models specialized for AV perception. Our findings demonstrate that V2LMs inherently exhibit superior robustness against unseen attacks without requiring adversarial training, maintaining significantly higher accuracy than conventional DNNs under adversarial conditions. We evaluate two deployment strategies: Solo Mode, where individual V2LMs handle specific perception tasks, and Tandem Mode, where a single unified V2LM is fine-tuned for multiple tasks simultaneously. Experimental results reveal that DNNs suffer performance drops of 33% to 46% under attacks, whereas V2LMs maintain adversarial accuracy with reductions of less than 8% on average. The Tandem Mode further offers a memory-efficient alternative while achieving comparable robustness to Solo Mode. We also explore integrating V2LMs as parallel components to AV perception to enhance resilience against adversarial threats. Our results suggest that V2LMs offer a promising path toward more secure and resilient AV perception systems.

摘要: 自动驾驶汽车（AV）依赖深度神经网络（DNN）来执行交通标志识别（TSB）、自动车道定中心（ALC）和车辆检测（VD）等关键任务。然而，这些模型很容易受到可能导致错误分类并损害安全性的攻击。传统的防御机制，包括对抗训练，通常会降低良性准确性，并且无法针对不可见的攻击进行概括。在这项工作中，我们介绍了车辆视觉语言模型（V2 LM），这是专门用于AV感知的微调视觉语言模型。我们的研究结果表明，V2 LM本质上对不可见的攻击表现出卓越的鲁棒性，无需对抗训练，在对抗条件下保持比传统DNN显着更高的准确性。我们评估了两种部署策略：Solo模式（单个V2 LM处理特定的感知任务）和Tandem模式（单个统一V2 LM同时针对多个任务进行微调）。实验结果显示，DNN在攻击下性能下降33%至46%，而V2 LM保持对抗准确性，平均下降不到8%。Tandem模式进一步提供了一种内存高效的替代方案，同时实现了与Solo模式相当的稳健性。我们还探索将V2 LM集成为AV感知的并行组件，以增强对抗威胁的弹性。我们的结果表明，V2 LM为更安全和更有弹性的AV感知系统提供了一条有希望的途径。



## **11. Bias Amplification in RAG: Poisoning Knowledge Retrieval to Steer LLMs**

RAG中的偏见放大：毒害Steer LLM的知识检索 cs.LG

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11415v1) [paper-pdf](http://arxiv.org/pdf/2506.11415v1)

**Authors**: Linlin Wang, Tianqing Zhu, Laiqiao Qin, Longxiang Gao, Wanlei Zhou

**Abstract**: In Large Language Models, Retrieval-Augmented Generation (RAG) systems can significantly enhance the performance of large language models by integrating external knowledge. However, RAG also introduces new security risks. Existing research focuses mainly on how poisoning attacks in RAG systems affect model output quality, overlooking their potential to amplify model biases. For example, when querying about domestic violence victims, a compromised RAG system might preferentially retrieve documents depicting women as victims, causing the model to generate outputs that perpetuate gender stereotypes even when the original query is gender neutral. To show the impact of the bias, this paper proposes a Bias Retrieval and Reward Attack (BRRA) framework, which systematically investigates attack pathways that amplify language model biases through a RAG system manipulation. We design an adversarial document generation method based on multi-objective reward functions, employ subspace projection techniques to manipulate retrieval results, and construct a cyclic feedback mechanism for continuous bias amplification. Experiments on multiple mainstream large language models demonstrate that BRRA attacks can significantly enhance model biases in dimensions. In addition, we explore a dual stage defense mechanism to effectively mitigate the impacts of the attack. This study reveals that poisoning attacks in RAG systems directly amplify model output biases and clarifies the relationship between RAG system security and model fairness. This novel potential attack indicates that we need to keep an eye on the fairness issues of the RAG system.

摘要: 在大型语言模型中，检索增强生成（RAG）系统可以通过集成外部知识来显着增强大型语言模型的性能。然而，RAG也带来了新的安全风险。现有的研究主要关注RAG系统中的中毒攻击如何影响模型输出质量，而忽视了它们放大模型偏差的潜力。例如，当查询家庭暴力受害者时，受损的RAG系统可能会优先检索将女性描述为受害者的文件，从而导致模型生成的输出即使在原始查询是性别中立的情况下也会延续性别刻板印象。为了展示偏见的影响，本文提出了一个偏差检索和奖励攻击（BRRA）框架，该框架系统地研究通过RAG系统操纵放大语言模型偏差的攻击途径。我们设计了一种基于多目标奖励函数的对抗性文档生成方法，采用子空间投影技术来操纵检索结果，并构建了连续偏差放大的循环反馈机制。对多个主流大型语言模型的实验表明，BRRA攻击可以显着增强模型维度偏差。此外，我们还探索了双阶段防御机制，以有效减轻攻击的影响。该研究表明，RAG系统中的中毒攻击直接放大了模型输出偏差，并澄清了RAG系统安全性与模型公平性之间的关系。这种新颖的潜在攻击表明我们需要密切关注RAG系统的公平性问题。



## **12. PLeak: Prompt Leaking Attacks against Large Language Model Applications**

PLeak：针对大型语言模型应用程序的提示泄露攻击 cs.CR

To appear in the Proceedings of The ACM Conference on Computer and  Communications Security (CCS), 2024

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2405.06823v3) [paper-pdf](http://arxiv.org/pdf/2405.06823v3)

**Authors**: Bo Hui, Haolin Yuan, Neil Gong, Philippe Burlina, Yinzhi Cao

**Abstract**: Large Language Models (LLMs) enable a new ecosystem with many downstream applications, called LLM applications, with different natural language processing tasks. The functionality and performance of an LLM application highly depend on its system prompt, which instructs the backend LLM on what task to perform. Therefore, an LLM application developer often keeps a system prompt confidential to protect its intellectual property. As a result, a natural attack, called prompt leaking, is to steal the system prompt from an LLM application, which compromises the developer's intellectual property. Existing prompt leaking attacks primarily rely on manually crafted queries, and thus achieve limited effectiveness.   In this paper, we design a novel, closed-box prompt leaking attack framework, called PLeak, to optimize an adversarial query such that when the attacker sends it to a target LLM application, its response reveals its own system prompt. We formulate finding such an adversarial query as an optimization problem and solve it with a gradient-based method approximately. Our key idea is to break down the optimization goal by optimizing adversary queries for system prompts incrementally, i.e., starting from the first few tokens of each system prompt step by step until the entire length of the system prompt.   We evaluate PLeak in both offline settings and for real-world LLM applications, e.g., those on Poe, a popular platform hosting such applications. Our results show that PLeak can effectively leak system prompts and significantly outperforms not only baselines that manually curate queries but also baselines with optimized queries that are modified and adapted from existing jailbreaking attacks. We responsibly reported the issues to Poe and are still waiting for their response. Our implementation is available at this repository: https://github.com/BHui97/PLeak.

摘要: 大型语言模型（LLM）支持一个具有许多下游应用程序（称为LLM应用程序）的新生态系统，这些应用程序具有不同的自然语言处理任务。LLM应用程序的功能和性能高度取决于其系统提示符，系统提示符指示后台LLM执行什么任务。因此，LLM应用程序开发人员通常会对系统进行保密，以保护其知识产权。因此，一种称为提示泄露的自然攻击是从LLM应用程序窃取系统提示，这会损害开发人员的知识产权。现有的提示泄露攻击主要依赖于手动构建的查询，因此效果有限。   本文中，我们设计了一种新颖的封闭式提示泄露攻击框架，称为PLeak，来优化对抗性查询，以便当攻击者将其发送到目标LLM应用程序时，其响应会显示其自己的系统提示。我们将寻找这样的对抗性查询作为一个优化问题，并大致使用基于梯度的方法来解决它。我们的关键想法是通过逐步优化系统提示的对手查询来分解优化目标，即从每个系统提示的前几个标记开始，一步一步地直到系统提示的整个长度。   我们在离线设置和现实世界的LLM应用程序中评估PLeak，例如，Poe上的用户，Poe是托管此类应用程序的流行平台。我们的结果表明，PLeak可以有效地泄露系统提示，并且不仅显着优于手动策划查询的基线，而且还优于具有根据现有越狱攻击修改和改编的优化查询的基线。我们负责任地向Poe报告了这些问题，目前仍在等待他们的回应。我们的实现可以在以下存储库中找到：https://github.com/BHui97/PLeak。



## **13. Byzantine Outside, Curious Inside: Reconstructing Data Through Malicious Updates**

外部拜占庭，内部好奇：通过恶意更新重建数据 cs.LG

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11413v1) [paper-pdf](http://arxiv.org/pdf/2506.11413v1)

**Authors**: Kai Yue, Richeng Jin, Chau-Wai Wong, Huaiyu Dai

**Abstract**: Federated learning (FL) enables decentralized machine learning without sharing raw data, allowing multiple clients to collaboratively learn a global model. However, studies reveal that privacy leakage is possible under commonly adopted FL protocols. In particular, a server with access to client gradients can synthesize data resembling the clients' training data. In this paper, we introduce a novel threat model in FL, named the maliciously curious client, where a client manipulates its own gradients with the goal of inferring private data from peers. This attacker uniquely exploits the strength of a Byzantine adversary, traditionally aimed at undermining model robustness, and repurposes it to facilitate data reconstruction attack. We begin by formally defining this novel client-side threat model and providing a theoretical analysis that demonstrates its ability to achieve significant reconstruction success during FL training. To demonstrate its practical impact, we further develop a reconstruction algorithm that combines gradient inversion with malicious update strategies. Our analysis and experimental results reveal a critical blind spot in FL defenses: both server-side robust aggregation and client-side privacy mechanisms may fail against our proposed attack. Surprisingly, standard server- and client-side defenses designed to enhance robustness or privacy may unintentionally amplify data leakage. Compared to the baseline approach, a mistakenly used defense may instead improve the reconstructed image quality by 10-15%.

摘要: 联邦学习（FL）支持分散式机器学习，无需共享原始数据，允许多个客户端协作学习全局模型。然而，研究表明，隐私泄漏是可能的，在普遍采用的FL协议。特别地，可以访问客户端梯度的服务器可以合成类似于客户端的训练数据的数据。在本文中，我们介绍了一种新的威胁模型在FL，命名为恶意好奇的客户端，其中客户端操纵自己的梯度，从同行推断私人数据的目标。该攻击者独特地利用了拜占庭对手的力量，传统上旨在破坏模型稳健性，并重新利用它来促进数据重建攻击。我们首先正式定义这个新颖的客户端威胁模型，并提供理论分析，证明其在FL培训期间取得重大重建成功的能力。为了展示其实际影响，我们进一步开发了一种将梯度逆与恶意更新策略相结合的重建算法。我们的分析和实验结果揭示了FL防御中的一个关键盲点：服务器端稳健聚合和客户端隐私机制都可能无法抵御我们提出的攻击。令人惊讶的是，旨在增强稳健性或隐私性的标准服务器和客户端防御可能会无意中放大数据泄露。与基线方法相比，错误使用的防御可能会将重建的图像质量提高10- 15%。



## **14. Towards Robust Recommendation: A Review and an Adversarial Robustness Evaluation Library**

迈向稳健推荐：评论和对抗稳健性评估库 cs.IR

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2404.17844v3) [paper-pdf](http://arxiv.org/pdf/2404.17844v3)

**Authors**: Lei Cheng, Xiaowen Huang, Jitao Sang, Jian Yu

**Abstract**: Recently, recommender system has achieved significant success. However, due to the openness of recommender systems, they remain vulnerable to malicious attacks. Additionally, natural noise in training data and issues such as data sparsity can also degrade the performance of recommender systems. Therefore, enhancing the robustness of recommender systems has become an increasingly important research topic. In this survey, we provide a comprehensive overview of the robustness of recommender systems. Based on our investigation, we categorize the robustness of recommender systems into adversarial robustness and non-adversarial robustness. In the adversarial robustness, we introduce the fundamental principles and classical methods of recommender system adversarial attacks and defenses. In the non-adversarial robustness, we analyze non-adversarial robustness from the perspectives of data sparsity, natural noise, and data imbalance. Additionally, we summarize commonly used datasets and evaluation metrics for evaluating the robustness of recommender systems. Finally, we also discuss the current challenges in the field of recommender system robustness and potential future research directions. Additionally, to facilitate fair and efficient evaluation of attack and defense methods in adversarial robustness, we propose an adversarial robustness evaluation library--ShillingREC, and we conduct evaluations of basic attack models and recommendation models. ShillingREC project is released at https://github.com/chengleileilei/ShillingREC.

摘要: 最近，推荐系统取得了显着的成功。然而，由于推荐系统的开放性，它们仍然容易受到恶意攻击。此外，训练数据中的自然噪音和数据稀疏性等问题也会降低推荐系统的性能。因此，增强推荐系统的鲁棒性已成为一个越来越重要的研究课题。在本调查中，我们全面概述了推荐系统的稳健性。根据我们的调查，我们将推荐系统的鲁棒性分为对抗性鲁棒性和非对抗性鲁棒性。在对抗鲁棒性方面，我们介绍了推荐系统对抗攻击和防御的基本原则和经典方法。在非对抗鲁棒性方面，我们从数据稀疏性、自然噪音和数据不平衡的角度分析非对抗鲁棒性。此外，我们总结了常用的数据集和评价指标，用于评估推荐系统的鲁棒性。最后，我们还讨论了目前在推荐系统鲁棒性领域的挑战和潜在的未来研究方向。此外，为了便于公平和有效地评估攻击和防御方法的对抗鲁棒性，我们提出了一个对抗鲁棒性评估库-ShillingREC，我们进行了基本的攻击模型和推荐模型的评估。ShillingREC项目发布于https://github.com/chengleileilei/ShillingREC。



## **15. Deception Against Data-Driven Linear-Quadratic Control**

针对数据驱动线性二次控制的欺骗 eess.SY

16 pages, 5 figures

**SubmitDate**: 2025-06-13    [abs](http://arxiv.org/abs/2506.11373v1) [paper-pdf](http://arxiv.org/pdf/2506.11373v1)

**Authors**: Filippos Fotiadis, Aris Kanellopoulos, Kyriakos G. Vamvoudakis, Ufuk Topcu

**Abstract**: Deception is a common defense mechanism against adversaries with an information disadvantage. It can force such adversaries to select suboptimal policies for a defender's benefit. We consider a setting where an adversary tries to learn the optimal linear-quadratic attack against a system, the dynamics of which it does not know. On the other end, a defender who knows its dynamics exploits its information advantage and injects a deceptive input into the system to mislead the adversary. The defender's aim is to then strategically design this deceptive input: it should force the adversary to learn, as closely as possible, a pre-selected attack that is different from the optimal one. We show that this deception design problem boils down to the solution of a coupled algebraic Riccati and a Lyapunov equation which, however, are challenging to tackle analytically. Nevertheless, we use a block successive over-relaxation algorithm to extract their solution numerically and prove the algorithm's convergence under certain conditions. We perform simulations on a benchmark aircraft, where we showcase how the proposed algorithm can mislead adversaries into learning attacks that are less performance-degrading.

摘要: 欺骗是针对具有信息劣势的对手的一种常见防御机制。它可以迫使此类对手为了防御者的利益而选择次优的政策。我们考虑这样一种情况：对手试图学习针对系统的最佳线性二次攻击，而它不知道系统的动态。另一方面，了解其动态的防御者利用其信息优势，向系统中注入欺骗性输入以误导对手。防御者的目标是从战略上设计这种欺骗性输入：它应该迫使对手尽可能接近地学习与最佳攻击不同的预先选择的攻击。我们表明，这个欺骗设计问题可以归结为耦合代数Riccati和李雅普诺夫方程的解，然而，通过分析来解决这一问题是具有挑战性的。尽管如此，我们使用块连续过松弛算法来数值提取它们的解，并证明了算法在一定条件下的收敛性。我们在基准飞机上进行模拟，展示了所提出的算法如何误导对手学习性能下降较小的攻击。



## **16. On the Stability of Graph Convolutional Neural Networks: A Probabilistic Perspective**

图卷积神经网络的稳定性：概率的角度 cs.LG

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.01213v3) [paper-pdf](http://arxiv.org/pdf/2506.01213v3)

**Authors**: Ning Zhang, Henry Kenlay, Li Zhang, Mihai Cucuringu, Xiaowen Dong

**Abstract**: Graph convolutional neural networks (GCNNs) have emerged as powerful tools for analyzing graph-structured data, achieving remarkable success across diverse applications. However, the theoretical understanding of the stability of these models, i.e., their sensitivity to small changes in the graph structure, remains in rather limited settings, hampering the development and deployment of robust and trustworthy models in practice. To fill this gap, we study how perturbations in the graph topology affect GCNN outputs and propose a novel formulation for analyzing model stability. Unlike prior studies that focus only on worst-case perturbations, our distribution-aware formulation characterizes output perturbations across a broad range of input data. This way, our framework enables, for the first time, a probabilistic perspective on the interplay between the statistical properties of the node data and perturbations in the graph topology. We conduct extensive experiments to validate our theoretical findings and demonstrate their benefits over existing baselines, in terms of both representation stability and adversarial attacks on downstream tasks. Our results demonstrate the practical significance of the proposed formulation and highlight the importance of incorporating data distribution into stability analysis.

摘要: 图卷积神经网络（GCNN）已成为分析图结构数据的强大工具，在各种应用中取得了显着的成功。然而，对这些模型稳定性的理论理解，即，它们对图形结构中的微小变化的敏感性仍然在相当有限的环境中，阻碍了在实践中稳健且值得信赖的模型的开发和部署。为了填补这一空白，我们研究了图布局中的扰动如何影响GCNN输出，并提出了一种用于分析模型稳定性的新公式。与以前的研究只关注最坏情况下的扰动不同，我们的分布感知公式描述了广泛的输入数据中的输出扰动。通过这种方式，我们的框架第一次实现了对节点数据的统计特性和图拓扑中的扰动之间相互作用的概率视角。我们进行了大量的实验来验证我们的理论研究结果，并证明它们在表示稳定性和对下游任务的对抗性攻击方面优于现有基线。我们的研究结果表明，所提出的配方的实际意义，并强调将数据分布到稳定性分析的重要性。



## **17. Improving LLM Safety Alignment with Dual-Objective Optimization**

通过双目标优化改善LLM安全一致性 cs.CL

ICML 2025

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2503.03710v2) [paper-pdf](http://arxiv.org/pdf/2503.03710v2)

**Authors**: Xuandong Zhao, Will Cai, Tianneng Shi, David Huang, Licong Lin, Song Mei, Dawn Song

**Abstract**: Existing training-time safety alignment techniques for large language models (LLMs) remain vulnerable to jailbreak attacks. Direct preference optimization (DPO), a widely deployed alignment method, exhibits limitations in both experimental and theoretical contexts as its loss function proves suboptimal for refusal learning. Through gradient-based analysis, we identify these shortcomings and propose an improved safety alignment that disentangles DPO objectives into two components: (1) robust refusal training, which encourages refusal even when partial unsafe generations are produced, and (2) targeted unlearning of harmful knowledge. This approach significantly increases LLM robustness against a wide range of jailbreak attacks, including prefilling, suffix, and multi-turn attacks across both in-distribution and out-of-distribution scenarios. Furthermore, we introduce a method to emphasize critical refusal tokens by incorporating a reward-based token-level weighting mechanism for refusal learning, which further improves the robustness against adversarial exploits. Our research also suggests that robustness to jailbreak attacks is correlated with token distribution shifts in the training process and internal representations of refusal and harmful tokens, offering valuable directions for future research in LLM safety alignment. The code is available at https://github.com/wicai24/DOOR-Alignment

摘要: 现有的大型语言模型（LLM）训练时安全对齐技术仍然容易受到越狱攻击。直接偏好优化（DPO）是一种广泛应用的对齐方法，在实验和理论背景下都表现出局限性，因为其损失函数被证明对于拒绝学习来说次优。通过基于梯度的分析，我们发现了这些缺点，并提出了一种改进的安全调整，将DPO目标分解为两个部分：（1）稳健的拒绝训练，即使在产生部分不安全的世代时也鼓励拒绝，和（2）有针对性地忘记有害知识。这种方法显着提高了LLM针对各种越狱攻击的稳健性，包括跨分发和跨分发场景的预填充、后缀和多回合攻击。此外，我们引入了一种强调关键拒绝令牌的方法，通过结合基于奖励的令牌级加权机制进行拒绝学习，这进一步提高了针对对抗性利用的鲁棒性。我们的研究还表明，对越狱攻击的稳健性与训练过程中的代币分布变化以及拒绝和有害代币的内部表示相关，为LLM安全性调整的未来研究提供了有价值的方向。该代码可在https://github.com/wicai24/DOOR-Alignment上获取



## **18. Weak-to-Strong Jailbreaking on Large Language Models**

大型语言模型上的弱到强越狱 cs.CL

ICML 2025

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2401.17256v3) [paper-pdf](http://arxiv.org/pdf/2401.17256v3)

**Authors**: Xuandong Zhao, Xianjun Yang, Tianyu Pang, Chao Du, Lei Li, Yu-Xiang Wang, William Yang Wang

**Abstract**: Large language models (LLMs) are vulnerable to jailbreak attacks - resulting in harmful, unethical, or biased text generations. However, existing jailbreaking methods are computationally costly. In this paper, we propose the weak-to-strong jailbreaking attack, an efficient inference time attack for aligned LLMs to produce harmful text. Our key intuition is based on the observation that jailbroken and aligned models only differ in their initial decoding distributions. The weak-to-strong attack's key technical insight is using two smaller models (a safe and an unsafe one) to adversarially modify a significantly larger safe model's decoding probabilities. We evaluate the weak-to-strong attack on 5 diverse open-source LLMs from 3 organizations. The results show our method can increase the misalignment rate to over 99% on two datasets with just one forward pass per example. Our study exposes an urgent safety issue that needs to be addressed when aligning LLMs. As an initial attempt, we propose a defense strategy to protect against such attacks, but creating more advanced defenses remains challenging. The code for replicating the method is available at https://github.com/XuandongZhao/weak-to-strong

摘要: 大型语言模型（LLM）很容易受到越狱攻击，从而导致有害、不道德或有偏见的文本生成。然而，现有的越狱方法计算成本很高。本文中，我们提出了弱到强越狱攻击，这是一种针对对齐LLM的有效推理时间攻击，以产生有害文本。我们的关键直觉是基于这样的观察：越狱和对齐的模型仅在其初始解码分布上有所不同。从弱到强攻击的关键技术见解是使用两个较小的模型（一个安全的模型和一个不安全的模型）来对抗性地修改明显更大的安全模型的解码概率。我们评估了对来自3个组织的5个不同开源LLM的弱到强攻击。结果表明，我们的方法可以将两个数据集的未对准率提高到99%以上，每个示例只需向前传递一次。我们的研究揭示了在调整LLM时需要解决的紧迫安全问题。作为初步尝试，我们提出了一种防御策略来抵御此类攻击，但创建更先进的防御仍然具有挑战性。复制该方法的代码可在https://github.com/XuandongZhao/weak-to-strong上获取



## **19. Lattice Climber Attack: Adversarial attacks for randomized mixtures of classifiers**

格子攀登者攻击：对分类器随机混合的对抗攻击 cs.LG

17 pages including bibliography + 13 pages of supplementary material.  Extended version of the article accepted at ECML 2025

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.10888v1) [paper-pdf](http://arxiv.org/pdf/2506.10888v1)

**Authors**: Lucas Gnecco-Heredia, Benjamin Negrevergne, Yann Chevaleyre

**Abstract**: Finite mixtures of classifiers (a.k.a. randomized ensembles) have been proposed as a way to improve robustness against adversarial attacks. However, existing attacks have been shown to not suit this kind of classifier. In this paper, we discuss the problem of attacking a mixture in a principled way and introduce two desirable properties of attacks based on a geometrical analysis of the problem (effectiveness and maximality). We then show that existing attacks do not meet both of these properties. Finally, we introduce a new attack called {\em lattice climber attack} with theoretical guarantees in the binary linear setting, and demonstrate its performance by conducting experiments on synthetic and real datasets.

摘要: 分类器的有限混合（也称为随机集合）已经被提出作为提高对抗性攻击的鲁棒性的一种方式。然而，现有的攻击已被证明不适合这种分类器。在本文中，我们讨论的问题，攻击的混合物的原则性的方式，并介绍了两个理想的属性的攻击的基础上的几何分析的问题（有效性和极大性）。然后我们表明现有的攻击不满足这两个属性。最后，我们在二进制线性设置中引入了一种名为{\em lattack}的新攻击，具有理论保证，并通过对合成和真实数据集进行实验来证明其性能。



## **20. Unveiling the Role of Randomization in Multiclass Adversarial Classification: Insights from Graph Theory**

揭示随机化在多类对抗分类中的作用：来自图论的见解 cs.LG

9 pages (main), 30 in total. Camera-ready version, accepted at  AISTATS 2025. Erratum: Figure 3 was wrong, the three balls had a common  intersection when they were not supposed to. Fixed the value of radius in  tikz code

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2503.14299v2) [paper-pdf](http://arxiv.org/pdf/2503.14299v2)

**Authors**: Lucas Gnecco-Heredia, Matteo Sammut, Muni Sreenivas Pydi, Rafael Pinot, Benjamin Negrevergne, Yann Chevaleyre

**Abstract**: Randomization as a mean to improve the adversarial robustness of machine learning models has recently attracted significant attention. Unfortunately, much of the theoretical analysis so far has focused on binary classification, providing only limited insights into the more complex multiclass setting. In this paper, we take a step toward closing this gap by drawing inspiration from the field of graph theory. Our analysis focuses on discrete data distributions, allowing us to cast the adversarial risk minimization problems within the well-established framework of set packing problems. By doing so, we are able to identify three structural conditions on the support of the data distribution that are necessary for randomization to improve robustness. Furthermore, we are able to construct several data distributions where (contrarily to binary classification) switching from a deterministic to a randomized solution significantly reduces the optimal adversarial risk. These findings highlight the crucial role randomization can play in enhancing robustness to adversarial attacks in multiclass classification.

摘要: 随机化作为提高机器学习模型对抗鲁棒性的一种手段最近引起了广泛关注。不幸的是，迄今为止的大部分理论分析都集中在二元分类上，对更复杂的多类环境仅提供了有限的见解。在本文中，我们通过从图论领域汲取灵感，朝着缩小这一差距迈出了一步。我们的分析重点是离散数据分布，使我们能够将对抗风险最小化问题置于集合包装问题的完善框架中。通过这样做，我们能够在数据分布的支持下识别出随机化以提高稳健性所需的三个结构条件。此外，我们能够构建多种数据分布，其中（与二元分类相反）从确定性解决方案切换到随机化解决方案可以显着降低最佳对抗风险。这些发现凸显了随机化在增强多类分类中对抗攻击的鲁棒性方面可以发挥的关键作用。



## **21. Breaking Distortion-free Watermarks in Large Language Models**

破解大型语言模型中的无失真水印 cs.CR

22 pages, 5 figures, 4 tables, earlier version presented at AAAI'25  Workshop on Preventing and Detecting LLM Generated Misinformation

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2502.18608v2) [paper-pdf](http://arxiv.org/pdf/2502.18608v2)

**Authors**: Shayleen Reynolds, Hengzhi He, Dung Daniel T. Ngo, Saheed Obitayo, Niccolò Dalmasso, Guang Cheng, Vamsi K. Potluru, Manuela Veloso

**Abstract**: In recent years, LLM watermarking has emerged as an attractive safeguard against AI-generated content, with promising applications in many real-world domains. However, there are growing concerns that the current LLM watermarking schemes are vulnerable to expert adversaries wishing to reverse-engineer the watermarking mechanisms. Prior work in breaking or stealing LLM watermarks mainly focuses on the distribution-modifying algorithm of Kirchenbauer et al. (2023), which perturbs the logit vector before sampling. In this work, we focus on reverse-engineering the other prominent LLM watermarking scheme, distortion-free watermarking (Kuditipudi et al. 2024), which preserves the underlying token distribution by using a hidden watermarking key sequence. We demonstrate that, even under a more sophisticated watermarking scheme, it is possible to compromise the LLM and carry out a spoofing attack, i.e. generate a large number of (potentially harmful) texts that can be attributed to the original watermarked LLM. Specifically, we propose using adaptive prompting and a sorting-based algorithm to accurately recover the underlying secret key for watermarking the LLM. Our empirical findings on LLAMA-3.1-8B-Instruct, Mistral-7B-Instruct, Gemma-7b, and OPT-125M challenge the current theoretical claims on the robustness and usability of the distortion-free watermarking techniques.

摘要: 近年来，LLM水印已成为针对人工智能生成内容的一种有吸引力的保护措施，在许多现实世界领域有着广阔的应用前景。然而，人们越来越担心当前的LLM水印方案容易受到希望对水印机制进行反向工程的专家对手的攻击。先前在破坏或窃取LLM水印方面的工作主要集中在Kirchenbauer等人（2023）的分布修改算法上，该算法在采样之前扰动logit向量。在这项工作中，我们专注于对另一种著名的LLM水印方案进行反向工程，即无失真水印（Kuditipudi等人，2024），该方案通过使用隐藏的水印密钥序列来保留底层令牌分布。我们证明，即使在更复杂的水印方案下，也有可能损害LLM并实施欺骗攻击，即生成大量（潜在有害）的文本，这些文本可归因于原始的水印LLM。具体来说，我们建议使用自适应提示和基于排序的算法来准确地恢复底层秘密密钥，用于对LLM进行水印。我们对LLAMA-3.1- 8B-Direct、Mistral-7 B-Direct、Gemma-7 b和OPT-125 M的实证研究结果挑战了当前关于无失真水印技术的稳健性和可用性的理论主张。



## **22. Efficiency Robustness of Dynamic Deep Learning Systems**

动态深度学习系统的效率稳健性 cs.LG

Accepted to USENIX Security '25

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.10831v1) [paper-pdf](http://arxiv.org/pdf/2506.10831v1)

**Authors**: Ravishka Rathnasuriya, Tingxi Li, Zexin Xu, Zihe Song, Mirazul Haque, Simin Chen, Wei Yang

**Abstract**: Deep Learning Systems (DLSs) are increasingly deployed in real-time applications, including those in resourceconstrained environments such as mobile and IoT devices. To address efficiency challenges, Dynamic Deep Learning Systems (DDLSs) adapt inference computation based on input complexity, reducing overhead. While this dynamic behavior improves efficiency, such behavior introduces new attack surfaces. In particular, efficiency adversarial attacks exploit these dynamic mechanisms to degrade system performance. This paper systematically explores efficiency robustness of DDLSs, presenting the first comprehensive taxonomy of efficiency attacks. We categorize these attacks based on three dynamic behaviors: (i) attacks on dynamic computations per inference, (ii) attacks on dynamic inference iterations, and (iii) attacks on dynamic output production for downstream tasks. Through an in-depth evaluation, we analyze adversarial strategies that target DDLSs efficiency and identify key challenges in securing these systems. In addition, we investigate existing defense mechanisms, demonstrating their limitations against increasingly popular efficiency attacks and the necessity for novel mitigation strategies to secure future adaptive DDLSs.

摘要: 深度学习系统（SLS）越来越多地部署在实时应用程序中，包括移动和物联网设备等资源受限环境中的应用程序。为了解决效率挑战，动态深度学习系统（DDLS）根据输入复杂性调整推理计算，从而减少了系统开销。虽然这种动态行为提高了效率，但这种行为会引入新的攻击表面。特别是，效率对抗攻击利用这些动态机制来降低系统性能。本文系统地探讨了DDLS的效率稳健性，提出了效率攻击的第一个全面分类。我们根据三种动态行为对这些攻击进行分类：（i）对每次推理的动态计算的攻击，（ii）对动态推理迭代的攻击，以及（iii）对下游任务的动态输出产生的攻击。通过深入的评估，我们分析了针对DDLS效率的对抗策略，并确定了保护这些系统的关键挑战。此外，我们还研究了现有的防御机制，证明了它们对日益流行的效率攻击的局限性，以及采用新型缓解策略来保护未来自适应DDLS的必要性。



## **23. Unsourced Adversarial CAPTCHA: A Bi-Phase Adversarial CAPTCHA Framework**

无源对抗验证码：两阶段对抗验证码框架 cs.CV

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.10685v1) [paper-pdf](http://arxiv.org/pdf/2506.10685v1)

**Authors**: Xia Du, Xiaoyuan Liu, Jizhe Zhou, Zheng Lin, Chi-man Pun, Zhe Chen, Wei Ni, Jun Luo

**Abstract**: With the rapid advancements in deep learning, traditional CAPTCHA schemes are increasingly vulnerable to automated attacks powered by deep neural networks (DNNs). Existing adversarial attack methods often rely on original image characteristics, resulting in distortions that hinder human interpretation and limit applicability in scenarios lacking initial input images. To address these challenges, we propose the Unsourced Adversarial CAPTCHA (UAC), a novel framework generating high-fidelity adversarial examples guided by attacker-specified text prompts. Leveraging a Large Language Model (LLM), UAC enhances CAPTCHA diversity and supports both targeted and untargeted attacks. For targeted attacks, the EDICT method optimizes dual latent variables in a diffusion model for superior image quality. In untargeted attacks, especially for black-box scenarios, we introduce bi-path unsourced adversarial CAPTCHA (BP-UAC), a two-step optimization strategy employing multimodal gradients and bi-path optimization for efficient misclassification. Experiments show BP-UAC achieves high attack success rates across diverse systems, generating natural CAPTCHAs indistinguishable to humans and DNNs.

摘要: 随着深度学习的快速发展，传统的CAPTCHA方案越来越容易受到深度神经网络（DNN）支持的自动攻击的影响。现有的对抗攻击方法通常依赖于原始图像特征，从而导致失真，阻碍人类解释并限制在缺乏初始输入图像的场景中的适用性。为了解决这些挑战，我们提出了无源对抗验证码（UAC），这是一种新颖的框架，可以在攻击者指定的文本提示的指导下生成高保真对抗示例。利用大型语言模型（LLM），UAC增强了验证码多样性，并支持有针对性和无针对性的攻击。对于有针对性的攻击，EDICT方法优化扩散模型中的双重潜在变量，以获得卓越的图像质量。在无目标攻击中，特别是对于黑匣子场景，我们引入了双路径无源对抗性CAPTCHA（BP-UAC），这是一种两步优化策略，采用多峰梯度和双路径优化来实现高效的误分类。实验表明，BP-UAC在不同系统中实现了很高的攻击成功率，生成人类和DNN难以区分的自然验证码。



## **24. Experimental Verification of Entangled States in the Adversarial Scenario**

对抗场景中纠缠状态的实验验证 quant-ph

9 pages, 5 figures

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.10655v1) [paper-pdf](http://arxiv.org/pdf/2506.10655v1)

**Authors**: Wen-Hao Zhang, Zihao Li, Gong-Chu Li, Xu-Song Hong, Huangjun Zhu, Geng Chen, Chuan-Feng Li, Guang-Can Guo

**Abstract**: Efficient verification of entangled states is crucial to many applications in quantum information processing. However, the effectiveness of standard quantum state verification (QSV) is based on the condition of independent and identical distribution (IID), which impedes its applications in many practical scenarios. Here we demonstrate a defensive QSV protocol, which is effective in all kinds of non-IID scenarios, including the extremely challenging adversarial scenario. To this end, we build a high-speed preparation-and-measurement apparatus controlled by quantum random-number generators. Our experiments clearly show that standard QSV protocols often provide unreliable fidelity certificates in non-IID scenarios. In sharp contrast, the defensive QSV protocol based on a homogeneous strategy can provide reliable and nearly tight fidelity certificates at comparable high efficiency, even under malicious attacks. Moreover, our scheme is robust against the imperfections in a realistic experiment, which is very appealing to practical applications.

摘要: 纠缠态的有效验证对于量子信息处理的许多应用至关重要。然而，标准量子状态验证（QSV）的有效性是基于独立相同分布（IID）条件的，这阻碍了其在许多实际场景中的应用。在这里，我们演示了一种防御性QSV协议，该协议在各种非IID场景中都有效，包括极具挑战性的对抗场景。为此，我们建造了一个由量子随机数发生器控制的高速描述和测量装置。我们的实验清楚地表明，标准QSV协议通常在非IID场景中提供不可靠的保真度证书。与此形成鲜明对比的是，基于同质策略的防御性QSV协议即使在恶意攻击下也可以以相当高的效率提供可靠且近乎严格的保真度证书。此外，我们的方案对现实实验中的不完善性具有鲁棒性，这对实际应用非常有吸引力。



## **25. Assessing the Resilience of Automotive Intrusion Detection Systems to Adversarial Manipulation**

评估汽车入侵检测系统对对抗操纵的弹性 cs.CR

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.10620v1) [paper-pdf](http://arxiv.org/pdf/2506.10620v1)

**Authors**: Stefano Longari, Paolo Cerracchio, Michele Carminati, Stefano Zanero

**Abstract**: The security of modern vehicles has become increasingly important, with the controller area network (CAN) bus serving as a critical communication backbone for various Electronic Control Units (ECUs). The absence of robust security measures in CAN, coupled with the increasing connectivity of vehicles, makes them susceptible to cyberattacks. While intrusion detection systems (IDSs) have been developed to counter such threats, they are not foolproof. Adversarial attacks, particularly evasion attacks, can manipulate inputs to bypass detection by IDSs. This paper extends our previous work by investigating the feasibility and impact of gradient-based adversarial attacks performed with different degrees of knowledge against automotive IDSs. We consider three scenarios: white-box (attacker with full system knowledge), grey-box (partial system knowledge), and the more realistic black-box (no knowledge of the IDS' internal workings or data). We evaluate the effectiveness of the proposed attacks against state-of-the-art IDSs on two publicly available datasets. Additionally, we study effect of the adversarial perturbation on the attack impact and evaluate real-time feasibility by precomputing evasive payloads for timed injection based on bus traffic. Our results demonstrate that, besides attacks being challenging due to the automotive domain constraints, their effectiveness is strongly dependent on the dataset quality, the target IDS, and the attacker's degree of knowledge.

摘要: 现代车辆的安全性变得越来越重要，控制器局域网（CAN）总线作为各种电子控制单元（ECU）的关键通信骨干。CAN缺乏强大的安全措施，加上车辆的连接性越来越强，使它们容易受到网络攻击。虽然入侵检测系统（IDS）已经被开发出来以应对这些威胁，但它们并不是万无一失的。对抗性攻击，特别是逃避攻击，可以操纵输入以绕过IDS的检测。本文扩展了我们之前的工作，研究了在不同知识程度下针对汽车IDS进行的基于梯度的对抗攻击的可行性和影响。我们考虑三种情况：白盒（具有完整系统知识的攻击者）、灰盒（部分系统知识）和更现实的黑盒（不了解IDS的内部工作或数据）。我们评估了针对两个公开可用数据集的最先进IDS的拟议攻击的有效性。此外，我们还研究了对抗性扰动对攻击影响的影响，并通过根据公交车流量预先计算定时注入的规避有效负载来评估实时可行性。我们的结果表明，除了攻击因汽车领域限制而具有挑战性外，它们的有效性还严重依赖于数据集质量、目标IDS和攻击者的知识程度。



## **26. Guardians of the Agentic System: Preventing Many Shots Jailbreak with Agentic System**

紧急系统的守护者：用紧急系统防止多次枪击越狱 cs.CR

18 pages, 7 figures

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2502.16750v4) [paper-pdf](http://arxiv.org/pdf/2502.16750v4)

**Authors**: Saikat Barua, Mostafizur Rahman, Md Jafor Sadek, Rafiul Islam, Shehenaz Khaled, Ahmedul Kabir

**Abstract**: The autonomous AI agents using large language models can create undeniable values in all span of the society but they face security threats from adversaries that warrants immediate protective solutions because trust and safety issues arise. Considering the many-shot jailbreaking and deceptive alignment as some of the main advanced attacks, that cannot be mitigated by the static guardrails used during the supervised training, points out a crucial research priority for real world robustness. The combination of static guardrails in dynamic multi-agent system fails to defend against those attacks. We intend to enhance security for LLM-based agents through the development of new evaluation frameworks which identify and counter threats for safe operational deployment. Our work uses three examination methods to detect rogue agents through a Reverse Turing Test and analyze deceptive alignment through multi-agent simulations and develops an anti-jailbreaking system by testing it with GEMINI 1.5 pro and llama-3.3-70B, deepseek r1 models using tool-mediated adversarial scenarios. The detection capabilities are strong such as 94\% accuracy for GEMINI 1.5 pro yet the system suffers persistent vulnerabilities when under long attacks as prompt length increases attack success rates (ASR) and diversity metrics become ineffective in prediction while revealing multiple complex system faults. The findings demonstrate the necessity of adopting flexible security systems based on active monitoring that can be performed by the agents themselves together with adaptable interventions by system admin as the current models can create vulnerabilities that can lead to the unreliable and vulnerable system. So, in our work, we try to address such situations and propose a comprehensive framework to counteract the security issues.

摘要: 使用大型语言模型的自主人工智能代理可以在社会各个领域创造不可否认的价值，但它们面临着来自对手的安全威胁，需要立即采取保护性解决方案，因为信任和安全问题的出现。考虑到多次越狱和欺骗性对齐是一些主要的高级攻击，无法通过监督训练期间使用的静态护栏来缓解这一点，这指出了现实世界鲁棒性的一个关键研究优先事项。动态多代理系统中静态护栏的组合无法抵御这些攻击。我们打算通过开发新的评估框架来增强基于LLM的代理的安全性，该评估框架可以识别和应对安全运营部署的威胁。我们的工作使用三种检查方法通过反向图灵测试检测流氓代理，并通过多代理模拟分析欺骗性对齐，并通过使用GEMINI 1.5 pro和llama-3.3- 70 B、deepseek r1模型进行测试来开发反越狱系统，使用工具介导的对抗场景。检测能力很强，例如GEMINI 1.5 pro的准确率为94%，但系统在长时间攻击下会遭受持久漏洞，因为提示长度会增加攻击成功率（ASB），多样性指标在预测中变得无效，同时揭示了多个复杂的系统故障。研究结果表明，有必要采用基于主动监控的灵活安全系统，主动监控可以由代理本身执行，并由系统管理员进行自适应的干预，因为当前的模型可能会创建漏洞，从而导致系统不可靠和脆弱。因此，在我们的工作中，我们试图解决此类情况，并提出一个全面的框架来应对安全问题。



## **27. Don't Lag, RAG: Training-Free Adversarial Detection Using RAG**

不要落后，RAG：使用RAG进行免训练对抗检测 cs.AI

Accepted at VecDB @ ICML 2025

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2504.04858v2) [paper-pdf](http://arxiv.org/pdf/2504.04858v2)

**Authors**: Roie Kazoom, Raz Lapid, Moshe Sipper, Ofer Hadar

**Abstract**: Adversarial patch attacks pose a major threat to vision systems by embedding localized perturbations that mislead deep models. Traditional defense methods often require retraining or fine-tuning, making them impractical for real-world deployment. We propose a training-free Visual Retrieval-Augmented Generation (VRAG) framework that integrates Vision-Language Models (VLMs) for adversarial patch detection. By retrieving visually similar patches and images that resemble stored attacks in a continuously expanding database, VRAG performs generative reasoning to identify diverse attack types, all without additional training or fine-tuning. We extensively evaluate open-source large-scale VLMs, including Qwen-VL-Plus, Qwen2.5-VL-72B, and UI-TARS-72B-DPO, alongside Gemini-2.0, a closed-source model. Notably, the open-source UI-TARS-72B-DPO model achieves up to 95 percent classification accuracy, setting a new state-of-the-art for open-source adversarial patch detection. Gemini-2.0 attains the highest overall accuracy, 98 percent, but remains closed-source. Experimental results demonstrate VRAG's effectiveness in identifying a variety of adversarial patches with minimal human annotation, paving the way for robust, practical defenses against evolving adversarial patch attacks.

摘要: 对抗性补丁攻击通过嵌入误导深度模型的局部扰动，对视觉系统构成重大威胁。传统的防御方法通常需要重新培训或微调，这使得它们对于现实世界的部署来说不切实际。我们提出了一个免训练的视觉检索增强生成（VRAG）框架，该框架集成了用于对抗性补丁检测的视觉语言模型（VLM）。通过检索视觉上相似的补丁和图像，这些补丁和图像类似于不断扩展的数据库中存储的攻击，VRAG执行生成式推理以识别不同的攻击类型，而所有这些都无需额外的训练或微调。我们广泛评估了开源大型VLM，包括Qwen-VL-Plus、Qwen2.5-VL-72 B和UI-TARS-72 B-DPO，以及Gemini-2.0（一种闭源模型）。值得注意的是，开源UI-TARS-72 B-DPO模型实现了高达95%的分类准确率，为开源对抗补丁检测奠定了新的最新水平。Gemini-2.0的总体准确率达到了最高的98%，但仍然是闭源的。实验结果证明了VRAG在以最少的人类注释识别各种对抗补丁方面的有效性，为针对不断发展的对抗补丁攻击的稳健、实用的防御铺平了道路。



## **28. Boosting Adversarial Transferability for Hyperspectral Image Classification Using 3D Structure-invariant Transformation and Intermediate Feature Distance**

使用3D结构不变变换和中间特征距离提高高光谱图像分类的对抗可移植性 cs.CV

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.10459v1) [paper-pdf](http://arxiv.org/pdf/2506.10459v1)

**Authors**: Chun Liu, Bingqian Zhu, Tao Xu, Zheng Zheng, Zheng Li, Wei Yang, Zhigang Han, Jiayao Wang

**Abstract**: Deep Neural Networks (DNNs) are vulnerable to adversarial attacks, which pose security challenges to hyperspectral image (HSI) classification technologies based on DNNs. In the domain of natural images, numerous transfer-based adversarial attack methods have been studied. However, HSIs differ from natural images due to their high-dimensional and rich spectral information. Current research on HSI adversarial examples remains limited and faces challenges in fully utilizing the structural and feature information of images. To address these issues, this paper proposes a novel method to enhance the transferability of the adversarial examples for HSI classification models. First, while keeping the image structure unchanged, the proposed method randomly divides the image into blocks in both spatial and spectral dimensions. Then, various transformations are applied on a block by block basis to increase input diversity and mitigate overfitting. Second, a feature distancing loss targeting intermediate layers is designed, which measures the distance between the amplified features of the original examples and the features of the adversarial examples as the primary loss, while the output layer prediction serves as the auxiliary loss. This guides the perturbation to disrupt the features of the true class in adversarial examples, effectively enhancing transferability. Extensive experiments demonstrate that the adversarial examples generated by the proposed method achieve effective transferability to black-box models on two public HSI datasets. Furthermore, the method maintains robust attack performance even under defense strategies.

摘要: 深度神经网络（DNN）容易受到对抗攻击，这对基于DNN的高光谱图像（HSI）分类技术构成了安全挑战。在自然图像领域，人们研究了多种基于传输的对抗攻击方法。然而，HS因其多维且丰富的光谱信息而与自然图像不同。目前对HSI对抗示例的研究仍然有限，并且在充分利用图像的结构和特征信息方面面临挑战。为了解决这些问题，本文提出了一种新颖的方法来增强HSI分类模型对抗性示例的可移植性。首先，在保持图像结构不变的情况下，提出的方法将图像随机分为空间和光谱维度的块。然后，逐块应用各种转换，以增加输入多样性并减轻过拟合。其次，设计了针对中间层的特征距离损失，测量原始示例的放大特征与对抗示例的特征之间的距离作为主要损失，而输出层预测作为辅助损失。这引导扰动破坏对抗性示例中真实类的特征，有效地增强了可移植性。大量实验表明，所提出的方法生成的对抗示例可以有效地转移到两个公共HSI数据集上的黑匣子模型。此外，即使在防御策略下，该方法也能保持稳健的攻击性能。



## **29. TooBadRL: Trigger Optimization to Boost Effectiveness of Backdoor Attacks on Deep Reinforcement Learning**

TooBadRL：触发优化以提高后门攻击对深度强化学习的有效性 cs.CR

**SubmitDate**: 2025-06-12    [abs](http://arxiv.org/abs/2506.09562v2) [paper-pdf](http://arxiv.org/pdf/2506.09562v2)

**Authors**: Songze Li, Mingxuan Zhang, Kang Wei, Shouling Ji

**Abstract**: Deep reinforcement learning (DRL) has achieved remarkable success in a wide range of sequential decision-making domains, including robotics, healthcare, smart grids, and finance. Recent research demonstrates that attackers can efficiently exploit system vulnerabilities during the training phase to execute backdoor attacks, producing malicious actions when specific trigger patterns are present in the state observations. However, most existing backdoor attacks rely primarily on simplistic and heuristic trigger configurations, overlooking the potential efficacy of trigger optimization. To address this gap, we introduce TooBadRL (Trigger Optimization to Boost Effectiveness of Backdoor Attacks on DRL), the first framework to systematically optimize DRL backdoor triggers along three critical axes, i.e., temporal, spatial, and magnitude. Specifically, we first introduce a performance-aware adaptive freezing mechanism for injection timing. Then, we formulate dimension selection as a cooperative game, utilizing Shapley value analysis to identify the most influential state variable for the injection dimension. Furthermore, we propose a gradient-based adversarial procedure to optimize the injection magnitude under environment constraints. Evaluations on three mainstream DRL algorithms and nine benchmark tasks show that TooBadRL significantly improves attack success rates, while ensuring minimal degradation of normal task performance. These results highlight the previously underappreciated importance of principled trigger optimization in DRL backdoor attacks. The source code of TooBadRL can be found at https://github.com/S3IC-Lab/TooBadRL.

摘要: 深度强化学习（DRL）在广泛的顺序决策领域取得了巨大的成功，包括机器人、医疗保健、智能电网和金融。最近的研究表明，攻击者可以在训练阶段有效地利用系统漏洞来执行后门攻击，当状态观测中存在特定的触发模式时，就会产生恶意行为。然而，大多数现有的后门攻击主要依赖于简单化和启发式的触发器配置，忽略了触发器优化的潜在功效。为了解决这个问题，我们引入了TooBadRL（触发器优化以提高DRL后门攻击的有效性），这是第一个沿着三个关键轴系统优化DRL后门触发器的框架，即，时间、空间和幅度。具体来说，我们首先引入了一种用于注射定时的性能感知自适应冻结机制。然后，我们将维度选择制定为合作博弈，利用Shapley值分析来识别对注入维度最有影响力的状态变量。此外，我们提出了一种基于梯度的对抗程序来优化环境约束下的注入量。对三种主流DRL算法和九个基准任务的评估表明，TooBadRL显着提高了攻击成功率，同时确保正常任务性能的下降最小。这些结果凸显了DRL后门攻击中原则性触发优化的重要性之前被低估。TooBadRL的源代码可在https://github.com/S3IC-Lab/TooBadRL上找到。



## **30. Securing Large Language Models: Threats, Vulnerabilities and Responsible Practices**

保护大型语言模型：威胁、漏洞和负责任的实践 cs.CR

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2403.12503v2) [paper-pdf](http://arxiv.org/pdf/2403.12503v2)

**Authors**: Sara Abdali, Richard Anarfi, CJ Barberan, Jia He, Erfan Shayegani

**Abstract**: Large language models (LLMs) have significantly transformed the landscape of Natural Language Processing (NLP). Their impact extends across a diverse spectrum of tasks, revolutionizing how we approach language understanding and generations. Nevertheless, alongside their remarkable utility, LLMs introduce critical security and risk considerations. These challenges warrant careful examination to ensure responsible deployment and safeguard against potential vulnerabilities. This research paper thoroughly investigates security and privacy concerns related to LLMs from five thematic perspectives: security and privacy concerns, vulnerabilities against adversarial attacks, potential harms caused by misuses of LLMs, mitigation strategies to address these challenges while identifying limitations of current strategies. Lastly, the paper recommends promising avenues for future research to enhance the security and risk management of LLMs.

摘要: 大型语言模型（LLM）显着改变了自然语言处理（NLP）的格局。它们的影响范围涵盖了各种各样的任务，彻底改变了我们对语言理解和生成的方式。然而，除了具有非凡的实用性之外，LLM还引入了关键的安全和风险考虑因素。这些挑战值得仔细检查，以确保负责任的部署并防范潜在漏洞。本研究论文从五个主题角度彻底调查了与LLM相关的安全和隐私问题：安全和隐私问题、对抗性攻击的漏洞、滥用LLM造成的潜在伤害、应对这些挑战的缓解策略，同时识别当前策略的局限性。最后，本文为未来的研究推荐了有希望的途径，以增强LLC的安全性和风险管理。



## **31. AURA: A Multi-Agent Intelligence Framework for Knowledge-Enhanced Cyber Threat Attribution**

AURA：一个用于知识增强型网络威胁归因的多智能体情报框架 cs.CR

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.10175v1) [paper-pdf](http://arxiv.org/pdf/2506.10175v1)

**Authors**: Nanda Rani, Sandeep Kumar Shukla

**Abstract**: Effective attribution of Advanced Persistent Threats (APTs) increasingly hinges on the ability to correlate behavioral patterns and reason over complex, varied threat intelligence artifacts. We present AURA (Attribution Using Retrieval-Augmented Agents), a multi-agent, knowledge-enhanced framework for automated and interpretable APT attribution. AURA ingests diverse threat data including Tactics, Techniques, and Procedures (TTPs), Indicators of Compromise (IoCs), malware details, adversarial tools, and temporal information, which are processed through a network of collaborative agents. These agents are designed for intelligent query rewriting, context-enriched retrieval from structured threat knowledge bases, and natural language justification of attribution decisions. By combining Retrieval-Augmented Generation (RAG) with Large Language Models (LLMs), AURA enables contextual linking of threat behaviors to known APT groups and supports traceable reasoning across multiple attack phases. Experiments on recent APT campaigns demonstrate AURA's high attribution consistency, expert-aligned justifications, and scalability. This work establishes AURA as a promising direction for advancing transparent, data-driven, and scalable threat attribution using multi-agent intelligence.

摘要: 高级持续威胁（APT）的有效归因越来越依赖于将行为模式和推理与复杂、多样化的威胁情报制品相关联的能力。我们提出了AURA（使用检索增强代理的归因），这是一个用于自动化和可解释APT归因的多代理、知识增强框架。AURA吸收各种威胁数据，包括战术、技术和程序（TTP）、妥协指标（IoC）、恶意软件详细信息、对抗工具和临时信息，这些数据通过协作代理网络进行处理。这些代理旨在智能查询重写、从结构化威胁知识库进行上下文丰富检索以及属性决策的自然语言证明。通过将检索增强生成（RAG）与大型语言模型（LLM）相结合，AURA能够将威胁行为与已知的APT组进行上下文链接，并支持跨多个攻击阶段的可追溯推理。最近的APT活动的实验证明AURA的高归因一致性，专家对齐的理由，和可扩展性。这项工作将AURA确立为使用多智能体智能推进透明，数据驱动和可扩展的威胁归因的有前途的方向。



## **32. LLMail-Inject: A Dataset from a Realistic Adaptive Prompt Injection Challenge**

LL Mail-Injects：来自现实自适应提示注入挑战的数据集 cs.CR

Dataset at:  https://huggingface.co/datasets/microsoft/llmail-inject-challenge

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09956v1) [paper-pdf](http://arxiv.org/pdf/2506.09956v1)

**Authors**: Sahar Abdelnabi, Aideen Fay, Ahmed Salem, Egor Zverev, Kai-Chieh Liao, Chi-Huang Liu, Chun-Chih Kuo, Jannis Weigend, Danyael Manlangit, Alex Apostolov, Haris Umair, João Donato, Masayuki Kawakita, Athar Mahboob, Tran Huu Bach, Tsun-Han Chiang, Myeongjin Cho, Hajin Choi, Byeonghyeon Kim, Hyeonjin Lee, Benjamin Pannell, Conor McCauley, Mark Russinovich, Andrew Paverd, Giovanni Cherubin

**Abstract**: Indirect Prompt Injection attacks exploit the inherent limitation of Large Language Models (LLMs) to distinguish between instructions and data in their inputs. Despite numerous defense proposals, the systematic evaluation against adaptive adversaries remains limited, even when successful attacks can have wide security and privacy implications, and many real-world LLM-based applications remain vulnerable. We present the results of LLMail-Inject, a public challenge simulating a realistic scenario in which participants adaptively attempted to inject malicious instructions into emails in order to trigger unauthorized tool calls in an LLM-based email assistant. The challenge spanned multiple defense strategies, LLM architectures, and retrieval configurations, resulting in a dataset of 208,095 unique attack submissions from 839 participants. We release the challenge code, the full dataset of submissions, and our analysis demonstrating how this data can provide new insights into the instruction-data separation problem. We hope this will serve as a foundation for future research towards practical structural solutions to prompt injection.

摘要: 间接提示注入攻击利用大型语言模型（LLM）的固有限制来区分其输入中的指令和数据。尽管有许多防御提案，但针对自适应对手的系统评估仍然有限，即使成功的攻击可能会产生广泛的安全和隐私影响，并且许多基于现实世界的LLM应用程序仍然容易受到攻击。我们展示了LLMail-Injects的结果，这是一个模拟现实场景的公开挑战，其中参与者自适应地尝试将恶意指令注入电子邮件中，以在基于LLM的电子邮件助手中触发未经授权的工具调用。该挑战涵盖多种防御策略、LLM架构和检索配置，产生了来自839名参与者的208，095份独特攻击提交的数据集。我们发布了挑战代码、完整的提交数据集以及我们的分析，展示了这些数据如何为描述-数据分离问题提供新的见解。我们希望这将成为未来研究的基础，以推动注入的实用结构性解决方案。



## **33. Generate-then-Verify: Reconstructing Data from Limited Published Statistics**

生成然后验证：从有限的已发布统计数据重建数据 stat.ML

First two authors contributed equally. Remaining authors are ordered  alphabetically

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2504.21199v2) [paper-pdf](http://arxiv.org/pdf/2504.21199v2)

**Authors**: Terrance Liu, Eileen Xiao, Adam Smith, Pratiksha Thaker, Zhiwei Steven Wu

**Abstract**: We study the problem of reconstructing tabular data from aggregate statistics, in which the attacker aims to identify interesting claims about the sensitive data that can be verified with 100% certainty given the aggregates. Successful attempts in prior work have conducted studies in settings where the set of published statistics is rich enough that entire datasets can be reconstructed with certainty. In our work, we instead focus on the regime where many possible datasets match the published statistics, making it impossible to reconstruct the entire private dataset perfectly (i.e., when approaches in prior work fail). We propose the problem of partial data reconstruction, in which the goal of the adversary is to instead output a $\textit{subset}$ of rows and/or columns that are $\textit{guaranteed to be correct}$. We introduce a novel integer programming approach that first $\textbf{generates}$ a set of claims and then $\textbf{verifies}$ whether each claim holds for all possible datasets consistent with the published aggregates. We evaluate our approach on the housing-level microdata from the U.S. Decennial Census release, demonstrating that privacy violations can still persist even when information published about such data is relatively sparse.

摘要: 我们研究从聚合统计数据重建表格数据的问题，其中攻击者的目标是识别有关敏感数据的有趣声明，这些声明可以在给定聚合物的情况下以100%的确定性进行验证。之前的工作中的成功尝试是在已发布的统计数据集足够丰富的环境中进行的研究，以至于可以确定地重建整个数据集。在我们的工作中，我们专注于许多可能的数据集与已发布的统计数据相匹配的制度，这使得不可能完美地重建整个私人数据集（即，当之前工作中的方法失败时）。我们提出了部分数据重建的问题，其中对手的目标是输出$\texit {subset}$的行和/或列，$\texit {保证是正确的}$。我们引入了一种新的整数规划方法，首先$\textbf{生成}$一组索赔，然后$\textbf{验证}$是否每个索赔持有所有可能的数据集一致的发布的聚合。我们评估了我们对美国十年一次的人口普查发布的住房层面微观数据的方法，表明即使发布的有关此类数据的信息相对较少，侵犯隐私的行为仍然存在。



## **34. Apollo: A Posteriori Label-Only Membership Inference Attack Towards Machine Unlearning**

Apollo：针对机器取消学习的后验纯标签成员推理攻击 cs.LG

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09923v1) [paper-pdf](http://arxiv.org/pdf/2506.09923v1)

**Authors**: Liou Tang, James Joshi, Ashish Kundu

**Abstract**: Machine Unlearning (MU) aims to update Machine Learning (ML) models following requests to remove training samples and their influences on a trained model efficiently without retraining the original ML model from scratch. While MU itself has been employed to provide privacy protection and regulatory compliance, it can also increase the attack surface of the model. Existing privacy inference attacks towards MU that aim to infer properties of the unlearned set rely on the weaker threat model that assumes the attacker has access to both the unlearned model and the original model, limiting their feasibility toward real-life scenarios. We propose a novel privacy attack, A Posteriori Label-Only Membership Inference Attack towards MU, Apollo, that infers whether a data sample has been unlearned, following a strict threat model where an adversary has access to the label-output of the unlearned model only. We demonstrate that our proposed attack, while requiring less access to the target model compared to previous attacks, can achieve relatively high precision on the membership status of the unlearned samples.

摘要: 机器非学习（MU）旨在根据请求更新机器学习（ML）模型，以有效地删除训练样本及其对训练模型的影响，而无需从头重新训练原始ML模型。虽然MU本身被用来提供隐私保护和监管合规性，但它也会增加模型的攻击面。现有的针对MU的隐私推断攻击旨在推断未学习集的属性，依赖于较弱的威胁模型，该模型假设攻击者可以访问未学习的模型和原始模型，从而限制了其在现实生活场景中的可行性。我们提出了一种新颖的隐私攻击，即针对MU、Apollo的后验标签成员资格推断攻击，它遵循严格的威胁模型，其中对手只能访问未学习的模型的标签输出。我们证明，与之前的攻击相比，我们提出的攻击虽然需要更少的访问目标模型，但可以对未学习样本的成员身份状态实现相对高的精确度。



## **35. A look at adversarial attacks on radio waveforms from discrete latent space**

从离散潜在空间看无线电波的对抗性攻击 cs.LG

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09896v1) [paper-pdf](http://arxiv.org/pdf/2506.09896v1)

**Authors**: Attanasia Garuso, Silvija Kokalj-Filipovic, Yagna Kaasaragadda

**Abstract**: Having designed a VQVAE that maps digital radio waveforms into discrete latent space, and yields a perfectly classifiable reconstruction of the original data, we here analyze the attack suppressing properties of VQVAE when an adversarial attack is performed on high-SNR radio-frequency (RF) data-points. To target amplitude modulations from a subset of digitally modulated waveform classes, we first create adversarial attacks that preserve the phase between the in-phase and quadrature component whose values are adversarially changed. We compare them with adversarial attacks of the same intensity where phase is not preserved. We test the classification accuracy of such adversarial examples on a classifier trained to deliver 100% accuracy on the original data. To assess the ability of VQVAE to suppress the strength of the attack, we evaluate the classifier accuracy on the reconstructions by VQVAE of the adversarial datapoints and show that VQVAE substantially decreases the effectiveness of the attack. We also compare the I/Q plane diagram of the attacked data, their reconstructions and the original data. Finally, using multiple methods and metrics, we compare the probability distribution of the VQVAE latent space with and without attack. Varying the attack strength, we observe interesting properties of the discrete space, which may help detect the attacks.

摘要: 设计了一个VQVAE，它将数字无线电波形映射到离散潜在空间，并产生原始数据的完美可分类重建，我们在这里分析了当对高SNR射频（RF）数据点执行对抗性攻击时VQVAE的攻击抑制特性。为了针对数字调制波类子集的幅度调制，我们首先创建对抗攻击，以保留同相和正相分量之间的相，其值会发生对抗性变化。我们将它们与相同强度的对抗性攻击进行比较，其中不保留相。我们在经过训练的分类器上测试此类对抗性示例的分类准确性，该分类器可在原始数据上提供100%准确性。为了评估VQVAE抑制攻击强度的能力，我们评估了VQVAE对对抗数据点重建的分类器准确性，并表明VQVAE大大降低了攻击的有效性。我们还比较了受攻击数据的I/Q平面图、它们的重建和原始数据。最后，我们使用多种方法和指标，比较了有攻击和没有攻击的VQVAE潜在空间的概率分布。通过改变攻击强度，我们观察到离散空间的有趣属性，这可能有助于检测攻击。



## **36. One Pic is All it Takes: Poisoning Visual Document Retrieval Augmented Generation with a Single Image**

一张图片即可：用单个图像毒害视觉文档检索增强生成 cs.CL

19 pages, 7 figures

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2504.02132v2) [paper-pdf](http://arxiv.org/pdf/2504.02132v2)

**Authors**: Ezzeldin Shereen, Dan Ristea, Shae McFadden, Burak Hasircioglu, Vasilios Mavroudis, Chris Hicks

**Abstract**: Multi-modal retrieval augmented generation (M-RAG) is instrumental for inhibiting hallucinations in large multi-modal models (LMMs) through the use of a factual knowledge base (KB). However, M-RAG introduces new attack vectors for adversaries that aim to disrupt the system by injecting malicious entries into the KB. In this paper, we present the first poisoning attack against M-RAG targeting visual document retrieval applications where the KB contains images of document pages. We propose two attacks, each of which require injecting only a single adversarial image into the KB. Firstly, we propose a universal attack that, for any potential user query, influences the response to cause a denial-of-service (DoS) in the M-RAG system. Secondly, we present a targeted attack against one or a group of user queries, with the goal of spreading targeted misinformation. For both attacks, we use a multi-objective gradient-based adversarial approach to craft the injected image while optimizing for both retrieval and generation. We evaluate our attacks against several visual document retrieval datasets, a diverse set of state-of-the-art retrievers (embedding models) and generators (LMMs), demonstrating the attack effectiveness in both the universal and targeted settings. We additionally present results including commonly used defenses, various attack hyper-parameter settings, ablations, and attack transferability.

摘要: 多模式检索增强生成（M-RAG）有助于通过使用事实知识库（KB）来抑制大型多模式模型（LSYS）中的幻觉。然而，M-RAG为对手引入了新的攻击载体，旨在通过将恶意条目注入知识库来破坏系统。本文中，我们提出了针对M-RAG的第一次中毒攻击，目标是KB包含文档页面图像的视觉文档检索应用程序。我们提出了两种攻击，每种攻击只需要将单个对抗图像注入到KB中。首先，我们提出了一种通用攻击，对于任何潜在的用户查询，该攻击都会影响响应，从而在M-RAG系统中引起拒绝服务（DPS）。其次，我们对一个或一组用户查询进行有针对性的攻击，目标是传播有针对性的错误信息。对于这两种攻击，我们使用基于多目标梯度的对抗方法来制作注入的图像，同时优化检索和生成。我们评估了对多个视觉文档检索数据集、一组不同的最先进检索器（嵌入模型）和生成器（LSYS）的攻击，展示了在通用和目标设置中的攻击有效性。我们还提供了包括常用防御、各种攻击超参数设置、消融和攻击可转移性在内的结果。



## **37. CROW: Eliminating Backdoors from Large Language Models via Internal Consistency Regularization**

CROW：通过内部一致性规范化消除大型语言模型的后门 cs.CL

Accepted at ICML 2025, 20 pages

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2411.12768v2) [paper-pdf](http://arxiv.org/pdf/2411.12768v2)

**Authors**: Nay Myat Min, Long H. Pham, Yige Li, Jun Sun

**Abstract**: Large Language Models (LLMs) are vulnerable to backdoor attacks that manipulate outputs via hidden triggers. Existing defense methods--designed for vision/text classification tasks--fail for text generation. We propose Internal Consistency Regularization (CROW), a defense leveraging the observation that backdoored models exhibit unstable layer-wise hidden representations when triggered, while clean models show smooth transitions. CROW enforces consistency across layers via adversarial perturbations and regularization during finetuning, neutralizing backdoors without requiring clean reference models or trigger knowledge--only a small clean dataset. Experiments across Llama-2 (7B, 13B), CodeLlama (7B, 13B), and Mistral-7B demonstrate CROW's effectiveness: it achieves significant reductions in attack success rates across diverse backdoor strategies (sentiment steering, targeted refusal, code injection) while preserving generative performance. CROW's architecture-agnostic design enables practical deployment.

摘要: 大型语言模型（LLM）容易受到后门攻击，这些攻击通过隐藏触发器操纵输出。现有的防御方法（专为视觉/文本分类任务设计）无法生成文本。我们提出了内部一致性正规化（CROW），这是一种利用以下观察结果的防御，即后门模型在触发时表现出不稳定的分层隐藏表示，而干净模型则表现出平滑的过渡。CROW在微调期间通过对抗性扰动和正规化来强制跨层的一致性，中和后门，而不需要干净的参考模型或触发知识--只需一个小的干净数据集。Llama-2（7 B，13 B）、CodeLlama（7 B，13 B）和Mistral-7 B的实验证明了CROW的有效性：它在各种后门策略（情绪引导、定向拒绝、代码注入）上显着降低攻击成功率，同时保持生成性能。CROW的架构不可知设计可以实现实际部署。



## **38. Distributionally and Adversarially Robust Logistic Regression via Intersecting Wasserstein Balls**

通过交叉Wasserstein Balls进行分布和反向稳健逻辑回归 math.OC

9 main pages + 25 pages of appendices

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2407.13625v4) [paper-pdf](http://arxiv.org/pdf/2407.13625v4)

**Authors**: Aras Selvi, Eleonora Kreacic, Mohsen Ghassemi, Vamsi Potluru, Tucker Balch, Manuela Veloso

**Abstract**: Adversarially robust optimization (ARO) has emerged as the *de facto* standard for training models that hedge against adversarial attacks in the test stage. While these models are robust against adversarial attacks, they tend to suffer severely from overfitting. To address this issue, some successful methods replace the empirical distribution in the training stage with alternatives including *(i)* a worst-case distribution residing in an ambiguity set, resulting in a distributionally robust (DR) counterpart of ARO; *(ii)* a mixture of the empirical distribution with a distribution induced by an auxiliary (*e.g.*, synthetic, external, out-of-domain) dataset. Inspired by the former, we study the Wasserstein DR counterpart of ARO for logistic regression and show it admits a tractable convex optimization reformulation. Adopting the latter setting, we revise the DR approach by intersecting its ambiguity set with another ambiguity set built using the auxiliary dataset, which offers a significant improvement whenever the Wasserstein distance between the data generating and auxiliary distributions can be estimated. We study the underlying optimization problem, develop efficient solution algorithms, and demonstrate that the proposed method outperforms benchmark approaches on standard datasets.

摘要: 对抗鲁棒优化（ARO）已成为在测试阶段对冲对抗攻击的训练模型的“事实上的”标准。虽然这些模型对对抗攻击很强，但它们往往会严重遭受过度匹配的影响。为了解决这个问题，一些成功的方法用替代方案取代训练阶段中的经验分布，包括 *（i）* 驻留在模糊集中的最坏情况分布，从而产生ARO的分布稳健（DR）对应物; *（ii）* 经验分布与由辅助（* 例如 *，合成的、外部的、域外的）数据集。受前者的启发，我们研究了ARO的Wasserstein DR逻辑回归，并表明它允许易于处理的凸优化重新公式化。采用后一种设置，我们通过将其模糊度集与使用辅助数据集构建的另一个模糊度集相交来修改DR方法，每当可以估计数据生成和辅助分布之间的Wasserstein距离时，这都会提供显着的改进。我们研究潜在的优化问题，开发有效的解决方案算法，并证明所提出的方法优于标准数据集的基准方法。



## **39. Evasion Attacks Against Bayesian Predictive Models**

针对Bayesian预测模型的规避攻击 stat.ML

Accepted as an oral presentation at UAI'25

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09640v1) [paper-pdf](http://arxiv.org/pdf/2506.09640v1)

**Authors**: Pablo G. Arce, Roi Naveiro, David Ríos Insua

**Abstract**: There is an increasing interest in analyzing the behavior of machine learning systems against adversarial attacks. However, most of the research in adversarial machine learning has focused on studying weaknesses against evasion or poisoning attacks to predictive models in classical setups, with the susceptibility of Bayesian predictive models to attacks remaining underexplored. This paper introduces a general methodology for designing optimal evasion attacks against such models. We investigate two adversarial objectives: perturbing specific point predictions and altering the entire posterior predictive distribution. For both scenarios, we propose novel gradient-based attacks and study their implementation and properties in various computational setups.

摘要: 人们对分析机器学习系统对抗对抗攻击的行为越来越感兴趣。然而，对抗性机器学习的大部分研究都集中在研究经典设置中针对预测模型的规避或毒害攻击的弱点，而Bayesian预测模型对攻击的易感性仍然没有得到充分的研究。本文介绍了一种设计针对此类模型的最佳规避攻击的通用方法。我们研究了两个对抗目标：扰乱特定点预测和改变整个后验预测分布。对于这两种情况，我们提出了新型的基于梯度的攻击，并研究了它们在各种计算设置中的实现和属性。



## **40. Effective Red-Teaming of Policy-Adherent Agents**

有效的政策遵守人员红色团队 cs.MA

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09600v1) [paper-pdf](http://arxiv.org/pdf/2506.09600v1)

**Authors**: Itay Nakash, George Kour, Koren Lazar, Matan Vetzler, Guy Uziel, Ateret Anaby-Tavor

**Abstract**: Task-oriented LLM-based agents are increasingly used in domains with strict policies, such as refund eligibility or cancellation rules. The challenge lies in ensuring that the agent consistently adheres to these rules and policies, appropriately refusing any request that would violate them, while still maintaining a helpful and natural interaction. This calls for the development of tailored design and evaluation methodologies to ensure agent resilience against malicious user behavior. We propose a novel threat model that focuses on adversarial users aiming to exploit policy-adherent agents for personal benefit. To address this, we present CRAFT, a multi-agent red-teaming system that leverages policy-aware persuasive strategies to undermine a policy-adherent agent in a customer-service scenario, outperforming conventional jailbreak methods such as DAN prompts, emotional manipulation, and coercive. Building upon the existing tau-bench benchmark, we introduce tau-break, a complementary benchmark designed to rigorously assess the agent's robustness against manipulative user behavior. Finally, we evaluate several straightforward yet effective defense strategies. While these measures provide some protection, they fall short, highlighting the need for stronger, research-driven safeguards to protect policy-adherent agents from adversarial attacks

摘要: 以任务为导向的基于LLM的代理越来越多地用于具有严格政策（例如退款资格或取消规则）的领域。挑战在于确保代理始终遵守这些规则和政策，适当拒绝任何违反规则和政策的请求，同时仍然保持有用且自然的交互。这需要开发量身定制的设计和评估方法，以确保代理针对恶意用户行为的弹性。我们提出了一种新颖的威胁模型，重点关注旨在利用遵守政策的代理来谋取个人利益的对抗用户。为了解决这个问题，我们提出了CRAFT，这是一个多代理红色团队系统，它利用政策感知的说服策略来破坏客户服务场景中遵守政策的代理，优于传统的越狱方法，例如DAN提示、情绪操纵和胁迫。在现有的tau-table基准的基础上，我们引入了tau-break，这是一个补充基准，旨在严格评估代理针对操纵用户行为的稳健性。最后，我们评估了几种简单但有效的防御策略。虽然这些措施提供了一些保护，但它们还不够，这凸显了需要更强大的、以研究为驱动的保障措施来保护遵守政策的代理人免受对抗性攻击



## **41. RSafe: Incentivizing proactive reasoning to build robust and adaptive LLM safeguards**

RSafe：激励积极推理，以建立强大且自适应的LLM保障措施 cs.AI

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.07736v2) [paper-pdf](http://arxiv.org/pdf/2506.07736v2)

**Authors**: Jingnan Zheng, Xiangtian Ji, Yijun Lu, Chenhang Cui, Weixiang Zhao, Gelei Deng, Zhenkai Liang, An Zhang, Tat-Seng Chua

**Abstract**: Large Language Models (LLMs) continue to exhibit vulnerabilities despite deliberate safety alignment efforts, posing significant risks to users and society. To safeguard against the risk of policy-violating content, system-level moderation via external guard models-designed to monitor LLM inputs and outputs and block potentially harmful content-has emerged as a prevalent mitigation strategy. Existing approaches of training guard models rely heavily on extensive human curated datasets and struggle with out-of-distribution threats, such as emerging harmful categories or jailbreak attacks. To address these limitations, we propose RSafe, an adaptive reasoning-based safeguard that conducts guided safety reasoning to provide robust protection within the scope of specified safety policies. RSafe operates in two stages: 1) guided reasoning, where it analyzes safety risks of input content through policy-guided step-by-step reasoning, and 2) reinforced alignment, where rule-based RL optimizes its reasoning paths to align with accurate safety prediction. This two-stage training paradigm enables RSafe to internalize safety principles to generalize safety protection capability over unseen or adversarial safety violation scenarios. During inference, RSafe accepts user-specified safety policies to provide enhanced safeguards tailored to specific safety requirements.

摘要: 尽管采取了刻意的安全调整措施，大型语言模型（LLM）仍然表现出漏洞，给用户和社会带来了重大风险。为了防范违反政策内容的风险，通过外部防护模型进行系统级审核（旨在监控LLM输入和输出并阻止潜在有害内容）已成为一种流行的缓解策略。训练警卫模型的现有方法严重依赖于大量的人类策划的数据集，并与分发外威胁作斗争，例如新出现的有害类别或越狱攻击。为了解决这些限制，我们提出RSafe，这是一种基于自适应推理的保护措施，它进行引导式安全推理，以在指定安全政策范围内提供强有力的保护。RSafe分两个阶段运行：1）引导推理，通过政策引导的分步推理来分析输入内容的安全风险，2）强化对齐，基于规则的RL优化其推理路径以与准确的安全预测保持一致。这种两阶段培训范式使RSafe能够内化安全原则，以概括针对不可见或对抗性安全违规场景的安全保护能力。在推理过程中，RSafe接受用户指定的安全政策，以提供针对特定安全要求的增强的保障措施。



## **42. AngleRoCL: Angle-Robust Concept Learning for Physically View-Invariant T2I Adversarial Patches**

AngleRoCL：针对物理观点不变T2 I对抗补丁的角度稳健概念学习 cs.CV

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09538v1) [paper-pdf](http://arxiv.org/pdf/2506.09538v1)

**Authors**: Wenjun Ji, Yuxiang Fu, Luyang Ying, Deng-Ping Fan, Yuyi Wang, Ming-Ming Cheng, Ivor Tsang, Qing Guo

**Abstract**: Cutting-edge works have demonstrated that text-to-image (T2I) diffusion models can generate adversarial patches that mislead state-of-the-art object detectors in the physical world, revealing detectors' vulnerabilities and risks. However, these methods neglect the T2I patches' attack effectiveness when observed from different views in the physical world (i.e., angle robustness of the T2I adversarial patches). In this paper, we study the angle robustness of T2I adversarial patches comprehensively, revealing their angle-robust issues, demonstrating that texts affect the angle robustness of generated patches significantly, and task-specific linguistic instructions fail to enhance the angle robustness. Motivated by the studies, we introduce Angle-Robust Concept Learning (AngleRoCL), a simple and flexible approach that learns a generalizable concept (i.e., text embeddings in implementation) representing the capability of generating angle-robust patches. The learned concept can be incorporated into textual prompts and guides T2I models to generate patches with their attack effectiveness inherently resistant to viewpoint variations. Through extensive simulation and physical-world experiments on five SOTA detectors across multiple views, we demonstrate that AngleRoCL significantly enhances the angle robustness of T2I adversarial patches compared to baseline methods. Our patches maintain high attack success rates even under challenging viewing conditions, with over 50% average relative improvement in attack effectiveness across multiple angles. This research advances the understanding of physically angle-robust patches and provides insights into the relationship between textual concepts and physical properties in T2I-generated contents.

摘要: 最前沿的作品表明，文本到图像（T2 I）扩散模型可以生成对抗补丁，误导物理世界中最先进的对象检测器，揭示检测器的漏洞和风险。然而，当从物理世界的不同角度观察时，这些方法忽视了T2 I补丁的攻击有效性（即，T2 I对抗补丁的角度稳健性）。本文全面研究了T2 I对抗补丁的角度鲁棒性，揭示了它们的角度鲁棒性问题，证明文本对生成补丁的角度鲁棒性有显着影响，而特定任务的语言指令未能增强角度鲁棒性。受这些研究的启发，我们引入了角度稳健概念学习（AngleRoCL），这是一种简单灵活的方法，可以学习可概括的概念（即，实现中的文本嵌入）表示生成角度稳健补丁的能力。学习到的概念可以被整合到文本提示中，并引导T2 I模型生成攻击有效性本质上可以抵抗观点变化的补丁。通过对多个视图的五个SOTA检测器进行广泛的模拟和物理世界实验，我们证明与基线方法相比，AngleRoCL显着增强了T2 I对抗斑块的角度稳健性。即使在具有挑战性的观看条件下，我们的补丁也能保持很高的攻击成功率，多个角度的攻击有效性平均相对提高超过50%。这项研究促进了对物理角度稳健补丁的理解，并深入了解T2 I生成的内容中的文本概念和物理属性之间的关系。



## **43. GenBreak: Red Teaming Text-to-Image Generators Using Large Language Models**

GenBreak：使用大型语言模型的Red协作文本到图像生成器 cs.CR

27 pages, 7 figures

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.10047v1) [paper-pdf](http://arxiv.org/pdf/2506.10047v1)

**Authors**: Zilong Wang, Xiang Zheng, Xiaosen Wang, Bo Wang, Xingjun Ma, Yu-Gang Jiang

**Abstract**: Text-to-image (T2I) models such as Stable Diffusion have advanced rapidly and are now widely used in content creation. However, these models can be misused to generate harmful content, including nudity or violence, posing significant safety risks. While most platforms employ content moderation systems, underlying vulnerabilities can still be exploited by determined adversaries. Recent research on red-teaming and adversarial attacks against T2I models has notable limitations: some studies successfully generate highly toxic images but use adversarial prompts that are easily detected and blocked by safety filters, while others focus on bypassing safety mechanisms but fail to produce genuinely harmful outputs, neglecting the discovery of truly high-risk prompts. Consequently, there remains a lack of reliable tools for evaluating the safety of defended T2I models. To address this gap, we propose GenBreak, a framework that fine-tunes a red-team large language model (LLM) to systematically explore underlying vulnerabilities in T2I generators. Our approach combines supervised fine-tuning on curated datasets with reinforcement learning via interaction with a surrogate T2I model. By integrating multiple reward signals, we guide the LLM to craft adversarial prompts that enhance both evasion capability and image toxicity, while maintaining semantic coherence and diversity. These prompts demonstrate strong effectiveness in black-box attacks against commercial T2I generators, revealing practical and concerning safety weaknesses.

摘要: 稳定扩散等文本到图像（T2 I）模型发展迅速，现已广泛用于内容创建。然而，这些模型可能会被滥用来生成有害内容，包括裸体或暴力，从而构成重大安全风险。虽然大多数平台都采用内容审核系统，但潜在漏洞仍然可能被坚定的对手利用。最近关于针对T2 I模型的红色组队和对抗性攻击的研究存在显着的局限性：一些研究成功地生成了剧毒图像，但使用了容易被安全过滤器检测和阻止的对抗性提示，而另一些研究则专注于绕过安全机制，但未能产生真正有害的输出，忽视了真正高风险提示的发现。因此，仍然缺乏可靠的工具来评估防御T2 I模型的安全性。为了解决这一差距，我们提出了GenBreak，这是一个对红队大型语言模型（LLM）进行微调的框架，以系统性地探索T2 I生成器中的潜在漏洞。我们的方法将对策划数据集的监督微调与通过与替代T2 I模型交互的强化学习相结合。通过集成多个奖励信号，我们引导LLM设计对抗性提示，以增强规避能力和图像毒性，同时保持语义一致性和多样性。这些提示在针对商用T2 I发生器的黑匣子攻击中表现出强大的有效性，揭示了实际且令人担忧的安全弱点。



## **44. On the Privacy Risks of Spiking Neural Networks: A Membership Inference Analysis**

尖峰神经网络的隐私风险：成员推断分析 cs.LG

14 pages, 6 figures

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2502.13191v4) [paper-pdf](http://arxiv.org/pdf/2502.13191v4)

**Authors**: Junyi Guan, Abhijith Sharma, Chong Tian, Salem Lahlou

**Abstract**: Spiking Neural Networks (SNNs) are increasingly explored for their energy efficiency and robustness in real-world applications, yet their privacy risks remain largely unexamined. In this work, we investigate the susceptibility of SNNs to Membership Inference Attacks (MIAs) -- a major privacy threat where an adversary attempts to determine whether a given sample was part of the training dataset. While prior work suggests that SNNs may offer inherent robustness due to their discrete, event-driven nature, we find that its resilience diminishes as latency (T) increases. Furthermore, we introduce an input dropout strategy under black box setting, that significantly enhances membership inference in SNNs. Our findings challenge the assumption that SNNs are inherently more secure, and even though they are expected to be better, our results reveal that SNNs exhibit privacy vulnerabilities that are equally comparable to Artificial Neural Networks (ANNs). Our code is available at https://github.com/sharmaabhijith/MIA_SNN.

摘要: 尖峰神经网络（SNN）在现实世界应用中因其能源效率和鲁棒性而受到越来越多的探索，但其隐私风险在很大程度上仍未得到审查。在这项工作中，我们调查了SNN对成员推断攻击（MIA）的敏感性--这是一种主要的隐私威胁，对手试图确定给定样本是否是训练数据集的一部分。虽然之前的工作表明SNN由于其离散的、事件驱动的性质而可能提供固有的鲁棒性，但我们发现其弹性随着延迟（T）的增加而减弱。此外，我们在黑匣子设置下引入了输入丢弃策略，这显着增强了SNN中的成员推断。我们的研究结果挑战了SNN本质上更安全的假设，尽管它们预计会更好，但我们的结果表明SNN表现出与人工神经网络（ANN）同等可比的隐私漏洞。我们的代码可在https://github.com/sharmaabhijith/MIA_SNN上获取。



## **45. LLMs Cannot Reliably Judge (Yet?): A Comprehensive Assessment on the Robustness of LLM-as-a-Judge**

LLM无法可靠地判断（还吗？）：法学硕士作为法官稳健性的综合评估 cs.CR

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09443v1) [paper-pdf](http://arxiv.org/pdf/2506.09443v1)

**Authors**: Songze Li, Chuokun Xu, Jiaying Wang, Xueluan Gong, Chen Chen, Jirui Zhang, Jun Wang, Kwok-Yan Lam, Shouling Ji

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable intelligence across various tasks, which has inspired the development and widespread adoption of LLM-as-a-Judge systems for automated model testing, such as red teaming and benchmarking. However, these systems are susceptible to adversarial attacks that can manipulate evaluation outcomes, raising concerns about their robustness and, consequently, their trustworthiness. Existing evaluation methods adopted by LLM-based judges are often piecemeal and lack a unified framework for comprehensive assessment. Furthermore, prompt template and model selections for improving judge robustness have been rarely explored, and their performance in real-world settings remains largely unverified. To address these gaps, we introduce RobustJudge, a fully automated and scalable framework designed to systematically evaluate the robustness of LLM-as-a-Judge systems. RobustJudge investigates the impact of attack methods and defense strategies (RQ1), explores the influence of prompt template and model selection (RQ2), and assesses the robustness of real-world LLM-as-a-Judge applications (RQ3).Our main findings are: (1) LLM-as-a-Judge systems are still vulnerable to a range of adversarial attacks, including Combined Attack and PAIR, while defense mechanisms such as Re-tokenization and LLM-based Detectors offer improved protection; (2) Robustness is highly sensitive to the choice of prompt template and judge models. Our proposed prompt template optimization method can improve robustness, and JudgeLM-13B demonstrates strong performance as a robust open-source judge; (3) Applying RobustJudge to Alibaba's PAI platform reveals previously unreported vulnerabilities. The source code of RobustJudge is provided at https://github.com/S3IC-Lab/RobustJudge.

摘要: 大型语言模型（LLM）在各种任务中表现出了非凡的智能，这激发了LLM作为法官系统的开发和广泛采用，用于自动化模型测试，例如红色团队和基准测试。然而，这些系统很容易受到对抗攻击，这些攻击可以操纵评估结果，从而引发人们对其稳健性的担忧，从而对其可信度。LLM法官采用的现有评估方法往往是零碎的，缺乏统一的综合评估框架。此外，很少探索用于提高判断稳健性的提示模板和模型选择，而且它们在现实世界环境中的性能在很大程度上仍然未经验证。为了解决这些差距，我们引入了RobustJudge，这是一个全自动化和可扩展的框架，旨在系统性评估法学硕士即法官系统的稳健性。RobustJudge调查攻击方法和防御策略的影响（MQ 1），探索提示模板和模型选择的影响（MQ 2），并评估现实世界的LLM作为法官应用程序的稳健性（MQ 3）。我们的主要发现是：（1）法学硕士作为法官系统仍然容易受到一系列对抗攻击，包括联合攻击和PAIR，而重新标记化和基于LLM的检测器等防御机制提供了更好的保护;（2）鲁棒性对提示模板和判断模型的选择高度敏感。我们提出的提示模板优化方法可以提高稳健性，JudggeLM-13 B作为稳健的开源法官表现出了强大的性能;（3）将RobustJudge应用于阿里巴巴的PRI平台，揭示了之前未报告的漏洞。RobustJudge的源代码可访问https://github.com/S3IC-Lab/RobustJudge。



## **46. AdversariaL attacK sAfety aLIgnment(ALKALI): Safeguarding LLMs through GRACE: Geometric Representation-Aware Contrastive Enhancement- Introducing Adversarial Vulnerability Quality Index (AVQI)**

对抗性漏洞质量指数（AVQI）：通过GRACE保护LLM：几何表示-感知对比增强-引入对抗性漏洞质量指数（AVQI） cs.CL

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.08885v2) [paper-pdf](http://arxiv.org/pdf/2506.08885v2)

**Authors**: Danush Khanna, Krishna Kumar, Basab Ghosh, Vinija Jain, Vasu Sharma, Aman Chadha, Amitava Das

**Abstract**: Adversarial threats against LLMs are escalating faster than current defenses can adapt. We expose a critical geometric blind spot in alignment: adversarial prompts exploit latent camouflage, embedding perilously close to the safe representation manifold while encoding unsafe intent thereby evading surface level defenses like Direct Preference Optimization (DPO), which remain blind to the latent geometry. We introduce ALKALI, the first rigorously curated adversarial benchmark and the most comprehensive to date spanning 9,000 prompts across three macro categories, six subtypes, and fifteen attack families. Evaluation of 21 leading LLMs reveals alarmingly high Attack Success Rates (ASRs) across both open and closed source models, exposing an underlying vulnerability we term latent camouflage, a structural blind spot where adversarial completions mimic the latent geometry of safe ones. To mitigate this vulnerability, we introduce GRACE - Geometric Representation Aware Contrastive Enhancement, an alignment framework coupling preference learning with latent space regularization. GRACE enforces two constraints: latent separation between safe and adversarial completions, and adversarial cohesion among unsafe and jailbreak behaviors. These operate over layerwise pooled embeddings guided by a learned attention profile, reshaping internal geometry without modifying the base model, and achieve up to 39% ASR reduction. Moreover, we introduce AVQI, a geometry aware metric that quantifies latent alignment failure via cluster separation and compactness. AVQI reveals when unsafe completions mimic the geometry of safe ones, offering a principled lens into how models internally encode safety. We make the code publicly available at https://anonymous.4open.science/r/alkali-B416/README.md.

摘要: 针对LLM的对抗威胁升级的速度超出了当前防御系统的适应能力。我们暴露了对齐中的一个关键几何盲点：对抗性提示利用潜在伪装，危险地嵌入到靠近安全表示集合的地方，同时编码不安全的意图，从而规避直接偏好优化（DPO）等表面级别的防御，这些防御仍然对潜在的几何形状视而不见。我们引入了ALKARI，这是第一个经过严格策划的对抗基准，也是迄今为止最全面的基准，涵盖三个宏类别、六个亚型和十五个攻击家族的9，000个提示。对21个领先LLM的评估显示，开放源和封闭源模型的攻击成功率（ASB）都高得惊人，暴露了我们称之为“潜在伪装”的潜在漏洞，这是一个结构盲点，对抗性完成模仿安全的潜在几何形状。为了缓解这一漏洞，我们引入了GRACE -几何表示感知对比增强，这是一个将偏好学习与潜在空间正规化结合起来的对齐框架。GRACE强制执行两个限制：安全完成和对抗完成之间的潜在分离，以及不安全和越狱行为之间的对抗凝聚力。这些在学习注意力配置文件的指导下通过分层池嵌入进行操作，在不修改基本模型的情况下重塑内部几何形状，并实现高达39%的ASB降低。此外，我们还引入了AVQI，这是一种几何感知指标，通过集群分离和紧凑性量化潜在的对齐失败。AVQI揭示了不安全完工何时模仿安全完工的几何形状，为模型如何内部编码安全性提供了原则性的视角。我们在https://anonymous.4open.science/r/alkali-B416/README.md上公开该代码。



## **47. Adversarial Surrogate Risk Bounds for Binary Classification**

二元分类的对抗性代理风险界限 cs.LG

37 pages, 2 figures

**SubmitDate**: 2025-06-11    [abs](http://arxiv.org/abs/2506.09348v1) [paper-pdf](http://arxiv.org/pdf/2506.09348v1)

**Authors**: Natalie S. Frank

**Abstract**: A central concern in classification is the vulnerability of machine learning models to adversarial attacks. Adversarial training is one of the most popular techniques for training robust classifiers, which involves minimizing an adversarial surrogate risk. Recent work characterized when a minimizing sequence of an adversarial surrogate risk is also a minimizing sequence of the adversarial classification risk for binary classification -- a property known as adversarial consistency. However, these results do not address the rate at which the adversarial classification risk converges to its optimal value for such a sequence of functions that minimize the adversarial surrogate. This paper provides surrogate risk bounds that quantify that convergence rate. Additionally, we derive distribution-dependent surrogate risk bounds in the standard (non-adversarial) learning setting, that may be of independent interest.

摘要: 分类中的一个核心问题是机器学习模型对对抗攻击的脆弱性。对抗性训练是训练稳健分类器最流行的技术之一，它涉及最大限度地减少对抗性代理风险。最近的工作的特点是，对抗性替代风险的最小化序列也是二元分类的对抗性分类风险的最小化序列--这一属性称为对抗性一致性。然而，这些结果并没有解决对抗性分类风险收敛到最佳值的速度，以使对抗性替代物最小化。本文提供了量化收敛率的替代风险界限。此外，我们在标准（非对抗性）学习环境中推导出依赖于分布的替代风险界限，这可能具有独立的兴趣。



## **48. PatchGuard: Adversarially Robust Anomaly Detection and Localization through Vision Transformers and Pseudo Anomalies**

PatchGuard：通过视觉变换器和伪异常进行逆向鲁棒异常检测和定位 cs.CV

Accepted to the Conference on Computer Vision and Pattern Recognition  (CVPR) 2025

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2506.09237v1) [paper-pdf](http://arxiv.org/pdf/2506.09237v1)

**Authors**: Mojtaba Nafez, Amirhossein Koochakian, Arad Maleki, Jafar Habibi, Mohammad Hossein Rohban

**Abstract**: Anomaly Detection (AD) and Anomaly Localization (AL) are crucial in fields that demand high reliability, such as medical imaging and industrial monitoring. However, current AD and AL approaches are often susceptible to adversarial attacks due to limitations in training data, which typically include only normal, unlabeled samples. This study introduces PatchGuard, an adversarially robust AD and AL method that incorporates pseudo anomalies with localization masks within a Vision Transformer (ViT)-based architecture to address these vulnerabilities. We begin by examining the essential properties of pseudo anomalies, and follow it by providing theoretical insights into the attention mechanisms required to enhance the adversarial robustness of AD and AL systems. We then present our approach, which leverages Foreground-Aware Pseudo-Anomalies to overcome the deficiencies of previous anomaly-aware methods. Our method incorporates these crafted pseudo-anomaly samples into a ViT-based framework, with adversarial training guided by a novel loss function designed to improve model robustness, as supported by our theoretical analysis. Experimental results on well-established industrial and medical datasets demonstrate that PatchGuard significantly outperforms previous methods in adversarial settings, achieving performance gains of $53.2\%$ in AD and $68.5\%$ in AL, while also maintaining competitive accuracy in non-adversarial settings. The code repository is available at https://github.com/rohban-lab/PatchGuard .

摘要: 异常检测（AD）和异常定位（AL）在医学成像和工业监控等要求高可靠性的领域至关重要。然而，由于训练数据的限制，当前的AD和AL方法通常容易受到对抗攻击，训练数据通常只包括正常的、未标记的样本。本研究引入了PatchGuard，这是一种对抗稳健的AD和AL方法，它在基于Vision Transformer（ViT）的架构中将伪异常与定位屏蔽结合起来，以解决这些漏洞。我们首先研究伪异常的基本属性，然后提供增强AD和AL系统对抗鲁棒性所需的注意机制的理论见解。然后，我们介绍了我们的方法，该方法利用前景感知伪异常来克服之前异常感知方法的缺陷。我们的方法将这些精心设计的伪异常样本整合到基于ViT的框架中，并在旨在提高模型稳健性的新型损失函数指导下进行对抗训练，正如我们的理论分析所支持的那样。在成熟的工业和医疗数据集上的实验结果表明，PatchGuard在对抗环境中的表现显着优于之前的方法，在AD中实现了53.2%美元的性能收益，在AL中实现了68.5%美元的性能收益，同时在非对抗环境中还保持了有竞争力的准确性。代码存储库可在https://github.com/rohban-lab/PatchGuard上获取。



## **49. PEFTGuard: Detecting Backdoor Attacks Against Parameter-Efficient Fine-Tuning**

PEFTGuard：检测后门攻击对抗参数高效微调 cs.CR

21 pages, 7 figures

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2411.17453v2) [paper-pdf](http://arxiv.org/pdf/2411.17453v2)

**Authors**: Zhen Sun, Tianshuo Cong, Yule Liu, Chenhao Lin, Xinlei He, Rongmao Chen, Xingshuo Han, Xinyi Huang

**Abstract**: Fine-tuning is an essential process to improve the performance of Large Language Models (LLMs) in specific domains, with Parameter-Efficient Fine-Tuning (PEFT) gaining popularity due to its capacity to reduce computational demands through the integration of low-rank adapters. These lightweight adapters, such as LoRA, can be shared and utilized on open-source platforms. However, adversaries could exploit this mechanism to inject backdoors into these adapters, resulting in malicious behaviors like incorrect or harmful outputs, which pose serious security risks to the community. Unfortunately, few current efforts concentrate on analyzing the backdoor patterns or detecting the backdoors in the adapters. To fill this gap, we first construct and release PADBench, a comprehensive benchmark that contains 13,300 benign and backdoored adapters fine-tuned with various datasets, attack strategies, PEFT methods, and LLMs. Moreover, we propose PEFTGuard, the first backdoor detection framework against PEFT-based adapters. Extensive evaluation upon PADBench shows that PEFTGuard outperforms existing detection methods, achieving nearly perfect detection accuracy (100%) in most cases. Notably, PEFTGuard exhibits zero-shot transferability on three aspects, including different attacks, PEFT methods, and adapter ranks. In addition, we consider various adaptive attacks to demonstrate the high robustness of PEFTGuard. We further explore several possible backdoor mitigation defenses, finding fine-mixing to be the most effective method. We envision that our benchmark and method can shed light on future LLM backdoor detection research.

摘要: 微调是提高特定领域大型语言模型（LLM）性能的重要过程，参数高效微调（PEFT）因其能够通过集成低级适配器来减少计算需求而越来越受欢迎。这些轻量级适配器（例如LoRA）可以在开源平台上共享和使用。然而，对手可能会利用这种机制向这些适配器注入后门，导致不正确或有害输出等恶意行为，从而给社区带来严重的安全风险。不幸的是，目前很少有工作专注于分析后门模式或检测适配器中的后门。为了填补这一空白，我们首先构建并发布PADBench，这是一个全面的基准测试，包含13，300个良性和后门适配器，经过各种数据集、攻击策略、PEFT方法和LLM微调。此外，我们提出了PEFTGuard，第一个后门检测框架对基于PEFT的适配器。对PADBench的广泛评估表明，PEFTGuard优于现有的检测方法，在大多数情况下实现了近乎完美的检测准确率（100%）。值得注意的是，PEFTGuard在三个方面表现出零射击可移植性，包括不同的攻击，PEFT方法和适配器等级。此外，我们还考虑各种自适应攻击来证明PEFTGuard的高稳健性。我们进一步探索了几种可能的后门缓解防御措施，发现精细混合是最有效的方法。我们设想我们的基准和方法可以为未来的LLM后门检测研究提供线索。



## **50. Unified Breakdown Analysis for Byzantine Robust Gossip**

拜占庭鲁棒流言的统一分解分析 math.OC

**SubmitDate**: 2025-06-10    [abs](http://arxiv.org/abs/2410.10418v3) [paper-pdf](http://arxiv.org/pdf/2410.10418v3)

**Authors**: Renaud Gaucher, Aymeric Dieuleveut, Hadrien Hendrikx

**Abstract**: In decentralized machine learning, different devices communicate in a peer-to-peer manner to collaboratively learn from each other's data. Such approaches are vulnerable to misbehaving (or Byzantine) devices. We introduce F-RG, a general framework for building robust decentralized algorithms with guarantees arising from robust-sum-like aggregation rules F. We then investigate the notion of *breakdown point*, and show an upper bound on the number of adversaries that decentralized algorithms can tolerate. We introduce a practical robust aggregation rule, coined CS+, such that CS+-RG has a near-optimal breakdown. Other choices of aggregation rules lead to existing algorithms such as ClippedGossip or NNA. We give experimental evidence to validate the effectiveness of CS+-RG and highlight the gap with NNA, in particular against a novel attack tailored to decentralized communications.

摘要: 在去中心化机器学习中，不同的设备以点对点的方式进行通信，以协作地从彼此的数据中学习。此类方法很容易受到行为不当（或拜占庭式）设备的影响。我们引入了F-RG，这是一个用于构建稳健去中心化算法的通用框架，其保证源自稳健和类聚合规则F。然后，我们研究 * 崩溃点 * 的概念，并给出去中心化算法可以容忍的对手数量的上限。我们引入了一个实用的鲁棒聚合规则，即CS+，这样CS+-RG就有了接近最优的分解。聚合规则的其他选择导致现有算法，例如ClipedGossip或NNA。我们提供了实验证据来验证CS+-RG的有效性，并强调了与NNA的差距，特别是针对去中心化通信量身定制的新型攻击。



