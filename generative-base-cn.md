# 泛生成模型 - 基础/可解释性
**update at 2026-01-25 10:36:50**

按分类器置信度从高到低排序。

## **1. Attacks on Approximate Caches in Text-to-Image Diffusion Models**

文本到图像扩散模型中近似缓存的攻击研究 cs.CR

Accepted by Usenix Security 2026

**SubmitDate**: 2026-01-22    [abs](http://arxiv.org/abs/2508.20424v3) [paper-pdf](https://arxiv.org/pdf/2508.20424v3)

**Confidence**: 0.95

**Authors**: Desen Sun, Shuncheng Jie, Sihang Liu

**Abstract**: Diffusion models are a powerful class of generative models that produce images and other content from user prompts, but they are computationally intensive. To mitigate this cost, recent academic and industry work has adopted approximate caching, which reuses intermediate states from similar prompts in a cache. While efficient, this optimization introduces new security risks by breaking isolation among users. This paper provides a comprehensive assessment of the security vulnerabilities introduced by approximate caching. First, we demonstrate a remote covert channel established with the approximate cache, where a sender injects prompts with special keywords into the cache system and a receiver can recover that even after days, to exchange information. Second, we introduce a prompt stealing attack using the approximate cache, where an attacker can recover existing cached prompts from hits. Finally, we introduce a poisoning attack that embeds the attacker's logos into the previously stolen prompt, leading to unexpected logo rendering for the requests that hit the poisoned cache prompts. These attacks are all performed remotely through the serving system, demonstrating severe security vulnerabilities in approximate caching. The code for this work is available.

摘要: 扩散模型是一类强大的生成模型，能够根据用户提示生成图像等内容，但其计算成本高昂。为降低开销，近期学术界和工业界采用了近似缓存技术，通过缓存相似提示的中间状态实现复用。尽管这一优化提升了效率，却因打破用户间隔离而引入了新的安全风险。本文全面评估了近似缓存带来的安全漏洞。首先，我们展示了利用近似缓存建立的远程隐蔽信道：发送方通过特殊关键词将提示注入缓存系统，接收方即使在数日后仍能恢复信息以实现数据交换。其次，我们提出了基于近似缓存的提示窃取攻击，攻击者可通过缓存命中恢复已缓存的原始提示。最后，我们设计了一种投毒攻击，将攻击者标识嵌入先前窃取的提示中，导致命中污染缓存的请求意外渲染该标识。所有攻击均可通过服务系统远程实施，揭示了近似缓存存在的严重安全漏洞。本研究的代码已公开。



## **2. A Two-Stage Globally-Diverse Adversarial Attack for Vision-Language Pre-training Models**

面向视觉语言预训练模型的两阶段全局多样性对抗攻击 cs.CV

Accepted to ICASSP 2026

**SubmitDate**: 2026-01-18    [abs](http://arxiv.org/abs/2601.12304v1) [paper-pdf](https://arxiv.org/pdf/2601.12304v1)

**Confidence**: 0.95

**Authors**: Wutao Chen, Huaqin Zou, Chen Wan, Lifeng Huang

**Abstract**: Vision-language pre-training (VLP) models are vulnerable to adversarial examples, particularly in black-box scenarios. Existing multimodal attacks often suffer from limited perturbation diversity and unstable multi-stage pipelines. To address these challenges, we propose 2S-GDA, a two-stage globally-diverse attack framework. The proposed method first introduces textual perturbations through a globally-diverse strategy by combining candidate text expansion with globally-aware replacement. To enhance visual diversity, image-level perturbations are generated using multi-scale resizing and block-shuffle rotation. Extensive experiments on VLP models demonstrate that 2S-GDA consistently improves attack success rates over state-of-the-art methods, with gains of up to 11.17\% in black-box settings. Our framework is modular and can be easily combined with existing methods to further enhance adversarial transferability.

摘要: 视觉语言预训练（VLP）模型易受对抗样本攻击，尤其在黑盒场景下。现有多模态攻击方法常面临扰动多样性有限和多阶段流程不稳定的问题。为解决这些挑战，我们提出了2S-GDA——一个两阶段全局多样性攻击框架。该方法首先通过结合候选文本扩展与全局感知替换的全局多样性策略引入文本扰动。为增强视觉多样性，采用多尺度调整和分块随机旋转生成图像级扰动。在VLP模型上的大量实验表明，2S-GDA相比现有最优方法持续提升攻击成功率，在黑盒设置下最高提升11.17%。该框架采用模块化设计，可轻松与现有方法结合以进一步增强对抗迁移性。



## **3. One Perturbation is Enough: On Generating Universal Adversarial Perturbations against Vision-Language Pre-training Models**

一次扰动足矣：针对视觉语言预训练模型的通用对抗扰动生成 cs.CV

Accepted by ICCV-2025

**SubmitDate**: 2025-12-22    [abs](http://arxiv.org/abs/2406.05491v4) [paper-pdf](https://arxiv.org/pdf/2406.05491v4)

**Confidence**: 0.95

**Authors**: Hao Fang, Jiawei Kong, Wenbo Yu, Bin Chen, Jiawei Li, Hao Wu, Shutao Xia, Ke Xu

**Abstract**: Vision-Language Pre-training (VLP) models have exhibited unprecedented capability in many applications by taking full advantage of the multimodal alignment. However, previous studies have shown they are vulnerable to maliciously crafted adversarial samples. Despite recent success, these methods are generally instance-specific and require generating perturbations for each input sample. In this paper, we reveal that VLP models are also vulnerable to the instance-agnostic universal adversarial perturbation (UAP). Specifically, we design a novel Contrastive-training Perturbation Generator with Cross-modal conditions (C-PGC) to achieve the attack. In light that the pivotal multimodal alignment is achieved through the advanced contrastive learning technique, we devise to turn this powerful weapon against themselves, i.e., employ a malicious version of contrastive learning to train the C-PGC based on our carefully crafted positive and negative image-text pairs for essentially destroying the alignment relationship learned by VLP models. Besides, C-PGC fully utilizes the characteristics of Vision-and-Language (V+L) scenarios by incorporating both unimodal and cross-modal information as effective guidance. Extensive experiments show that C-PGC successfully forces adversarial samples to move away from their original area in the VLP model's feature space, thus essentially enhancing attacks across various victim models and V+L tasks. The GitHub repository is available at https://github.com/ffhibnese/CPGC_VLP_Universal_Attacks.

摘要: 视觉语言预训练（VLP）模型通过充分利用多模态对齐能力，在许多应用中展现出前所未有的性能。然而，先前研究表明它们容易受到恶意构造的对抗样本攻击。尽管近期方法取得一定成功，但这些方法通常是实例特定的，需要为每个输入样本生成扰动。本文揭示了VLP模型同样容易受到实例无关的通用对抗扰动（UAP）攻击。具体而言，我们设计了一种新颖的跨模态条件对比训练扰动生成器（C-PGC）来实现攻击。鉴于关键的多模态对齐是通过先进的对比学习技术实现的，我们决定将这一强大武器反制其自身——即采用恶意版本的对比学习来训练C-PGC，该方法基于我们精心构建的正负图像-文本对，旨在从根本上破坏VLP模型学习到的对齐关系。此外，C-PGC充分利用视觉与语言（V+L）场景的特性，将单模态和跨模态信息同时作为有效指导。大量实验表明，C-PGC成功迫使对抗样本在VLP模型特征空间中远离其原始区域，从而显著增强了对不同受害模型和V+L任务的攻击效果。GitHub仓库地址：https://github.com/ffhibnese/CPGC_VLP_Universal_Attacks。



## **4. Towards Effective Prompt Stealing Attack against Text-to-Image Diffusion Models**

面向文本到图像扩散模型的有效提示词窃取攻击研究 cs.CR

This paper proposes an effective training-free, proxy-in-the-loop, and search-based prompt-stealing scheme against T2I models

**SubmitDate**: 2026-01-21    [abs](http://arxiv.org/abs/2508.06837v2) [paper-pdf](https://arxiv.org/pdf/2508.06837v2)

**Confidence**: 0.95

**Authors**: Shiqian Zhao, Chong Wang, Yiming Li, Yihao Huang, Wenjie Qu, Siew-Kei Lam, Yi Xie, Kangjie Chen, Jie Zhang, Tianwei Zhang

**Abstract**: Text-to-Image (T2I) models, represented by DALL$\cdot$E and Midjourney, have gained huge popularity for creating realistic images. The quality of these images relies on the carefully engineered prompts, which have become valuable intellectual property. While skilled prompters showcase their AI-generated art on markets to attract buyers, this business incidentally exposes them to \textit{prompt stealing attacks}. Existing state-of-the-art attack techniques reconstruct the prompts from a fixed set of modifiers (i.e., style descriptions) with model-specific training, which exhibit restricted adaptability and effectiveness to diverse showcases (i.e., target images) and diffusion models.   To alleviate these limitations, we propose Prometheus, a training-free, proxy-in-the-loop, search-based prompt-stealing attack, which reverse-engineers the valuable prompts of the showcases by interacting with a local proxy model. It consists of three innovative designs. First, we introduce dynamic modifiers, as a supplement to static modifiers used in prior works. These dynamic modifiers provide more details specific to the showcases, and we exploit NLP analysis to generate them on the fly. Second, we design a contextual matching algorithm to sort both dynamic and static modifiers. This offline process helps reduce the search space of the subsequent step. Third, we interact with a local proxy model to invert the prompts with a greedy search algorithm. Based on the feedback guidance, we refine the prompt to achieve higher fidelity. The evaluation results show that Prometheus successfully extracts prompts from popular platforms like PromptBase and AIFrog against diverse victim models, including Midjourney, Leonardo.ai, and DALL$\cdot$E, with an ASR improvement of 25.0\%. We also validate that Prometheus is resistant to extensive potential defenses, further highlighting its severity in practice.

摘要: 以DALL·E和Midjourney为代表的文本到图像（T2I）模型因能生成逼真图像而广受欢迎。这些图像的质量依赖于精心设计的提示词，这些提示词已成为宝贵的知识产权。当熟练的提示工程师在市场上展示其AI生成的艺术作品以吸引买家时，这种商业模式无意中使他们暴露于“提示词窃取攻击”。现有的最先进攻击技术通过模型特定训练从一组固定的修饰符（即风格描述）中重建提示词，这些方法对多样化展示（即目标图像）和扩散模型的适应性和有效性有限。为缓解这些限制，我们提出了Prometheus——一种无需训练、基于代理循环搜索的提示词窃取攻击方法，通过与本地代理模型交互来逆向推导展示作品中的有价值提示词。该方法包含三项创新设计：首先，我们引入动态修饰符作为对先前工作中静态修饰符的补充。这些动态修饰符能提供针对展示作品的更多细节，我们利用自然语言处理分析技术实时生成它们。其次，我们设计了上下文匹配算法来对动态和静态修饰符进行排序。这种离线处理有助于缩减后续步骤的搜索空间。第三，我们通过与本地代理模型交互，采用贪心搜索算法逆向推导提示词。基于反馈指导，我们优化提示词以实现更高的保真度。评估结果表明，Prometheus成功从PromptBase和AIFrog等流行平台提取提示词，针对包括Midjourney、Leonardo.ai和DALL·E在内的多种受害模型，攻击成功率提升了25.0%。我们还验证了Prometheus能抵抗多种潜在防御措施，进一步凸显了其在实践中的严重性。



## **5. Inference Attacks Against Graph Generative Diffusion Models**

针对图生成扩散模型的推理攻击 cs.LG

This work has been accepted by USENIX Security 2026

**SubmitDate**: 2026-01-07    [abs](http://arxiv.org/abs/2601.03701v1) [paper-pdf](https://arxiv.org/pdf/2601.03701v1)

**Confidence**: 0.95

**Authors**: Xiuling Wang, Xin Huang, Guibo Luo, Jianliang Xu

**Abstract**: Graph generative diffusion models have recently emerged as a powerful paradigm for generating complex graph structures, effectively capturing intricate dependencies and relationships within graph data. However, the privacy risks associated with these models remain largely unexplored. In this paper, we investigate information leakage in such models through three types of black-box inference attacks. First, we design a graph reconstruction attack, which can reconstruct graphs structurally similar to those training graphs from the generated graphs. Second, we propose a property inference attack to infer the properties of the training graphs, such as the average graph density and the distribution of densities, from the generated graphs. Third, we develop two membership inference attacks to determine whether a given graph is present in the training set. Extensive experiments on three different types of graph generative diffusion models and six real-world graphs demonstrate the effectiveness of these attacks, significantly outperforming the baseline approaches. Finally, we propose two defense mechanisms that mitigate these inference attacks and achieve a better trade-off between defense strength and target model utility than existing methods. Our code is available at https://zenodo.org/records/17946102.

摘要: 图生成扩散模型最近已成为生成复杂图结构的强大范式，能有效捕捉图数据中复杂的依赖关系和关联。然而，这些模型相关的隐私风险在很大程度上仍未得到探索。本文通过三种类型的黑盒推理攻击，研究了此类模型中的信息泄露问题。首先，我们设计了一种图重构攻击，能够从生成的图中重构出与训练图结构相似的图。其次，我们提出了一种属性推理攻击，用于从生成的图中推断训练图的属性，如图的平均密度和密度分布。第三，我们开发了两种成员推理攻击，以判断给定图是否存在于训练集中。在三种不同类型的图生成扩散模型和六个真实世界图上的大量实验证明了这些攻击的有效性，其性能显著优于基线方法。最后，我们提出了两种防御机制来缓解这些推理攻击，并在防御强度与目标模型效用之间实现了比现有方法更好的权衡。我们的代码可在 https://zenodo.org/records/17946102 获取。



## **6. T2VAttack: Adversarial Attack on Text-to-Video Diffusion Models**

T2VAttack：针对文本到视频扩散模型的对抗攻击 cs.CV

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2512.23953v1) [paper-pdf](https://arxiv.org/pdf/2512.23953v1)

**Confidence**: 0.95

**Authors**: Changzhen Li, Yuecong Min, Jie Zhang, Zheng Yuan, Shiguang Shan, Xilin Chen

**Abstract**: The rapid evolution of Text-to-Video (T2V) diffusion models has driven remarkable advancements in generating high-quality, temporally coherent videos from natural language descriptions. Despite these achievements, their vulnerability to adversarial attacks remains largely unexplored. In this paper, we introduce T2VAttack, a comprehensive study of adversarial attacks on T2V diffusion models from both semantic and temporal perspectives. Considering the inherently dynamic nature of video data, we propose two distinct attack objectives: a semantic objective to evaluate video-text alignment and a temporal objective to assess the temporal dynamics. To achieve an effective and efficient attack process, we propose two adversarial attack methods: (i) T2VAttack-S, which identifies semantically or temporally critical words in prompts and replaces them with synonyms via greedy search, and (ii) T2VAttack-I, which iteratively inserts optimized words with minimal perturbation to the prompt. By combining these objectives and strategies, we conduct a comprehensive evaluation on the adversarial robustness of several state-of-the-art T2V models, including ModelScope, CogVideoX, Open-Sora, and HunyuanVideo. Our experiments reveal that even minor prompt modifications, such as the substitution or insertion of a single word, can cause substantial degradation in semantic fidelity and temporal dynamics, highlighting critical vulnerabilities in current T2V diffusion models.

摘要: 文本到视频（T2V）扩散模型的快速发展，推动了根据自然语言描述生成高质量、时序一致视频的显著进步。尽管取得了这些成就，其对抗攻击的脆弱性在很大程度上仍未得到探索。本文提出T2VAttack，从语义和时序两个角度对T2V扩散模型的对抗攻击进行全面研究。考虑到视频数据固有的动态特性，我们提出了两个不同的攻击目标：评估视频-文本对齐的语义目标，以及评估时序动态的时序目标。为实现高效有效的攻击过程，我们提出了两种对抗攻击方法：（i）T2VAttack-S，识别提示中语义或时序关键词语，并通过贪婪搜索将其替换为同义词；（ii）T2VAttack-I，迭代插入优化词语，对提示的扰动最小。通过结合这些目标和策略，我们对多个最先进的T2V模型（包括ModelScope、CogVideoX、Open-Sora和HunyuanVideo）的对抗鲁棒性进行了全面评估。实验表明，即使对提示进行微小修改（如替换或插入单个词语），也可能导致语义保真度和时序动态的显著下降，这凸显了当前T2V扩散模型的关键脆弱性。



## **7. Data-Chain Backdoor: Do You Trust Diffusion Models as Generative Data Supplier?**

数据链后门：您是否信任扩散模型作为生成式数据供应商？ cs.CR

**SubmitDate**: 2025-12-12    [abs](http://arxiv.org/abs/2512.15769v1) [paper-pdf](https://arxiv.org/pdf/2512.15769v1)

**Confidence**: 0.95

**Authors**: Junchi Lu, Xinke Li, Yuheng Liu, Qi Alfred Chen

**Abstract**: The increasing use of generative models such as diffusion models for synthetic data augmentation has greatly reduced the cost of data collection and labeling in downstream perception tasks. However, this new data source paradigm may introduce important security concerns. This work investigates backdoor propagation in such emerging generative data supply chains, namely Data-Chain Backdoor (DCB). Specifically, we find that open-source diffusion models can become hidden carriers of backdoors. Their strong distribution-fitting ability causes them to memorize and reproduce backdoor triggers during generation, which are subsequently inherited by downstream models, resulting in severe security risks. This threat is particularly concerning under clean-label attack scenarios, as it remains effective while having negligible impact on the utility of the synthetic data. Furthermore, we discover an Early-Stage Trigger Manifestation (ESTM) phenomenon: backdoor trigger patterns tend to surface more explicitly in the early, high-noise stages of the diffusion model's reverse generation process before being subtly integrated into the final samples. Overall, this work reveals a previously underexplored threat in generative data pipelines and provides initial insights toward mitigating backdoor risks in synthetic data generation.

摘要: 扩散模型等生成模型在合成数据增强中的日益广泛应用，显著降低了下游感知任务中数据收集与标注的成本。然而，这种新型数据源范式可能引发重要的安全隐患。本研究探讨了此类新兴生成式数据供应链中的后门传播问题，即数据链后门（DCB）。具体而言，我们发现开源扩散模型可能成为后门的隐蔽载体。其强大的分布拟合能力使其在生成过程中记忆并复现后门触发器，这些触发器随后被下游模型继承，导致严重的安全风险。这种威胁在干净标签攻击场景下尤为值得关注，因为它在保持合成数据效用几乎不受影响的同时仍能有效实施。此外，我们发现了早期阶段触发器显现（ESTM）现象：后门触发模式倾向于在扩散模型反向生成过程的早期高噪声阶段更明显地显现，随后才被微妙地整合到最终样本中。总体而言，本研究揭示了生成式数据管道中一个先前未被充分探索的威胁，并为缓解合成数据生成中的后门风险提供了初步见解。



## **8. Membership and Dataset Inference Attacks on Large Audio Generative Models**

针对大型音频生成模型的成员推断与数据集推断攻击 cs.LG

NeurIPS 2025 AI for Music Workshop NeurIPS 2025 Workshop on Creativity & Generative AI

**SubmitDate**: 2025-12-10    [abs](http://arxiv.org/abs/2512.09654v1) [paper-pdf](https://arxiv.org/pdf/2512.09654v1)

**Confidence**: 0.95

**Authors**: Jakub Proboszcz, Paweł Kochanski, Karol Korszun, Donato Crisostomi, Giorgio Strano, Emanuele Rodolà, Kamil Deja, Jan Dubinski

**Abstract**: Generative audio models, based on diffusion and autoregressive architectures, have advanced rapidly in both quality and expressiveness. This progress, however, raises pressing copyright concerns, as such models are often trained on vast corpora of artistic and commercial works. A central question is whether one can reliably verify if an artist's material was included in training, thereby providing a means for copyright holders to protect their content. In this work, we investigate the feasibility of such verification through membership inference attacks (MIA) on open-source generative audio models, which attempt to determine whether a specific audio sample was part of the training set. Our empirical results show that membership inference alone is of limited effectiveness at scale, as the per-sample membership signal is weak for models trained on large and diverse datasets. However, artists and media owners typically hold collections of works rather than isolated samples. Building on prior work in text and vision domains, in this work we focus on dataset inference (DI), which aggregates diverse membership evidence across multiple samples. We find that DI is successful in the audio domain, offering a more practical mechanism for assessing whether an artist's works contributed to model training. Our results suggest DI as a promising direction for copyright protection and dataset accountability in the era of large audio generative models.

摘要: 基于扩散和自回归架构的生成式音频模型在质量和表现力方面发展迅速。然而，这种进步引发了紧迫的版权问题，因为此类模型通常是在大量艺术和商业作品语料库上进行训练的。核心问题在于能否可靠地验证某艺术家的作品是否被包含在训练数据中，从而为版权持有者提供保护其内容的手段。本研究通过针对开源生成式音频模型的成员推断攻击（MIA）来探讨此类验证的可行性，该攻击旨在判断特定音频样本是否属于训练集。我们的实证结果表明，仅凭成员推断在大规模场景下效果有限，因为对于在大型多样化数据集上训练的模型，单个样本的成员信号较弱。然而，艺术家和媒体所有者通常持有作品集合而非孤立样本。基于先前在文本和视觉领域的研究，本研究重点关注数据集推断（DI），该方法通过聚合多个样本的多样化成员证据。我们发现DI在音频领域是成功的，为评估艺术家的作品是否对模型训练有贡献提供了更实用的机制。我们的结果表明，在大规模音频生成模型时代，DI是版权保护和数据集问责制的一个有前景的方向。



## **9. Towards Irreversible Machine Unlearning for Diffusion Models**

面向扩散模型的不可逆机器遗忘研究 cs.LG

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03564v1) [paper-pdf](https://arxiv.org/pdf/2512.03564v1)

**Confidence**: 0.95

**Authors**: Xun Yuan, Zilong Zhao, Jiayu Li, Aryan Pasikhani, Prosanta Gope, Biplab Sikdar

**Abstract**: Diffusion models are renowned for their state-of-the-art performance in generating synthetic images. However, concerns related to safety, privacy, and copyright highlight the need for machine unlearning, which can make diffusion models forget specific training data and prevent the generation of sensitive or unwanted content. Current machine unlearning methods for diffusion models are primarily designed for conditional diffusion models and focus on unlearning specific data classes or features. Among these methods, finetuning-based machine unlearning methods are recognized for their efficiency and effectiveness, which update the parameters of pre-trained diffusion models by minimizing carefully designed loss functions. However, in this paper, we propose a novel attack named Diffusion Model Relearning Attack (DiMRA), which can reverse the finetuning-based machine unlearning methods, posing a significant vulnerability of this kind of technique. Without prior knowledge of the unlearning elements, DiMRA optimizes the unlearned diffusion model on an auxiliary dataset to reverse the unlearning, enabling the model to regenerate previously unlearned elements. To mitigate this vulnerability, we propose a novel machine unlearning method for diffusion models, termed as Diffusion Model Unlearning by Memorization (DiMUM). Unlike traditional methods that focus on forgetting, DiMUM memorizes alternative data or features to replace targeted unlearning data or features in order to prevent generating such elements. In our experiments, we demonstrate the effectiveness of DiMRA in reversing state-of-the-art finetuning-based machine unlearning methods for diffusion models, highlighting the need for more robust solutions. We extensively evaluate DiMUM, demonstrating its superior ability to preserve the generative performance of diffusion models while enhancing robustness against DiMRA.

摘要: 扩散模型以其在生成合成图像方面的最先进性能而闻名。然而，与安全、隐私和版权相关的问题凸显了机器遗忘的必要性，这可以使扩散模型忘记特定的训练数据，并防止生成敏感或不想要的内容。当前针对扩散模型的机器遗忘方法主要设计用于条件扩散模型，并专注于遗忘特定的数据类别或特征。在这些方法中，基于微调的机器遗忘方法因其效率和有效性而受到认可，它们通过最小化精心设计的损失函数来更新预训练扩散模型的参数。然而，在本文中，我们提出了一种名为扩散模型再学习攻击（DiMRA）的新型攻击方法，它可以逆转基于微调的机器遗忘方法，揭示了此类技术的重大漏洞。在不知道遗忘元素的情况下，DiMRA在辅助数据集上优化已遗忘的扩散模型以逆转遗忘过程，使模型能够重新生成先前遗忘的元素。为了缓解这一漏洞，我们提出了一种针对扩散模型的新型机器遗忘方法，称为基于记忆的扩散模型遗忘（DiMUM）。与专注于遗忘的传统方法不同，DiMUM通过记忆替代数据或特征来替换目标遗忘数据或特征，从而防止生成此类元素。在我们的实验中，我们证明了DiMRA在逆转最先进的基于微调的扩散模型机器遗忘方法方面的有效性，强调了需要更鲁棒的解决方案。我们广泛评估了DiMUM，展示了其在保持扩散模型生成性能的同时，增强了对DiMRA攻击的鲁棒性。



## **10. BadBlocks: Lightweight and Stealthy Backdoor Threat in Text-to-Image Diffusion Models**

BadBlocks：文本到图像扩散模型中的轻量级隐蔽后门威胁 cs.CR

**SubmitDate**: 2025-12-30    [abs](http://arxiv.org/abs/2508.03221v4) [paper-pdf](https://arxiv.org/pdf/2508.03221v4)

**Confidence**: 0.95

**Authors**: Yu Pan, Jiahao Chen, Wenjie Wang, Bingrong Dai, Junjun Yang

**Abstract**: Diffusion models have recently achieved remarkable success in image generation, yet growing evidence shows their vulnerability to backdoor attacks, where adversaries implant covert triggers to manipulate outputs. While existing defenses can detect many such attacks via visual inspection and neural network-based analysis, we identify a more lightweight and stealthy threat, termed BadBlocks. BadBlocks selectively contaminates specific blocks within the UNet architecture while preserving the normal behavior of the remaining components. Compared with prior methods, it requires only about 30% of the computation and 20% of the GPU time, yet achieves high attack success rates with minimal perceptual degradation. Extensive experiments demonstrate that BadBlocks can effectively evade state-of-the-art defenses, particularly attention-based detection frameworks. Ablation studies further reveal that effective backdoor injection does not require fine-tuning the entire network and highlight the critical role of certain layers in backdoor mapping. Overall, BadBlocks substantially lowers the barrier for backdooring large-scale diffusion models, even on consumer-grade GPUs.

摘要: 扩散模型近期在图像生成领域取得了显著成功，但越来越多的证据表明其易受后门攻击，攻击者可通过植入隐蔽触发器来操纵输出。虽然现有防御方法能通过视觉检查和基于神经网络的分析检测许多此类攻击，但我们发现了一种更为轻量级和隐蔽的威胁，称为BadBlocks。BadBlocks选择性地污染UNet架构中的特定模块，同时保持其余组件的正常行为。与先前方法相比，它仅需约30%的计算量和20%的GPU时间，却能以最小的感知退化实现高攻击成功率。大量实验表明，BadBlocks能有效规避最先进的防御机制，特别是基于注意力的检测框架。消融研究进一步揭示，有效的后门注入无需微调整个网络，并突显了某些层在后门映射中的关键作用。总体而言，BadBlocks显著降低了在消费级GPU上对大规模扩散模型植入后门的门槛。



## **11. Low Resource Reconstruction Attacks Through Benign Prompts**

通过良性提示实现低资源重建攻击 cs.LG

**SubmitDate**: 2026-01-07    [abs](http://arxiv.org/abs/2507.07947v3) [paper-pdf](https://arxiv.org/pdf/2507.07947v3)

**Confidence**: 0.95

**Authors**: Sol Yarkoni, Mahmood Sharif, Roi Livni

**Abstract**: Recent advances in generative models, such as diffusion models, have raised concerns related to privacy, copyright infringement, and data stewardship. To better understand and control these risks, prior work has introduced techniques and attacks that reconstruct images, or parts of images, from training data. While these results demonstrate that training data can be recovered, existing methods often rely on high computational resources, partial access to the training set, or carefully engineered prompts.   In this work, we present a new attack that requires low resources, assumes little to no access to the training data, and identifies seemingly benign prompts that can lead to potentially risky image reconstruction. We further show that such reconstructions may occur unintentionally, even for users without specialized knowledge. For example, we observe that for one existing model, the prompt ``blue Unisex T-Shirt'' generates the face of a real individual. Moreover, by combining the identified vulnerabilities with real-world prompt data, we discover prompts that reproduce memorized visual elements.   Our approach builds on insights from prior work and leverages domain knowledge to expose a fundamental vulnerability arising from the use of scraped e-commerce data, where templated layouts and images are closely tied to pattern-like textual prompts.   The code for our attack is publicly available at https://github.com/TheSolY/lr-tmi.

摘要: 生成模型（如扩散模型）的最新进展引发了关于隐私、版权侵权和数据管理的担忧。为了更好地理解和控制这些风险，先前的研究引入了从训练数据中重建图像或部分图像的技术和攻击方法。虽然这些结果表明训练数据可以被恢复，但现有方法通常依赖于高计算资源、对训练集的部分访问权限或精心设计的提示。在本研究中，我们提出了一种新的攻击方法，该方法所需资源较低，几乎不需要访问训练数据，并能识别看似良性的提示，这些提示可能导致潜在风险的图像重建。我们进一步证明，即使对于没有专业知识的用户，此类重建也可能无意中发生。例如，我们观察到在某个现有模型中，提示“蓝色中性T恤”会生成真实人物的面部图像。此外，通过将已识别的漏洞与现实世界的提示数据相结合，我们发现了能够重现记忆视觉元素的提示。我们的方法基于先前研究的见解，并利用领域知识揭示了因使用抓取的电子商务数据而产生的基本漏洞，其中模板化布局和图像与模式化文本提示紧密关联。我们的攻击代码已在 https://github.com/TheSolY/lr-tmi 公开提供。



## **12. Towards Dataset Copyright Evasion Attack against Personalized Text-to-Image Diffusion Models**

面向个性化文本到图像扩散模型的数据集版权规避攻击研究 cs.CV

Accepted by IEEE Transactions on Information Forensics and Security

**SubmitDate**: 2025-12-23    [abs](http://arxiv.org/abs/2505.02824v2) [paper-pdf](https://arxiv.org/pdf/2505.02824v2)

**Confidence**: 0.95

**Authors**: Kuofeng Gao, Yufei Zhu, Yiming Li, Jiawang Bai, Yong Yang, Zhifeng Li, Shu-Tao Xia

**Abstract**: Text-to-image (T2I) diffusion models enable high-quality image generation conditioned on textual prompts. However, fine-tuning these pre-trained models for personalization raises concerns about unauthorized dataset usage. To address this issue, dataset ownership verification (DOV) has recently been proposed, which embeds watermarks into fine-tuning datasets via backdoor techniques. These watermarks remain dormant on benign samples but produce owner-specified outputs when triggered. Despite its promise, the robustness of DOV against copyright evasion attacks (CEA) remains unexplored. In this paper, we investigate how adversaries can circumvent these mechanisms, enabling models trained on watermarked datasets to bypass ownership verification. We begin by analyzing the limitations of potential attacks achieved by backdoor removal, including TPD and T2IShield. In practice, TPD suffers from inconsistent effectiveness due to randomness, while T2IShield fails when watermarks are embedded as local image patches. To this end, we introduce CEAT2I, the first CEA specifically targeting DOV in T2I diffusion models. CEAT2I consists of three stages: (1) motivated by the observation that T2I models converge faster on watermarked samples with respect to intermediate features rather than training loss, we reliably detect watermarked samples; (2) we iteratively ablate tokens from the prompts of detected samples and monitor feature shifts to identify trigger tokens; and (3) we apply a closed-form concept erasure method to remove the injected watermarks. Extensive experiments demonstrate that CEAT2I effectively evades state-of-the-art DOV mechanisms while preserving model performance. The code is available at https://github.com/csyufei/CEAT2I.

摘要: 文本到图像（T2I）扩散模型能够根据文本提示生成高质量图像。然而，对这些预训练模型进行个性化微调引发了关于未经授权使用数据集的担忧。为解决此问题，近期提出了数据集所有权验证（DOV）方法，该方法通过后门技术将水印嵌入微调数据集中。这些水印在良性样本上保持休眠状态，但在触发时会产生所有者指定的输出。尽管前景广阔，但DOV对抗版权规避攻击（CEA）的鲁棒性尚未得到探索。本文研究了攻击者如何规避这些机制，使在带水印数据集上训练的模型能够绕过所有权验证。我们首先分析了通过后门移除实现潜在攻击（包括TPD和T2IShield）的局限性。实践中，TPD因随机性导致效果不稳定，而T2IShield在水印以局部图像块形式嵌入时会失效。为此，我们提出了CEAT2I——首个专门针对T2I扩散模型中DOV的CEA方法。CEAT2I包含三个阶段：（1）基于观察到T2I模型在带水印样本上相对于中间特征（而非训练损失）收敛更快的现象，我们可靠地检测带水印样本；（2）迭代消融检测样本提示中的token，并通过监测特征偏移来识别触发token；（3）应用闭式概念擦除方法移除注入的水印。大量实验表明，CEAT2I在保持模型性能的同时，能有效规避最先进的DOV机制。代码发布于https://github.com/csyufei/CEAT2I。



## **13. Dynamic Attention Analysis for Backdoor Detection in Text-to-Image Diffusion Models**

基于动态注意力分析的文本到图像扩散模型后门检测 cs.CV

Accepted by TPAMI

**SubmitDate**: 2025-12-01    [abs](http://arxiv.org/abs/2504.20518v3) [paper-pdf](https://arxiv.org/pdf/2504.20518v3)

**Confidence**: 0.95

**Authors**: Zhongqi Wang, Jie Zhang, Shiguang Shan, Xilin Chen

**Abstract**: Recent studies have revealed that text-to-image diffusion models are vulnerable to backdoor attacks, where attackers implant stealthy textual triggers to manipulate model outputs. Previous backdoor detection methods primarily focus on the static features of backdoor samples. However, a vital property of diffusion models is their inherent dynamism. This study introduces a novel backdoor detection perspective named Dynamic Attention Analysis (DAA), showing that these dynamic characteristics serve as better indicators for backdoor detection. Specifically, by examining the dynamic evolution of cross-attention maps, we observe that backdoor samples exhibit distinct feature evolution patterns at the $<$EOS$>$ token compared to benign samples. To quantify these dynamic anomalies, we first introduce DAA-I, which treats the tokens' attention maps as spatially independent and measures dynamic feature using the Frobenius norm. Furthermore, to better capture the interactions between attention maps and refine the feature, we propose a dynamical system-based approach, referred to as DAA-S. This model formulates the spatial correlations among attention maps using a graph-based state equation and we theoretically analyze the global asymptotic stability of this method. Extensive experiments across six representative backdoor attack scenarios demonstrate that our approach significantly surpasses existing detection methods, achieving an average F1 Score of 79.27% and an AUC of 86.27%. The code is available at https://github.com/Robin-WZQ/DAA.

摘要: 近期研究表明，文本到图像扩散模型易受后门攻击，攻击者可通过植入隐蔽的文本触发器来操纵模型输出。现有的后门检测方法主要关注后门样本的静态特征，但扩散模型的关键特性在于其固有的动态性。本研究提出了一种名为动态注意力分析（DAA）的新型后门检测视角，证明这些动态特征能更有效地指示后门存在。具体而言，通过分析交叉注意力图的动态演化过程，我们发现后门样本在<$<$EOS$>$>标记处表现出与良性样本截然不同的特征演化模式。为量化这些动态异常，我们首先提出DAA-I方法，将各标记的注意力图视为空间独立，并使用Frobenius范数度量动态特征。进一步地，为更好地捕捉注意力图间的相互作用并优化特征提取，我们提出基于动力系统的方法DAA-S。该方法通过基于图的状态方程建模注意力图间的空间相关性，并从理论上分析了该方法的全局渐近稳定性。在六种典型后门攻击场景中的大量实验表明，我们的方法显著优于现有检测技术，平均F1分数达到79.27%，AUC达到86.27%。代码已开源：https://github.com/Robin-WZQ/DAA。



## **14. Targeted Data Protection for Diffusion Model by Matching Training Trajectory**

通过匹配训练轨迹实现扩散模型的定向数据保护 cs.AI

AAAI 2026

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2512.10433v1) [paper-pdf](https://arxiv.org/pdf/2512.10433v1)

**Confidence**: 0.95

**Authors**: Hojun Lee, Mijin Koo, Yeji Song, Nojun Kwak

**Abstract**: Recent advancements in diffusion models have made fine-tuning text-to-image models for personalization increasingly accessible, but have also raised significant concerns regarding unauthorized data usage and privacy infringement. Current protection methods are limited to passively degrading image quality, failing to achieve stable control. While Targeted Data Protection (TDP) offers a promising paradigm for active redirection toward user-specified target concepts, existing TDP attempts suffer from poor controllability due to snapshot-matching approaches that fail to account for complete learning dynamics. We introduce TAFAP (Trajectory Alignment via Fine-tuning with Adversarial Perturbations), the first method to successfully achieve effective TDP by controlling the entire training trajectory. Unlike snapshot-based methods whose protective influence is easily diluted as training progresses, TAFAP employs trajectory-matching inspired by dataset distillation to enforce persistent, verifiable transformations throughout fine-tuning. We validate our method through extensive experiments, demonstrating the first successful targeted transformation in diffusion models with simultaneous control over both identity and visual patterns. TAFAP significantly outperforms existing TDP attempts, achieving robust redirection toward target concepts while maintaining high image quality. This work enables verifiable safeguards and provides a new framework for controlling and tracing alterations in diffusion model outputs.

摘要: 扩散模型的最新进展使得针对个性化需求微调文本到图像模型变得更加便捷，但也引发了关于未经授权数据使用和隐私侵犯的重大担忧。当前保护方法仅限于被动降低图像质量，无法实现稳定控制。虽然定向数据保护（TDP）为主动重定向至用户指定目标概念提供了有前景的范式，但现有TDP尝试因采用快照匹配方法而可控性较差，未能考虑完整的学习动态。我们提出了TAFAP（基于对抗性扰动的微调轨迹对齐），这是首个通过控制整个训练轨迹成功实现有效TDP的方法。与基于快照的方法（其保护效果易随训练进程被稀释）不同，TAFAP采用受数据集蒸馏启发的轨迹匹配技术，在整个微调过程中强制执行持久且可验证的转换。我们通过大量实验验证了该方法，首次在扩散模型中成功实现了对身份和视觉模式同时控制的定向转换。TAFAP显著优于现有TDP尝试，在保持高图像质量的同时实现了向目标概念的稳健重定向。这项工作为扩散模型输出提供了可验证的防护机制，并为控制和追踪模型修改提供了新框架。



## **15. Bob's Confetti: Phonetic Memorization Attacks in Music and Video Generation**

Bob的彩屑：音乐与视频生成中的语音记忆攻击 cs.SD

**SubmitDate**: 2025-10-29    [abs](http://arxiv.org/abs/2507.17937v3) [paper-pdf](https://arxiv.org/pdf/2507.17937v3)

**Confidence**: 0.95

**Authors**: Jaechul Roh, Zachary Novack, Yuefeng Peng, Niloofar Mireshghallah, Taylor Berg-Kirkpatrick, Amir Houmansadr

**Abstract**: Generative AI systems for music and video commonly use text-based filters to prevent the regurgitation of copyrighted material. We expose a fundamental flaw in this approach by introducing Adversarial PhoneTic Prompting (APT), a novel attack that bypasses these safeguards by exploiting phonetic memorization. The APT attack replaces iconic lyrics with homophonic but semantically unrelated alternatives (e.g., "mom's spaghetti" becomes "Bob's confetti"), preserving acoustic structure while altering meaning; we identify high-fidelity phonetic matches using CMU pronouncing dictionary. We demonstrate that leading Lyrics-to-Song (L2S) models like SUNO and YuE regenerate songs with striking melodic and rhythmic similarity to their copyrighted originals when prompted with these altered lyrics. More surprisingly, this vulnerability extends across modalities. When prompted with phonetically modified lyrics from a song, a Text-to-Video (T2V) model like Veo 3 reconstructs visual scenes from the original music video-including specific settings and character archetypes-despite the absence of any visual cues in the prompt. Our findings reveal that models memorize deep, structural patterns tied to acoustics, not just verbatim text. This phonetic-to-visual leakage represents a critical vulnerability in transcript-conditioned generative models, rendering simple copyright filters ineffective and raising urgent concerns about the secure deployment of multimodal AI systems. Demo examples are available at our project page (https://jrohsc.github.io/music_attack/).

摘要: 音乐和视频生成式AI系统通常使用基于文本的过滤器来防止受版权保护内容的复现。我们通过引入对抗性语音提示（APT）攻击，揭示了这种方法的一个根本缺陷。APT是一种新颖的攻击方式，通过利用语音记忆绕过这些安全防护。该攻击将标志性歌词替换为同音异义但语义无关的替代词（例如“mom's spaghetti”变为“Bob's confetti”），在保留声学结构的同时改变语义；我们使用CMU发音词典识别高保真度的语音匹配。研究表明，当使用这些修改后的歌词进行提示时，SUNO和YuE等主流歌词转歌曲（L2S）模型生成的歌曲与其受版权保护的原作在旋律和节奏上具有惊人的相似性。更令人惊讶的是，这种漏洞跨越了模态边界。当使用歌曲的语音修改歌词进行提示时，Veo 3等文本转视频（T2V）模型能够重建原始音乐视频中的视觉场景——包括特定场景和角色原型——尽管提示中没有任何视觉线索。我们的发现表明，模型记忆的是与声学相关的深层结构模式，而不仅仅是逐字文本。这种语音到视觉的泄漏揭示了转录条件生成模型的关键漏洞，使得简单的版权过滤器失效，并对多模态AI系统的安全部署提出了紧迫关切。演示示例可在项目页面（https://jrohsc.github.io/music_attack/）查看。



## **16. Adversarial Attacks and Robust Defenses in Speaker Embedding based Zero-Shot Text-to-Speech System**

基于说话人嵌入的零样本文本到语音系统中的对抗攻击与鲁棒防御 eess.AS

**SubmitDate**: 2025-10-05    [abs](http://arxiv.org/abs/2410.04017v2) [paper-pdf](https://arxiv.org/pdf/2410.04017v2)

**Confidence**: 0.95

**Authors**: Ze Li, Yao Shi, Yunfei Xu, Ming Li

**Abstract**: Speaker embedding based zero-shot Text-to-Speech (TTS) systems enable high-quality speech synthesis for unseen speakers using minimal data. However, these systems are vulnerable to adversarial attacks, where an attacker introduces imperceptible perturbations to the original speaker's audio waveform, leading to synthesized speech sounds like another person. This vulnerability poses significant security risks, including speaker identity spoofing and unauthorized voice manipulation. This paper investigates two primary defense strategies to address these threats: adversarial training and adversarial purification. Adversarial training enhances the model's robustness by integrating adversarial examples during the training process, thereby improving resistance to such attacks. Adversarial purification, on the other hand, employs diffusion probabilistic models to revert adversarially perturbed audio to its clean form. Experimental results demonstrate that these defense mechanisms can significantly reduce the impact of adversarial perturbations, enhancing the security and reliability of speaker embedding based zero-shot TTS systems in adversarial environments.

摘要: 基于说话人嵌入的零样本文本到语音（TTS）系统能够利用少量数据为未见过的说话人生成高质量语音。然而，这些系统易受对抗攻击的影响，攻击者通过对原始说话人音频波形引入难以察觉的扰动，使合成语音听起来像另一个人。这种脆弱性带来了重大的安全风险，包括说话人身份欺骗和未经授权的语音操纵。本文研究了两种主要的防御策略来应对这些威胁：对抗训练和对抗净化。对抗训练通过在训练过程中融入对抗样本来增强模型的鲁棒性，从而提高对此类攻击的抵抗力。而对抗净化则采用扩散概率模型将受对抗扰动的音频恢复至原始纯净形式。实验结果表明，这些防御机制能显著降低对抗扰动的影响，增强基于说话人嵌入的零样本TTS系统在对抗环境中的安全性和可靠性。



## **17. Backdoor Attacks on Open Vocabulary Object Detectors via Multi-Modal Prompt Tuning**

基于多模态提示调优的开放词汇目标检测器后门攻击 cs.CV

Accepted to AAAI 2026

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2511.12735v2) [paper-pdf](https://arxiv.org/pdf/2511.12735v2)

**Confidence**: 0.85

**Authors**: Ankita Raj, Chetan Arora

**Abstract**: Open-vocabulary object detectors (OVODs) unify vision and language to detect arbitrary object categories based on text prompts, enabling strong zero-shot generalization to novel concepts. As these models gain traction in high-stakes applications such as robotics, autonomous driving, and surveillance, understanding their security risks becomes crucial. In this work, we conduct the first study of backdoor attacks on OVODs and reveal a new attack surface introduced by prompt tuning. We propose TrAP (Trigger-Aware Prompt tuning), a multi-modal backdoor injection strategy that jointly optimizes prompt parameters in both image and text modalities along with visual triggers. TrAP enables the attacker to implant malicious behavior using lightweight, learnable prompt tokens without retraining the base model weights, thus preserving generalization while embedding a hidden backdoor. We adopt a curriculum-based training strategy that progressively shrinks the trigger size, enabling effective backdoor activation using small trigger patches at inference. Experiments across multiple datasets show that TrAP achieves high attack success rates for both object misclassification and object disappearance attacks, while also improving clean image performance on downstream datasets compared to the zero-shot setting. Code: https://github.com/rajankita/TrAP

摘要: 开放词汇目标检测器（OVODs）通过融合视觉与语言模态，能够基于文本提示检测任意对象类别，实现了对新概念的强大零样本泛化能力。随着这类模型在机器人、自动驾驶和监控等高风险应用中的普及，理解其安全风险变得至关重要。本研究首次对OVODs的后门攻击进行系统性探索，揭示了提示调优技术引入的新型攻击面。我们提出TrAP（触发感知提示调优）——一种多模态后门注入策略，通过联合优化图像与文本模态的提示参数及视觉触发器，使攻击者能够利用轻量级可学习提示令牌植入恶意行为，无需重新训练基础模型权重，在保持泛化能力的同时嵌入隐蔽后门。采用渐进式课程训练策略逐步缩小触发器尺寸，实现在推理阶段使用微小触发补丁即可有效激活后门。跨数据集实验表明，TrAP在目标误分类和目标消失攻击中均实现高成功率，同时相比零样本设置在下游数据集上提升了干净图像的检测性能。代码：https://github.com/rajankita/TrAP



## **18. RAVEN: Erasing Invisible Watermarks via Novel View Synthesis**

RAVEN：通过新颖视角合成消除隐形水印 cs.CV

13 pages

**SubmitDate**: 2026-01-13    [abs](http://arxiv.org/abs/2601.08832v1) [paper-pdf](https://arxiv.org/pdf/2601.08832v1)

**Confidence**: 0.85

**Authors**: Fahad Shamshad, Nils Lukas, Karthik Nandakumar

**Abstract**: Invisible watermarking has become a critical mechanism for authenticating AI-generated image content, with major platforms deploying watermarking schemes at scale. However, evaluating the vulnerability of these schemes against sophisticated removal attacks remains essential to assess their reliability and guide robust design. In this work, we expose a fundamental vulnerability in invisible watermarks by reformulating watermark removal as a view synthesis problem. Our key insight is that generating a perceptually consistent alternative view of the same semantic content, akin to re-observing a scene from a shifted perspective, naturally removes the embedded watermark while preserving visual fidelity. This reveals a critical gap: watermarks robust to pixel-space and frequency-domain attacks remain vulnerable to semantic-preserving viewpoint transformations. We introduce a zero-shot diffusion-based framework that applies controlled geometric transformations in latent space, augmented with view-guided correspondence attention to maintain structural consistency during reconstruction. Operating on frozen pre-trained models without detector access or watermark knowledge, our method achieves state-of-the-art watermark suppression across 15 watermarking methods--outperforming 14 baseline attacks while maintaining superior perceptual quality across multiple datasets.

摘要: 隐形水印已成为认证AI生成图像内容的关键机制，各大平台已大规模部署水印方案。然而，评估这些方案面对复杂去除攻击的脆弱性，对于评估其可靠性并指导稳健设计至关重要。本研究通过将水印去除重新定义为视角合成问题，揭示了隐形水印的根本性脆弱性。我们的核心洞见是：生成具有感知一致性的同一语义内容的替代视角（类似于从偏移视角重新观察场景），能在保持视觉保真度的同时自然去除嵌入水印。这揭示了一个关键缺陷：对像素空间和频域攻击具有鲁棒性的水印，在保持语义的视角变换面前依然脆弱。我们提出了一种基于零样本扩散的框架，在潜在空间施加受控几何变换，并通过视角引导的对应注意力增强以保持重建过程中的结构一致性。该方法在冻结的预训练模型上运行，无需水印检测器或先验知识，在15种水印方法上实现了最先进的水印抑制效果——在多个数据集上超越14种基线攻击方法的同时保持了卓越的感知质量。



## **19. Prototypicality Bias Reveals Blindspots in Multimodal Evaluation Metrics**

原型性偏差揭示多模态评估指标的盲点 cs.CV

First version

**SubmitDate**: 2026-01-10    [abs](http://arxiv.org/abs/2601.04946v2) [paper-pdf](https://arxiv.org/pdf/2601.04946v2)

**Confidence**: 0.85

**Authors**: Subhadeep Roy, Gagan Bhatia, Steffen Eger

**Abstract**: Automatic metrics are now central to evaluating text-to-image models, often substituting for human judgment in benchmarking and large-scale filtering. However, it remains unclear whether these metrics truly prioritize semantic correctness or instead favor visually and socially prototypical images learned from biased data distributions. We identify and study prototypicality bias as a systematic failure mode in multimodal evaluation. We introduce a controlled contrastive benchmark ProtoBias (Prototypical Bias), spanning Animals, Objects, and Demography images, where semantically correct but non-prototypical images are paired with subtly incorrect yet prototypical adversarial counterparts. This setup enables a directional evaluation of whether metrics follow textual semantics or default to prototypes. Our results show that widely used metrics, including CLIPScore, PickScore, and VQA-based scores, frequently misrank these pairs, while even LLM-as-Judge systems exhibit uneven robustness in socially grounded cases. Human evaluations consistently favour semantic correctness with larger decision margins. Motivated by these findings, we propose ProtoScore, a robust 7B-parameter metric that substantially reduces failure rates and suppresses misranking, while running at orders of magnitude faster than the inference time of GPT-5, approaching the robustness of much larger closed-source judges.

摘要: 自动评估指标现已成为评估文本到图像模型的核心工具，常在基准测试和大规模筛选中替代人类判断。然而，这些指标究竟真正优先考虑语义正确性，还是更倾向于从有偏数据分布中学到的视觉和社会原型图像，目前尚不明确。我们识别并研究了多模态评估中的原型性偏差这一系统性失效模式。我们引入了一个受控对比基准ProtoBias（原型偏差），涵盖动物、物体和人口统计图像，其中语义正确但非原型的图像与微妙错误但原型的对抗性对应图像配对。这种设置能够定向评估指标是遵循文本语义还是默认选择原型。我们的结果表明，包括CLIPScore、PickScore和基于VQA的评分在内的广泛使用指标经常对这些配对进行错误排序，而即使是LLM-as-Judge系统在社会基础案例中也表现出不均匀的鲁棒性。人类评估始终更倾向于语义正确性，且决策边界更大。基于这些发现，我们提出了ProtoScore，这是一个鲁棒的70亿参数指标，能显著降低失败率并抑制错误排序，同时运行速度比GPT-5的推理时间快数个数量级，接近更大规模闭源评估器的鲁棒性。



## **20. Cryptanalysis of Pseudorandom Error-Correcting Codes**

伪随机纠错码的密码分析 cs.CR

**SubmitDate**: 2025-12-19    [abs](http://arxiv.org/abs/2512.17310v1) [paper-pdf](https://arxiv.org/pdf/2512.17310v1)

**Confidence**: 0.85

**Authors**: Tianrui Wang, Anyu Wang, Tianshuo Cong, Delong Ran, Jinyuan Liu, Xiaoyun Wang

**Abstract**: Pseudorandom error-correcting codes (PRC) is a novel cryptographic primitive proposed at CRYPTO 2024. Due to the dual capability of pseudorandomness and error correction, PRC has been recognized as a promising foundational component for watermarking AI-generated content. However, the security of PRC has not been thoroughly analyzed, especially with concrete parameters or even in the face of cryptographic attacks. To fill this gap, we present the first cryptanalysis of PRC. We first propose three attacks to challenge the undetectability and robustness assumptions of PRC. Among them, two attacks aim to distinguish PRC-based codewords from plain vectors, and one attack aims to compromise the decoding process of PRC. Our attacks successfully undermine the claimed security guarantees across all parameter configurations. Notably, our attack can detect the presence of a watermark with overwhelming probability at a cost of $2^{22}$ operations. We also validate our approach by attacking real-world large generative models such as DeepSeek and Stable Diffusion. To mitigate our attacks, we further propose three defenses to enhance the security of PRC, including parameter suggestions, implementation suggestions, and constructing a revised key generation algorithm. Our proposed revised key generation function effectively prevents the occurrence of weak keys. However, we highlight that the current PRC-based watermarking scheme still cannot achieve a 128-bit security under our parameter suggestions due to the inherent configurations of large generative models, such as the maximum output length of large language models.

摘要: 伪随机纠错码（PRC）是CRYPTO 2024会议上提出的一种新型密码学原语。凭借其兼具伪随机性和纠错能力的双重特性，PRC已被视为AI生成内容水印技术中极具前景的基础组件。然而，PRC的安全性尚未得到深入分析，特别是在具体参数设置下甚至面临密码攻击时。为填补这一空白，我们首次对PRC进行了密码分析。首先提出三种攻击方法，挑战PRC的不可检测性和鲁棒性假设：其中两种攻击旨在区分基于PRC的码字与普通向量，另一种攻击则针对PRC的解码过程。我们的攻击成功破坏了所有参数配置下宣称的安全保证。值得注意的是，仅需2^22次运算即可以压倒性概率检测水印存在。我们还通过攻击DeepSeek和Stable Diffusion等实际大型生成模型验证了方法的有效性。为应对这些攻击，我们进一步提出三项防御措施以增强PRC安全性，包括参数建议、实施建议以及构建改进的密钥生成算法。我们提出的改进密钥生成函数能有效防止弱密钥出现。但需要指出的是，由于大型生成模型（如大语言模型的最大输出长度）的固有配置限制，基于PRC的水印方案在当前参数建议下仍无法实现128比特安全强度。



## **21. Optimization-Guided Diffusion for Interactive Scene Generation**

优化引导的扩散模型用于交互式场景生成 cs.CV

**SubmitDate**: 2025-12-11    [abs](http://arxiv.org/abs/2512.07661v2) [paper-pdf](https://arxiv.org/pdf/2512.07661v2)

**Confidence**: 0.85

**Authors**: Shihao Li, Naisheng Ye, Tianyu Li, Kashyap Chitta, Tuo An, Peng Su, Boyang Wang, Haiou Liu, Chen Lv, Hongyang Li

**Abstract**: Realistic and diverse multi-agent driving scenes are crucial for evaluating autonomous vehicles, but safety-critical events which are essential for this task are rare and underrepresented in driving datasets. Data-driven scene generation offers a low-cost alternative by synthesizing complex traffic behaviors from existing driving logs. However, existing models often lack controllability or yield samples that violate physical or social constraints, limiting their usability. We present OMEGA, an optimization-guided, training-free framework that enforces structural consistency and interaction awareness during diffusion-based sampling from a scene generation model. OMEGA re-anchors each reverse diffusion step via constrained optimization, steering the generation towards physically plausible and behaviorally coherent trajectories. Building on this framework, we formulate ego-attacker interactions as a game-theoretic optimization in the distribution space, approximating Nash equilibria to generate realistic, safety-critical adversarial scenarios. Experiments on nuPlan and Waymo show that OMEGA improves generation realism, consistency, and controllability, increasing the ratio of physically and behaviorally valid scenes from 32.35% to 72.27% for free exploration capabilities, and from 11% to 80% for controllability-focused generation. Our approach can also generate $5\times$ more near-collision frames with a time-to-collision under three seconds while maintaining the overall scene realism.

摘要: 真实且多样的多智能体驾驶场景对于评估自动驾驶车辆至关重要，但该任务所需的安全关键事件在驾驶数据集中极为罕见且代表性不足。数据驱动的场景生成通过从现有驾驶日志中合成复杂的交通行为，提供了一种低成本的替代方案。然而，现有模型往往缺乏可控性，或产生违反物理或社会约束的样本，限制了其实用性。我们提出了OMEGA，一种优化引导、无需训练的框架，在基于扩散模型的场景生成采样过程中强制执行结构一致性和交互感知。OMEGA通过约束优化重新锚定每个反向扩散步骤，引导生成过程朝向物理合理且行为一致的轨迹。基于此框架，我们将自车-攻击者交互建模为分布空间中的博弈论优化问题，通过近似纳什均衡来生成真实的安全关键对抗场景。在nuPlan和Waymo数据集上的实验表明，OMEGA显著提升了生成的真实性、一致性和可控性：对于自由探索能力，物理和行为有效场景的比例从32.35%提升至72.27%；对于侧重可控性的生成任务，该比例从11%提升至80%。我们的方法还能在保持整体场景真实性的同时，生成时间碰撞小于三秒的近碰撞帧数增加5倍。



## **22. Rethinking Security in Semantic Communication: Latent Manipulation as a New Threat**

重新思考语义通信安全：潜在操纵作为一种新型威胁 cs.CR

8 pages, 6 figures

**SubmitDate**: 2025-12-03    [abs](http://arxiv.org/abs/2512.03361v1) [paper-pdf](https://arxiv.org/pdf/2512.03361v1)

**Confidence**: 0.85

**Authors**: Zhiyuan Xi, Kun Zhu

**Abstract**: Deep learning-based semantic communication (SemCom) has emerged as a promising paradigm for next-generation wireless networks, offering superior transmission efficiency by extracting and conveying task-relevant semantic latent representations rather than raw data. However, the openness of the wireless medium and the intrinsic vulnerability of semantic latent representations expose such systems to previously unrecognized security risks. In this paper, we uncover a fundamental latent-space vulnerability that enables Man-in-the-Middle (MitM) attacker to covertly manipulate the transmitted semantics while preserving the statistical properties of the transmitted latent representations. We first present a Diffusion-based Re-encoding Attack (DiR), wherein the attacker employs a diffusion model to synthesize an attacker-designed semantic variant, and re-encodes it into a valid latent representation compatible with the SemCom decoder. Beyond this model-dependent pathway, we further propose a model-agnostic and training-free Test-Time Adaptation Latent Manipulation attack (TTA-LM), in which the attacker perturbs and steers the intercepted latent representation toward an attacker-specified semantic target by leveraging the gradient of a target loss function. In contrast to diffusion-based manipulation, TTA-LM does not rely on any generative model and does not impose modality-specific or task-specific assumptions, thereby enabling efficient and broadly applicable latent-space tampering across diverse SemCom architectures. Extensive experiments on representative semantic communication architectures demonstrate that both attacks can significantly alter the decoded semantics while preserving natural latent-space distributions, making the attacks covert and difficult to detect.

摘要: 基于深度学习的语义通信已成为下一代无线网络的前沿范式，通过提取和传输任务相关的语义潜在表征而非原始数据，实现了卓越的传输效率。然而，无线介质的开放性和语义潜在表征的内在脆弱性，使此类系统面临先前未被认识的安全风险。本文揭示了一种根本性的潜在空间脆弱性，使得中间人攻击者能够在保持传输潜在表征统计特性的同时，隐蔽地操纵传输语义。我们首先提出基于扩散模型的重编码攻击，攻击者利用扩散模型合成攻击者设计的语义变体，并将其重新编码为与语义通信解码器兼容的有效潜在表征。超越这种模型依赖路径，我们进一步提出模型无关且无需训练的在测试时自适应潜在操纵攻击，攻击者通过利用目标损失函数的梯度，将截获的潜在表征扰动并导向攻击者指定的语义目标。与基于扩散的操纵相比，TTA-LM不依赖任何生成模型，也不施加模态特定或任务特定的假设，从而能够在多样化的语义通信架构中实现高效且广泛适用的潜在空间篡改。在代表性语义通信架构上的大量实验表明，两种攻击均能在保持自然潜在空间分布的同时显著改变解码语义，使得攻击具有隐蔽性且难以检测。



## **23. Counterfeit Answers: Adversarial Forgery against OCR-Free Document Visual Question Answering**

伪造答案：针对无OCR文档视觉问答的对抗性伪造攻击 cs.CV

**SubmitDate**: 2025-12-04    [abs](http://arxiv.org/abs/2512.04554v1) [paper-pdf](https://arxiv.org/pdf/2512.04554v1)

**Confidence**: 0.85

**Authors**: Marco Pintore, Maura Pintor, Dimosthenis Karatzas, Battista Biggio

**Abstract**: Document Visual Question Answering (DocVQA) enables end-to-end reasoning grounded on information present in a document input. While recent models have shown impressive capabilities, they remain vulnerable to adversarial attacks. In this work, we introduce a novel attack scenario that aims to forge document content in a visually imperceptible yet semantically targeted manner, allowing an adversary to induce specific or generally incorrect answers from a DocVQA model. We develop specialized attack algorithms that can produce adversarially forged documents tailored to different attackers' goals, ranging from targeted misinformation to systematic model failure scenarios. We demonstrate the effectiveness of our approach against two end-to-end state-of-the-art models: Pix2Struct, a vision-language transformer that jointly processes image and text through sequence-to-sequence modeling, and Donut, a transformer-based model that directly extracts text and answers questions from document images. Our findings highlight critical vulnerabilities in current DocVQA systems and call for the development of more robust defenses.

摘要: 文档视觉问答（DocVQA）支持基于文档输入信息的端到端推理。尽管最新模型展现出卓越能力，但仍易受对抗攻击。本研究提出一种新颖的攻击场景，旨在以视觉不可察觉但语义目标明确的方式伪造文档内容，使攻击者能够诱导DocVQA模型产生特定或普遍错误的答案。我们开发了专用攻击算法，可针对不同攻击目标生成对抗性伪造文档，涵盖定向误导信息至系统性模型失效场景。我们在两种端到端最先进模型上验证了方法的有效性：Pix2Struct（通过序列到序列建模联合处理图像与文本的视觉语言Transformer）和Donut（直接从文档图像提取文本并回答问题的基于Transformer的模型）。我们的研究揭示了当前DocVQA系统的关键脆弱性，呼吁开发更强大的防御机制。



## **24. Exploiting Leaderboards for Large-Scale Distribution of Malicious Models**

利用排行榜大规模分发恶意模型 cs.LG

**SubmitDate**: 2025-07-11    [abs](http://arxiv.org/abs/2507.08983v1) [paper-pdf](https://arxiv.org/pdf/2507.08983v1)

**Confidence**: 0.85

**Authors**: Anshuman Suri, Harsh Chaudhari, Yuefeng Peng, Ali Naseh, Amir Houmansadr, Alina Oprea

**Abstract**: While poisoning attacks on machine learning models have been extensively studied, the mechanisms by which adversaries can distribute poisoned models at scale remain largely unexplored. In this paper, we shed light on how model leaderboards -- ranked platforms for model discovery and evaluation -- can serve as a powerful channel for adversaries for stealthy large-scale distribution of poisoned models. We present TrojanClimb, a general framework that enables injection of malicious behaviors while maintaining competitive leaderboard performance. We demonstrate its effectiveness across four diverse modalities: text-embedding, text-generation, text-to-speech and text-to-image, showing that adversaries can successfully achieve high leaderboard rankings while embedding arbitrary harmful functionalities, from backdoors to bias injection. Our findings reveal a significant vulnerability in the machine learning ecosystem, highlighting the urgent need to redesign leaderboard evaluation mechanisms to detect and filter malicious (e.g., poisoned) models, while exposing broader security implications for the machine learning community regarding the risks of adopting models from unverified sources.

摘要: 尽管针对机器学习模型的投毒攻击已被广泛研究，但攻击者大规模分发投毒模型的机制仍鲜有探索。本文揭示了模型排行榜——用于模型发现和评估的排名平台——如何成为攻击者隐秘大规模分发投毒模型的有力渠道。我们提出了TrojanClimb这一通用框架，能够在保持竞争力的排行榜性能的同时注入恶意行为。我们在文本嵌入、文本生成、文本转语音和文本转图像四种不同模态上验证了其有效性，表明攻击者能够在嵌入任意有害功能（从后门到偏见注入）的同时成功获得较高的排行榜排名。我们的研究揭示了机器学习生态系统中的一个重大漏洞，强调了重新设计排行榜评估机制以检测和过滤恶意（如投毒）模型的迫切需求，同时向机器学习社区揭示了从未经验证来源采用模型所带来的更广泛安全风险。



## **25. Multi-Faceted Multimodal Monosemanticity**

多模态单义性的多面性研究 cs.CV

**SubmitDate**: 2025-05-23    [abs](http://arxiv.org/abs/2502.14888v3) [paper-pdf](https://arxiv.org/pdf/2502.14888v3)

**Confidence**: 0.85

**Authors**: Hanqi Yan, Xiangxiang Cui, Lu Yin, Paul Pu Liang, Yulan He, Yifei Wang

**Abstract**: Humans experience the world through multiple modalities, such as, vision, language, and speech, making it natural to explore the commonality and distinctions among them. In this work, we take a data-driven approach to address this question by analyzing interpretable, monosemantic features extracted from deep multimodal models. Specifically, we investigate CLIP, a prominent visual-language representation model trained on massive image-text pairs. Building on prior research in single-modal interpretability, we develop a set of multi-modal interpretability tools and measures designed to disentangle and analyze features learned from CLIP. Specifically, we introduce the Modality Dominance Score (MDS) to attribute each CLIP feature to a specific modality. We then map CLIP features into a more interpretable space, enabling us to categorize them into three distinct classes: vision features (single-modal), language features (single-modal), and visual-language features (cross-modal). Interestingly, this data-driven categorization closely aligns with human intuitive understandings of different modalities. We further show that this modality decomposition can benefit multiple downstream tasks, including reducing bias in gender detection, generating cross-modal adversarial examples, and enabling modal-specific feature control in text-to-image generation. These results indicate that large-scale multimodal models, when equipped with task-agnostic interpretability tools, can offer valuable insights into the relationships between different data modalities.

摘要: 人类通过视觉、语言、语音等多种模态感知世界，因此探索这些模态之间的共性与差异具有天然必要性。本研究采用数据驱动方法，通过分析从深度多模态模型中提取的可解释单义特征来探讨这一问题。具体而言，我们研究了基于海量图文对训练的视觉-语言表征模型CLIP。在单模态可解释性研究的基础上，我们开发了一套多模态可解释性工具与度量方法，旨在解耦和分析CLIP学习到的特征。我们特别引入了模态主导分数（MDS）来将每个CLIP特征归因于特定模态。随后将CLIP特征映射到更具可解释性的空间，将其划分为三类：视觉特征（单模态）、语言特征（单模态）和视觉-语言特征（跨模态）。有趣的是，这种数据驱动的分类方式与人类对不同模态的直观理解高度吻合。我们进一步证明这种模态分解可提升多个下游任务的性能，包括降低性别检测中的偏差、生成跨模态对抗样本，以及在文生图任务中实现模态特异性特征控制。这些结果表明，配备任务无关可解释性工具的大规模多模态模型，能够为不同数据模态间的关系提供有价值的见解。



