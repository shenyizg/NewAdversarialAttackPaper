# Latest Adversarial Attack Papers
**update at 2025-05-16 16:51:10**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. The Ephemeral Threat: Assessing the Security of Algorithmic Trading Systems powered by Deep Learning**

短暂的威胁：评估深度学习支持的虚拟交易系统的安全性 cs.CR

To appear at ACM CODASPY 2025. 12 pages

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2505.10430v1) [paper-pdf](http://arxiv.org/pdf/2505.10430v1)

**Authors**: Advije Rizvani, Giovanni Apruzzese, Pavel Laskov

**Abstract**: We study the security of stock price forecasting using Deep Learning (DL) in computational finance. Despite abundant prior research on the vulnerability of DL to adversarial perturbations, such work has hitherto hardly addressed practical adversarial threat models in the context of DL-powered algorithmic trading systems (ATS). Specifically, we investigate the vulnerability of ATS to adversarial perturbations launched by a realistically constrained attacker. We first show that existing literature has paid limited attention to DL security in the financial domain, which is naturally attractive for adversaries. Then, we formalize the concept of ephemeral perturbations (EP), which can be used to stage a novel type of attack tailored for DL-based ATS. Finally, we carry out an end-to-end evaluation of our EP against a profitable ATS. Our results reveal that the introduction of small changes to the input stock prices not only (i) induces the DL model to behave incorrectly but also (ii) leads the whole ATS to make suboptimal buy/sell decisions, resulting in a worse financial performance of the targeted ATS.

摘要: 我们在计算金融中使用深度学习（DL）研究股价预测的安全性。尽管之前对DL对对抗性扰动的脆弱性进行了大量研究，但迄今为止，此类工作几乎没有解决DL驱动算法交易系统（ATS）背景下的实际对抗性威胁模型。具体来说，我们研究了ATS对现实约束的攻击者发起的对抗性扰动的脆弱性。我们首先表明，现有文献对金融领域的DL安全关注有限，这对对手来说自然有吸引力。然后，我们形式化了短暂扰动（EP）的概念，它可用于发起一种为基于DL的ATS量身定制的新型攻击。最后，我们针对盈利的ATS对我们的EP进行端到端评估。我们的结果表明，对输入股票价格的微小变化不仅（i）导致DL模型行为错误，而且（ii）导致整个ATS做出次优的买入/卖出决策，导致目标ATS的财务表现更差。



## **2. A Unified and Scalable Membership Inference Method for Visual Self-supervised Encoder via Part-aware Capability**

通过部件感知能力的视觉自我监督编码器统一且可扩展的隶属度推理方法 cs.CV

An extension of our ACM CCS2024 conference paper (arXiv:2404.02462).  We show the impacts of scaling from both data and model aspects on membership  inference for self-supervised visual encoders

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2505.10351v1) [paper-pdf](http://arxiv.org/pdf/2505.10351v1)

**Authors**: Jie Zhu, Jirong Zha, Ding Li, Leye Wang

**Abstract**: Self-supervised learning shows promise in harnessing extensive unlabeled data, but it also confronts significant privacy concerns, especially in vision. In this paper, we perform membership inference on visual self-supervised models in a more realistic setting: self-supervised training method and details are unknown for an adversary when attacking as he usually faces a black-box system in practice. In this setting, considering that self-supervised model could be trained by completely different self-supervised paradigms, e.g., masked image modeling and contrastive learning, with complex training details, we propose a unified membership inference method called PartCrop. It is motivated by the shared part-aware capability among models and stronger part response on the training data. Specifically, PartCrop crops parts of objects in an image to query responses within the image in representation space. We conduct extensive attacks on self-supervised models with different training protocols and structures using three widely used image datasets. The results verify the effectiveness and generalization of PartCrop. Moreover, to defend against PartCrop, we evaluate two common approaches, i.e., early stop and differential privacy, and propose a tailored method called shrinking crop scale range. The defense experiments indicate that all of them are effective. Finally, besides prototype testing on toy visual encoders and small-scale image datasets, we quantitatively study the impacts of scaling from both data and model aspects in a realistic scenario and propose a scalable PartCrop-v2 by introducing two structural improvements to PartCrop. Our code is at https://github.com/JiePKU/PartCrop.

摘要: 自我监督学习在利用大量未标记数据方面显示出希望，但它也面临着严重的隐私问题，尤其是在视觉方面。在本文中，我们在更现实的环境中对视觉自我监督模型进行隶属度推断：自我监督训练方法和细节对于对手来说是未知的，因为他在实践中通常会面临黑匣子系统。在这种情况下，考虑到自我监督模型可以通过完全不同的自我监督范式来训练，例如，针对掩蔽图像建模和对比学习，在复杂的训练细节下，我们提出了一种称为PartCrop的统一隶属度推理方法。它的动机是模型之间共享的部件感知能力以及对训练数据更强的部件响应。具体来说，PartCrop裁剪图像中对象的部分，以查询表示空间中图像内的响应。我们使用三个广泛使用的图像数据集，对具有不同训练协议和结构的自我监督模型进行广泛攻击。结果验证了PartCrop的有效性和通用性。此外，为了防御PartCrop，我们评估了两种常见的方法，即，早期停止和差异隐私，并提出了一种量身定制的方法，称为缩小作物规模范围。防御实验表明，这些方法都是有效的。最后，除了在玩具视觉编码器和小规模图像数据集上进行原型测试外，我们还从数据和模型两个方面定量研究了现实场景中缩放的影响，并通过对PartCrop引入两个结构改进，提出了一个可扩展的PartCrop v2。我们的代码位于https://github.com/JiePKU/PartCrop。



## **3. DAPPER: A Performance-Attack-Resilient Tracker for RowHammer Defense**

DAPPER：RowHammer防御的性能-攻击-弹性跟踪器 cs.CR

The initial version of this paper was submitted to MICRO 2024 on  April 18, 2024. The final version was presented at HPCA 2025  (https://hpca-conf.org/2025) and is 16 pages long, including references

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2501.18857v2) [paper-pdf](http://arxiv.org/pdf/2501.18857v2)

**Authors**: Jeonghyun Woo, Prashant J. Nair

**Abstract**: RowHammer vulnerabilities pose a significant threat to modern DRAM-based systems, where rapid activation of DRAM rows can induce bit-flips in neighboring rows. To mitigate this, state-of-the-art host-side RowHammer mitigations typically rely on shared counters or tracking structures. While these optimizations benefit benign applications, they are vulnerable to Performance Attacks (Perf-Attacks), where adversaries exploit shared structures to reduce DRAM bandwidth for co-running benign applications by increasing DRAM accesses for RowHammer counters or triggering repetitive refreshes required for the early reset of structures, significantly degrading performance.   In this paper, we propose secure hashing mechanisms to thwart adversarial attempts to capture the mapping of shared structures. We propose DAPPER, a novel low-cost tracker resilient to Perf-Attacks even at ultra-low RowHammer thresholds. We first present a secure hashing template in the form of DAPPER-S. We then develop DAPPER-H, an enhanced version of DAPPER-S, incorporating double-hashing, novel reset strategies, and mitigative refresh techniques. Our security analysis demonstrates the effectiveness of DAPPER-H against both RowHammer and Perf-Attacks. Experiments with 57 workloads from SPEC2006, SPEC2017, TPC, Hadoop, MediaBench, and YCSB show that, even at an ultra-low RowHammer threshold of 500, DAPPER-H incurs only a 0.9% slowdown in the presence of Perf-Attacks while using only 96KB of SRAM per 32GB of DRAM memory.

摘要: RowHammer漏洞对现代基于RAM的系统构成了重大威胁，其中快速激活RAM行可能会导致邻近行中的位翻转。为了缓解这种情况，最先进的主机端RowHammer缓解措施通常依赖于共享计数器或跟踪结构。虽然这些优化有利于良性应用程序，但它们很容易受到性能攻击（Perf-Attacks）的影响，对手利用共享结构通过增加RowHammer计数器的RAM访问或触发早期重置结构所需的重复刷新来减少并行运行的良性应用程序的RAM带宽，从而显着降低性能。   在本文中，我们提出了安全哈希机制来阻止捕获共享结构映射的对抗尝试。我们提出了DAPPER，这是一种新型的低成本跟踪器，即使在超低RowHammer阈值下也能抵御Perf-Attack。我们首先以DAPPER-S形式呈现一个安全哈希模板。然后，我们开发了DAPPER-H，这是DAPPER-S的增强版本，融合了双哈希、新颖的重置策略和缓解刷新技术。我们的安全分析证明了DAPPER-H对抗RowHammer和Perf-Attacks的有效性。对来自SPEC 2006、SPEC 2017、TBC、Hadoop、MediaBench和YCSB的57个工作负载进行的实验表明，即使在RowHammer的超低阈值500下，DAPPER-H在Perf-Attacks的存在下也只会减慢0.9%，而每32 GB内存仅使用96 KB的静态存储器。



## **4. Scaling Laws for Black box Adversarial Attacks**

黑匣子对抗攻击的缩放定律 cs.LG

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2411.16782v2) [paper-pdf](http://arxiv.org/pdf/2411.16782v2)

**Authors**: Chuan Liu, Huanran Chen, Yichi Zhang, Yinpeng Dong, Jun Zhu

**Abstract**: Adversarial examples usually exhibit good cross-model transferability, enabling attacks on black-box models with limited information about their architectures and parameters, which are highly threatening in commercial black-box scenarios. Model ensembling is an effective strategy to improve the transferability of adversarial examples by attacking multiple surrogate models. However, since prior studies usually adopt few models in the ensemble, there remains an open question of whether scaling the number of models can further improve black-box attacks. Inspired by the scaling law of large foundation models, we investigate the scaling laws of black-box adversarial attacks in this work. Through theoretical analysis and empirical evaluations, we conclude with clear scaling laws that using more surrogate models enhances adversarial transferability. Comprehensive experiments verify the claims on standard image classifiers, diverse defended models and multimodal large language models using various adversarial attack methods. Specifically, by scaling law, we achieve 90%+ transfer attack success rate on even proprietary models like GPT-4o. Further visualization indicates that there is also a scaling law on the interpretability and semantics of adversarial perturbations.

摘要: 对抗性示例通常表现出良好的跨模型可移植性，从而能够在有关其架构和参数的有限信息的情况下对黑匣子模型进行攻击，这在商业黑匣子场景中具有高度威胁性。模型集成是通过攻击多个代理模型来提高对抗性示例可移植性的有效策略。然而，由于之前的研究通常在整体中采用很少的模型，因此扩大模型数量是否可以进一步改善黑匣子攻击仍然是一个悬而未决的问题。受大型基金会模型缩放定律的启发，我们在这项工作中研究了黑匣子对抗攻击的缩放定律。通过理论分析和实证评估，我们得出了明确的缩放定律，即使用更多的代理模型增强了对抗性可转让性。全面的实验验证了标准图像分类器、多样化防御模型和使用各种对抗攻击方法的多模式大型语言模型的主张。具体来说，通过缩放定律，即使是GPT-4 o等专有模型，我们也能实现90%以上的传输攻击成功率。进一步的可视化表明，对抗性扰动的可解释性和语义也存在缩放定律。



## **5. Sybil-based Virtual Data Poisoning Attacks in Federated Learning**

联邦学习中基于Sybil的虚拟数据中毒攻击 cs.CR

7 pages, 6 figures, accepted by IEEE Codit 2025

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2505.09983v1) [paper-pdf](http://arxiv.org/pdf/2505.09983v1)

**Authors**: Changxun Zhu, Qilong Wu, Lingjuan Lyu, Shibei Xue

**Abstract**: Federated learning is vulnerable to poisoning attacks by malicious adversaries. Existing methods often involve high costs to achieve effective attacks. To address this challenge, we propose a sybil-based virtual data poisoning attack, where a malicious client generates sybil nodes to amplify the poisoning model's impact. To reduce neural network computational complexity, we develop a virtual data generation method based on gradient matching. We also design three schemes for target model acquisition, applicable to online local, online global, and offline scenarios. In simulation, our method outperforms other attack algorithms since our method can obtain a global target model under non-independent uniformly distributed data.

摘要: 联邦学习很容易受到恶意对手的毒害攻击。现有方法通常需要很高的成本来实现有效的攻击。为了应对这一挑战，我们提出了一种基于sybil的虚拟数据中毒攻击，其中恶意客户端生成sybil节点来放大中毒模型的影响。为了降低神经网络计算复杂性，我们开发了一种基于梯度匹配的虚拟数据生成方法。我们还设计了三种目标型号获取方案，适用于线上本地、线上全球和线下场景。在模拟中，我们的方法优于其他攻击算法，因为我们的方法可以在非独立均匀分布数据下获得全局目标模型。



## **6. Adversarial Attack on Large Language Models using Exponentiated Gradient Descent**

使用指数梯度下降对大型语言模型的对抗攻击 cs.LG

Accepted to International Joint Conference on Neural Networks (IJCNN)  2025

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2505.09820v1) [paper-pdf](http://arxiv.org/pdf/2505.09820v1)

**Authors**: Sajib Biswas, Mao Nishino, Samuel Jacob Chacko, Xiuwen Liu

**Abstract**: As Large Language Models (LLMs) are widely used, understanding them systematically is key to improving their safety and realizing their full potential. Although many models are aligned using techniques such as reinforcement learning from human feedback (RLHF), they are still vulnerable to jailbreaking attacks. Some of the existing adversarial attack methods search for discrete tokens that may jailbreak a target model while others try to optimize the continuous space represented by the tokens of the model's vocabulary. While techniques based on the discrete space may prove to be inefficient, optimization of continuous token embeddings requires projections to produce discrete tokens, which might render them ineffective. To fully utilize the constraints and the structures of the space, we develop an intrinsic optimization technique using exponentiated gradient descent with the Bregman projection method to ensure that the optimized one-hot encoding always stays within the probability simplex. We prove the convergence of the technique and implement an efficient algorithm that is effective in jailbreaking several widely used LLMs. We demonstrate the efficacy of the proposed technique using five open-source LLMs on four openly available datasets. The results show that the technique achieves a higher success rate with great efficiency compared to three other state-of-the-art jailbreaking techniques. The source code for our implementation is available at: https://github.com/sbamit/Exponentiated-Gradient-Descent-LLM-Attack

摘要: 随着大型语言模型（LLM）的广泛使用，系统性地理解它们是提高其安全性并充分发挥其潜力的关键。尽管许多模型都使用人类反馈强化学习（RL HF）等技术进行了调整，但它们仍然容易受到越狱攻击。现有的一些对抗攻击方法搜索可能越狱目标模型的离散令牌，而另一些方法则试图优化模型词汇表中的令牌所代表的连续空间。虽然基于离散空间的技术可能被证明效率低下，但连续令牌嵌入的优化需要投影来产生离散令牌，这可能会使它们无效。为了充分利用空间的约束和结构，我们使用Bregman投影方法的指数梯度下降开发了一种内在优化技术，以确保优化的一次性编码始终保持在概率单形内。我们证明了该技术的收敛性，并实现了一种有效的算法，该算法可以有效越狱几种广泛使用的LLM。我们在四个公开可用的数据集上使用五个开源LLM来证明所提出技术的有效性。结果表明，与其他三种最先进的越狱技术相比，该技术的成功率更高，效率更高。我们实现的源代码可访问：https://github.com/sbamit/Exponentiated-Gradient-Descent-LLM-Attack



## **7. Self-Consuming Generative Models with Adversarially Curated Data**

具有对抗策划数据的自我消费生成模型 cs.LG

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2505.09768v1) [paper-pdf](http://arxiv.org/pdf/2505.09768v1)

**Authors**: Xiukun Wei, Xueru Zhang

**Abstract**: Recent advances in generative models have made it increasingly difficult to distinguish real data from model-generated synthetic data. Using synthetic data for successive training of future model generations creates "self-consuming loops", which may lead to model collapse or training instability. Furthermore, synthetic data is often subject to human feedback and curated by users based on their preferences. Ferbach et al. (2024) recently showed that when data is curated according to user preferences, the self-consuming retraining loop drives the model to converge toward a distribution that optimizes those preferences. However, in practice, data curation is often noisy or adversarially manipulated. For example, competing platforms may recruit malicious users to adversarially curate data and disrupt rival models. In this paper, we study how generative models evolve under self-consuming retraining loops with noisy and adversarially curated data. We theoretically analyze the impact of such noisy data curation on generative models and identify conditions for the robustness of the retraining process. Building on this analysis, we design attack algorithms for competitive adversarial scenarios, where a platform with a limited budget employs malicious users to misalign a rival's model from actual user preferences. Experiments on both synthetic and real-world datasets demonstrate the effectiveness of the proposed algorithms.

摘要: 生成模型的最新进展使得区分真实数据与模型生成的合成数据变得越来越困难。使用合成数据对未来模型世代进行连续训练会创建“自消费循环”，这可能会导致模型崩溃或训练不稳定。此外，合成数据通常会受到人类反馈的影响，并由用户根据他们的偏好进行策划。Ferbach等人（2024）最近表明，当根据用户偏好策划数据时，自消费再培训循环会推动模型向优化这些偏好的分布收敛。然而，在实践中，数据策展通常会受到干扰或不利操纵。例如，竞争平台可能会招募恶意用户来对抗性地策划数据并破坏竞争对手的模型。在本文中，我们研究生成模型如何在具有噪音和敌对策划数据的自消费再培训循环下发展。我们从理论上分析了这种有噪音的数据策展对生成模型的影响，并确定再培训过程稳健性的条件。在此分析的基础上，我们设计了针对竞争性对抗场景的攻击算法，在这种场景中，预算有限的平台使用恶意用户来使竞争对手的模型与实际用户偏好不一致。在合成数据集和真实数据集上的实验证明了所提算法的有效性。



## **8. Adversarial Suffix Filtering: a Defense Pipeline for LLMs**

对抗性后缀过滤：LLM的防御管道 cs.LG

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2505.09602v1) [paper-pdf](http://arxiv.org/pdf/2505.09602v1)

**Authors**: David Khachaturov, Robert Mullins

**Abstract**: Large Language Models (LLMs) are increasingly embedded in autonomous systems and public-facing environments, yet they remain susceptible to jailbreak vulnerabilities that may undermine their security and trustworthiness. Adversarial suffixes are considered to be the current state-of-the-art jailbreak, consistently outperforming simpler methods and frequently succeeding even in black-box settings. Existing defenses rely on access to the internal architecture of models limiting diverse deployment, increase memory and computation footprints dramatically, or can be bypassed with simple prompt engineering methods. We introduce $\textbf{Adversarial Suffix Filtering}$ (ASF), a lightweight novel model-agnostic defensive pipeline designed to protect LLMs against adversarial suffix attacks. ASF functions as an input preprocessor and sanitizer that detects and filters adversarially crafted suffixes in prompts, effectively neutralizing malicious injections. We demonstrate that ASF provides comprehensive defense capabilities across both black-box and white-box attack settings, reducing the attack efficacy of state-of-the-art adversarial suffix generation methods to below 4%, while only minimally affecting the target model's capabilities in non-adversarial scenarios.

摘要: 大型语言模型（LLM）越来越多地嵌入到自治系统和面向公众的环境中，但它们仍然容易受到越狱漏洞的影响，这可能会损害其安全性和可信度。对抗性后缀被认为是当前最先进的越狱方法，其性能始终优于更简单的方法，即使在黑匣子环境中也经常取得成功。现有的防御依赖于对模型内部架构的访问，从而限制了多样化部署、大幅增加内存和计算占用空间，或者可以通过简单的即时工程方法绕过。我们引入了$\textBF{对抗后缀过滤}$（SAF），这是一个轻量级的新颖模型不可知防御管道，旨在保护LLM免受对抗后缀攻击。ADF充当输入预处理器和消毒器，可以检测和过滤提示中反向制作的后缀，有效地中和恶意注入。我们证明，ADF在黑匣子和白盒攻击环境中提供全面的防御能力，将最先进的对抗性后缀生成方法的攻击功效降低到4%以下，而对目标模型在非对抗性场景中的能力的影响微乎其微。



## **9. I Know What You Said: Unveiling Hardware Cache Side-Channels in Local Large Language Model Inference**

我知道你说什么：揭开本地大型语言模型推理中的硬件缓存侧通道 cs.CR

Submitted for review in January 22, 2025

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2505.06738v2) [paper-pdf](http://arxiv.org/pdf/2505.06738v2)

**Authors**: Zibo Gao, Junjie Hu, Feng Guo, Yixin Zhang, Yinglong Han, Siyuan Liu, Haiyang Li, Zhiqiang Lv

**Abstract**: Large Language Models (LLMs) that can be deployed locally have recently gained popularity for privacy-sensitive tasks, with companies such as Meta, Google, and Intel playing significant roles in their development. However, the security of local LLMs through the lens of hardware cache side-channels remains unexplored. In this paper, we unveil novel side-channel vulnerabilities in local LLM inference: token value and token position leakage, which can expose both the victim's input and output text, thereby compromising user privacy. Specifically, we found that adversaries can infer the token values from the cache access patterns of the token embedding operation, and deduce the token positions from the timing of autoregressive decoding phases. To demonstrate the potential of these leaks, we design a novel eavesdropping attack framework targeting both open-source and proprietary LLM inference systems. The attack framework does not directly interact with the victim's LLM and can be executed without privilege.   We evaluate the attack on a range of practical local LLM deployments (e.g., Llama, Falcon, and Gemma), and the results show that our attack achieves promising accuracy. The restored output and input text have an average edit distance of 5.2% and 17.3% to the ground truth, respectively. Furthermore, the reconstructed texts achieve average cosine similarity scores of 98.7% (input) and 98.0% (output).

摘要: 可以在本地部署的大型语言模型（LLM）最近在隐私敏感任务中越来越受欢迎，Meta、谷歌和英特尔等公司在其开发中发挥了重要作用。然而，通过硬件缓存侧通道的视角来探讨本地LLM的安全性仍然有待探索。在本文中，我们揭示了本地LLM推断中的新型侧通道漏洞：令牌值和令牌位置泄露，它可以暴露受害者的输入和输出文本，从而损害用户隐私。具体来说，我们发现对手可以从令牌嵌入操作的缓存访问模式中推断令牌值，并从自回归解码阶段的时间推断令牌位置。为了证明这些泄漏的潜力，我们设计了一个新的窃听攻击框架，针对开源和专有的LLM推理系统。攻击框架不直接与受害者的LLM交互，并且可以在没有特权的情况下执行。   我们评估了对一系列实际本地LLM部署的攻击（例如，Llama，Falcon和Gemma），结果表明我们的攻击达到了很好的准确性。恢复的输出和输入文本与地面真相的平均编辑距离分别为5.2%和17.3%。此外，重建的文本的平均cos相似度评分为98.7%（输入）和98.0%（输出）。



## **10. Evaluating the Robustness of Adversarial Defenses in Malware Detection Systems**

评估恶意软件检测系统中对抗性防御的稳健性 cs.CR

Submitted to IEEE Transactions on Information Forensics and Security  (T-IFS), 13 pages, 4 figures

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2505.09342v1) [paper-pdf](http://arxiv.org/pdf/2505.09342v1)

**Authors**: Mostafa Jafari, Alireza Shameli-Sendi

**Abstract**: Machine learning is a key tool for Android malware detection, effectively identifying malicious patterns in apps. However, ML-based detectors are vulnerable to evasion attacks, where small, crafted changes bypass detection. Despite progress in adversarial defenses, the lack of comprehensive evaluation frameworks in binary-constrained domains limits understanding of their robustness. We introduce two key contributions. First, Prioritized Binary Rounding, a technique to convert continuous perturbations into binary feature spaces while preserving high attack success and low perturbation size. Second, the sigma-binary attack, a novel adversarial method for binary domains, designed to achieve attack goals with minimal feature changes. Experiments on the Malscan dataset show that sigma-binary outperforms existing attacks and exposes key vulnerabilities in state-of-the-art defenses. Defenses equipped with adversary detectors, such as KDE, DLA, DNN+, and ICNN, exhibit significant brittleness, with attack success rates exceeding 90% using fewer than 10 feature modifications and reaching 100% with just 20. Adversarially trained defenses, including AT-rFGSM-k, AT-MaxMA, improves robustness under small budgets but remains vulnerable to unrestricted perturbations, with attack success rates of 99.45% and 96.62%, respectively. Although PAD-SMA demonstrates strong robustness against state-of-the-art gradient-based adversarial attacks by maintaining an attack success rate below 16.55%, the sigma-binary attack significantly outperforms these methods, achieving a 94.56% success rate under unrestricted perturbations. These findings highlight the critical need for precise method like sigma-binary to expose hidden vulnerabilities in existing defenses and support the development of more resilient malware detection systems.

摘要: 机器学习是Android恶意软件检测的关键工具，可以有效识别应用程序中的恶意模式。然而，基于ML的检测器很容易受到规避攻击，因为小的精心设计的更改会绕过检测。尽管对抗性防御取得了进展，但二进制约束领域缺乏全面的评估框架限制了对其稳健性的理解。我们介绍两个关键贡献。首先，优先二进制舍入，一种将连续扰动转换为二进制特征空间的技术，同时保持高攻击成功率和低扰动大小。第二，sigma-binary攻击，一种针对二进制域的新型对抗方法，旨在以最小的特征变化实现攻击目标。在Malscan数据集上的实验表明，sigma-binary优于现有的攻击，并暴露了最先进防御中的关键漏洞。配备了攻击者检测器的防御系统，如KDE、DLA、DNN+和ICNN，表现出明显的脆弱性，攻击成功率超过90%，使用不到10个功能修改，只需20个即可达到100%。经过对抗训练的防御，包括AT-rFGSM-k、AT-MaxMA，可以在小预算下提高稳健性，但仍然容易受到不受限制的干扰，攻击成功率分别为99.45%和96.62%。尽管PAD-SM通过将攻击成功率保持在16.55%以下，表现出对最先进的基于梯度的对抗攻击的强大鲁棒性，但西格玛二进制攻击的表现显着优于这些方法，在不受限制的扰动下实现了94.56%的成功率。这些发现凸显了迫切需要西格玛二进制等精确方法来暴露现有防御中隐藏的漏洞并支持开发更具弹性的恶意软件检测系统。



## **11. Beyond Optimal Fault Tolerance**

超越最佳故障容忍度 cs.DC

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2501.06044v5) [paper-pdf](http://arxiv.org/pdf/2501.06044v5)

**Authors**: Andrew Lewis-Pye, Tim Roughgarden

**Abstract**: The optimal fault-tolerance achievable by any protocol has been characterized in a wide range of settings. For example, for state machine replication (SMR) protocols operating in the partially synchronous setting, it is possible to simultaneously guarantee consistency against $\alpha$-bounded adversaries (i.e., adversaries that control less than an $\alpha$ fraction of the participants) and liveness against $\beta$-bounded adversaries if and only if $\alpha + 2\beta \leq 1$.   This paper characterizes to what extent "better-than-optimal" fault-tolerance guarantees are possible for SMR protocols when the standard consistency requirement is relaxed to allow a bounded number $r$ of consistency violations. We prove that bounding rollback is impossible without additional timing assumptions and investigate protocols that tolerate and recover from consistency violations whenever message delays around the time of an attack are bounded by a parameter $\Delta^*$ (which may be arbitrarily larger than the parameter $\Delta$ that bounds post-GST message delays in the partially synchronous model). Here, a protocol's fault-tolerance can be a non-constant function of $r$, and we prove, for each $r$, matching upper and lower bounds on the optimal "recoverable fault-tolerance" achievable by any SMR protocol. For example, for protocols that guarantee liveness against 1/3-bounded adversaries in the partially synchronous setting, a 5/9-bounded adversary can always cause one consistency violation but not two, and a 2/3-bounded adversary can always cause two consistency violations but not three. Our positive results are achieved through a generic "recovery procedure" that can be grafted on to any accountable SMR protocol and restores consistency following a violation while rolling back only transactions that were finalized in the previous $2\Delta^*$ timesteps.

摘要: 任何协议都可以实现的最佳故障容差在各种设置中都有其特征。例如，对于在部分同步设置中运行的状态机复制（SVR）协议，可以同时保证针对$\Alpha $-有界对手（即，控制少于参与者$\Alpha $一部分的对手）和针对$\Beta $的活力-有界的对手当且仅当$\Alpha +2\Beta\leq 1 $。   本文描述了当放宽标准一致性要求以允许有界数量$r $的一致性违规时，SVR协议在多大程度上可能实现“优于最优”的公差保证。我们证明，如果没有额外的时间假设，限制回滚是不可能的，并研究每当攻击时间左右的消息延迟受到参数$\Delta '*$（该参数可以任意大于部分同步模型中限制后GST消息延迟的参数$\Delta $）时，能够容忍一致性违规并从一致性违规中恢复的协议。在这里，协议的故障容限可以是$r $的非常函数，并且我们证明，对于每个$r $，任何SVR协议可实现的最佳“可恢复故障容限”的上下限和下限匹配。例如，对于在部分同步设置中保证针对1/3有界对手的活性的协议，5/9有界对手总是会导致一次一致性违规，但不会导致两次一致性违规，而2/3有界对手总是会导致两次一致性违规，但不会导致三次。我们的积极结果是通过通用的“恢复程序”实现的，该程序可以移植到任何负责任的SVR协议上，并在违规后恢复一致性，同时仅回滚在之前$2\Delta '*$时间步中完成的事务。



## **12. Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction**

可靠地限制假阳性：通过多尺度保形预测的零镜头机器生成文本检测框架 cs.CL

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2505.05084v2) [paper-pdf](http://arxiv.org/pdf/2505.05084v2)

**Authors**: Xiaowei Zhu, Yubing Ren, Yanan Cao, Xixun Lin, Fang Fang, Yangxi Li

**Abstract**: The rapid advancement of large language models has raised significant concerns regarding their potential misuse by malicious actors. As a result, developing effective detectors to mitigate these risks has become a critical priority. However, most existing detection methods focus excessively on detection accuracy, often neglecting the societal risks posed by high false positive rates (FPRs). This paper addresses this issue by leveraging Conformal Prediction (CP), which effectively constrains the upper bound of FPRs. While directly applying CP constrains FPRs, it also leads to a significant reduction in detection performance. To overcome this trade-off, this paper proposes a Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction (MCP), which both enforces the FPR constraint and improves detection performance. This paper also introduces RealDet, a high-quality dataset that spans a wide range of domains, ensuring realistic calibration and enabling superior detection performance when combined with MCP. Empirical evaluations demonstrate that MCP effectively constrains FPRs, significantly enhances detection performance, and increases robustness against adversarial attacks across multiple detectors and datasets.

摘要: 大型语言模型的快速发展引发了人们对其潜在被恶意行为者滥用的严重担忧。因此，开发有效的探测器来减轻这些风险已成为当务之急。然而，大多数现有的检测方法过度关注检测准确性，往往忽视了高假阳性率（FPR）带来的社会风险。本文通过利用保形预测（CP）来解决这个问题，该预测有效地限制了FPR的上界。虽然直接应用CP约束FPR，但也会导致检测性能显着降低。为了克服这种权衡，本文提出了一种通过多尺度保形预测（LCP）的零镜头机器生成文本检测框架，该框架既强制执行FPR约束又提高检测性能。本文还介绍了RealDet，这是一个跨越广泛领域的高质量数据集，可确保真实的校准并在与HCP结合时实现卓越的检测性能。经验评估表明，LCP有效地约束了FPR，显着增强了检测性能，并增强了针对多个检测器和数据集的对抗攻击的鲁棒性。



## **13. BridgePure: Limited Protection Leakage Can Break Black-Box Data Protection**

BridgePure：有限的保护泄漏可以打破黑匣子数据保护 cs.LG

29 pages,18 figures

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2412.21061v2) [paper-pdf](http://arxiv.org/pdf/2412.21061v2)

**Authors**: Yihan Wang, Yiwei Lu, Xiao-Shan Gao, Gautam Kamath, Yaoliang Yu

**Abstract**: Availability attacks, or unlearnable examples, are defensive techniques that allow data owners to modify their datasets in ways that prevent unauthorized machine learning models from learning effectively while maintaining the data's intended functionality. It has led to the release of popular black-box tools (e.g., APIs) for users to upload personal data and receive protected counterparts. In this work, we show that such black-box protections can be substantially compromised if a small set of unprotected in-distribution data is available. Specifically, we propose a novel threat model of protection leakage, where an adversary can (1) easily acquire (unprotected, protected) pairs by querying the black-box protections with a small unprotected dataset; and (2) train a diffusion bridge model to build a mapping between unprotected and protected data. This mapping, termed BridgePure, can effectively remove the protection from any previously unseen data within the same distribution. BridgePure demonstrates superior purification performance on classification and style mimicry tasks, exposing critical vulnerabilities in black-box data protection. We suggest that practitioners implement multi-level countermeasures to mitigate such risks.

摘要: 可用性攻击或不可学习的示例是防御性技术，允许数据所有者修改其数据集，以防止未经授权的机器学习模型在维护数据的预期功能的同时有效学习。它导致了流行的黑匣子工具（例如，API）供用户上传个人数据并接收受保护的对应数据。在这项工作中，我们表明，如果有一小组未受保护的分发数据可用，这种黑匣子保护可能会受到严重损害。具体来说，我们提出了一种新型的保护泄漏威胁模型，其中对手可以（1）通过使用较小的未受保护数据集查询黑匣子保护来轻松获取（未受保护的、受保护的）对;以及（2）训练扩散桥模型来构建未受保护的和受保护的数据之间的映射。这种名为BridgePure的映射可以有效地消除对同一分发中任何以前未见过的数据的保护。BridgePure在分类和风格模仿任务方面表现出卓越的净化性能，暴露了黑匣子数据保护中的关键漏洞。我们建议从业者实施多层次的应对措施来降低此类风险。



## **14. SAFE-SiP: Secure Authentication Framework for System-in-Package Using Multi-party Computation**

SAFE-SiP：使用多方计算的系统级包安全认证框架 cs.CR

Accepted for GLSVLSI 2025, New Orleans, LA, USA

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.09002v1) [paper-pdf](http://arxiv.org/pdf/2505.09002v1)

**Authors**: Ishraq Tashdid, Tasnuva Farheen, Sazadur Rahman

**Abstract**: The emergence of chiplet-based heterogeneous integration is transforming the semiconductor, AI, and high-performance computing industries by enabling modular designs and improved scalability. However, assembling chiplets from multiple vendors after fabrication introduces a complex supply chain that raises serious security concerns, including counterfeiting, overproduction, and unauthorized access. Current solutions often depend on dedicated security chiplets or changes to the timing flow, which assume a trusted SiP integrator. This assumption can expose chiplet signatures to other vendors and create new attack surfaces. This work addresses those vulnerabilities using Multi-party Computation (MPC), which enables zero-trust authentication without disclosing sensitive information to any party. We present SAFE-SiP, a scalable authentication framework that garbles chiplet signatures and uses MPC for verifying integrity, effectively blocking unauthorized access and adversarial inference. SAFE-SiP removes the need for a dedicated security chiplet and ensures secure authentication, even in untrusted integration scenarios. We evaluated SAFE-SiP on five RISC-V-based System-in-Package (SiP) designs. Experimental results show that SAFE-SiP incurs minimal power overhead, an average area overhead of only 3.05%, and maintains a computational complexity of 2^192, offering a highly efficient and scalable security solution.

摘要: 基于芯片的异类集成的出现通过实现模块化设计和提高可扩展性，正在改变半导体、人工智能和高性能计算行业。然而，在制造后从多个供应商组装小芯片会引入复杂的供应链，引发严重的安全问题，包括假冒、过度生产和未经授权的访问。当前的解决方案通常依赖于专用的安全小芯片或对计时流程的更改，这假设有值得信赖的SiP集成商。这个假设可能会将小芯片签名暴露给其他供应商并创建新的攻击面。这项工作使用多方计算（MPC）来解决这些漏洞，该计算可以实现零信任身份验证，而不会向任何一方泄露敏感信息。我们提出了SAFE-SiP，这是一种可扩展的认证框架，它可以混淆小芯片签名并使用MPC来验证完整性，有效地阻止未经授权的访问和对抗性推断。SAFE-SiP消除了对专用安全小芯片的需要，并确保安全身份验证，即使在不受信任的集成场景中也是如此。我们评估了五种基于RISC-V的系统级封装（SiP）设计的SAFE-SiP。实验结果表明，SAFE-SiP的功耗最小，平均面积开销仅为3.05%，计算复杂度为2#192，提供高效且可扩展的安全解决方案。



## **15. Towards Adaptive Meta-Gradient Adversarial Examples for Visual Tracking**

视觉跟踪的自适应元梯度对抗示例 cs.CV

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.08999v1) [paper-pdf](http://arxiv.org/pdf/2505.08999v1)

**Authors**: Wei-Long Tian, Peng Gao, Xiao Liu, Long Xu, Hamido Fujita, Hanan Aljuai, Mao-Li Wang

**Abstract**: In recent years, visual tracking methods based on convolutional neural networks and Transformers have achieved remarkable performance and have been successfully applied in fields such as autonomous driving. However, the numerous security issues exposed by deep learning models have gradually affected the reliable application of visual tracking methods in real-world scenarios. Therefore, how to reveal the security vulnerabilities of existing visual trackers through effective adversarial attacks has become a critical problem that needs to be addressed. To this end, we propose an adaptive meta-gradient adversarial attack (AMGA) method for visual tracking. This method integrates multi-model ensembles and meta-learning strategies, combining momentum mechanisms and Gaussian smoothing, which can significantly enhance the transferability and attack effectiveness of adversarial examples. AMGA randomly selects models from a large model repository, constructs diverse tracking scenarios, and iteratively performs both white- and black-box adversarial attacks in each scenario, optimizing the gradient directions of each model. This paradigm minimizes the gap between white- and black-box adversarial attacks, thus achieving excellent attack performance in black-box scenarios. Extensive experimental results on large-scale datasets such as OTB2015, LaSOT, and GOT-10k demonstrate that AMGA significantly improves the attack performance, transferability, and deception of adversarial examples. Codes and data are available at https://github.com/pgao-lab/AMGA.

摘要: 近年来，基于卷积神经网络和变形金刚的视觉跟踪方法取得了显着的性能，并成功应用于自动驾驶等领域。然而，深度学习模型暴露出的众多安全问题逐渐影响了视觉跟踪方法在现实场景中的可靠应用。因此，如何通过有效的对抗攻击来揭示现有视觉跟踪器的安全漏洞已成为需要解决的关键问题。为此，我们提出了一种用于视觉跟踪的自适应元梯度对抗攻击（AMGA）方法。该方法集成了多模型集成和元学习策略，结合动量机制和高斯平滑，可以显著增强对抗性样本的可移植性和攻击有效性。AMGA从大型模型库中随机选择模型，构建不同的跟踪场景，并在每个场景中迭代执行白盒和黑盒对抗攻击，优化每个模型的梯度方向。这种模式最大限度地减少了白盒和黑盒对抗攻击之间的差距，从而在黑盒场景中实现了出色的攻击性能。在OTB 2015、LaSOT和GOT-10 k等大规模数据集上的大量实验结果表明，AMGA显著提高了攻击性能、可转移性和对抗性示例的欺骗性。代码和数据可访问https://github.com/pgao-lab/AMGA。



## **16. On the Impact of Uncertainty and Calibration on Likelihood-Ratio Membership Inference Attacks**

不确定性和校准对可能性比隶属推理攻击的影响 cs.IT

16 pages, 23 figures

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2402.10686v4) [paper-pdf](http://arxiv.org/pdf/2402.10686v4)

**Authors**: Meiyi Zhu, Caili Guo, Chunyan Feng, Osvaldo Simeone

**Abstract**: In a membership inference attack (MIA), an attacker exploits the overconfidence exhibited by typical machine learning models to determine whether a specific data point was used to train a target model. In this paper, we analyze the performance of the likelihood ratio attack (LiRA) within an information-theoretical framework that allows the investigation of the impact of the aleatoric uncertainty in the true data generation process, of the epistemic uncertainty caused by a limited training data set, and of the calibration level of the target model. We compare three different settings, in which the attacker receives decreasingly informative feedback from the target model: confidence vector (CV) disclosure, in which the output probability vector is released; true label confidence (TLC) disclosure, in which only the probability assigned to the true label is made available by the model; and decision set (DS) disclosure, in which an adaptive prediction set is produced as in conformal prediction. We derive bounds on the advantage of an MIA adversary with the aim of offering insights into the impact of uncertainty and calibration on the effectiveness of MIAs. Simulation results demonstrate that the derived analytical bounds predict well the effectiveness of MIAs.

摘要: 在隶属推理攻击（MIA）中，攻击者利用典型机器学习模型表现出的过度自信来确定特定数据点是否用于训练目标模型。在本文中，我们分析了似然比攻击（LiRA）的性能在一个信息理论框架内，允许调查的影响任意的不确定性在真实的数据生成过程中，由有限的训练数据集造成的认知不确定性，和目标模型的校准水平。我们比较了三种不同的设置，其中攻击者从目标模型收到的信息反馈越来越少：置信向量（CV）披露，其中输出概率向量被释放;真实标签置信度（TLC）披露，其中只有分配给真实标签的概率由模型提供;以及决策集（DS）公开，其中如在共形预测中一样产生自适应预测集。我们得出了MIA对手的优势界限，旨在深入了解不确定性和校准对MIA有效性的影响。仿真结果表明，推导出的分析界能够很好地预测MIA的有效性。



## **17. DFA-CON: A Contrastive Learning Approach for Detecting Copyright Infringement in DeepFake Art**

DFA-CON：检测DeepFake Art版权侵权的对比学习方法 cs.CV

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.08552v1) [paper-pdf](http://arxiv.org/pdf/2505.08552v1)

**Authors**: Haroon Wahab, Hassan Ugail, Irfan Mehmood

**Abstract**: Recent proliferation of generative AI tools for visual content creation-particularly in the context of visual artworks-has raised serious concerns about copyright infringement and forgery. The large-scale datasets used to train these models often contain a mixture of copyrighted and non-copyrighted artworks. Given the tendency of generative models to memorize training patterns, they are susceptible to varying degrees of copyright violation. Building on the recently proposed DeepfakeArt Challenge benchmark, this work introduces DFA-CON, a contrastive learning framework designed to detect copyright-infringing or forged AI-generated art. DFA-CON learns a discriminative representation space, posing affinity among original artworks and their forged counterparts within a contrastive learning framework. The model is trained across multiple attack types, including inpainting, style transfer, adversarial perturbation, and cutmix. Evaluation results demonstrate robust detection performance across most attack types, outperforming recent pretrained foundation models. Code and model checkpoints will be released publicly upon acceptance.

摘要: 最近用于视觉内容创作的生成性人工智能工具的激增--特别是在视觉艺术品的背景下--引发了人们对版权侵权和伪造的严重担忧。用于训练这些模型的大规模数据集通常包含受版权保护和非受版权保护的艺术品的混合物。鉴于生成模型记忆训练模式的倾向，它们很容易受到不同程度的版权侵犯。这项工作以最近提出的DeepfakeArt Challenge基准为基础，引入了DFA-CON，这是一种对比学习框架，旨在检测侵犯版权或伪造的人工智能生成的艺术品。DFA-CON学习区分性表示空间，在对比学习框架内在原始艺术品及其伪造作品之间建立了亲和力。该模型跨多种攻击类型进行训练，包括修补、风格转移、对抗性扰动和cutmix。评估结果表明，大多数攻击类型都具有稳健的检测性能，优于最近预训练的基础模型。代码和模型检查点将在接受后公开发布。



## **18. Minimax rates of convergence for nonparametric regression under adversarial attacks**

对抗攻击下非参数回归的极小极大收敛率 math.ST

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2410.09402v2) [paper-pdf](http://arxiv.org/pdf/2410.09402v2)

**Authors**: Jingfu Peng, Yuhong Yang

**Abstract**: Recent research shows the susceptibility of machine learning models to adversarial attacks, wherein minor but maliciously chosen perturbations of the input can significantly degrade model performance. In this paper, we theoretically analyse the limits of robustness against such adversarial attacks in a nonparametric regression setting, by examining the minimax rates of convergence in an adversarial sup-norm. Our work reveals that the minimax rate under adversarial attacks in the input is the same as sum of two terms: one represents the minimax rate in the standard setting without adversarial attacks, and the other reflects the maximum deviation of the true regression function value within the target function class when subjected to the input perturbations. The optimal rates under the adversarial setup can be achieved by an adversarial plug-in procedure constructed from a minimax optimal estimator in the corresponding standard setting. Two specific examples are given to illustrate the established minimax results.

摘要: 最近的研究表明，机器学习模型容易受到对抗攻击，其中对输入进行微小但恶意选择的干扰可能会显着降低模型性能。在本文中，我们通过检查对抗性超规范中的极小极大收敛率，从理论上分析了非参数回归设置中针对此类对抗性攻击的鲁棒性的限制。我们的工作表明，输入中对抗性攻击下的极小最大率与两项的和相同：一项代表没有对抗性攻击的标准设置中的极小最大率，另一项反映了目标函数类内真实回归函数值的最大偏差时受到输入扰动。对抗设置下的最佳速率可以通过对抗插件程序来实现，该程序由相应标准设置中的极小最佳估计器构建。给出了两个具体的例子来说明所建立的极小极大结果。



## **19. Quantum Support Vector Regression for Robust Anomaly Detection**

量子支持量回归用于鲁棒异常检测 quant-ph

Submitted to IEEE International Conference on Quantum Computing and  Engineering (QCE) 2025

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.01012v2) [paper-pdf](http://arxiv.org/pdf/2505.01012v2)

**Authors**: Kilian Tscharke, Maximilian Wendlinger, Sebastian Issel, Pascal Debus

**Abstract**: Anomaly Detection (AD) is critical in data analysis, particularly within the domain of IT security. In recent years, Machine Learning (ML) algorithms have emerged as a powerful tool for AD in large-scale data. In this study, we explore the potential of quantum ML approaches, specifically quantum kernel methods, for the application to robust AD. We build upon previous work on Quantum Support Vector Regression (QSVR) for semisupervised AD by conducting a comprehensive benchmark on IBM quantum hardware using eleven datasets. Our results demonstrate that QSVR achieves strong classification performance and even outperforms the noiseless simulation on two of these datasets. Moreover, we investigate the influence of - in the NISQ-era inevitable - quantum noise on the performance of the QSVR. Our findings reveal that the model exhibits robustness to depolarizing, phase damping, phase flip, and bit flip noise, while amplitude damping and miscalibration noise prove to be more disruptive. Finally, we explore the domain of Quantum Adversarial Machine Learning and demonstrate that QSVR is highly vulnerable to adversarial attacks and that noise does not improve the adversarial robustness of the model.

摘要: 异常检测（AD）在数据分析中至关重要，尤其是在IT安全领域。近年来，机器学习（ML）算法已成为大规模数据中AD的强大工具。在这项研究中，我们探索了量子ML方法（特别是量子核方法）应用于稳健AD的潜力。我们在之前针对半监督AD的量子支持量回归（QSVR）工作的基础上，使用11个数据集对IBM量子硬件进行了全面的基准测试。我们的结果表明，QSVR实现了强大的分类性能，甚至优于对其中两个数据集的无噪模拟。此外，我们还研究了在NISQ时代不可避免的量子噪音对QSVR性能的影响。我们的研究结果表明，该模型对去极化、相衰减、相翻转和位翻转噪音表现出鲁棒性，而幅度衰减和失调噪音被证明更具破坏性。最后，我们探索了量子对抗机器学习领域，并证明QSVR极易受到对抗攻击，并且噪音不会提高模型的对抗鲁棒性。



## **20. SHAP-based Explanations are Sensitive to Feature Representation**

基于SHAP的描述对特征表示敏感 cs.LG

Accepted to ACM FAccT 2025

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.08345v1) [paper-pdf](http://arxiv.org/pdf/2505.08345v1)

**Authors**: Hyunseung Hwang, Andrew Bell, Joao Fonseca, Venetia Pliatsika, Julia Stoyanovich, Steven Euijong Whang

**Abstract**: Local feature-based explanations are a key component of the XAI toolkit. These explanations compute feature importance values relative to an ``interpretable'' feature representation. In tabular data, feature values themselves are often considered interpretable. This paper examines the impact of data engineering choices on local feature-based explanations. We demonstrate that simple, common data engineering techniques, such as representing age with a histogram or encoding race in a specific way, can manipulate feature importance as determined by popular methods like SHAP. Notably, the sensitivity of explanations to feature representation can be exploited by adversaries to obscure issues like discrimination. While the intuition behind these results is straightforward, their systematic exploration has been lacking. Previous work has focused on adversarial attacks on feature-based explainers by biasing data or manipulating models. To the best of our knowledge, this is the first study demonstrating that explainers can be misled by standard, seemingly innocuous data engineering techniques.

摘要: 基于本地特征的解释是XAI工具包的关键组件。这些解释计算相对于“可解释”特征表示的特征重要性值。在表格数据中，特征值本身通常被认为是可解释的。本文探讨了数据工程选择对基于局部特征的解释的影响。我们证明了简单，常见的数据工程技术，如用直方图表示年龄或以特定方式编码种族，可以操纵由SHAP等流行方法确定的特征重要性。值得注意的是，对手可以利用解释对特征表示的敏感性来掩盖歧视等问题。虽然这些结果背后的直觉是直截了当的，但缺乏系统的探索。以前的工作主要集中在通过偏置数据或操纵模型对基于特征的解释器进行对抗性攻击。据我们所知，这是第一项证明解释者可能会被标准的、看似无害的数据工程技术误导的研究。



## **21. Robustness Analysis against Adversarial Patch Attacks in Fully Unmanned Stores**

完全无人店中对抗补丁攻击的鲁棒性分析 cs.CR

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.08835v1) [paper-pdf](http://arxiv.org/pdf/2505.08835v1)

**Authors**: Hyunsik Na, Wonho Lee, Seungdeok Roh, Sohee Park, Daeseon Choi

**Abstract**: The advent of convenient and efficient fully unmanned stores equipped with artificial intelligence-based automated checkout systems marks a new era in retail. However, these systems have inherent artificial intelligence security vulnerabilities, which are exploited via adversarial patch attacks, particularly in physical environments. This study demonstrated that adversarial patches can severely disrupt object detection models used in unmanned stores, leading to issues such as theft, inventory discrepancies, and interference. We investigated three types of adversarial patch attacks -- Hiding, Creating, and Altering attacks -- and highlighted their effectiveness. We also introduce the novel color histogram similarity loss function by leveraging attacker knowledge of the color information of a target class object. Besides the traditional confusion-matrix-based attack success rate, we introduce a new bounding-boxes-based metric to analyze the practical impact of these attacks. Starting with attacks on object detection models trained on snack and fruit datasets in a digital environment, we evaluated the effectiveness of adversarial patches in a physical testbed that mimicked a real unmanned store with RGB cameras and realistic conditions. Furthermore, we assessed the robustness of these attacks in black-box scenarios, demonstrating that shadow attacks can enhance success rates of attacks even without direct access to model parameters. Our study underscores the necessity for robust defense strategies to protect unmanned stores from adversarial threats. Highlighting the limitations of the current defense mechanisms in real-time detection systems and discussing various proactive measures, we provide insights into improving the robustness of object detection models and fortifying unmanned retail environments against these attacks.

摘要: 配备基于人工智能的自动结账系统的便捷、高效的全无人商店的出现标志着零售业的新时代。然而，这些系统具有固有的人工智能安全漏洞，这些漏洞会通过对抗补丁攻击被利用，特别是在物理环境中。这项研究表明，对抗补丁会严重扰乱无人商店中使用的物体检测模型，导致盗窃、库存差异和干扰等问题。我们调查了三种类型的对抗性补丁攻击--隐藏攻击、创建攻击和更改攻击--并强调了它们的有效性。我们还通过利用攻击者对目标类对象颜色信息的了解，引入了新颖的颜色矩形相似性损失函数。除了传统的基于混淆矩阵的攻击成功率之外，我们还引入了一种新的基于边界盒的指标来分析这些攻击的实际影响。从对数字环境中在零食和水果数据集上训练的对象检测模型的攻击开始，我们在物理测试床上评估了对抗补丁的有效性，该测试床上模拟了具有RB摄像机和现实条件的真实无人商店。此外，我们评估了这些攻击在黑匣子场景中的稳健性，证明即使在不直接访问模型参数的情况下，影子攻击也可以提高攻击的成功率。我们的研究强调了强有力的防御策略的必要性，以保护无人悬挂物免受敌对威胁。我们强调了实时检测系统中当前防御机制的局限性，并讨论了各种主动措施，提供了提高物体检测模型稳健性和加强无人零售环境抵御这些攻击的见解。



## **22. Red Teaming the Mind of the Machine: A Systematic Evaluation of Prompt Injection and Jailbreak Vulnerabilities in LLMs**

Red联手机器思维：LLM中即时注射和越狱漏洞的系统评估 cs.CR

7 Pages, 6 Figures

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.04806v2) [paper-pdf](http://arxiv.org/pdf/2505.04806v2)

**Authors**: Chetan Pathade

**Abstract**: Large Language Models (LLMs) are increasingly integrated into consumer and enterprise applications. Despite their capabilities, they remain susceptible to adversarial attacks such as prompt injection and jailbreaks that override alignment safeguards. This paper provides a systematic investigation of jailbreak strategies against various state-of-the-art LLMs. We categorize over 1,400 adversarial prompts, analyze their success against GPT-4, Claude 2, Mistral 7B, and Vicuna, and examine their generalizability and construction logic. We further propose layered mitigation strategies and recommend a hybrid red-teaming and sandboxing approach for robust LLM security.

摘要: 大型语言模型（LLM）越来越多地集成到消费者和企业应用程序中。尽管它们有能力，但它们仍然容易受到对抗攻击，例如超越对齐保障措施的立即注射和越狱。本文对针对各种最先进的法学硕士的越狱策略进行了系统调查。我们对1，400多个对抗提示进行了分类，分析了它们对GPT-4、Claude 2、Mistral 7 B和Vicuna的成功，并检查它们的概括性和构造逻辑。我们进一步提出分层缓解策略，并推荐混合红色团队和沙箱方法以实现强大的LLM安全性。



## **23. Removing Watermarks with Partial Regeneration using Semantic Information**

使用语义信息通过部分再生去除水印 cs.CV

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.08234v1) [paper-pdf](http://arxiv.org/pdf/2505.08234v1)

**Authors**: Krti Tallam, John Kevin Cava, Caleb Geniesse, N. Benjamin Erichson, Michael W. Mahoney

**Abstract**: As AI-generated imagery becomes ubiquitous, invisible watermarks have emerged as a primary line of defense for copyright and provenance. The newest watermarking schemes embed semantic signals - content-aware patterns that are designed to survive common image manipulations - yet their true robustness against adaptive adversaries remains under-explored. We expose a previously unreported vulnerability and introduce SemanticRegen, a three-stage, label-free attack that erases state-of-the-art semantic and invisible watermarks while leaving an image's apparent meaning intact. Our pipeline (i) uses a vision-language model to obtain fine-grained captions, (ii) extracts foreground masks with zero-shot segmentation, and (iii) inpaints only the background via an LLM-guided diffusion model, thereby preserving salient objects and style cues. Evaluated on 1,000 prompts across four watermarking systems - TreeRing, StegaStamp, StableSig, and DWT/DCT - SemanticRegen is the only method to defeat the semantic TreeRing watermark (p = 0.10 > 0.05) and reduces bit-accuracy below 0.75 for the remaining schemes, all while maintaining high perceptual quality (masked SSIM = 0.94 +/- 0.01). We further introduce masked SSIM (mSSIM) to quantify fidelity within foreground regions, showing that our attack achieves up to 12 percent higher mSSIM than prior diffusion-based attackers. These results highlight an urgent gap between current watermark defenses and the capabilities of adaptive, semantics-aware adversaries, underscoring the need for watermarking algorithms that are resilient to content-preserving regenerative attacks.

摘要: 随着人工智能生成的图像变得无处不在，隐形水印已成为版权和出处的主要防线。最新的水印方案嵌入了语义信号--旨在在常见图像操作中生存的内容感知模式--但它们对自适应对手的真正鲁棒性仍然没有得到充分探索。我们揭露了一个以前未报告的漏洞，并引入SemanticRegen，这是一种三阶段、无标签攻击，可以擦除最先进的语义和隐形水印，同时保留图像的明显含义。我们的管道（i）使用视觉语言模型来获得细粒度的字幕，（ii）通过零镜头分割提取前景模板，以及（iii）通过LLM引导的扩散模型仅修补背景，从而保留显着的对象和风格线索。对四种水印系统（TreeRing、StegaStamp、StableSig和DWT/CT）的1，000个提示进行了评估- SemanticRegen是击败语义TreeRing水印（p = 0.10 > 0.05）并将剩余方案的位准确度降低到0.75以下的唯一方法，同时保持高感知质量（掩蔽SSIM = 0.94 +/- 0.01）。我们进一步引入掩蔽SSIM（mSSIM）来量化前景区域内的保真度，表明我们的攻击比之前的基于扩散的攻击者获得的mSSIM高出12%。这些结果凸显了当前的水印防御与自适应、语义感知的对手的能力之间的紧迫差距，强调了对能够抵御内容保留再生攻击的水印算法的需求。



## **24. AI and Generative AI Transforming Disaster Management: A Survey of Damage Assessment and Response Techniques**

人工智能和生成式人工智能改变灾害管理：灾害评估和响应技术综述 cs.CY

Accepted in IEEE Compsac 2025

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.08202v1) [paper-pdf](http://arxiv.org/pdf/2505.08202v1)

**Authors**: Aman Raj, Lakshit Arora, Sanjay Surendranath Girija, Shashank Kapoor, Dipen Pradhan, Ankit Shetgaonkar

**Abstract**: Natural disasters, including earthquakes, wildfires and cyclones, bear a huge risk on human lives as well as infrastructure assets. An effective response to disaster depends on the ability to rapidly and efficiently assess the intensity of damage. Artificial Intelligence (AI) and Generative Artificial Intelligence (GenAI) presents a breakthrough solution, capable of combining knowledge from multiple types and sources of data, simulating realistic scenarios of disaster, and identifying emerging trends at a speed previously unimaginable. In this paper, we present a comprehensive review on the prospects of AI and GenAI in damage assessment for various natural disasters, highlighting both its strengths and limitations. We talk about its application to multimodal data such as text, image, video, and audio, and also cover major issues of data privacy, security, and ethical use of the technology during crises. The paper also recognizes the threat of Generative AI misuse, in the form of dissemination of misinformation and for adversarial attacks. Finally, we outline avenues of future research, emphasizing the need for secure, reliable, and ethical Generative AI systems for disaster management in general. We believe that this work represents the first comprehensive survey of Gen-AI techniques being used in the field of Disaster Assessment and Response.

摘要: 地震、野火和飓风等自然灾害对人类生命和基础设施资产构成巨大风险。有效应对灾难取决于快速有效评估损害强度的能力。人工智能（AI）和生成人工智能（GenAI）提供了一种突破性的解决方案，能够结合来自多种类型和数据来源的知识，模拟现实的灾难场景，并以以前难以想象的速度识别新兴趋势。在本文中，我们全面回顾了人工智能和GenAI在各种自然灾害损害评估中的前景，强调了其优势和局限性。我们讨论了它在文本、图像、视频和音频等多模式数据中的应用，还涵盖了数据隐私、安全和危机期间技术的道德使用等主要问题。该论文还认识到了生成性人工智能滥用的威胁，其形式是传播错误信息和对抗性攻击。最后，我们概述了未来研究的途径，强调需要安全、可靠和道德的生成性人工智能系统来进行总体灾难管理。我们相信这项工作代表了对灾难评估和响应领域使用的Gen-AI技术的首次全面调查。



## **25. FlippedRAG: Black-Box Opinion Manipulation Adversarial Attacks to Retrieval-Augmented Generation Models**

FlippedRAG：黑匣子观点操纵对检索增强一代模型的对抗攻击 cs.IR

arXiv admin note: text overlap with arXiv:2407.13757

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2501.02968v3) [paper-pdf](http://arxiv.org/pdf/2501.02968v3)

**Authors**: Zhuo Chen, Jiawei Liu, Yuyang Gong, Miaokun Chen, Haotan Liu, Qikai Cheng, Fan Zhang, Wei Lu, Xiaozhong Liu, Xiaofeng Wang

**Abstract**: Retrieval-Augmented Generation (RAG) enriches LLMs by dynamically retrieving external knowledge, reducing hallucinations and satisfying real-time information needs. While existing research mainly targets RAG's performance and efficiency, emerging studies highlight critical security concerns. Yet, current adversarial approaches remain limited, mostly addressing white-box scenarios or heuristic black-box attacks without fully investigating vulnerabilities in the retrieval phase. Additionally, prior works mainly focus on factoid QA tasks, their attacks lack complexity and can be easily corrected by advanced LLMs. In this paper, we investigate a more realistic and critical threat scenario: adversarial attacks intended for opinion manipulation against black-box RAG models, particularly on controversial topics. Specifically, we propose FlippedRAG, a transfer-based adversarial attack against black-box RAG systems. We first demonstrate that the underlying retriever of a black-box RAG system can be reverse-engineered, enabling us to train a surrogate retriever. Leveraging the surrogate retriever, we further craft target poisoning triggers, altering vary few documents to effectively manipulate both retrieval and subsequent generation. Extensive empirical results show that FlippedRAG substantially outperforms baseline methods, improving the average attack success rate by 16.7%. FlippedRAG achieves on average a 50% directional shift in the opinion polarity of RAG-generated responses, ultimately causing a notable 20% shift in user cognition. Furthermore, we evaluate the performance of several potential defensive measures, concluding that existing mitigation strategies remain insufficient against such sophisticated manipulation attacks. These results highlight an urgent need for developing innovative defensive solutions to ensure the security and trustworthiness of RAG systems.

摘要: 检索增强生成（RAG）通过动态检索外部知识，减少幻觉和满足实时信息需求来丰富LLM。虽然现有的研究主要针对RAG的性能和效率，但新兴的研究突出了关键的安全问题。然而，目前的对抗性方法仍然有限，主要是解决白盒场景或启发式黑盒攻击，而没有充分调查检索阶段的漏洞。此外，以前的工作主要集中在事实问答任务，他们的攻击缺乏复杂性，可以很容易地纠正先进的LLM。在本文中，我们研究了一种更现实、更关键的威胁场景：旨在针对黑匣子RAG模型进行意见操纵的对抗性攻击，特别是在有争议的主题上。具体来说，我们提出了FlippedRAG，这是一种针对黑匣子RAG系统的基于传输的对抗攻击。我们首先证明黑匣子RAG系统的底层检索器可以进行反向工程，使我们能够训练代理检索器。利用替代检索器，我们进一步制作目标中毒触发器，改变不同的少数文档，以有效地操纵检索和后续生成。广泛的实证结果表明，FlippedRAG的性能大大优于基线方法，将平均攻击成功率提高了16.7%。FlippedRAG平均实现了RAG生成的响应的意见两极50%的方向性转变，最终导致用户认知发生了20%的显着转变。此外，我们评估了几种潜在防御措施的性能，得出的结论是，现有的缓解策略仍然不足以应对此类复杂的操纵攻击。这些结果凸显了开发创新防御解决方案的迫切需要，以确保RAG系统的安全性和可信性。



## **26. PoisonCatcher: Revealing and Identifying LDP Poisoning Attacks in IIoT**

Poison Catcher：揭露和识别IIoT中的自民党中毒攻击 cs.CR

14 pages,7 figures, 2 tables

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2412.15704v2) [paper-pdf](http://arxiv.org/pdf/2412.15704v2)

**Authors**: Lisha Shuai, Shaofeng Tan, Nan Zhang, Jiamin Zhang, Min Zhang, Xiaolong Yang

**Abstract**: Local Differential Privacy (LDP), a robust privacy-protection model, is widely adopted in the Industrial Internet of Things (IIoT) due to its lightweight, decentralized, and scalable. However, its perturbation-based privacy-protection mechanism hinders distinguishing between any two data, thereby facilitating LDP poisoning attacks. The exposed physical-layer vulnerabilities and resource-constrained prevalent at the IIoT edge not only facilitate such attacks but also render existing LDP poisoning defenses, all of which are deployed at the edge and rely on ample resources, impractical.   This work proposes a LDP poisoning defense for IIoT in the resource-rich aggregator. We first reveal key poisoning attack modes occurring within the LDP-utilized IIoT data-collection process, detailing how IIoT vulnerabilities enable attacks, and then formulate a general attack model and derive the poisoned data's indistinguishability. This work subsequently analyzes the poisoning impacts on aggregated data based on industrial process correlation, revealing the distortion of statistical query results' temporal similarity and the resulting disruption of inter-attribute correlation, and uncovering the intriguing paradox that adversaries' attempts to stabilize their poisoning actions for stealth are difficult to maintain. Given these findings, we propose PoisonCatcher, a solution for identifying poisoned data, which includes time-series detectors based on temporal similarity, attribute correlation, and pattern stability metrics to detect poisoned attributes, and a latent-bias feature miner for identifying poisons. Experiments on the real-world dataset indicate that PoisonCatcher successfully identifies poisoned data, demonstrating robust identification capabilities with F2 scores above 90.7\% under various attack settings.

摘要: 本地差异隐私（LDP）是一种强大的隐私保护模型，因其轻量级、去中心化和可扩展性而在工业物联网（IIoT）中广泛采用。然而，其基于扰动的隐私保护机制阻碍了区分任何两个数据，从而促进了SDP中毒攻击。IIoT边缘普遍存在的暴露物理层漏洞和资源限制不仅助长了此类攻击，而且还使现有的SDP中毒防御变得不切实际，所有这些防御都部署在边缘并依赖充足的资源。   这项工作为资源丰富的聚合器中的IIoT提出了一种自民党中毒防御。我们首先揭示了在LDP利用的IIoT数据收集过程中发生的关键中毒攻击模式，详细说明了IIoT漏洞如何实现攻击，然后制定了一个通用的攻击模型，并推导出中毒数据的不可恢复性。这项工作随后基于工业过程相关性分析了中毒对聚合数据的影响，揭示了统计查询结果时间相似性的扭曲以及由此产生的属性间相关性的破坏，并揭示了一个有趣的悖论，即对手试图稳定其中毒行为以进行潜行是难以维持的。鉴于这些发现，我们提出了PoisonCatcher，这是一种识别中毒数据的解决方案，其中包括基于时间相似性、属性相关性和模式稳定性指标的时间序列检测器来检测中毒属性，以及用于识别毒物的潜在偏差特征挖掘器。对现实世界数据集的实验表明，PoisonCatcher成功识别出中毒数据，展示了强大的识别能力，在各种攻击设置下F2得分高于90.7%。



## **27. Sharp Gaussian approximations for Decentralized Federated Learning**

分散式联邦学习的尖锐高斯逼近 stat.ML

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.08125v1) [paper-pdf](http://arxiv.org/pdf/2505.08125v1)

**Authors**: Soham Bonnerjee, Sayar Karmakar, Wei Biao Wu

**Abstract**: Federated Learning has gained traction in privacy-sensitive collaborative environments, with local SGD emerging as a key optimization method in decentralized settings. While its convergence properties are well-studied, asymptotic statistical guarantees beyond convergence remain limited. In this paper, we present two generalized Gaussian approximation results for local SGD and explore their implications. First, we prove a Berry-Esseen theorem for the final local SGD iterates, enabling valid multiplier bootstrap procedures. Second, motivated by robustness considerations, we introduce two distinct time-uniform Gaussian approximations for the entire trajectory of local SGD. The time-uniform approximations support Gaussian bootstrap-based tests for detecting adversarial attacks. Extensive simulations are provided to support our theoretical results.

摘要: 联邦学习在隐私敏感的协作环境中获得了吸引力，本地SGD成为分散环境中的关键优化方法。虽然它的收敛性得到了很好的研究，但超越收敛的渐近统计保证仍然有限。在本文中，我们提出了两个广义高斯近似结果的局部SGD，并探讨其含义。首先，我们证明了最终的本地SGD迭代的Berry-Esseen定理，使有效的乘数引导程序。其次，出于鲁棒性的考虑，我们引入了两个不同的时间均匀高斯近似的整个轨迹的本地SGD。时间均匀近似支持基于高斯引导的测试，用于检测对抗性攻击。提供了大量的模拟来支持我们的理论结果。



## **28. LiteLMGuard: Seamless and Lightweight On-Device Prompt Filtering for Safeguarding Small Language Models against Quantization-induced Risks and Vulnerabilities**

LiteLMGGuard：无缝且轻量级的设备上提示过滤，用于保护小语言模型免受量化引发的风险和漏洞的影响 cs.CR

14 pages, 18 figures, and 4 tables

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.05619v2) [paper-pdf](http://arxiv.org/pdf/2505.05619v2)

**Authors**: Kalyan Nakka, Jimmy Dani, Ausmit Mondal, Nitesh Saxena

**Abstract**: The growing adoption of Large Language Models (LLMs) has influenced the development of their lighter counterparts-Small Language Models (SLMs)-to enable on-device deployment across smartphones and edge devices. These SLMs offer enhanced privacy, reduced latency, server-free functionality, and improved user experience. However, due to resource constraints of on-device environment, SLMs undergo size optimization through compression techniques like quantization, which can inadvertently introduce fairness, ethical and privacy risks. Critically, quantized SLMs may respond to harmful queries directly, without requiring adversarial manipulation, raising significant safety and trust concerns.   To address this, we propose LiteLMGuard (LLMG), an on-device prompt guard that provides real-time, prompt-level defense for quantized SLMs. Additionally, our prompt guard is designed to be model-agnostic such that it can be seamlessly integrated with any SLM, operating independently of underlying architectures. Our LLMG formalizes prompt filtering as a deep learning (DL)-based prompt answerability classification task, leveraging semantic understanding to determine whether a query should be answered by any SLM. Using our curated dataset, Answerable-or-Not, we trained and fine-tuned several DL models and selected ELECTRA as the candidate, with 97.75% answerability classification accuracy.   Our safety effectiveness evaluations demonstrate that LLMG defends against over 87% of harmful prompts, including both direct instruction and jailbreak attack strategies. We further showcase its ability to mitigate the Open Knowledge Attacks, where compromised SLMs provide unsafe responses without adversarial prompting. In terms of prompt filtering effectiveness, LLMG achieves near state-of-the-art filtering accuracy of 94%, with an average latency of 135 ms, incurring negligible overhead for users.

摘要: 大型语言模型（LLM）的日益采用影响了其更轻的同类产品--小型语言模型（SLM）--的发展，以实现跨智能手机和边缘设备的设备上部署。这些STM提供增强的隐私、减少的延迟、无服务器功能和改善的用户体验。然而，由于设备上环境的资源限制，STM通过量化等压缩技术进行尺寸优化，这可能会无意中引入公平性、道德和隐私风险。至关重要的是，量化的SLC可以直接响应有害查询，而不需要对抗性操纵，从而引发重大的安全和信任问题。   为了解决这个问题，我们提出了LiteLMGard（LLMG），这是一种设备上提示保护，为量化的STM提供实时、预算级防御。此外，我们的提示卫士设计为模型不可知，因此它可以与任何SPL无缝集成，独立于底层架构运行。我们的LLMG将提示过滤形式化为基于深度学习（DL）的提示可回答性分类任务，利用语义理解来确定查询是否应该由任何SPL回答。使用我们精心策划的数据集“可供选择”，我们训练和微调了几个DL模型，并选择ELECTRA作为候选模型，其回答性分类准确率为97.75%。   我们的安全有效性评估表明，LLMG可以抵御超过87%的有害提示，包括直接指令和越狱攻击策略。我们进一步展示了其缓解开放知识攻击的能力，其中受攻击的STM在没有对抗提示的情况下提供不安全的响应。在即时过滤有效性方面，LLMG实现了94%的接近最先进的过滤准确率，平均延迟为135 ms，为用户带来的负担可以忽略不计。



## **29. Dynamical Low-Rank Compression of Neural Networks with Robustness under Adversarial Attacks**

对抗攻击下具有鲁棒性的神经网络动态低阶压缩 cs.LG

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.08022v1) [paper-pdf](http://arxiv.org/pdf/2505.08022v1)

**Authors**: Steffen Schotthöfer, H. Lexie Yang, Stefan Schnake

**Abstract**: Deployment of neural networks on resource-constrained devices demands models that are both compact and robust to adversarial inputs. However, compression and adversarial robustness often conflict. In this work, we introduce a dynamical low-rank training scheme enhanced with a novel spectral regularizer that controls the condition number of the low-rank core in each layer. This approach mitigates the sensitivity of compressed models to adversarial perturbations without sacrificing clean accuracy. The method is model- and data-agnostic, computationally efficient, and supports rank adaptivity to automatically compress the network at hand. Extensive experiments across standard architectures, datasets, and adversarial attacks show the regularized networks can achieve over 94% compression while recovering or improving adversarial accuracy relative to uncompressed baselines.

摘要: 在资源受限的设备上部署神经网络需要紧凑且对对抗输入稳健的模型。然而，压缩和对抗鲁棒性经常发生冲突。在这项工作中，我们引入了一种动态低等级训练方案，该方案通过新型谱正规化器增强，该算法控制每层中低等级核心的条件数。这种方法降低了压缩模型对对抗性扰动的敏感性，而不会牺牲清晰的准确性。该方法与模型和数据无关，计算效率高，并且支持等级自适应性以自动压缩手头的网络。跨标准架构、数据集和对抗性攻击的广泛实验表明，正规化网络可以实现超过94%的压缩，同时恢复或提高相对于未压缩基线的对抗性准确性。



## **30. SCA: Improve Semantic Consistent in Unrestricted Adversarial Attacks via DDPM Inversion**

SCA：通过DDPM倒置提高无限制对抗攻击中的语义一致性 cs.CV

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2410.02240v6) [paper-pdf](http://arxiv.org/pdf/2410.02240v6)

**Authors**: Zihao Pan, Lifeng Chen, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Systems based on deep neural networks are vulnerable to adversarial attacks. Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often result in substantial semantic distortions in the denoised output and suffer from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes a Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our code can be found at https://github.com/Pan-Zihao/SCA.

摘要: 基于深度神经网络的系统很容易受到对抗攻击。不受限制的对抗攻击通常操纵图像的语义内容（例如，颜色或纹理）来创建既有效又逼真的对抗示例。最近的作品利用扩散倒置过程将图像映射到潜在空间，其中通过引入扰动来操纵高级语义。然而，它们通常会导致去噪输出中出现严重的语义扭曲，并且效率低下。在这项研究中，我们提出了一种名为语义一致的无限制对抗攻击（SCA）的新型框架，该框架采用倒置方法来提取编辑友好的噪音图，并利用多模式大型语言模型（MLLM）来提供整个过程的语义指导。在MLLM提供丰富的语义信息的情况下，我们使用一系列编辑友好的噪音图来执行每一步的DDPM去噪过程，并利用DeliverSolver ++加速这一过程，实现具有语义一致性的高效采样。与现有方法相比，我们的框架能够高效生成表现出最小可辨别的语义变化的对抗性示例。因此，我们首次引入语义一致的对抗示例（SCAE）。大量的实验和可视化已经证明了SCA的高效率，特别是平均比最先进的攻击快12倍。我们的代码可在https://github.com/Pan-Zihao/SCA上找到。



## **31. Must Read: A Systematic Survey of Computational Persuasion**

必读：计算说服的系统调查 cs.CL

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.07775v1) [paper-pdf](http://arxiv.org/pdf/2505.07775v1)

**Authors**: Nimet Beyza Bozdag, Shuhaib Mehri, Xiaocheng Yang, Hyeonjeong Ha, Zirui Cheng, Esin Durmus, Jiaxuan You, Heng Ji, Gokhan Tur, Dilek Hakkani-Tür

**Abstract**: Persuasion is a fundamental aspect of communication, influencing decision-making across diverse contexts, from everyday conversations to high-stakes scenarios such as politics, marketing, and law. The rise of conversational AI systems has significantly expanded the scope of persuasion, introducing both opportunities and risks. AI-driven persuasion can be leveraged for beneficial applications, but also poses threats through manipulation and unethical influence. Moreover, AI systems are not only persuaders, but also susceptible to persuasion, making them vulnerable to adversarial attacks and bias reinforcement. Despite rapid advancements in AI-generated persuasive content, our understanding of what makes persuasion effective remains limited due to its inherently subjective and context-dependent nature. In this survey, we provide a comprehensive overview of computational persuasion, structured around three key perspectives: (1) AI as a Persuader, which explores AI-generated persuasive content and its applications; (2) AI as a Persuadee, which examines AI's susceptibility to influence and manipulation; and (3) AI as a Persuasion Judge, which analyzes AI's role in evaluating persuasive strategies, detecting manipulation, and ensuring ethical persuasion. We introduce a taxonomy for computational persuasion research and discuss key challenges, including evaluating persuasiveness, mitigating manipulative persuasion, and developing responsible AI-driven persuasive systems. Our survey outlines future research directions to enhance the safety, fairness, and effectiveness of AI-powered persuasion while addressing the risks posed by increasingly capable language models.

摘要: 说服是沟通的一个基本方面，影响不同背景下的决策，从日常对话到政治、营销和法律等高风险场景。对话人工智能系统的兴起显着扩大了说服的范围，带来了机会和风险。人工智能驱动的说服可以用于有益的应用，但也会通过操纵和不道德影响构成威胁。此外，人工智能系统不仅是说服者，而且容易受到说服，使它们容易受到对抗攻击和偏见强化的影响。尽管人工智能生成的说服性内容取得了迅速的进步，但由于其固有的主观性和上下文依赖性，我们对说服有效的理解仍然有限。在这项调查中，我们围绕三个关键角度提供了计算说服力的全面概述：（1）人工智能作为说服者，探索人工智能生成的说服力内容及其应用;（2）人工智能作为说服者，考察人工智能对影响力和操纵的敏感性;（3）人工智能作为说服法官，分析人工智能在评估说服策略、检测操纵和确保道德说服方面的作用。我们介绍了计算说服研究的分类法，并讨论了关键挑战，包括评估说服力、减轻操纵说服以及开发负责任的人工智能驱动说服系统。我们的调查概述了未来的研究方向，以增强人工智能说服的安全性、公平性和有效性，同时解决能力日益增强的语言模型带来的风险。



## **32. Trial and Trust: Addressing Byzantine Attacks with Comprehensive Defense Strategy**

审判与信任：以全面的防御战略应对拜占庭袭击 cs.LG

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.07614v1) [paper-pdf](http://arxiv.org/pdf/2505.07614v1)

**Authors**: Gleb Molodtsov, Daniil Medyakov, Sergey Skorik, Nikolas Khachaturov, Shahane Tigranyan, Vladimir Aletov, Aram Avetisyan, Martin Takáč, Aleksandr Beznosikov

**Abstract**: Recent advancements in machine learning have improved performance while also increasing computational demands. While federated and distributed setups address these issues, their structure is vulnerable to malicious influences. In this paper, we address a specific threat, Byzantine attacks, where compromised clients inject adversarial updates to derail global convergence. We combine the trust scores concept with trial function methodology to dynamically filter outliers. Our methods address the critical limitations of previous approaches, allowing functionality even when Byzantine nodes are in the majority. Moreover, our algorithms adapt to widely used scaled methods like Adam and RMSProp, as well as practical scenarios, including local training and partial participation. We validate the robustness of our methods by conducting extensive experiments on both synthetic and real ECG data collected from medical institutions. Furthermore, we provide a broad theoretical analysis of our algorithms and their extensions to aforementioned practical setups. The convergence guarantees of our methods are comparable to those of classical algorithms developed without Byzantine interference.

摘要: 机器学习的最新进展提高了性能，同时也增加了计算需求。虽然联邦和分布式设置可以解决这些问题，但其结构很容易受到恶意影响。在本文中，我们解决了一个特定的威胁，即拜占庭攻击，其中受影响的客户端注入对抗性更新以破坏全球融合。我们将信任分数概念与尝试函数方法相结合，以动态过滤离群值。我们的方法解决了以前方法的关键局限性，即使在拜占庭节点占多数时也允许功能。此外，我们的算法适用于Adam和RMSProp等广泛使用的缩放方法，以及实际场景，包括本地训练和部分参与。我们通过对从医疗机构收集的合成和真实心电图数据进行广泛的实验来验证我们方法的稳健性。此外，我们还对算法及其对上述实际设置的扩展进行了广泛的理论分析。我们方法的收敛保证与没有拜占庭干扰而开发的经典算法的收敛保证相当。



## **33. SecReEvalBench: A Multi-turned Security Resilience Evaluation Benchmark for Large Language Models**

SecReEvalBench：大型语言模型的多角度安全弹性评估基准 cs.CR

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.07584v1) [paper-pdf](http://arxiv.org/pdf/2505.07584v1)

**Authors**: Huining Cui, Wei Liu

**Abstract**: The increasing deployment of large language models in security-sensitive domains necessitates rigorous evaluation of their resilience against adversarial prompt-based attacks. While previous benchmarks have focused on security evaluations with limited and predefined attack domains, such as cybersecurity attacks, they often lack a comprehensive assessment of intent-driven adversarial prompts and the consideration of real-life scenario-based multi-turn attacks. To address this gap, we present SecReEvalBench, the Security Resilience Evaluation Benchmark, which defines four novel metrics: Prompt Attack Resilience Score, Prompt Attack Refusal Logic Score, Chain-Based Attack Resilience Score and Chain-Based Attack Rejection Time Score. Moreover, SecReEvalBench employs six questioning sequences for model assessment: one-off attack, successive attack, successive reverse attack, alternative attack, sequential ascending attack with escalating threat levels and sequential descending attack with diminishing threat levels. In addition, we introduce a dataset customized for the benchmark, which incorporates both neutral and malicious prompts, categorised across seven security domains and sixteen attack techniques. In applying this benchmark, we systematically evaluate five state-of-the-art open-weighted large language models, Llama 3.1, Gemma 2, Mistral v0.3, DeepSeek-R1 and Qwen 3. Our findings offer critical insights into the strengths and weaknesses of modern large language models in defending against evolving adversarial threats. The SecReEvalBench dataset is publicly available at https://kaggle.com/datasets/5a7ee22cf9dab6c93b55a73f630f6c9b42e936351b0ae98fbae6ddaca7fe248d, which provides a groundwork for advancing research in large language model security.

摘要: 大型语言模型在安全敏感领域的部署越来越多，需要严格评估它们对抗基于预算的敌对攻击的弹性。虽然之前的基准侧重于有限且预定义的攻击域（例如网络安全攻击）的安全评估，但它们通常缺乏对意图驱动的对抗提示的全面评估以及对现实生活中基于情景的多回合攻击的考虑。为了解决这一差距，我们提出了SecReEvalBench，安全韧性评估基准，它定义了四个新颖的指标：即时攻击韧性分数、即时攻击拒绝逻辑分数、基于链的攻击韧性分数和基于链的攻击拒绝时间分数。此外，SecReEvalBench采用六个提问序列进行模型评估：一次性攻击、连续攻击、连续反向攻击、替代攻击、威胁级别不断上升的顺序上升攻击和威胁级别不断下降的顺序下降攻击。此外，我们还引入了一个为基准定制的数据集，其中包含中性和恶意提示，分为七个安全域和十六种攻击技术。在应用该基准时，我们系统地评估了五个最先进的开放加权大型语言模型：Llama 3.1、Gemma 2、Mistral v0.3、DeepSeek-R1和Qwen 3。我们的研究结果为现代大型语言模型在防御不断变化的对抗威胁方面的优势和弱点提供了重要的见解。SecReEvalBench数据集可在https：//kaggle.com/guardets/5a7ee22CF9dab6c93b55a73f630f6c9 b42 e936351 b 0ae 98 fbae 6ddaca 7 fe 248 d上公开，为推进大型语言模型安全性研究提供了基础。



## **34. GRADA: Graph-based Reranker against Adversarial Documents Attack**

GRADA：针对对抗文档攻击的基于图形的重新搜索器 cs.IR

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.07546v1) [paper-pdf](http://arxiv.org/pdf/2505.07546v1)

**Authors**: Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xuanli He, Pasquale Minervini, Youcheng Sun, Qiongkai Xu

**Abstract**: Retrieval Augmented Generation (RAG) frameworks improve the accuracy of large language models (LLMs) by integrating external knowledge from retrieved documents, thereby overcoming the limitations of models' static intrinsic knowledge. However, these systems are susceptible to adversarial attacks that manipulate the retrieval process by introducing documents that are adversarial yet semantically similar to the query. Notably, while these adversarial documents resemble the query, they exhibit weak similarity to benign documents in the retrieval set. Thus, we propose a simple yet effective Graph-based Reranking against Adversarial Document Attacks (GRADA) framework aiming at preserving retrieval quality while significantly reducing the success of adversaries. Our study evaluates the effectiveness of our approach through experiments conducted on five LLMs: GPT-3.5-Turbo, GPT-4o, Llama3.1-8b, Llama3.1-70b, and Qwen2.5-7b. We use three datasets to assess performance, with results from the Natural Questions dataset demonstrating up to an 80% reduction in attack success rates while maintaining minimal loss in accuracy.

摘要: 检索增强生成（RAG）框架通过集成来自检索文档的外部知识来提高大型语言模型（LLM）的准确性，从而克服模型静态内在知识的局限性。然而，这些系统很容易受到对抗性攻击，这些攻击通过引入对抗性但在语义上与查询相似的文档来操纵检索过程。值得注意的是，虽然这些对抗性文档类似于查询，但它们与检索集中的良性文档表现出弱的相似性。因此，我们提出了一个简单而有效的基于图形的对抗性文档攻击重新排名（GRADA）框架，旨在保留检索质量，同时显着降低对手的成功。我们的研究通过在五个LLM上进行的实验来评估我们方法的有效性：GPT-3.5-Turbo、GPT-4 o、Llama 3.1 -8b、Llama 3.1 - 70 b和Qwen 2.5 - 7 b。我们使用三个数据集来评估性能，Natural Questions数据集的结果表明，攻击成功率可降低高达80%，同时保持最小的准确性损失。



## **35. Beyond Boundaries: A Comprehensive Survey of Transferable Attacks on AI Systems**

超越边界：对人工智能系统的可转移攻击的全面调查 cs.CR

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2311.11796v2) [paper-pdf](http://arxiv.org/pdf/2311.11796v2)

**Authors**: Guangjing Wang, Ce Zhou, Yuanda Wang, Bocheng Chen, Hanqing Guo, Qiben Yan

**Abstract**: As Artificial Intelligence (AI) systems increasingly underpin critical applications, from autonomous vehicles to biometric authentication, their vulnerability to transferable attacks presents a growing concern. These attacks, designed to generalize across instances, domains, models, tasks, modalities, or even hardware platforms, pose severe risks to security, privacy, and system integrity. This survey delivers the first comprehensive review of transferable attacks across seven major categories, including evasion, backdoor, data poisoning, model stealing, model inversion, membership inference, and side-channel attacks. We introduce a unified six-dimensional taxonomy: cross-instance, cross-domain, cross-modality, cross-model, cross-task, and cross-hardware, which systematically captures the diverse transfer pathways of adversarial strategies. Through this framework, we examine both the underlying mechanics and practical implications of transferable attacks on AI systems. Furthermore, we review cutting-edge methods for enhancing attack transferability, organized around data augmentation and optimization strategies. By consolidating fragmented research and identifying critical future directions, this work provides a foundational roadmap for understanding, evaluating, and defending against transferable threats in real-world AI systems.

摘要: 随着人工智能（AI）系统越来越多地支持从自动驾驶汽车到生物识别认证的关键应用，它们对可转移攻击的脆弱性引起了越来越多的担忧。这些攻击旨在跨实例、域、模型、任务、模式甚至硬件平台进行推广，对安全性、隐私和系统完整性构成了严重风险。这项调查首次全面审查了七个主要类别的可转移攻击，包括规避、后门、数据中毒、模型窃取、模型倒置、成员资格推断和侧通道攻击。我们引入了统一的六维分类法：跨实例、跨领域、跨模式、跨模型、跨任务和跨硬件，它系统地捕捉对抗策略的不同转移途径。通过这个框架，我们研究了对人工智能系统的可转移攻击的潜在机制和实际影响。此外，我们还回顾了增强攻击可转移性的尖端方法，并围绕数据增强和优化策略进行组织。通过整合碎片化的研究并确定关键的未来方向，这项工作为理解、评估和防御现实世界人工智能系统中的可转移威胁提供了基础路线图。



## **36. Decentralized Adversarial Training over Graphs**

图上的分散对抗训练 cs.LG

arXiv admin note: text overlap with arXiv:2303.01936

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2303.13326v3) [paper-pdf](http://arxiv.org/pdf/2303.13326v3)

**Authors**: Ying Cao, Elsa Rizk, Stefan Vlaski, Ali H. Sayed

**Abstract**: The vulnerability of machine learning models to adversarial attacks has been attracting considerable attention in recent years. Most existing studies focus on the behavior of stand-alone single-agent learners. In comparison, this work studies adversarial training over graphs, where individual agents are subjected to perturbations of varied strength levels across space. It is expected that interactions by linked agents, and the heterogeneity of the attack models that are possible over the graph, can help enhance robustness in view of the coordination power of the group. Using a min-max formulation of distributed learning, we develop a decentralized adversarial training framework for multi-agent systems. Specifically, we devise two decentralized adversarial training algorithms by relying on two popular decentralized learning strategies--diffusion and consensus. We analyze the convergence properties of the proposed framework for strongly-convex, convex, and non-convex environments, and illustrate the enhanced robustness to adversarial attacks.

摘要: 近年来，机器学习模型对对抗性攻击的脆弱性引起了相当大的关注。现有的大多数研究集中在独立的单智能体学习者的行为。相比之下，这项工作研究了图上的对抗训练，其中单个代理受到空间上不同强度水平的扰动。预计链接代理的交互以及图上可能的攻击模型的异质性可以帮助增强组的协调能力的鲁棒性。使用分布式学习的最小-最大公式，我们为多智能体系统开发了一个去中心化的对抗训练框架。具体来说，我们通过依赖两种流行的去中心化学习策略--扩散和共识，设计了两种去中心化对抗训练算法。我们分析了所提出的框架在强凸、凸和非凸环境下的收敛特性，并说明了对对抗攻击的增强的鲁棒性。



## **37. AttackBench: Evaluating Gradient-based Attacks for Adversarial Examples**

AttackBench：评估对抗示例的基于恶意的攻击 cs.LG

Paper accepted at AAAI2025. Project page and leaderboard:  https://attackbench.github.io

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2404.19460v3) [paper-pdf](http://arxiv.org/pdf/2404.19460v3)

**Authors**: Antonio Emanuele Cinà, Jérôme Rony, Maura Pintor, Luca Demetrio, Ambra Demontis, Battista Biggio, Ismail Ben Ayed, Fabio Roli

**Abstract**: Adversarial examples are typically optimized with gradient-based attacks. While novel attacks are continuously proposed, each is shown to outperform its predecessors using different experimental setups, hyperparameter settings, and number of forward and backward calls to the target models. This provides overly-optimistic and even biased evaluations that may unfairly favor one particular attack over the others. In this work, we aim to overcome these limitations by proposing AttackBench, i.e., the first evaluation framework that enables a fair comparison among different attacks. To this end, we first propose a categorization of gradient-based attacks, identifying their main components and differences. We then introduce our framework, which evaluates their effectiveness and efficiency. We measure these characteristics by (i) defining an optimality metric that quantifies how close an attack is to the optimal solution, and (ii) limiting the number of forward and backward queries to the model, such that all attacks are compared within a given maximum query budget. Our extensive experimental analysis compares more than $100$ attack implementations with a total of over $800$ different configurations against CIFAR-10 and ImageNet models, highlighting that only very few attacks outperform all the competing approaches. Within this analysis, we shed light on several implementation issues that prevent many attacks from finding better solutions or running at all. We release AttackBench as a publicly-available benchmark, aiming to continuously update it to include and evaluate novel gradient-based attacks for optimizing adversarial examples.

摘要: 对抗性示例通常通过基于梯度的攻击进行优化。虽然新型攻击不断被提出，但使用不同的实验设置、超参数设置以及对目标模型的前向和后向调用次数，每种攻击都优于其前辈。这提供了过于乐观甚至有偏见的评估，可能不公平地支持一种特定的攻击而不是其他攻击。在这项工作中，我们的目标是通过提出AttackBench来克服这些限制，即第一个评估框架，可以公平地比较不同的攻击。为此，我们首先提出了基于梯度的攻击的分类，确定其主要组成部分和差异。然后我们介绍我们的框架，该框架评估其有效性和效率。我们通过（i）定义一个最佳性指标来衡量这些特征，该指标量化攻击与最佳解决方案的接近程度，以及（ii）限制对模型的前向和后向查询的数量，以便在给定的最大查询预算内比较所有攻击。我们广泛的实验分析将超过100美元的攻击实施与针对CIFAR-10和ImageNet模型的总计超过800美元的不同配置进行了比较，强调只有极少数攻击的性能优于所有竞争方法。在此分析中，我们揭示了几个实现问题，这些问题阻碍了许多攻击找到更好的解决方案或根本无法运行。我们将AttackBench作为公开基准发布，旨在不断更新它，以纳入和评估新颖的基于梯度的攻击，以优化对抗性示例。



## **38. No Query, No Access**

无查询，无访问 cs.CL

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.07258v1) [paper-pdf](http://arxiv.org/pdf/2505.07258v1)

**Authors**: Wenqiang Wang, Siyuan Liang, Yangshijie Zhang, Xiaojun Jia, Hao Lin, Xiaochun Cao

**Abstract**: Textual adversarial attacks mislead NLP models, including Large Language Models (LLMs), by subtly modifying text. While effective, existing attacks often require knowledge of the victim model, extensive queries, or access to training data, limiting real-world feasibility. To overcome these constraints, we introduce the \textbf{Victim Data-based Adversarial Attack (VDBA)}, which operates using only victim texts. To prevent access to the victim model, we create a shadow dataset with publicly available pre-trained models and clustering methods as a foundation for developing substitute models. To address the low attack success rate (ASR) due to insufficient information feedback, we propose the hierarchical substitution model design, generating substitute models to mitigate the failure of a single substitute model at the decision boundary.   Concurrently, we use diverse adversarial example generation, employing various attack methods to generate and select the adversarial example with better similarity and attack effectiveness. Experiments on the Emotion and SST5 datasets show that VDBA outperforms state-of-the-art methods, achieving an ASR improvement of 52.08\% while significantly reducing attack queries to 0. More importantly, we discover that VDBA poses a significant threat to LLMs such as Qwen2 and the GPT family, and achieves the highest ASR of 45.99% even without access to the API, confirming that advanced NLP models still face serious security risks. Our codes can be found at https://anonymous.4open.science/r/VDBA-Victim-Data-based-Adversarial-Attack-36EC/

摘要: 文本对抗攻击通过微妙地修改文本来误导NLP模型，包括大型语言模型（LLM）。虽然有效，但现有的攻击通常需要了解受害者模型、广泛的查询或访问训练数据，从而限制了现实世界的可行性。为了克服这些限制，我们引入了\textBF{基于受害者数据的对抗攻击（VDBA）}，它仅使用受害者文本来操作。为了防止访问受害者模型，我们创建了一个影子数据集，其中包含公开可用的预训练模型和集群方法，作为开发替代模型的基础。为了解决由于信息反馈不足而导致的攻击成功率（ASB）低的问题，我们提出了分层替代模型设计，生成替代模型以减轻单个替代模型在决策边界的失败。   同时，我们使用多样化的对抗性示例生成，采用各种攻击方法来生成并选择具有更好相似性和攻击有效性的对抗性示例。Emoy和CST 5数据集的实验表明，VDBA优于最先进的方法，实现了52.08%的ASB改进，同时将攻击查询显着减少到0。更重要的是，我们发现VDBA对Qwen 2和GPT系列等LLM构成了重大威胁，即使在不访问API的情况下也能达到45.99%的最高ASB，证实高级NLP模型仍然面临严重的安全风险。我们的代码可在https://anonymous.4open.science/r/VDBA-Victim-Data-based-Adversarial-Attack-36EC/上找到



## **39. Approximate Energetic Resilience of Nonlinear Systems under Partial Loss of Control Authority**

部分失去控制权下非线性系统的能量弹性 math.OC

20 pages, 3 figures, 1 table

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2502.07603v2) [paper-pdf](http://arxiv.org/pdf/2502.07603v2)

**Authors**: Ram Padmanabhan, Melkior Ornik

**Abstract**: In this paper, we quantify the resilience of nonlinear dynamical systems by studying the increased energy used by all inputs of a system that suffers a partial loss of control authority, either through actuator malfunctions or through adversarial attacks. To quantify the maximal increase in energy, we introduce the notion of an energetic resilience metric. Prior work in this particular setting does not consider general nonlinear dynamical systems. In developing this framework, we first consider the special case of linear driftless systems and recall the energies in the control signal in the nominal and malfunctioning systems. Using these energies, we derive a bound on the energetic resilience metric. For general nonlinear systems, we first obtain a condition on the mean value of the control signal in both the nominal and malfunctioning systems, which allows us to approximate the energy in the control. We then obtain a worst-case approximation of this energy for the malfunctioning system, over all malfunctioning inputs. Assuming this approximation is exact, we derive bounds on the energetic resilience metric when control authority is lost over one actuator. A set of simulation examples demonstrate that the metric is useful in quantifying the resilience of the system without significant conservatism, despite the approximations used in obtaining control energies for nonlinear systems.

摘要: 在本文中，我们通过研究由于致动器故障或对抗性攻击而部分失去控制权的系统的所有输入所使用的能量增加，来量化非线性动态系统的弹性。为了量化能量的最大增加，我们引入了能量弹性指标的概念。之前在这个特定环境下的工作没有考虑一般的非线性动力系统。在开发这个框架时，我们首先考虑线性无漂移系统的特殊情况，并回忆正常和故障系统中控制信号中的能量。使用这些能量，我们推导出能量弹性指标的界限。对于一般非线性系统，我们首先获得正常和故障系统中控制信号平均值的条件，这使我们能够逼近控制中的能量。然后，我们在所有故障输入上获得故障系统的该能量的最坏情况近似值。假设这种逼近是精确的，当一个致动器失去控制权时，我们推导出能量弹性指标的界限。一组模拟示例表明，尽管在获取非线性系统的控制能量时使用了近似值，但该指标对于量化系统的弹性是有用的，而无需显着的保守性。



## **40. AugMixCloak: A Defense against Membership Inference Attacks via Image Transformation**

AugMixCloak：通过图像转换防御会员推断攻击 cs.LG

**SubmitDate**: 2025-05-11    [abs](http://arxiv.org/abs/2505.07149v1) [paper-pdf](http://arxiv.org/pdf/2505.07149v1)

**Authors**: Heqing Ren, Chao Feng, Alberto Huertas, Burkhard Stiller

**Abstract**: Traditional machine learning (ML) raises serious privacy concerns, while federated learning (FL) mitigates the risk of data leakage by keeping data on local devices. However, the training process of FL can still leak sensitive information, which adversaries may exploit to infer private data. One of the most prominent threats is the membership inference attack (MIA), where the adversary aims to determine whether a particular data record was part of the training set.   This paper addresses this problem through a two-stage defense called AugMixCloak. The core idea is to apply data augmentation and principal component analysis (PCA)-based information fusion to query images, which are detected by perceptual hashing (pHash) as either identical to or highly similar to images in the training set. Experimental results show that AugMixCloak successfully defends against both binary classifier-based MIA and metric-based MIA across five datasets and various decentralized FL (DFL) topologies. Compared with regularization-based defenses, AugMixCloak demonstrates stronger protection. Compared with confidence score masking, AugMixCloak exhibits better generalization.

摘要: 传统的机器学习（ML）会引发严重的隐私问题，而联邦学习（FL）则通过将数据保留在本地设备上来减轻数据泄露的风险。然而，FL的训练过程仍然可能泄露敏感信息，对手可能会利用这些信息来推断私人数据。最突出的威胁之一是成员推断攻击（MIA），对手的目标是确定特定数据记录是否是训练集的一部分。   本文通过名为AugMixCloak的两阶段防御来解决这个问题。核心思想是将数据增强和基于主成分分析（PCA）的信息融合应用于查询图像，通过感知哈希（p哈希）检测这些图像与训练集中的图像相同或高度相似。实验结果表明，AugMixCloak在五个数据集和各种去中心化FL（DFL）布局中成功防御基于二进制分类器的MIA和基于度量的MIA。与基于规则化的防御相比，AugMixCloak表现出更强的保护。与置信评分掩蔽相比，AugMixCloak表现出更好的概括性。



## **41. Unleashing the potential of prompt engineering for large language models**

释放大型语言模型即时工程的潜力 cs.CL

v6 - Metadata updated (title, journal ref, DOI). PDF identical to v5  (original submission). Please cite the peer-reviewed Version of Record in  "Patterns" (DOI: 10.1016/j.patter.2025.101260)

**SubmitDate**: 2025-05-11    [abs](http://arxiv.org/abs/2310.14735v6) [paper-pdf](http://arxiv.org/pdf/2310.14735v6)

**Authors**: Banghao Chen, Zhaofeng Zhang, Nicolas Langrené, Shengxin Zhu

**Abstract**: This comprehensive review delves into the pivotal role of prompt engineering in unleashing the capabilities of Large Language Models (LLMs). The development of Artificial Intelligence (AI), from its inception in the 1950s to the emergence of advanced neural networks and deep learning architectures, has made a breakthrough in LLMs, with models such as GPT-4o and Claude-3, and in Vision-Language Models (VLMs), with models such as CLIP and ALIGN. Prompt engineering is the process of structuring inputs, which has emerged as a crucial technique to maximize the utility and accuracy of these models. This paper explores both foundational and advanced methodologies of prompt engineering, including techniques such as self-consistency, chain-of-thought, and generated knowledge, which significantly enhance model performance. Additionally, it examines the prompt method of VLMs through innovative approaches such as Context Optimization (CoOp), Conditional Context Optimization (CoCoOp), and Multimodal Prompt Learning (MaPLe). Critical to this discussion is the aspect of AI security, particularly adversarial attacks that exploit vulnerabilities in prompt engineering. Strategies to mitigate these risks and enhance model robustness are thoroughly reviewed. The evaluation of prompt methods is also addressed through both subjective and objective metrics, ensuring a robust analysis of their efficacy. This review also reflects the essential role of prompt engineering in advancing AI capabilities, providing a structured framework for future research and application.

摘要: 这篇全面的评论深入探讨了即时工程在释放大型语言模型（LLM）功能方面的关键作用。人工智能（AI）的发展，从20世纪50年代的诞生到先进神经网络和深度学习架构的出现，在LLM（如GPT-4 o和Claude-3）以及视觉语言模型（VLM）（如CLIP和ALIGN）等模型方面取得了突破。即时工程是结构化输入的过程，它已成为最大限度地提高这些模型的实用性和准确性的关键技术。本文探讨了即时工程的基础和高级方法，包括自一致性、思想链和生成知识等技术，这些技术显着增强模型性能。此外，它还通过上下文优化（CoOp）、条件上下文优化（CoCoOp）和多模式提示学习（MaPLe）等创新方法研究了VLM的提示方法。本次讨论的关键是人工智能安全方面，特别是利用即时工程中漏洞的对抗攻击。彻底审查了缓解这些风险和增强模型稳健性的策略。提示方法的评估还通过主观和客观指标来解决，确保对其功效进行稳健的分析。这篇评论还反映了快速工程在推进人工智能能力方面的重要作用，为未来的研究和应用提供了一个结构化的框架。



## **42. IM-BERT: Enhancing Robustness of BERT through the Implicit Euler Method**

IM-BERT：通过隐式欧拉方法增强BERT的鲁棒性 cs.CL

Accepted to EMNLP 2024 Main

**SubmitDate**: 2025-05-11    [abs](http://arxiv.org/abs/2505.06889v1) [paper-pdf](http://arxiv.org/pdf/2505.06889v1)

**Authors**: Mihyeon Kim, Juhyoung Park, Youngbin Kim

**Abstract**: Pre-trained Language Models (PLMs) have achieved remarkable performance on diverse NLP tasks through pre-training and fine-tuning. However, fine-tuning the model with a large number of parameters on limited downstream datasets often leads to vulnerability to adversarial attacks, causing overfitting of the model on standard datasets.   To address these issues, we propose IM-BERT from the perspective of a dynamic system by conceptualizing a layer of BERT as a solution of Ordinary Differential Equations (ODEs). Under the situation of initial value perturbation, we analyze the numerical stability of two main numerical ODE solvers: the explicit and implicit Euler approaches.   Based on these analyses, we introduce a numerically robust IM-connection incorporating BERT's layers. This strategy enhances the robustness of PLMs against adversarial attacks, even in low-resource scenarios, without introducing additional parameters or adversarial training strategies.   Experimental results on the adversarial GLUE (AdvGLUE) dataset validate the robustness of IM-BERT under various conditions. Compared to the original BERT, IM-BERT exhibits a performance improvement of approximately 8.3\%p on the AdvGLUE dataset. Furthermore, in low-resource scenarios, IM-BERT outperforms BERT by achieving 5.9\%p higher accuracy.

摘要: 通过预训练和微调，预训练语言模型（PLM）在各种NLP任务上取得了显着的性能。然而，在有限的下游数据集上使用大量参数对模型进行微调通常会导致对抗性攻击的脆弱性，从而导致模型在标准数据集上的过拟合。   为了解决这些问题，我们从动态系统的角度提出了IM-BERT，将BERT层概念化为常微分方程（ODE）的解。在初值摄动的情况下，我们分析了两种主要的常微分方程数值解法：显式和隐式欧拉方法的数值稳定性。   基于这些分析，我们引入了一种包含BERT层的数字鲁棒的IM连接。该策略增强了PLM对抗攻击的稳健性，即使在低资源场景中也是如此，而无需引入额外的参数或对抗训练策略。   对抗性GLUE（AdvGLUE）数据集的实验结果验证了IM-BERT在各种条件下的鲁棒性。与原始BERT相比，IM-BERT在AdvGLUE数据集上表现出约8.3%p的性能改进。此外，在低资源场景中，IM-BERT的准确性比BERT高出5.9%p。



## **43. DP-TRAE: A Dual-Phase Merging Transferable Reversible Adversarial Example for Image Privacy Protection**

DP-TRAE：一个用于图像隐私保护的双阶段合并可转移可逆对抗实例 cs.CR

12 pages, 5 figures

**SubmitDate**: 2025-05-11    [abs](http://arxiv.org/abs/2505.06860v1) [paper-pdf](http://arxiv.org/pdf/2505.06860v1)

**Authors**: Xia Du, Jiajie Zhu, Jizhe Zhou, Chi-man Pun, Zheng Lin, Cong Wu, Zhe Chen, Jun Luo

**Abstract**: In the field of digital security, Reversible Adversarial Examples (RAE) combine adversarial attacks with reversible data hiding techniques to effectively protect sensitive data and prevent unauthorized analysis by malicious Deep Neural Networks (DNNs). However, existing RAE techniques primarily focus on white-box attacks, lacking a comprehensive evaluation of their effectiveness in black-box scenarios. This limitation impedes their broader deployment in complex, dynamic environments. Further more, traditional black-box attacks are often characterized by poor transferability and high query costs, significantly limiting their practical applicability. To address these challenges, we propose the Dual-Phase Merging Transferable Reversible Attack method, which generates highly transferable initial adversarial perturbations in a white-box model and employs a memory augmented black-box strategy to effectively mislead target mod els. Experimental results demonstrate the superiority of our approach, achieving a 99.0% attack success rate and 100% recovery rate in black-box scenarios, highlighting its robustness in privacy protection. Moreover, we successfully implemented a black-box attack on a commercial model, further substantiating the potential of this approach for practical use.

摘要: 在数字安全领域，可逆对抗示例（RAE）将对抗攻击与可逆数据隐藏技术相结合，以有效保护敏感数据并防止恶意深度神经网络（DNN）进行未经授权的分析。然而，现有的RAE技术主要关注白盒攻击，缺乏对其在黑匣子场景中有效性的全面评估。这种限制阻碍了它们在复杂、动态的环境中更广泛的部署。此外，传统的黑匣子攻击往往具有可移植性差和查询成本高的特点，严重限制了其实际适用性。为了解决这些挑战，我们提出了双阶段合并可转移可逆攻击方法，该方法在白盒模型中生成高度可转移的初始对抗性扰动，并采用记忆增强黑匣子策略来有效地误导目标模型。实验结果证明了我们的方法的优越性，在黑匣子场景中攻击成功率达到99.0%，恢复率达到100%，凸显了其在隐私保护方面的鲁棒性。此外，我们成功地对商业模型实施了黑匣子攻击，进一步证实了这种方法的实际应用潜力。



## **44. Endless Jailbreaks with Bijection Learning**

通过双射学习实现无休止的越狱 cs.CL

**SubmitDate**: 2025-05-10    [abs](http://arxiv.org/abs/2410.01294v3) [paper-pdf](http://arxiv.org/pdf/2410.01294v3)

**Authors**: Brian R. Y. Huang, Maximilian Li, Leonard Tang

**Abstract**: Despite extensive safety measures, LLMs are vulnerable to adversarial inputs, or jailbreaks, which can elicit unsafe behaviors. In this work, we introduce bijection learning, a powerful attack algorithm which automatically fuzzes LLMs for safety vulnerabilities using randomly-generated encodings whose complexity can be tightly controlled. We leverage in-context learning to teach models bijective encodings, pass encoded queries to the model to bypass built-in safety mechanisms, and finally decode responses back into English. Our attack is extremely effective on a wide range of frontier language models. Moreover, by controlling complexity parameters such as number of key-value mappings in the encodings, we find a close relationship between the capability level of the attacked LLM and the average complexity of the most effective bijection attacks. Our work highlights that new vulnerabilities in frontier models can emerge with scale: more capable models are more severely jailbroken by bijection attacks.

摘要: 尽管采取了广泛的安全措施，但LLM很容易受到对抗性输入或越狱的影响，这可能会引发不安全的行为。在这项工作中，我们引入了双射学习，这是一种强大的攻击算法，它使用随机生成的编码来自动模糊LLM的安全漏洞，其复杂性可以严格控制。我们利用上下文学习来教授模型二元编码，将编码的查询传递给模型以绕过内置的安全机制，并最终将响应解码回英语。我们的攻击对各种前沿语言模型非常有效。此外，通过控制复杂性参数（例如编码中的关键-值映射的数量），我们发现受攻击的LLM的能力水平与最有效的双射攻击的平均复杂性之间存在密切关系。我们的工作强调，前沿模型中的新漏洞可能会随着规模的增加而出现：更有能力的模型会被双射攻击更严重地越狱。



## **45. Boundary-Guided Trajectory Prediction for Road Aware and Physically Feasible Autonomous Driving**

用于道路感知和物理可行自动驾驶的边界引导轨迹预测 cs.RO

Accepted in the 36th IEEE Intelligent Vehicles Symposium (IV 2025)

**SubmitDate**: 2025-05-10    [abs](http://arxiv.org/abs/2505.06740v1) [paper-pdf](http://arxiv.org/pdf/2505.06740v1)

**Authors**: Ahmed Abouelazm, Mianzhi Liu, Christian Hubschneider, Yin Wu, Daniel Slieter, J. Marius Zöllner

**Abstract**: Accurate prediction of surrounding road users' trajectories is essential for safe and efficient autonomous driving. While deep learning models have improved performance, challenges remain in preventing off-road predictions and ensuring kinematic feasibility. Existing methods incorporate road-awareness modules and enforce kinematic constraints but lack plausibility guarantees and often introduce trade-offs in complexity and flexibility. This paper proposes a novel framework that formulates trajectory prediction as a constrained regression guided by permissible driving directions and their boundaries. Using the agent's current state and an HD map, our approach defines the valid boundaries and ensures on-road predictions by training the network to learn superimposed paths between left and right boundary polylines. To guarantee feasibility, the model predicts acceleration profiles that determine the vehicle's travel distance along these paths while adhering to kinematic constraints. We evaluate our approach on the Argoverse-2 dataset against the HPTR baseline. Our approach shows a slight decrease in benchmark metrics compared to HPTR but notably improves final displacement error and eliminates infeasible trajectories. Moreover, the proposed approach has superior generalization to less prevalent maneuvers and unseen out-of-distribution scenarios, reducing the off-road rate under adversarial attacks from 66\% to just 1\%. These results highlight the effectiveness of our approach in generating feasible and robust predictions.

摘要: 准确预测周围道路使用者的轨迹对于安全高效的自动驾驶至关重要。虽然深度学习模型提高了性能，但在防止越野预测和确保运动学可行性方面仍然存在挑战。现有的方法包含道路感知模块并强制执行运动学约束，但缺乏合理性保证，并且经常在复杂性和灵活性方面引入权衡。本文提出了一种新颖的框架，将轨迹预测制定为由允许的驾驶方向及其边界引导的约束回归。使用代理的当前状态和高清地图，我们的方法定义有效边界，并通过训练网络学习左右边界多段线之间的叠加路径来确保道路预测。为了保证可行性，该模型预测加速度曲线，该曲线确定车辆沿着这些路径的行驶距离，同时遵守运动学约束。我们根据HTLR基线评估我们在Argoverse-2数据集上的方法。与HTLR相比，我们的方法显示基准指标略有下降，但显着改善了最终位移误差并消除了不可行的轨迹。此外，所提出的方法对不太普遍的机动和不可见的非分布场景具有更好的通用性，将对抗攻击下的越野率从66%降低到仅1%。这些结果凸显了我们的方法在生成可行且稳健的预测方面的有效性。



## **46. Practical Reasoning Interruption Attacks on Reasoning Large Language Models**

对推理大型语言模型的实用推理中断攻击 cs.CR

**SubmitDate**: 2025-05-10    [abs](http://arxiv.org/abs/2505.06643v1) [paper-pdf](http://arxiv.org/pdf/2505.06643v1)

**Authors**: Yu Cui, Cong Zuo

**Abstract**: Reasoning large language models (RLLMs) have demonstrated outstanding performance across a variety of tasks, yet they also expose numerous security vulnerabilities. Most of these vulnerabilities have centered on the generation of unsafe content. However, recent work has identified a distinct "thinking-stopped" vulnerability in DeepSeek-R1: under adversarial prompts, the model's reasoning process ceases at the system level and produces an empty final answer. Building upon this vulnerability, researchers developed a novel prompt injection attack, termed reasoning interruption attack, and also offered an initial analysis of its root cause. Through extensive experiments, we verify the previous analyses, correct key errors based on three experimental findings, and present a more rigorous explanation of the fundamental causes driving the vulnerability. Moreover, existing attacks typically require over 2,000 tokens, impose significant overhead, reduce practicality, and are easily detected. To overcome these limitations, we propose the first practical reasoning interruption attack. It succeeds with just 109 tokens by exploiting our newly uncovered "reasoning token overflow" (RTO) effect to overwrite the model's final answer, forcing it to return an invalid response. Experimental results demonstrate that our proposed attack is highly effective. Furthermore, we discover that the method for triggering RTO differs between the official DeepSeek-R1 release and common unofficial deployments. As a broadened application of RTO, we also construct a novel jailbreak attack that enables the transfer of unsafe content within the reasoning tokens into final answer, thereby exposing it to the user. Our work carries significant implications for enhancing the security of RLLMs.

摘要: 推理大型语言模型（RLLM）在各种任务中表现出出色的性能，但它们也暴露了许多安全漏洞。大多数漏洞都集中在不安全内容的生成上。然而，最近的工作在DeepSeek-R1中发现了一个明显的“思维停止”漏洞：在对抗性提示下，模型的推理过程在系统级别停止并产生空的最终答案。在此漏洞的基础上，研究人员开发了一种新型的即时注入攻击，称为推理中断攻击，并对其根本原因进行了初步分析。通过广泛的实验，我们验证了之前的分析，根据三个实验发现纠正了关键错误，并对驱动该漏洞的根本原因进行了更严格的解释。此外，现有的攻击通常需要超过2，000个令牌，造成大量的费用，降低实用性，并且很容易被检测到。为了克服这些限制，我们提出了第一个实际推理中断攻击。它利用我们新发现的“推理令牌溢出”（RTI）效应来覆盖模型的最终答案，迫使其返回无效响应，仅用109个令牌就成功了。实验结果表明我们提出的攻击非常有效。此外，我们发现官方DeepSeek-R1版本和常见的非官方部署之间触发RTI的方法有所不同。作为RTI的扩展应用，我们还构建了一种新颖的越狱攻击，可以将推理令牌中的不安全内容转移到最终答案中，从而将其暴露给用户。我们的工作对于增强LLLM的安全性具有重大影响。



## **47. TAROT: Towards Essentially Domain-Invariant Robustness with Theoretical Justification**

TAROT：通过理论证明迈向本质上领域不变的稳健性 cs.AI

Accepted in CVPR 2025 (19 pages, 7 figures)

**SubmitDate**: 2025-05-10    [abs](http://arxiv.org/abs/2505.06580v1) [paper-pdf](http://arxiv.org/pdf/2505.06580v1)

**Authors**: Dongyoon Yang, Jihu Lee, Yongdai Kim

**Abstract**: Robust domain adaptation against adversarial attacks is a critical research area that aims to develop models capable of maintaining consistent performance across diverse and challenging domains. In this paper, we derive a new generalization bound for robust risk on the target domain using a novel divergence measure specifically designed for robust domain adaptation. Building upon this, we propose a new algorithm named TAROT, which is designed to enhance both domain adaptability and robustness. Through extensive experiments, TAROT not only surpasses state-of-the-art methods in accuracy and robustness but also significantly enhances domain generalization and scalability by effectively learning domain-invariant features. In particular, TAROT achieves superior performance on the challenging DomainNet dataset, demonstrating its ability to learn domain-invariant representations that generalize well across different domains, including unseen ones. These results highlight the broader applicability of our approach in real-world domain adaptation scenarios.

摘要: 针对对抗性攻击的鲁棒域自适应是一个关键的研究领域，旨在开发能够在不同和具有挑战性的领域保持一致性能的模型。在本文中，我们推导出一个新的广义界的鲁棒风险的目标域使用一种新的分歧措施专门设计的鲁棒域适应。在此基础上，我们提出了一个新的算法名为TAROT，这是为了提高域的适应性和鲁棒性。通过大量的实验，TAROT不仅在准确性和鲁棒性方面超越了最先进的方法，而且通过有效地学习领域不变特征，显着提高了领域的泛化能力和可扩展性。特别是，TAROT在具有挑战性的DomainNet数据集上实现了卓越的性能，证明了其学习域不变表示的能力，这些表示在不同领域（包括不可见的领域）很好地概括。这些结果凸显了我们的方法在现实世界领域适应场景中的更广泛适用性。



## **48. Towards Robust Recommendation: A Review and an Adversarial Robustness Evaluation Library**

迈向稳健推荐：评论和对抗稳健性评估库 cs.IR

**SubmitDate**: 2025-05-10    [abs](http://arxiv.org/abs/2404.17844v2) [paper-pdf](http://arxiv.org/pdf/2404.17844v2)

**Authors**: Lei Cheng, Xiaowen Huang, Jitao Sang, Jian Yu

**Abstract**: Recently, recommender system has achieved significant success. However, due to the openness of recommender systems, they remain vulnerable to malicious attacks. Additionally, natural noise in training data and issues such as data sparsity can also degrade the performance of recommender systems. Therefore, enhancing the robustness of recommender systems has become an increasingly important research topic. In this survey, we provide a comprehensive overview of the robustness of recommender systems. Based on our investigation, we categorize the robustness of recommender systems into adversarial robustness and non-adversarial robustness. In the adversarial robustness, we introduce the fundamental principles and classical methods of recommender system adversarial attacks and defenses. In the non-adversarial robustness, we analyze non-adversarial robustness from the perspectives of data sparsity, natural noise, and data imbalance. Additionally, we summarize commonly used datasets and evaluation metrics for evaluating the robustness of recommender systems. Finally, we also discuss the current challenges in the field of recommender system robustness and potential future research directions. Additionally, to facilitate fair and efficient evaluation of attack and defense methods in adversarial robustness, we propose an adversarial robustness evaluation library--ShillingREC, and we conduct evaluations of basic attack models and recommendation models. ShillingREC project is released at https://github.com/chengleileilei/ShillingREC.

摘要: 最近，推荐系统取得了显着的成功。然而，由于推荐系统的开放性，它们仍然容易受到恶意攻击。此外，训练数据中的自然噪音和数据稀疏性等问题也会降低推荐系统的性能。因此，增强推荐系统的鲁棒性已成为一个越来越重要的研究课题。在本调查中，我们全面概述了推荐系统的稳健性。根据我们的调查，我们将推荐系统的鲁棒性分为对抗性鲁棒性和非对抗性鲁棒性。在对抗鲁棒性方面，我们介绍了推荐系统对抗攻击和防御的基本原则和经典方法。在非对抗鲁棒性方面，我们从数据稀疏性、自然噪音和数据不平衡的角度分析非对抗鲁棒性。此外，我们总结了常用的数据集和评价指标，用于评估推荐系统的鲁棒性。最后，我们还讨论了目前在推荐系统鲁棒性领域的挑战和潜在的未来研究方向。此外，为了便于公平和有效地评估攻击和防御方法的对抗鲁棒性，我们提出了一个对抗鲁棒性评估库-ShillingREC，我们进行了基本的攻击模型和推荐模型的评估。ShillingREC项目发布于https://github.com/chengleileilei/ShillingREC。



## **49. An In-kernel Forensics Engine for Investigating Evasive Attacks**

用于调查规避攻击的内核内取证引擎 cs.CR

**SubmitDate**: 2025-05-10    [abs](http://arxiv.org/abs/2505.06498v1) [paper-pdf](http://arxiv.org/pdf/2505.06498v1)

**Authors**: Javad Zhandi, Lalchandra Rampersaud, Amin Kharraz

**Abstract**: Over the years, adversarial attempts against critical services have become more effective and sophisticated in launching low-profile attacks. This trend has always been concerning. However, an even more alarming trend is the increasing difficulty of collecting relevant evidence about these attacks and the involved threat actors in the early stages before significant damage is done. This issue puts defenders at a significant disadvantage, as it becomes exceedingly difficult to understand the attack details and formulate an appropriate response. Developing robust forensics tools to collect evidence about modern threats has never been easy. One main challenge is to provide a robust trade-off between achieving sufficient visibility while leaving minimal detectable artifacts. This paper will introduce LASE, an open-source Low-Artifact Forensics Engine to perform threat analysis and forensics in Windows operating system. LASE augments current analysis tools by providing detailed, system-wide monitoring capabilities while minimizing detectable artifacts. We designed multiple deployment scenarios, showing LASE's potential in evidence gathering and threat reasoning in a real-world setting. By making LASE and its execution trace data available to the broader research community, this work encourages further exploration in the field by reducing the engineering costs for threat analysis and building a longitudinal behavioral analysis catalog for diverse security domains.

摘要: 多年来，针对关键服务的对抗尝试在发起低调攻击方面变得更加有效和复杂。这种趋势一直令人担忧。然而，一个更令人震惊的趋势是，在造成重大损害之前的早期阶段收集有关这些攻击和相关威胁行为者的相关证据的难度越来越大。这个问题使防御者处于明显的劣势，因为了解攻击细节并制定适当的应对措施变得极其困难。开发强大的取证工具来收集有关现代威胁的证据从来都不是一件容易的事。一个主要挑战是在实现足够的可见性同时留下最少的可检测伪影之间提供稳健的权衡。本文将介绍LASE，这是一个开源的低功耗取证引擎，用于在Windows操作系统中执行威胁分析和取证。LASE通过提供详细的系统范围监控功能，同时最大限度地减少可检测的伪影来增强当前的分析工具。我们设计了多种部署场景，展示了LASE在现实环境中证据收集和威胁推理方面的潜力。通过向更广泛的研究界提供LASE及其执行跟踪数据，这项工作通过降低威胁分析的工程成本并为不同安全领域构建纵向行为分析目录来鼓励该领域的进一步探索。



## **50. Fun-tuning: Characterizing the Vulnerability of Proprietary LLMs to Optimization-based Prompt Injection Attacks via the Fine-Tuning Interface**

有趣的调整：通过微调接口描述专有LLM对基于优化的提示注入攻击的脆弱性 cs.CR

**SubmitDate**: 2025-05-10    [abs](http://arxiv.org/abs/2501.09798v2) [paper-pdf](http://arxiv.org/pdf/2501.09798v2)

**Authors**: Andrey Labunets, Nishit V. Pandya, Ashish Hooda, Xiaohan Fu, Earlence Fernandes

**Abstract**: We surface a new threat to closed-weight Large Language Models (LLMs) that enables an attacker to compute optimization-based prompt injections. Specifically, we characterize how an attacker can leverage the loss-like information returned from the remote fine-tuning interface to guide the search for adversarial prompts. The fine-tuning interface is hosted by an LLM vendor and allows developers to fine-tune LLMs for their tasks, thus providing utility, but also exposes enough information for an attacker to compute adversarial prompts. Through an experimental analysis, we characterize the loss-like values returned by the Gemini fine-tuning API and demonstrate that they provide a useful signal for discrete optimization of adversarial prompts using a greedy search algorithm. Using the PurpleLlama prompt injection benchmark, we demonstrate attack success rates between 65% and 82% on Google's Gemini family of LLMs. These attacks exploit the classic utility-security tradeoff - the fine-tuning interface provides a useful feature for developers but also exposes the LLMs to powerful attacks.

摘要: 我们对封闭权重大型语言模型（LLM）提出了新的威胁，该威胁使攻击者能够计算基于优化的提示注入。具体来说，我们描述了攻击者如何利用从远程微调界面返回的类似损失的信息来指导搜索对抗性提示。微调接口由LLM供应商托管，允许开发人员针对其任务微调LLM，从而提供实用性，但也暴露了足够的信息供攻击者计算对抗提示。通过实验分析，我们描述了Gemini微调API返回的类似损失的值，并证明它们为使用贪婪搜索算法对对抗性提示的离散优化提供了有用的信号。使用PurpleLlama提示注入基准，我们证明了Google Gemini LLM系列的攻击成功率在65%至82%之间。这些攻击利用了经典的实用程序-安全权衡-微调接口为开发人员提供了有用的功能，但也使LLM面临强大的攻击。



