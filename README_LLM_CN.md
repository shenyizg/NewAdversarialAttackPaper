# Latest Large Language Model Attack Papers
**update at 2025-05-16 16:49:08**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. S3C2 Summit 2024-09: Industry Secure Software Supply Chain Summit**

S3 C2峰会2024-09：行业安全软件供应链峰会 cs.CR

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2505.10538v1) [paper-pdf](http://arxiv.org/pdf/2505.10538v1)

**Authors**: Imranur Rahman, Yasemin Acar, Michel Cukier, William Enck, Christian Kastner, Alexandros Kapravelos, Dominik Wermke, Laurie Williams

**Abstract**: While providing economic and software development value, software supply chains are only as strong as their weakest link. Over the past several years, there has been an exponential increase in cyberattacks, specifically targeting vulnerable links in critical software supply chains. These attacks disrupt the day-to-day functioning and threaten the security of nearly everyone on the internet, from billion-dollar companies and government agencies to hobbyist open-source developers. The ever-evolving threat of software supply chain attacks has garnered interest from the software industry and the US government in improving software supply chain security.   On September 20, 2024, three researchers from the NSF-backed Secure Software Supply Chain Center (S3C2) conducted a Secure Software Supply Chain Summit with a diverse set of 12 practitioners from 9 companies. The goals of the Summit were to: (1) to enable sharing between individuals from different companies regarding practical experiences and challenges with software supply chain security, (2) to help form new collaborations, (3) to share our observations from our previous summits with industry, and (4) to learn about practitioners' challenges to inform our future research direction. The summit consisted of discussions of six topics relevant to the companies represented, including updating vulnerable dependencies, component and container choice, malicious commits, building infrastructure, large language models, and reducing entire classes of vulnerabilities.

摘要: 在提供经济和软件开发价值的同时，软件供应链的强大程度取决于其最薄弱的环节。在过去的几年里，网络攻击呈指数级增加，特别是针对关键软件供应链中的脆弱环节。这些攻击扰乱了日常运作，并威胁到互联网上几乎所有人的安全，从价值数十亿美元的公司和政府机构到爱好者的开源开发人员。软件供应链攻击的不断变化的威胁引起了软件行业和美国政府对改善软件供应链安全的兴趣。   2024年9月20日，来自NSF支持的安全软件供应链中心（S3 C2）的三名研究人员与来自9家公司的12名从业者举行了安全软件供应链峰会。峰会的目标是：（1）实现来自不同公司的个人之间就软件供应链安全方面的实践经验和挑战进行分享，（2）帮助形成新的合作，（3）与行业分享我们在之前峰会上的观察，（4）了解从业者面临的挑战，为我们未来的研究方向提供信息。峰会讨论了与与会公司相关的六个主题，包括更新脆弱依赖关系、组件和容器选择、恶意提交、构建基础设施、大型语言模型以及减少整个漏洞类别。



## **2. MapExplorer: New Content Generation from Low-Dimensional Visualizations**

MapExplorer：来自低维可视化的新内容生成 cs.AI

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2412.18673v2) [paper-pdf](http://arxiv.org/pdf/2412.18673v2)

**Authors**: Xingjian Zhang, Ziyang Xiong, Shixuan Liu, Yutong Xie, Tolga Ergen, Dongsub Shim, Hua Xu, Honglak Lee, Qiaozhu Me

**Abstract**: Low-dimensional visualizations, or "projection maps," are widely used in scientific and creative domains to interpret large-scale and complex datasets. These visualizations not only aid in understanding existing knowledge spaces but also implicitly guide exploration into unknown areas. Although techniques such as t-SNE and UMAP can generate these maps, there exists no systematic method for leveraging them to generate new content. To address this, we introduce MapExplorer, a novel knowledge discovery task that translates coordinates within any projection map into coherent, contextually aligned textual content. This allows users to interactively explore and uncover insights embedded in the maps. To evaluate the performance of MapExplorer methods, we propose Atometric, a fine-grained metric inspired by ROUGE that quantifies logical coherence and alignment between generated and reference text. Experiments on diverse datasets demonstrate the versatility of MapExplorer in generating scientific hypotheses, crafting synthetic personas, and devising strategies for attacking large language models-even with simple baseline methods. By bridging visualization and generation, our work highlights the potential of MapExplorer to enable intuitive human-AI collaboration in large-scale data exploration.

摘要: 低维可视化或“投影地图”广泛用于科学和创意领域，以解释大规模和复杂的数据集。这些可视化不仅有助于理解现有的知识空间，而且还隐含地指导对未知领域的探索。尽管t-SNE和UMAP等技术可以生成这些地图，但不存在利用它们来生成新内容的系统方法。为了解决这个问题，我们引入了MapExplorer，这是一项新颖的知识发现任务，可以将任何投影地图内的坐标转换为连贯、上下文对齐的文本内容。这允许用户交互式探索和发现地图中嵌入的见解。为了评估MapExplorer方法的性能，我们提出了Atric，这是一种受ROUGE启发的细粒度指标，可以量化生成文本和参考文本之间的逻辑一致性和一致性。对不同数据集的实验证明了MapExplorer在生成科学假设、制作合成人物角色以及设计攻击大型语言模型的策略方面的多功能性--即使使用简单的基线方法。通过连接可视化和生成，我们的工作凸显了MapExplorer在大规模数据探索中实现直观的人机协作的潜力。



## **3. Scaling Laws for Black box Adversarial Attacks**

黑匣子对抗攻击的缩放定律 cs.LG

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2411.16782v2) [paper-pdf](http://arxiv.org/pdf/2411.16782v2)

**Authors**: Chuan Liu, Huanran Chen, Yichi Zhang, Yinpeng Dong, Jun Zhu

**Abstract**: Adversarial examples usually exhibit good cross-model transferability, enabling attacks on black-box models with limited information about their architectures and parameters, which are highly threatening in commercial black-box scenarios. Model ensembling is an effective strategy to improve the transferability of adversarial examples by attacking multiple surrogate models. However, since prior studies usually adopt few models in the ensemble, there remains an open question of whether scaling the number of models can further improve black-box attacks. Inspired by the scaling law of large foundation models, we investigate the scaling laws of black-box adversarial attacks in this work. Through theoretical analysis and empirical evaluations, we conclude with clear scaling laws that using more surrogate models enhances adversarial transferability. Comprehensive experiments verify the claims on standard image classifiers, diverse defended models and multimodal large language models using various adversarial attack methods. Specifically, by scaling law, we achieve 90%+ transfer attack success rate on even proprietary models like GPT-4o. Further visualization indicates that there is also a scaling law on the interpretability and semantics of adversarial perturbations.

摘要: 对抗性示例通常表现出良好的跨模型可移植性，从而能够在有关其架构和参数的有限信息的情况下对黑匣子模型进行攻击，这在商业黑匣子场景中具有高度威胁性。模型集成是通过攻击多个代理模型来提高对抗性示例可移植性的有效策略。然而，由于之前的研究通常在整体中采用很少的模型，因此扩大模型数量是否可以进一步改善黑匣子攻击仍然是一个悬而未决的问题。受大型基金会模型缩放定律的启发，我们在这项工作中研究了黑匣子对抗攻击的缩放定律。通过理论分析和实证评估，我们得出了明确的缩放定律，即使用更多的代理模型增强了对抗性可转让性。全面的实验验证了标准图像分类器、多样化防御模型和使用各种对抗攻击方法的多模式大型语言模型的主张。具体来说，通过缩放定律，即使是GPT-4 o等专有模型，我们也能实现90%以上的传输攻击成功率。进一步的可视化表明，对抗性扰动的可解释性和语义也存在缩放定律。



## **4. Dark LLMs: The Growing Threat of Unaligned AI Models**

黑暗LLM：不一致的人工智能模型日益增长的威胁 cs.CL

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2505.10066v1) [paper-pdf](http://arxiv.org/pdf/2505.10066v1)

**Authors**: Michael Fire, Yitzhak Elbazis, Adi Wasenstein, Lior Rokach

**Abstract**: Large Language Models (LLMs) rapidly reshape modern life, advancing fields from healthcare to education and beyond. However, alongside their remarkable capabilities lies a significant threat: the susceptibility of these models to jailbreaking. The fundamental vulnerability of LLMs to jailbreak attacks stems from the very data they learn from. As long as this training data includes unfiltered, problematic, or 'dark' content, the models can inherently learn undesirable patterns or weaknesses that allow users to circumvent their intended safety controls. Our research identifies the growing threat posed by dark LLMs models deliberately designed without ethical guardrails or modified through jailbreak techniques. In our research, we uncovered a universal jailbreak attack that effectively compromises multiple state-of-the-art models, enabling them to answer almost any question and produce harmful outputs upon request. The main idea of our attack was published online over seven months ago. However, many of the tested LLMs were still vulnerable to this attack. Despite our responsible disclosure efforts, responses from major LLM providers were often inadequate, highlighting a concerning gap in industry practices regarding AI safety. As model training becomes more accessible and cheaper, and as open-source LLMs proliferate, the risk of widespread misuse escalates. Without decisive intervention, LLMs may continue democratizing access to dangerous knowledge, posing greater risks than anticipated.

摘要: 大型语言模型（LLM）迅速重塑现代生活，推进从医疗保健到教育等领域的发展。然而，除了它们非凡的能力之外，还有一个重大威胁：这些模型容易越狱。LLM对越狱攻击的根本脆弱性源于他们从中学到的数据。只要此训练数据包括未经过滤的、有问题的或“黑暗”的内容，模型就可以本质上学习不希望的模式或弱点，从而允许用户规避其预期的安全控制。我们的研究确定了故意设计没有道德护栏或通过越狱技术进行修改的黑暗LLMS模型所构成的日益严重的威胁。在我们的研究中，我们发现了一种通用的越狱攻击，它有效地损害了多个最先进的模型，使它们能够回答几乎任何问题并根据请求产生有害输出。我们攻击的主要想法于七个多月前在网上发布。然而，许多经过测试的LLM仍然容易受到这种攻击。尽管我们做出了负责任的披露努力，但主要LLM提供商的回应往往不够充分，这凸显了人工智能安全方面的行业实践存在令人担忧的差距。随着模型训练变得更容易获得和更便宜，以及开源LLM的激增，广泛滥用的风险升级。如果没有果断的干预，LLM可能会继续使危险知识的获取民主化，带来比预期更大的风险。



## **5. Large Language Models for Cyber Security: A Systematic Literature Review**

网络安全的大型语言模型：系统性文献综述 cs.CR

56 pages,6 figures

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2405.04760v4) [paper-pdf](http://arxiv.org/pdf/2405.04760v4)

**Authors**: Hanxiang Xu, Shenao Wang, Ningke Li, Kailong Wang, Yanjie Zhao, Kai Chen, Ting Yu, Yang Liu, Haoyu Wang

**Abstract**: The rapid advancement of Large Language Models (LLMs) has opened up new opportunities for leveraging artificial intelligence in various domains, including cybersecurity. As the volume and sophistication of cyber threats continue to grow, there is an increasing need for intelligent systems that can automatically detect vulnerabilities, analyze malware, and respond to attacks. In this survey, we conduct a comprehensive review of the literature on the application of LLMs in cybersecurity (LLM4Security). By comprehensively collecting over 30K relevant papers and systematically analyzing 127 papers from top security and software engineering venues, we aim to provide a holistic view of how LLMs are being used to solve diverse problems across the cybersecurity domain. Through our analysis, we identify several key findings. First, we observe that LLMs are being applied to a wide range of cybersecurity tasks, including vulnerability detection, malware analysis, network intrusion detection, and phishing detection. Second, we find that the datasets used for training and evaluating LLMs in these tasks are often limited in size and diversity, highlighting the need for more comprehensive and representative datasets. Third, we identify several promising techniques for adapting LLMs to specific cybersecurity domains, such as fine-tuning, transfer learning, and domain-specific pre-training. Finally, we discuss the main challenges and opportunities for future research in LLM4Security, including the need for more interpretable and explainable models, the importance of addressing data privacy and security concerns, and the potential for leveraging LLMs for proactive defense and threat hunting. Overall, our survey provides a comprehensive overview of the current state-of-the-art in LLM4Security and identifies several promising directions for future research.

摘要: 大型语言模型（LLM）的快速发展为在包括网络安全在内的各个领域利用人工智能开辟了新的机会。随着网络威胁的数量和复杂性不断增长，对能够自动检测漏洞、分析恶意软件并响应攻击的智能系统的需求越来越大。在本调查中，我们对有关LLM在网络安全中应用的文献进行了全面审查（LLM4Security）。通过全面收集超过3万篇相关论文并系统分析来自顶级安全和软件工程场所的127篇论文，我们的目标是提供如何使用LLM来解决网络安全领域的各种问题的整体视图。通过我们的分析，我们确定了几个关键发现。首先，我们观察到LLM正在应用于广泛的网络安全任务，包括漏洞检测、恶意软件分析、网络入侵检测和网络钓鱼检测。其次，我们发现用于在这些任务中训练和评估LLM的数据集的规模和多样性通常受到限制，这凸显了对更全面和代表性的数据集的需求。第三，我们确定了几种有前途的技术，用于将LLM适应特定的网络安全领域，例如微调、迁移学习和特定领域的预训练。最后，我们讨论了LLM 4 Security未来研究的主要挑战和机遇，包括对更多可解释和可解释模型的需求、解决数据隐私和安全问题的重要性，以及利用LLM进行主动防御和威胁狩猎的潜力。总体而言，我们的调查全面概述了LLM 4 Security当前最新技术水平，并确定了未来研究的几个有前途的方向。



## **6. PIG: Privacy Jailbreak Attack on LLMs via Gradient-based Iterative In-Context Optimization**

PIG：通过基于对象的迭代上下文优化对LLM进行隐私越狱攻击 cs.CR

**SubmitDate**: 2025-05-15    [abs](http://arxiv.org/abs/2505.09921v1) [paper-pdf](http://arxiv.org/pdf/2505.09921v1)

**Authors**: Yidan Wang, Yanan Cao, Yubing Ren, Fang Fang, Zheng Lin, Binxing Fang

**Abstract**: Large Language Models (LLMs) excel in various domains but pose inherent privacy risks. Existing methods to evaluate privacy leakage in LLMs often use memorized prefixes or simple instructions to extract data, both of which well-alignment models can easily block. Meanwhile, Jailbreak attacks bypass LLM safety mechanisms to generate harmful content, but their role in privacy scenarios remains underexplored. In this paper, we examine the effectiveness of jailbreak attacks in extracting sensitive information, bridging privacy leakage and jailbreak attacks in LLMs. Moreover, we propose PIG, a novel framework targeting Personally Identifiable Information (PII) and addressing the limitations of current jailbreak methods. Specifically, PIG identifies PII entities and their types in privacy queries, uses in-context learning to build a privacy context, and iteratively updates it with three gradient-based strategies to elicit target PII. We evaluate PIG and existing jailbreak methods using two privacy-related datasets. Experiments on four white-box and two black-box LLMs show that PIG outperforms baseline methods and achieves state-of-the-art (SoTA) results. The results underscore significant privacy risks in LLMs, emphasizing the need for stronger safeguards. Our code is availble at \href{https://github.com/redwyd/PrivacyJailbreak}{https://github.com/redwyd/PrivacyJailbreak}.

摘要: 大型语言模型（LLM）在各个领域都表现出色，但也存在固有的隐私风险。评估LLM隐私泄露的现有方法通常使用记忆的前置码或简单指令来提取数据，而良好对齐的模型可以轻松阻止这两种情况。与此同时，越狱攻击绕过了LLM安全机制来生成有害内容，但它们在隐私场景中的作用仍然没有得到充分研究。在本文中，我们研究了越狱攻击在提取敏感信息、弥合LLC中隐私泄露和越狱攻击方面的有效性。此外，我们还提出了PIG，这是一种针对个人可识别信息（PRI）并解决当前越狱方法的局限性的新型框架。具体来说，PIG识别隐私查询中的PRI实体及其类型，使用上下文学习来构建隐私上下文，并使用三种基于梯度的策略迭代更新它以引出目标PRI。我们使用两个与隐私相关的数据集评估PIG和现有的越狱方法。对四个白盒和两个黑盒LLM的实验表明，PIG优于基线方法并实现了最先进的（SoTA）结果。结果强调了LLM中存在的重大隐私风险，强调了加强保护措施的必要性。我们的代码可在\href{https：//github.com/redwspe/PrivacyJailbreak}{https：//github.com/redwspe/PrivacyJailbreak}上获取。



## **7. Adversarial Attack on Large Language Models using Exponentiated Gradient Descent**

使用指数梯度下降对大型语言模型的对抗攻击 cs.LG

Accepted to International Joint Conference on Neural Networks (IJCNN)  2025

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2505.09820v1) [paper-pdf](http://arxiv.org/pdf/2505.09820v1)

**Authors**: Sajib Biswas, Mao Nishino, Samuel Jacob Chacko, Xiuwen Liu

**Abstract**: As Large Language Models (LLMs) are widely used, understanding them systematically is key to improving their safety and realizing their full potential. Although many models are aligned using techniques such as reinforcement learning from human feedback (RLHF), they are still vulnerable to jailbreaking attacks. Some of the existing adversarial attack methods search for discrete tokens that may jailbreak a target model while others try to optimize the continuous space represented by the tokens of the model's vocabulary. While techniques based on the discrete space may prove to be inefficient, optimization of continuous token embeddings requires projections to produce discrete tokens, which might render them ineffective. To fully utilize the constraints and the structures of the space, we develop an intrinsic optimization technique using exponentiated gradient descent with the Bregman projection method to ensure that the optimized one-hot encoding always stays within the probability simplex. We prove the convergence of the technique and implement an efficient algorithm that is effective in jailbreaking several widely used LLMs. We demonstrate the efficacy of the proposed technique using five open-source LLMs on four openly available datasets. The results show that the technique achieves a higher success rate with great efficiency compared to three other state-of-the-art jailbreaking techniques. The source code for our implementation is available at: https://github.com/sbamit/Exponentiated-Gradient-Descent-LLM-Attack

摘要: 随着大型语言模型（LLM）的广泛使用，系统性地理解它们是提高其安全性并充分发挥其潜力的关键。尽管许多模型都使用人类反馈强化学习（RL HF）等技术进行了调整，但它们仍然容易受到越狱攻击。现有的一些对抗攻击方法搜索可能越狱目标模型的离散令牌，而另一些方法则试图优化模型词汇表中的令牌所代表的连续空间。虽然基于离散空间的技术可能被证明效率低下，但连续令牌嵌入的优化需要投影来产生离散令牌，这可能会使它们无效。为了充分利用空间的约束和结构，我们使用Bregman投影方法的指数梯度下降开发了一种内在优化技术，以确保优化的一次性编码始终保持在概率单形内。我们证明了该技术的收敛性，并实现了一种有效的算法，该算法可以有效越狱几种广泛使用的LLM。我们在四个公开可用的数据集上使用五个开源LLM来证明所提出技术的有效性。结果表明，与其他三种最先进的越狱技术相比，该技术的成功率更高，效率更高。我们实现的源代码可访问：https://github.com/sbamit/Exponentiated-Gradient-Descent-LLM-Attack



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



## **10. FaceShield: Explainable Face Anti-Spoofing with Multimodal Large Language Models**

FaceShield：使用多模式大型语言模型的可解释面部反欺骗 cs.CV

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2505.09415v1) [paper-pdf](http://arxiv.org/pdf/2505.09415v1)

**Authors**: Hongyang Wang, Yichen Shi, Zhuofu Tao, Yuhao Gao, Liepiao Zhang, Xun Lin, Jun Feng, Xiaochen Yuan, Zitong Yu, Xiaochun Cao

**Abstract**: Face anti-spoofing (FAS) is crucial for protecting facial recognition systems from presentation attacks. Previous methods approached this task as a classification problem, lacking interpretability and reasoning behind the predicted results. Recently, multimodal large language models (MLLMs) have shown strong capabilities in perception, reasoning, and decision-making in visual tasks. However, there is currently no universal and comprehensive MLLM and dataset specifically designed for FAS task. To address this gap, we propose FaceShield, a MLLM for FAS, along with the corresponding pre-training and supervised fine-tuning (SFT) datasets, FaceShield-pre10K and FaceShield-sft45K. FaceShield is capable of determining the authenticity of faces, identifying types of spoofing attacks, providing reasoning for its judgments, and detecting attack areas. Specifically, we employ spoof-aware vision perception (SAVP) that incorporates both the original image and auxiliary information based on prior knowledge. We then use an prompt-guided vision token masking (PVTM) strategy to random mask vision tokens, thereby improving the model's generalization ability. We conducted extensive experiments on three benchmark datasets, demonstrating that FaceShield significantly outperforms previous deep learning models and general MLLMs on four FAS tasks, i.e., coarse-grained classification, fine-grained classification, reasoning, and attack localization. Our instruction datasets, protocols, and codes will be released soon.

摘要: 面部反欺骗（FAA）对于保护面部识别系统免受演示攻击至关重要。之前的方法将此任务视为分类问题，缺乏预测结果背后的解释性和推理。最近，多模式大型语言模型（MLLM）在视觉任务中表现出了强大的感知、推理和决策能力。然而，目前还没有通用、全面的MLLM和专门为FAA任务设计的数据集。为了解决这一差距，我们提出了FaceShield（一种适用于FAA的MLLM），以及相应的预训练和监督微调（SFT）数据集FaceShield-pre10 K和FaceShield-sft45 K。FaceShield能够确定人脸的真实性、识别欺骗攻击的类型、为其判断提供推理并检测攻击区域。具体来说，我们采用欺骗感知视觉感知（SAVP），它结合了原始图像和基于先验知识的辅助信息。然后，我们使用预算引导的视觉令牌掩蔽（PVTM）策略来随机掩蔽视觉令牌，从而提高模型的概括能力。我们对三个基准数据集进行了广泛的实验，证明FaceShield在四个FAA任务（即粗粒度分类、细粒度分类、推理和攻击定位。我们的指令数据集、协议和代码将很快发布。



## **11. What Features in Prompts Jailbreak LLMs? Investigating the Mechanisms Behind Attacks**

Prettts越狱LLMS有哪些功能？调查攻击背后的机制 cs.CR

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2411.03343v2) [paper-pdf](http://arxiv.org/pdf/2411.03343v2)

**Authors**: Nathalie Kirch, Constantin Weisser, Severin Field, Helen Yannakoudakis, Stephen Casper

**Abstract**: Jailbreaks have been a central focus of research regarding the safety and reliability of large language models (LLMs), yet the mechanisms underlying these attacks remain poorly understood. While previous studies have predominantly relied on linear methods to detect jailbreak attempts and model refusals, we take a different approach by examining both linear and non-linear features in prompts that lead to successful jailbreaks. First, we introduce a novel dataset comprising 10,800 jailbreak attempts spanning 35 diverse attack methods. Leveraging this dataset, we train probes to classify successful from unsuccessful jailbreaks using the latent representations corresponding to prompt tokens. Notably, we find that even when probes achieve high accuracy in predicting the success of jailbreaks, their performance often fails to generalize to unseen attack methods. This reveals that different jailbreaking strategies exploit different non-linear, non-universal features. Next, we demonstrate that non-linear probes provide a powerful tool for steering model behavior. Specifically, we use these probes to guide targeted latent space perturbations, enabling us to effectively modulate the model's robustness against jailbreaks. Overall, our findings challenge the assumption that jailbreaks can be fully understood through linear or simple universal prompt features alone, highlighting the importance of a nuanced understanding of the mechanisms behind LLM vulnerabilities.

摘要: 越狱一直是大型语言模型（LLM）安全性和可靠性研究的中心焦点，但人们对这些攻击的潜在机制仍然知之甚少。虽然之前的研究主要依赖线性方法来检测越狱尝试和模型拒绝，但我们采取了不同的方法，通过检查导致成功越狱的提示中的线性和非线性特征。首先，我们引入了一个新颖的数据集，其中包含10，800次越狱尝试，涵盖35种不同的攻击方法。利用该数据集，我们训练探测器使用与提示令牌对应的潜在表示对成功越狱和不成功越狱进行分类。值得注意的是，我们发现，即使探测器在预测越狱成功方面达到了很高的准确性，但它们的性能往往无法推广到看不见的攻击方法。这表明不同的越狱策略利用了不同的非线性、非普遍特征。接下来，我们证明非线性探针为引导模型行为提供了强大的工具。具体来说，我们使用这些探测器来引导有针对性的潜在空间扰动，使我们能够有效地调节模型针对越狱的鲁棒性。总体而言，我们的研究结果挑战了仅通过线性或简单的通用提示特征即可完全理解越狱的假设，凸显了细致入微地理解LLM漏洞背后机制的重要性。



## **12. Improving Network Threat Detection by Knowledge Graph, Large Language Model, and Imbalanced Learning**

通过知识图、大语言模型和不平衡学习改进网络威胁检测 cs.LG

Accepted by "Combining AI and OR/MS for Better Trustworthy Decision  Making" Bridge Program co-organized by AAAI and INFORMS as poster and demo

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2501.16393v2) [paper-pdf](http://arxiv.org/pdf/2501.16393v2)

**Authors**: Lili Zhang, Quanyan Zhu, Herman Ray, Ying Xie

**Abstract**: Network threat detection has been challenging due to the complexities of attack activities and the limitation of historical threat data to learn from. To help enhance the existing practices of using analytics, machine learning, and artificial intelligence methods to detect the network threats, we propose an integrated modelling framework, where Knowledge Graph is used to analyze the users' activity patterns, Imbalanced Learning techniques are used to prune and weigh Knowledge Graph, and LLM is used to retrieve and interpret the users' activities from Knowledge Graph. The proposed framework is applied to Agile Threat Detection through Online Sequential Learning. The preliminary results show the improved threat capture rate by 3%-4% and the increased interpretabilities of risk predictions based on the users' activities.

摘要: 由于攻击活动的复杂性和可供学习的历史威胁数据的局限性，网络威胁检测一直具有挑战性。为了帮助增强使用分析、机器学习和人工智能方法检测网络威胁的现有实践，我们提出了一个集成的建模框架，其中使用知识图来分析用户的活动模式，使用不平衡学习技术来修剪和加权知识图，使用LLM来从知识图中检索和解释用户的活动。所提出的框架通过在线顺序学习应用于敏捷威胁检测。初步结果显示，威胁捕获率提高了3%-4%，并且基于用户活动的风险预测的可解释性增强。



## **13. Reliably Bounding False Positives: A Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction**

可靠地限制假阳性：通过多尺度保形预测的零镜头机器生成文本检测框架 cs.CL

**SubmitDate**: 2025-05-14    [abs](http://arxiv.org/abs/2505.05084v2) [paper-pdf](http://arxiv.org/pdf/2505.05084v2)

**Authors**: Xiaowei Zhu, Yubing Ren, Yanan Cao, Xixun Lin, Fang Fang, Yangxi Li

**Abstract**: The rapid advancement of large language models has raised significant concerns regarding their potential misuse by malicious actors. As a result, developing effective detectors to mitigate these risks has become a critical priority. However, most existing detection methods focus excessively on detection accuracy, often neglecting the societal risks posed by high false positive rates (FPRs). This paper addresses this issue by leveraging Conformal Prediction (CP), which effectively constrains the upper bound of FPRs. While directly applying CP constrains FPRs, it also leads to a significant reduction in detection performance. To overcome this trade-off, this paper proposes a Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction (MCP), which both enforces the FPR constraint and improves detection performance. This paper also introduces RealDet, a high-quality dataset that spans a wide range of domains, ensuring realistic calibration and enabling superior detection performance when combined with MCP. Empirical evaluations demonstrate that MCP effectively constrains FPRs, significantly enhances detection performance, and increases robustness against adversarial attacks across multiple detectors and datasets.

摘要: 大型语言模型的快速发展引发了人们对其潜在被恶意行为者滥用的严重担忧。因此，开发有效的探测器来减轻这些风险已成为当务之急。然而，大多数现有的检测方法过度关注检测准确性，往往忽视了高假阳性率（FPR）带来的社会风险。本文通过利用保形预测（CP）来解决这个问题，该预测有效地限制了FPR的上界。虽然直接应用CP约束FPR，但也会导致检测性能显着降低。为了克服这种权衡，本文提出了一种通过多尺度保形预测（LCP）的零镜头机器生成文本检测框架，该框架既强制执行FPR约束又提高检测性能。本文还介绍了RealDet，这是一个跨越广泛领域的高质量数据集，可确保真实的校准并在与HCP结合时实现卓越的检测性能。经验评估表明，LCP有效地约束了FPR，显着增强了检测性能，并增强了针对多个检测器和数据集的对抗攻击的鲁棒性。



## **14. Securing RAG: A Risk Assessment and Mitigation Framework**

保护RAG：风险评估和缓解框架 cs.CR

8 pages, 3 figures, Sara Ott and Lukas Ammann contributed equally

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.08728v1) [paper-pdf](http://arxiv.org/pdf/2505.08728v1)

**Authors**: Lukas Ammann, Sara Ott, Christoph R. Landolt, Marco P. Lehmann

**Abstract**: Retrieval Augmented Generation (RAG) has emerged as the de facto industry standard for user-facing NLP applications, offering the ability to integrate data without re-training or fine-tuning Large Language Models (LLMs). This capability enhances the quality and accuracy of responses but also introduces novel security and privacy challenges, particularly when sensitive data is integrated. With the rapid adoption of RAG, securing data and services has become a critical priority. This paper first reviews the vulnerabilities of RAG pipelines, and outlines the attack surface from data pre-processing and data storage management to integration with LLMs. The identified risks are then paired with corresponding mitigations in a structured overview. In a second step, the paper develops a framework that combines RAG-specific security considerations, with existing general security guidelines, industry standards, and best practices. The proposed framework aims to guide the implementation of robust, compliant, secure, and trustworthy RAG systems.

摘要: 检索增强生成（RAG）已成为面向用户的NLP应用程序事实上的行业标准，提供集成数据的能力，无需重新训练或微调大型语言模型（LLM）。这种能力增强了响应的质量和准确性，但也带来了新的安全和隐私挑战，特别是在集成敏感数据时。随着RAG的迅速采用，保护数据和服务已成为首要任务。本文首先回顾了RAG管道的漏洞，概述了从数据预处理、数据存储管理到与LLM集成的攻击面。然后，在结构化概述中将识别的风险与相应的缓解措施配对。第二步，本文开发了一个框架，该框架将RAG特定的安全考虑因素与现有的通用安全准则、行业标准和最佳实践相结合。拟议的框架旨在指导稳健、合规、安全且值得信赖的RAG系统的实施。



## **15. Red Teaming the Mind of the Machine: A Systematic Evaluation of Prompt Injection and Jailbreak Vulnerabilities in LLMs**

Red联手机器思维：LLM中即时注射和越狱漏洞的系统评估 cs.CR

7 Pages, 6 Figures

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.04806v2) [paper-pdf](http://arxiv.org/pdf/2505.04806v2)

**Authors**: Chetan Pathade

**Abstract**: Large Language Models (LLMs) are increasingly integrated into consumer and enterprise applications. Despite their capabilities, they remain susceptible to adversarial attacks such as prompt injection and jailbreaks that override alignment safeguards. This paper provides a systematic investigation of jailbreak strategies against various state-of-the-art LLMs. We categorize over 1,400 adversarial prompts, analyze their success against GPT-4, Claude 2, Mistral 7B, and Vicuna, and examine their generalizability and construction logic. We further propose layered mitigation strategies and recommend a hybrid red-teaming and sandboxing approach for robust LLM security.

摘要: 大型语言模型（LLM）越来越多地集成到消费者和企业应用程序中。尽管它们有能力，但它们仍然容易受到对抗攻击，例如超越对齐保障措施的立即注射和越狱。本文对针对各种最先进的法学硕士的越狱策略进行了系统调查。我们对1，400多个对抗提示进行了分类，分析了它们对GPT-4、Claude 2、Mistral 7 B和Vicuna的成功，并检查它们的概括性和构造逻辑。我们进一步提出分层缓解策略，并推荐混合红色团队和沙箱方法以实现强大的LLM安全性。



## **16. LM-Scout: Analyzing the Security of Language Model Integration in Android Apps**

LM-Scout：分析Android应用程序中语言模型集成的安全性 cs.CR

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.08204v1) [paper-pdf](http://arxiv.org/pdf/2505.08204v1)

**Authors**: Muhammad Ibrahim, Gűliz Seray Tuncay, Z. Berkay Celik, Aravind Machiry, Antonio Bianchi

**Abstract**: Developers are increasingly integrating Language Models (LMs) into their mobile apps to provide features such as chat-based assistants. To prevent LM misuse, they impose various restrictions, including limits on the number of queries, input length, and allowed topics. However, if the LM integration is insecure, attackers can bypass these restrictions and gain unrestricted access to the LM, potentially harming developers' reputations and leading to significant financial losses.   This paper presents the first systematic study of insecure usage of LMs by Android apps. We first manually analyze a preliminary dataset of apps to investigate LM integration methods, construct a taxonomy that categorizes the LM usage restrictions implemented by the apps, and determine how to bypass them. Alarmingly, we can bypass restrictions in 127 out of 181 apps. Then, we develop LM-Scout, a fully automated tool to detect on a large-scale vulnerable usage of LMs in 2,950 mobile apps. LM-Scout shows that, in many cases (i.e., 120 apps), it is possible to find and exploit such security issues automatically. Finally, we identify the root causes for the identified issues and offer recommendations for secure LM integration.

摘要: 开发人员越来越多地将语言模型（LM）集成到其移动应用程序中，以提供基于聊天的助手等功能。为了防止LM滥用，他们施加了各种限制，包括对查询数量、输入长度和允许的主题的限制。然而，如果LM集成不安全，攻击者可以绕过这些限制并不受限制地访问LM，这可能会损害开发人员的声誉并导致重大财务损失。   本文首次对Android应用程序对LM的不安全使用进行了系统研究。我们首先手动分析应用程序的初步数据集以调查LM集成方法，构建对应用程序实施的LM使用限制进行分类的分类法，并确定如何绕过它们。令人震惊的是，我们可以绕过181个应用程序中的127个应用程序的限制。然后，我们开发了LM-Scout，这是一个全自动化工具，用于检测2，950个移动应用程序中LM的大规模漏洞使用情况。LM-Scout表明，在许多情况下（即，120个应用程序），就可以自动发现并利用此类安全问题。最后，我们找出所识别问题的根本原因，并提供安全LM集成的建议。



## **17. A Large-Scale Empirical Analysis of Custom GPTs' Vulnerabilities in the OpenAI Ecosystem**

OpenAI生态系统中自定义GPT漏洞的大规模实证分析 cs.CR

**SubmitDate**: 2025-05-13    [abs](http://arxiv.org/abs/2505.08148v1) [paper-pdf](http://arxiv.org/pdf/2505.08148v1)

**Authors**: Sunday Oyinlola Ogundoyin, Muhammad Ikram, Hassan Jameel Asghar, Benjamin Zi Hao Zhao, Dali Kaafar

**Abstract**: Millions of users leverage generative pretrained transformer (GPT)-based language models developed by leading model providers for a wide range of tasks. To support enhanced user interaction and customization, many platforms-such as OpenAI-now enable developers to create and publish tailored model instances, known as custom GPTs, via dedicated repositories or application stores. These custom GPTs empower users to browse and interact with specialized applications designed to meet specific needs. However, as custom GPTs see growing adoption, concerns regarding their security vulnerabilities have intensified. Existing research on these vulnerabilities remains largely theoretical, often lacking empirical, large-scale, and statistically rigorous assessments of associated risks.   In this study, we analyze 14,904 custom GPTs to assess their susceptibility to seven exploitable threats, such as roleplay-based attacks, system prompt leakage, phishing content generation, and malicious code synthesis, across various categories and popularity tiers within the OpenAI marketplace. We introduce a multi-metric ranking system to examine the relationship between a custom GPT's popularity and its associated security risks.   Our findings reveal that over 95% of custom GPTs lack adequate security protections. The most prevalent vulnerabilities include roleplay-based vulnerabilities (96.51%), system prompt leakage (92.20%), and phishing (91.22%). Furthermore, we demonstrate that OpenAI's foundational models exhibit inherent security weaknesses, which are often inherited or amplified in custom GPTs. These results highlight the urgent need for enhanced security measures and stricter content moderation to ensure the safe deployment of GPT-based applications.

摘要: 数百万用户利用领先模型提供商开发的基于生成式预训练Transformer（GPT）的语言模型来执行广泛的任务。为了支持增强的用户交互和定制，许多平台（例如OpenAI）现在使开发人员能够通过专用存储库或应用程序商店创建和发布定制的模型实例（称为自定义GPT）。这些自定义GPT使用户能够浏览专为满足特定需求而设计的专业应用程序和交互。然而，随着定制GPT的采用越来越多，对其安全漏洞的担忧也加剧了。关于这些脆弱性的现有研究基本上仍然是理论性的，通常缺乏对相关风险的经验性、大规模和统计上严格的评估。   在这项研究中，我们分析了14，904个自定义GPT，以评估它们对七种可利用威胁的敏感性，例如基于角色扮演的攻击、系统提示泄露、网络钓鱼内容生成和恶意代码合成，涵盖OpenAI市场内的各个类别和流行级别。我们引入多指标排名系统来检查自定义GPT的受欢迎程度与其相关安全风险之间的关系。   我们的调查结果显示，超过95%的定制GPT缺乏足够的安全保护。最常见的漏洞包括基于角色扮演的漏洞（96.51%）、系统提示泄露（92.20%）和网络钓鱼（91.22%）。此外，我们证明OpenAI的基础模型表现出固有的安全弱点，这些弱点通常在自定义GPT中继承或放大。这些结果凸显了迫切需要增强的安全措施和更严格的内容审核，以确保基于GPT的应用程序的安全部署。



## **18. LiteLMGuard: Seamless and Lightweight On-Device Prompt Filtering for Safeguarding Small Language Models against Quantization-induced Risks and Vulnerabilities**

LiteLMGGuard：无缝且轻量级的设备上提示过滤，用于保护小语言模型免受量化引发的风险和漏洞的影响 cs.CR

14 pages, 18 figures, and 4 tables

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.05619v2) [paper-pdf](http://arxiv.org/pdf/2505.05619v2)

**Authors**: Kalyan Nakka, Jimmy Dani, Ausmit Mondal, Nitesh Saxena

**Abstract**: The growing adoption of Large Language Models (LLMs) has influenced the development of their lighter counterparts-Small Language Models (SLMs)-to enable on-device deployment across smartphones and edge devices. These SLMs offer enhanced privacy, reduced latency, server-free functionality, and improved user experience. However, due to resource constraints of on-device environment, SLMs undergo size optimization through compression techniques like quantization, which can inadvertently introduce fairness, ethical and privacy risks. Critically, quantized SLMs may respond to harmful queries directly, without requiring adversarial manipulation, raising significant safety and trust concerns.   To address this, we propose LiteLMGuard (LLMG), an on-device prompt guard that provides real-time, prompt-level defense for quantized SLMs. Additionally, our prompt guard is designed to be model-agnostic such that it can be seamlessly integrated with any SLM, operating independently of underlying architectures. Our LLMG formalizes prompt filtering as a deep learning (DL)-based prompt answerability classification task, leveraging semantic understanding to determine whether a query should be answered by any SLM. Using our curated dataset, Answerable-or-Not, we trained and fine-tuned several DL models and selected ELECTRA as the candidate, with 97.75% answerability classification accuracy.   Our safety effectiveness evaluations demonstrate that LLMG defends against over 87% of harmful prompts, including both direct instruction and jailbreak attack strategies. We further showcase its ability to mitigate the Open Knowledge Attacks, where compromised SLMs provide unsafe responses without adversarial prompting. In terms of prompt filtering effectiveness, LLMG achieves near state-of-the-art filtering accuracy of 94%, with an average latency of 135 ms, incurring negligible overhead for users.

摘要: 大型语言模型（LLM）的日益采用影响了其更轻的同类产品--小型语言模型（SLM）--的发展，以实现跨智能手机和边缘设备的设备上部署。这些STM提供增强的隐私、减少的延迟、无服务器功能和改善的用户体验。然而，由于设备上环境的资源限制，STM通过量化等压缩技术进行尺寸优化，这可能会无意中引入公平性、道德和隐私风险。至关重要的是，量化的SLC可以直接响应有害查询，而不需要对抗性操纵，从而引发重大的安全和信任问题。   为了解决这个问题，我们提出了LiteLMGard（LLMG），这是一种设备上提示保护，为量化的STM提供实时、预算级防御。此外，我们的提示卫士设计为模型不可知，因此它可以与任何SPL无缝集成，独立于底层架构运行。我们的LLMG将提示过滤形式化为基于深度学习（DL）的提示可回答性分类任务，利用语义理解来确定查询是否应该由任何SPL回答。使用我们精心策划的数据集“可供选择”，我们训练和微调了几个DL模型，并选择ELECTRA作为候选模型，其回答性分类准确率为97.75%。   我们的安全有效性评估表明，LLMG可以抵御超过87%的有害提示，包括直接指令和越狱攻击策略。我们进一步展示了其缓解开放知识攻击的能力，其中受攻击的STM在没有对抗提示的情况下提供不安全的响应。在即时过滤有效性方面，LLMG实现了94%的接近最先进的过滤准确率，平均延迟为135 ms，为用户带来的负担可以忽略不计。



## **19. SCA: Improve Semantic Consistent in Unrestricted Adversarial Attacks via DDPM Inversion**

SCA：通过DDPM倒置提高无限制对抗攻击中的语义一致性 cs.CV

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2410.02240v6) [paper-pdf](http://arxiv.org/pdf/2410.02240v6)

**Authors**: Zihao Pan, Lifeng Chen, Weibin Wu, Yuhang Cao, Zibin Zheng

**Abstract**: Systems based on deep neural networks are vulnerable to adversarial attacks. Unrestricted adversarial attacks typically manipulate the semantic content of an image (e.g., color or texture) to create adversarial examples that are both effective and photorealistic. Recent works have utilized the diffusion inversion process to map images into a latent space, where high-level semantics are manipulated by introducing perturbations. However, they often result in substantial semantic distortions in the denoised output and suffer from low efficiency. In this study, we propose a novel framework called Semantic-Consistent Unrestricted Adversarial Attacks (SCA), which employs an inversion method to extract edit-friendly noise maps and utilizes a Multimodal Large Language Model (MLLM) to provide semantic guidance throughout the process. Under the condition of rich semantic information provided by MLLM, we perform the DDPM denoising process of each step using a series of edit-friendly noise maps and leverage DPM Solver++ to accelerate this process, enabling efficient sampling with semantic consistency. Compared to existing methods, our framework enables the efficient generation of adversarial examples that exhibit minimal discernible semantic changes. Consequently, we for the first time introduce Semantic-Consistent Adversarial Examples (SCAE). Extensive experiments and visualizations have demonstrated the high efficiency of SCA, particularly in being on average 12 times faster than the state-of-the-art attacks. Our code can be found at https://github.com/Pan-Zihao/SCA.

摘要: 基于深度神经网络的系统很容易受到对抗攻击。不受限制的对抗攻击通常操纵图像的语义内容（例如，颜色或纹理）来创建既有效又逼真的对抗示例。最近的作品利用扩散倒置过程将图像映射到潜在空间，其中通过引入扰动来操纵高级语义。然而，它们通常会导致去噪输出中出现严重的语义扭曲，并且效率低下。在这项研究中，我们提出了一种名为语义一致的无限制对抗攻击（SCA）的新型框架，该框架采用倒置方法来提取编辑友好的噪音图，并利用多模式大型语言模型（MLLM）来提供整个过程的语义指导。在MLLM提供丰富的语义信息的情况下，我们使用一系列编辑友好的噪音图来执行每一步的DDPM去噪过程，并利用DeliverSolver ++加速这一过程，实现具有语义一致性的高效采样。与现有方法相比，我们的框架能够高效生成表现出最小可辨别的语义变化的对抗性示例。因此，我们首次引入语义一致的对抗示例（SCAE）。大量的实验和可视化已经证明了SCA的高效率，特别是平均比最先进的攻击快12倍。我们的代码可在https://github.com/Pan-Zihao/SCA上找到。



## **20. Concept-Level Explainability for Auditing & Steering LLM Responses**

审计和指导LLM响应的概念级解释性 cs.CL

9 pages, 7 figures, Submission to Neurips 2025

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.07610v1) [paper-pdf](http://arxiv.org/pdf/2505.07610v1)

**Authors**: Kenza Amara, Rita Sevastjanova, Mennatallah El-Assady

**Abstract**: As large language models (LLMs) become widely deployed, concerns about their safety and alignment grow. An approach to steer LLM behavior, such as mitigating biases or defending against jailbreaks, is to identify which parts of a prompt influence specific aspects of the model's output. Token-level attribution methods offer a promising solution, but still struggle in text generation, explaining the presence of each token in the output separately, rather than the underlying semantics of the entire LLM response. We introduce ConceptX, a model-agnostic, concept-level explainability method that identifies the concepts, i.e., semantically rich tokens in the prompt, and assigns them importance based on the outputs' semantic similarity. Unlike current token-level methods, ConceptX also offers to preserve context integrity through in-place token replacements and supports flexible explanation goals, e.g., gender bias. ConceptX enables both auditing, by uncovering sources of bias, and steering, by modifying prompts to shift the sentiment or reduce the harmfulness of LLM responses, without requiring retraining. Across three LLMs, ConceptX outperforms token-level methods like TokenSHAP in both faithfulness and human alignment. Steering tasks boost sentiment shift by 0.252 versus 0.131 for random edits and lower attack success rates from 0.463 to 0.242, outperforming attribution and paraphrasing baselines. While prompt engineering and self-explaining methods sometimes yield safer responses, ConceptX offers a transparent and faithful alternative for improving LLM safety and alignment, demonstrating the practical value of attribution-based explainability in guiding LLM behavior.

摘要: 随着大型语言模型（LLM）的广泛部署，对其安全性和一致性的担忧日益加剧。引导LLM行为（例如减轻偏见或防范越狱）的一种方法是识别提示的哪些部分影响模型输出的特定方面。令牌级归因方法提供了一个有希望的解决方案，但在文本生成方面仍然很困难，分别解释输出中每个令牌的存在，而不是整个LLM响应的底层语义。我们引入ConceptX，这是一种模型不可知的概念级解释方法，可以识别概念，即提示中语义丰富的标记，并根据输出的语义相似性为其分配重要性。与当前的代币级方法不同，ConceptX还提供通过就地代币替换来保持上下文完整性，并支持灵活的解释目标，例如性别偏见。ConceptX通过发现偏见的来源来实现审计，并通过修改提示以改变情绪或减少LLM响应的危害性来实现引导，而无需再培训。在三个LLM中，ConceptX在忠诚度和人性化方面都优于TokenSHAP等代币级方法。随机编辑的引导任务使情绪转变提高了0.252和0.131，攻击成功率从0.463降低到0.242，优于归因和重述基线。虽然及时的工程和自我解释方法有时会产生更安全的响应，但ConceptX为提高LLM安全性和一致性提供了一种透明且忠实的替代方案，展示了基于属性的解释在指导LLM行为方面的实际价值。



## **21. SecReEvalBench: A Multi-turned Security Resilience Evaluation Benchmark for Large Language Models**

SecReEvalBench：大型语言模型的多角度安全弹性评估基准 cs.CR

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.07584v1) [paper-pdf](http://arxiv.org/pdf/2505.07584v1)

**Authors**: Huining Cui, Wei Liu

**Abstract**: The increasing deployment of large language models in security-sensitive domains necessitates rigorous evaluation of their resilience against adversarial prompt-based attacks. While previous benchmarks have focused on security evaluations with limited and predefined attack domains, such as cybersecurity attacks, they often lack a comprehensive assessment of intent-driven adversarial prompts and the consideration of real-life scenario-based multi-turn attacks. To address this gap, we present SecReEvalBench, the Security Resilience Evaluation Benchmark, which defines four novel metrics: Prompt Attack Resilience Score, Prompt Attack Refusal Logic Score, Chain-Based Attack Resilience Score and Chain-Based Attack Rejection Time Score. Moreover, SecReEvalBench employs six questioning sequences for model assessment: one-off attack, successive attack, successive reverse attack, alternative attack, sequential ascending attack with escalating threat levels and sequential descending attack with diminishing threat levels. In addition, we introduce a dataset customized for the benchmark, which incorporates both neutral and malicious prompts, categorised across seven security domains and sixteen attack techniques. In applying this benchmark, we systematically evaluate five state-of-the-art open-weighted large language models, Llama 3.1, Gemma 2, Mistral v0.3, DeepSeek-R1 and Qwen 3. Our findings offer critical insights into the strengths and weaknesses of modern large language models in defending against evolving adversarial threats. The SecReEvalBench dataset is publicly available at https://kaggle.com/datasets/5a7ee22cf9dab6c93b55a73f630f6c9b42e936351b0ae98fbae6ddaca7fe248d, which provides a groundwork for advancing research in large language model security.

摘要: 大型语言模型在安全敏感领域的部署越来越多，需要严格评估它们对抗基于预算的敌对攻击的弹性。虽然之前的基准侧重于有限且预定义的攻击域（例如网络安全攻击）的安全评估，但它们通常缺乏对意图驱动的对抗提示的全面评估以及对现实生活中基于情景的多回合攻击的考虑。为了解决这一差距，我们提出了SecReEvalBench，安全韧性评估基准，它定义了四个新颖的指标：即时攻击韧性分数、即时攻击拒绝逻辑分数、基于链的攻击韧性分数和基于链的攻击拒绝时间分数。此外，SecReEvalBench采用六个提问序列进行模型评估：一次性攻击、连续攻击、连续反向攻击、替代攻击、威胁级别不断上升的顺序上升攻击和威胁级别不断下降的顺序下降攻击。此外，我们还引入了一个为基准定制的数据集，其中包含中性和恶意提示，分为七个安全域和十六种攻击技术。在应用该基准时，我们系统地评估了五个最先进的开放加权大型语言模型：Llama 3.1、Gemma 2、Mistral v0.3、DeepSeek-R1和Qwen 3。我们的研究结果为现代大型语言模型在防御不断变化的对抗威胁方面的优势和弱点提供了重要的见解。SecReEvalBench数据集可在https：//kaggle.com/guardets/5a7ee22CF9dab6c93b55a73f630f6c9 b42 e936351 b 0ae 98 fbae 6ddaca 7 fe 248 d上公开，为推进大型语言模型安全性研究提供了基础。



## **22. SCAM: A Real-World Typographic Robustness Evaluation for Multimodal Foundation Models**

SCAM：多模式基础模型的真实印刷稳健性评估 cs.CV

Accepted at CVPR 2025 Workshop EVAL-FoMo-2

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2504.04893v3) [paper-pdf](http://arxiv.org/pdf/2504.04893v3)

**Authors**: Justus Westerhoff, Erblina Purelku, Jakob Hackstein, Jonas Loos, Leo Pinetzki, Lorenz Hufe

**Abstract**: Typographic attacks exploit the interplay between text and visual content in multimodal foundation models, causing misclassifications when misleading text is embedded within images. However, existing datasets are limited in size and diversity, making it difficult to study such vulnerabilities. In this paper, we introduce SCAM, the largest and most diverse dataset of real-world typographic attack images to date, containing 1,162 images across hundreds of object categories and attack words. Through extensive benchmarking of Vision-Language Models (VLMs) on SCAM, we demonstrate that typographic attacks significantly degrade performance, and identify that training data and model architecture influence the susceptibility to these attacks. Our findings reveal that typographic attacks persist in state-of-the-art Large Vision-Language Models (LVLMs) due to the choice of their vision encoder, though larger Large Language Models (LLMs) backbones help mitigate their vulnerability. Additionally, we demonstrate that synthetic attacks closely resemble real-world (handwritten) attacks, validating their use in research. Our work provides a comprehensive resource and empirical insights to facilitate future research toward robust and trustworthy multimodal AI systems. We publicly release the datasets introduced in this paper along with the code for evaluations at www.bliss.berlin/research/scam.

摘要: 排版攻击利用多模态基础模型中文本和视觉内容之间的相互作用，当误导性文本嵌入图像中时会导致错误分类。然而，现有的数据集在规模和多样性方面有限，因此难以研究这些脆弱性。在本文中，我们介绍了SCAM，这是迄今为止最大、最多样化的现实世界印刷攻击图像数据集，包含涵盖数百个对象类别和攻击词的1，162张图像。通过对SCAM上的视觉语言模型（VLM）进行广泛的基准测试，我们证明了排版攻击会显着降低性能，并确定训练数据和模型架构会影响对这些攻击的敏感性。我们的研究结果表明，由于视觉编码器的选择，印刷攻击在最先进的大型视觉语言模型（LVLM）中持续存在，尽管更大的大型语言模型（LLM）主干有助于减轻它们的脆弱性。此外，我们还证明合成攻击与现实世界（手写）攻击非常相似，验证了它们在研究中的用途。我们的工作提供了全面的资源和经验见解，以促进未来对稳健且值得信赖的多模式人工智能系统的研究。我们在www.bliss.berlin/research/scam上公开发布本文中介绍的数据集以及评估代码。



## **23. GRADA: Graph-based Reranker against Adversarial Documents Attack**

GRADA：针对对抗文档攻击的基于图形的重新搜索器 cs.IR

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.07546v1) [paper-pdf](http://arxiv.org/pdf/2505.07546v1)

**Authors**: Jingjie Zheng, Aryo Pradipta Gema, Giwon Hong, Xuanli He, Pasquale Minervini, Youcheng Sun, Qiongkai Xu

**Abstract**: Retrieval Augmented Generation (RAG) frameworks improve the accuracy of large language models (LLMs) by integrating external knowledge from retrieved documents, thereby overcoming the limitations of models' static intrinsic knowledge. However, these systems are susceptible to adversarial attacks that manipulate the retrieval process by introducing documents that are adversarial yet semantically similar to the query. Notably, while these adversarial documents resemble the query, they exhibit weak similarity to benign documents in the retrieval set. Thus, we propose a simple yet effective Graph-based Reranking against Adversarial Document Attacks (GRADA) framework aiming at preserving retrieval quality while significantly reducing the success of adversaries. Our study evaluates the effectiveness of our approach through experiments conducted on five LLMs: GPT-3.5-Turbo, GPT-4o, Llama3.1-8b, Llama3.1-70b, and Qwen2.5-7b. We use three datasets to assess performance, with results from the Natural Questions dataset demonstrating up to an 80% reduction in attack success rates while maintaining minimal loss in accuracy.

摘要: 检索增强生成（RAG）框架通过集成来自检索文档的外部知识来提高大型语言模型（LLM）的准确性，从而克服模型静态内在知识的局限性。然而，这些系统很容易受到对抗性攻击，这些攻击通过引入对抗性但在语义上与查询相似的文档来操纵检索过程。值得注意的是，虽然这些对抗性文档类似于查询，但它们与检索集中的良性文档表现出弱的相似性。因此，我们提出了一个简单而有效的基于图形的对抗性文档攻击重新排名（GRADA）框架，旨在保留检索质量，同时显着降低对手的成功。我们的研究通过在五个LLM上进行的实验来评估我们方法的有效性：GPT-3.5-Turbo、GPT-4 o、Llama 3.1 -8b、Llama 3.1 - 70 b和Qwen 2.5 - 7 b。我们使用三个数据集来评估性能，Natural Questions数据集的结果表明，攻击成功率可降低高达80%，同时保持最小的准确性损失。



## **24. No Query, No Access**

无查询，无访问 cs.CL

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.07258v1) [paper-pdf](http://arxiv.org/pdf/2505.07258v1)

**Authors**: Wenqiang Wang, Siyuan Liang, Yangshijie Zhang, Xiaojun Jia, Hao Lin, Xiaochun Cao

**Abstract**: Textual adversarial attacks mislead NLP models, including Large Language Models (LLMs), by subtly modifying text. While effective, existing attacks often require knowledge of the victim model, extensive queries, or access to training data, limiting real-world feasibility. To overcome these constraints, we introduce the \textbf{Victim Data-based Adversarial Attack (VDBA)}, which operates using only victim texts. To prevent access to the victim model, we create a shadow dataset with publicly available pre-trained models and clustering methods as a foundation for developing substitute models. To address the low attack success rate (ASR) due to insufficient information feedback, we propose the hierarchical substitution model design, generating substitute models to mitigate the failure of a single substitute model at the decision boundary.   Concurrently, we use diverse adversarial example generation, employing various attack methods to generate and select the adversarial example with better similarity and attack effectiveness. Experiments on the Emotion and SST5 datasets show that VDBA outperforms state-of-the-art methods, achieving an ASR improvement of 52.08\% while significantly reducing attack queries to 0. More importantly, we discover that VDBA poses a significant threat to LLMs such as Qwen2 and the GPT family, and achieves the highest ASR of 45.99% even without access to the API, confirming that advanced NLP models still face serious security risks. Our codes can be found at https://anonymous.4open.science/r/VDBA-Victim-Data-based-Adversarial-Attack-36EC/

摘要: 文本对抗攻击通过微妙地修改文本来误导NLP模型，包括大型语言模型（LLM）。虽然有效，但现有的攻击通常需要了解受害者模型、广泛的查询或访问训练数据，从而限制了现实世界的可行性。为了克服这些限制，我们引入了\textBF{基于受害者数据的对抗攻击（VDBA）}，它仅使用受害者文本来操作。为了防止访问受害者模型，我们创建了一个影子数据集，其中包含公开可用的预训练模型和集群方法，作为开发替代模型的基础。为了解决由于信息反馈不足而导致的攻击成功率（ASB）低的问题，我们提出了分层替代模型设计，生成替代模型以减轻单个替代模型在决策边界的失败。   同时，我们使用多样化的对抗性示例生成，采用各种攻击方法来生成并选择具有更好相似性和攻击有效性的对抗性示例。Emoy和CST 5数据集的实验表明，VDBA优于最先进的方法，实现了52.08%的ASB改进，同时将攻击查询显着减少到0。更重要的是，我们发现VDBA对Qwen 2和GPT系列等LLM构成了重大威胁，即使在不访问API的情况下也能达到45.99%的最高ASB，证实高级NLP模型仍然面临严重的安全风险。我们的代码可在https://anonymous.4open.science/r/VDBA-Victim-Data-based-Adversarial-Attack-36EC/上找到



## **25. One Trigger Token Is Enough: A Defense Strategy for Balancing Safety and Usability in Large Language Models**

一个触发令牌就足够了：平衡大型语言模型安全性和可用性的防御策略 cs.CR

**SubmitDate**: 2025-05-12    [abs](http://arxiv.org/abs/2505.07167v1) [paper-pdf](http://arxiv.org/pdf/2505.07167v1)

**Authors**: Haoran Gu, Handing Wang, Yi Mei, Mengjie Zhang, Yaochu Jin

**Abstract**: Large Language Models (LLMs) have been extensively used across diverse domains, including virtual assistants, automated code generation, and scientific research. However, they remain vulnerable to jailbreak attacks, which manipulate the models into generating harmful responses despite safety alignment. Recent studies have shown that current safety-aligned LLMs often undergo the shallow safety alignment, where the first few tokens largely determine whether the response will be harmful. Through comprehensive observations, we find that safety-aligned LLMs and various defense strategies generate highly similar initial tokens in their refusal responses, which we define as safety trigger tokens. Building on this insight, we propose \texttt{D-STT}, a simple yet effective defense algorithm that identifies and explicitly decodes safety trigger tokens of the given safety-aligned LLM to trigger the model's learned safety patterns. In this process, the safety trigger is constrained to a single token, which effectively preserves model usability by introducing minimum intervention in the decoding process. Extensive experiments across diverse jailbreak attacks and benign prompts demonstrate that \ours significantly reduces output harmfulness while preserving model usability and incurring negligible response time overhead, outperforming ten baseline methods.

摘要: 大型语言模型（LLM）已广泛用于各种领域，包括虚拟助手，自动代码生成和科学研究。然而，它们仍然容易受到越狱攻击，这些攻击操纵模型生成有害的响应，尽管安全对齐。最近的研究表明，当前的安全对齐LLM通常会经历浅安全对齐，其中前几个令牌在很大程度上决定了响应是否有害。通过全面的观察，我们发现，安全对齐的LLM和各种防御策略在其拒绝响应中生成高度相似的初始令牌，我们将其定义为安全触发令牌。基于这一见解，我们提出了\textttt {D-STT}，这是一种简单而有效的防御算法，可以识别和显式解码给定安全对齐LLM的安全触发令牌，以触发模型的学习安全模式。在此过程中，安全触发器被限制在单个令牌上，通过在解码过程中引入最小干预来有效地保留模型的可用性。针对各种越狱攻击和良性提示的广泛实验表明，我们的方法显着降低了输出危害性，同时保留了模型可用性并产生可忽略的响应时间负担，优于十种基线方法。



## **26. Revealing Weaknesses in Text Watermarking Through Self-Information Rewrite Attacks**

通过自信息重写攻击揭示文本水印的弱点 cs.LG

ICML 2025 Accpeted

**SubmitDate**: 2025-05-11    [abs](http://arxiv.org/abs/2505.05190v2) [paper-pdf](http://arxiv.org/pdf/2505.05190v2)

**Authors**: Yixin Cheng, Hongcheng Guo, Yangming Li, Leonid Sigal

**Abstract**: Text watermarking aims to subtly embed statistical signals into text by controlling the Large Language Model (LLM)'s sampling process, enabling watermark detectors to verify that the output was generated by the specified model. The robustness of these watermarking algorithms has become a key factor in evaluating their effectiveness. Current text watermarking algorithms embed watermarks in high-entropy tokens to ensure text quality. In this paper, we reveal that this seemingly benign design can be exploited by attackers, posing a significant risk to the robustness of the watermark. We introduce a generic efficient paraphrasing attack, the Self-Information Rewrite Attack (SIRA), which leverages the vulnerability by calculating the self-information of each token to identify potential pattern tokens and perform targeted attack. Our work exposes a widely prevalent vulnerability in current watermarking algorithms. The experimental results show SIRA achieves nearly 100% attack success rates on seven recent watermarking methods with only 0.88 USD per million tokens cost. Our approach does not require any access to the watermark algorithms or the watermarked LLM and can seamlessly transfer to any LLM as the attack model, even mobile-level models. Our findings highlight the urgent need for more robust watermarking.

摘要: 文本水印旨在通过控制大型语言模型（LLM）的采样过程将统计信号巧妙地嵌入到文本中，使水印检测器能够验证输出是否由指定模型生成。这些水印算法的鲁棒性已成为评估其有效性的关键因素。当前的文本水印算法将水印嵌入高熵令牌中以确保文本质量。在本文中，我们揭示了这种看似良性的设计可能会被攻击者利用，从而对水印的稳健性构成重大风险。我们引入了一种通用的高效解释攻击，即自我信息重写攻击（SIRA），它通过计算每个令牌的自我信息来利用漏洞来识别潜在的模式令牌并执行有针对性的攻击。我们的工作揭示了当前水印算法中广泛存在的漏洞。实验结果表明，SIRA对最近的七种水印方法的攻击成功率接近100%，每百万个代币的成本仅为0.88美元。我们的方法不需要对水印算法或带水印的LLM进行任何访问，并且可以无缝地转移到任何LLM作为攻击模型，甚至是移动级模型。我们的研究结果凸显了对更鲁棒的水印的迫切需求。



## **27. Unleashing the potential of prompt engineering for large language models**

释放大型语言模型即时工程的潜力 cs.CL

v6 - Metadata updated (title, journal ref, DOI). PDF identical to v5  (original submission). Please cite the peer-reviewed Version of Record in  "Patterns" (DOI: 10.1016/j.patter.2025.101260)

**SubmitDate**: 2025-05-11    [abs](http://arxiv.org/abs/2310.14735v6) [paper-pdf](http://arxiv.org/pdf/2310.14735v6)

**Authors**: Banghao Chen, Zhaofeng Zhang, Nicolas Langrené, Shengxin Zhu

**Abstract**: This comprehensive review delves into the pivotal role of prompt engineering in unleashing the capabilities of Large Language Models (LLMs). The development of Artificial Intelligence (AI), from its inception in the 1950s to the emergence of advanced neural networks and deep learning architectures, has made a breakthrough in LLMs, with models such as GPT-4o and Claude-3, and in Vision-Language Models (VLMs), with models such as CLIP and ALIGN. Prompt engineering is the process of structuring inputs, which has emerged as a crucial technique to maximize the utility and accuracy of these models. This paper explores both foundational and advanced methodologies of prompt engineering, including techniques such as self-consistency, chain-of-thought, and generated knowledge, which significantly enhance model performance. Additionally, it examines the prompt method of VLMs through innovative approaches such as Context Optimization (CoOp), Conditional Context Optimization (CoCoOp), and Multimodal Prompt Learning (MaPLe). Critical to this discussion is the aspect of AI security, particularly adversarial attacks that exploit vulnerabilities in prompt engineering. Strategies to mitigate these risks and enhance model robustness are thoroughly reviewed. The evaluation of prompt methods is also addressed through both subjective and objective metrics, ensuring a robust analysis of their efficacy. This review also reflects the essential role of prompt engineering in advancing AI capabilities, providing a structured framework for future research and application.

摘要: 这篇全面的评论深入探讨了即时工程在释放大型语言模型（LLM）功能方面的关键作用。人工智能（AI）的发展，从20世纪50年代的诞生到先进神经网络和深度学习架构的出现，在LLM（如GPT-4 o和Claude-3）以及视觉语言模型（VLM）（如CLIP和ALIGN）等模型方面取得了突破。即时工程是结构化输入的过程，它已成为最大限度地提高这些模型的实用性和准确性的关键技术。本文探讨了即时工程的基础和高级方法，包括自一致性、思想链和生成知识等技术，这些技术显着增强模型性能。此外，它还通过上下文优化（CoOp）、条件上下文优化（CoCoOp）和多模式提示学习（MaPLe）等创新方法研究了VLM的提示方法。本次讨论的关键是人工智能安全方面，特别是利用即时工程中漏洞的对抗攻击。彻底审查了缓解这些风险和增强模型稳健性的策略。提示方法的评估还通过主观和客观指标来解决，确保对其功效进行稳健的分析。这篇评论还反映了快速工程在推进人工智能能力方面的重要作用，为未来的研究和应用提供了一个结构化的框架。



## **28. IM-BERT: Enhancing Robustness of BERT through the Implicit Euler Method**

IM-BERT：通过隐式欧拉方法增强BERT的鲁棒性 cs.CL

Accepted to EMNLP 2024 Main

**SubmitDate**: 2025-05-11    [abs](http://arxiv.org/abs/2505.06889v1) [paper-pdf](http://arxiv.org/pdf/2505.06889v1)

**Authors**: Mihyeon Kim, Juhyoung Park, Youngbin Kim

**Abstract**: Pre-trained Language Models (PLMs) have achieved remarkable performance on diverse NLP tasks through pre-training and fine-tuning. However, fine-tuning the model with a large number of parameters on limited downstream datasets often leads to vulnerability to adversarial attacks, causing overfitting of the model on standard datasets.   To address these issues, we propose IM-BERT from the perspective of a dynamic system by conceptualizing a layer of BERT as a solution of Ordinary Differential Equations (ODEs). Under the situation of initial value perturbation, we analyze the numerical stability of two main numerical ODE solvers: the explicit and implicit Euler approaches.   Based on these analyses, we introduce a numerically robust IM-connection incorporating BERT's layers. This strategy enhances the robustness of PLMs against adversarial attacks, even in low-resource scenarios, without introducing additional parameters or adversarial training strategies.   Experimental results on the adversarial GLUE (AdvGLUE) dataset validate the robustness of IM-BERT under various conditions. Compared to the original BERT, IM-BERT exhibits a performance improvement of approximately 8.3\%p on the AdvGLUE dataset. Furthermore, in low-resource scenarios, IM-BERT outperforms BERT by achieving 5.9\%p higher accuracy.

摘要: 通过预训练和微调，预训练语言模型（PLM）在各种NLP任务上取得了显着的性能。然而，在有限的下游数据集上使用大量参数对模型进行微调通常会导致对抗性攻击的脆弱性，从而导致模型在标准数据集上的过拟合。   为了解决这些问题，我们从动态系统的角度提出了IM-BERT，将BERT层概念化为常微分方程（ODE）的解。在初值摄动的情况下，我们分析了两种主要的常微分方程数值解法：显式和隐式欧拉方法的数值稳定性。   基于这些分析，我们引入了一种包含BERT层的数字鲁棒的IM连接。该策略增强了PLM对抗攻击的稳健性，即使在低资源场景中也是如此，而无需引入额外的参数或对抗训练策略。   对抗性GLUE（AdvGLUE）数据集的实验结果验证了IM-BERT在各种条件下的鲁棒性。与原始BERT相比，IM-BERT在AdvGLUE数据集上表现出约8.3%p的性能改进。此外，在低资源场景中，IM-BERT的准确性比BERT高出5.9%p。



## **29. Benign Samples Matter! Fine-tuning On Outlier Benign Samples Severely Breaks Safety**

良性样本很重要！对异常值良性样本进行微调严重破坏安全性 cs.LG

26 pages, 13 figures

**SubmitDate**: 2025-05-11    [abs](http://arxiv.org/abs/2505.06843v1) [paper-pdf](http://arxiv.org/pdf/2505.06843v1)

**Authors**: Zihan Guan, Mengxuan Hu, Ronghang Zhu, Sheng Li, Anil Vullikanti

**Abstract**: Recent studies have uncovered a troubling vulnerability in the fine-tuning stage of large language models (LLMs): even fine-tuning on entirely benign datasets can lead to a significant increase in the harmfulness of LLM outputs. Building on this finding, our red teaming study takes this threat one step further by developing a more effective attack. Specifically, we analyze and identify samples within benign datasets that contribute most to safety degradation, then fine-tune LLMs exclusively on these samples. We approach this problem from an outlier detection perspective and propose Self-Inf-N, to detect and extract outliers for fine-tuning. Our findings reveal that fine-tuning LLMs on 100 outlier samples selected by Self-Inf-N in the benign datasets severely compromises LLM safety alignment. Extensive experiments across seven mainstream LLMs demonstrate that our attack exhibits high transferability across different architectures and remains effective in practical scenarios. Alarmingly, our results indicate that most existing mitigation strategies fail to defend against this attack, underscoring the urgent need for more robust alignment safeguards. Codes are available at https://github.com/GuanZihan/Benign-Samples-Matter.

摘要: 最近的研究发现了大型语言模型（LLM）微调阶段的一个令人不安的漏洞：即使对完全良性的数据集进行微调也可能导致LLM输出的危害性显着增加。在这一发现的基础上，我们的红色团队研究通过开发更有效的攻击来进一步推进这一威胁。具体来说，我们分析和识别良性数据集中对安全性下降影响最大的样本，然后专门对这些样本进行微调。我们从异常值检测的角度来解决这个问题，并提出Self-Inf-N来检测和提取异常值以进行微调。我们的研究结果表明，对Self-Inf-N在良性数据集中选择的100个异常值样本进行微调LLM会严重损害LLM的安全性对齐。针对七种主流LLM的广泛实验表明，我们的攻击在不同架构中表现出高度的可移植性，并且在实际场景中仍然有效。令人震惊的是，我们的结果表明，大多数现有的缓解策略都无法抵御这种攻击，这凸显了迫切需要更强大的对齐保障措施。代码可访问https://github.com/GuanZihan/Benign-Samples-Matter。



## **30. Diversity Helps Jailbreak Large Language Models**

多样性帮助越狱大型语言模型 cs.CL

**SubmitDate**: 2025-05-11    [abs](http://arxiv.org/abs/2411.04223v3) [paper-pdf](http://arxiv.org/pdf/2411.04223v3)

**Authors**: Weiliang Zhao, Daniel Ben-Levi, Wei Hao, Junfeng Yang, Chengzhi Mao

**Abstract**: We have uncovered a powerful jailbreak technique that leverages large language models' ability to diverge from prior context, enabling them to bypass safety constraints and generate harmful outputs. By simply instructing the LLM to deviate and obfuscate previous attacks, our method dramatically outperforms existing approaches, achieving up to a 62.83% higher success rate in compromising ten leading chatbots, including GPT-4, Gemini, and Llama, while using only 12.9% of the queries. This revelation exposes a critical flaw in current LLM safety training, suggesting that existing methods may merely mask vulnerabilities rather than eliminate them. Our findings sound an urgent alarm for the need to revolutionize testing methodologies to ensure robust and reliable LLM security.

摘要: 我们发现了一种强大的越狱技术，该技术利用大型语言模型脱离先前上下文的能力，使它们能够绕过安全约束并生成有害输出。通过简单地指示LLM偏离和混淆之前的攻击，我们的方法显着优于现有方法，在攻击包括GPT-4、Gemini和Llama在内的十个领先聊天机器人时，成功率提高了62.83%，而仅使用12.9%的查询。这一揭露暴露了当前LLM安全培训中的一个关键缺陷，表明现有方法可能只是掩盖了漏洞而不是消除漏洞。我们的发现敲响了紧急警报，需要彻底改变测试方法，以确保强大和可靠的LLM安全。



## **31. Practical Reasoning Interruption Attacks on Reasoning Large Language Models**

对推理大型语言模型的实用推理中断攻击 cs.CR

**SubmitDate**: 2025-05-10    [abs](http://arxiv.org/abs/2505.06643v1) [paper-pdf](http://arxiv.org/pdf/2505.06643v1)

**Authors**: Yu Cui, Cong Zuo

**Abstract**: Reasoning large language models (RLLMs) have demonstrated outstanding performance across a variety of tasks, yet they also expose numerous security vulnerabilities. Most of these vulnerabilities have centered on the generation of unsafe content. However, recent work has identified a distinct "thinking-stopped" vulnerability in DeepSeek-R1: under adversarial prompts, the model's reasoning process ceases at the system level and produces an empty final answer. Building upon this vulnerability, researchers developed a novel prompt injection attack, termed reasoning interruption attack, and also offered an initial analysis of its root cause. Through extensive experiments, we verify the previous analyses, correct key errors based on three experimental findings, and present a more rigorous explanation of the fundamental causes driving the vulnerability. Moreover, existing attacks typically require over 2,000 tokens, impose significant overhead, reduce practicality, and are easily detected. To overcome these limitations, we propose the first practical reasoning interruption attack. It succeeds with just 109 tokens by exploiting our newly uncovered "reasoning token overflow" (RTO) effect to overwrite the model's final answer, forcing it to return an invalid response. Experimental results demonstrate that our proposed attack is highly effective. Furthermore, we discover that the method for triggering RTO differs between the official DeepSeek-R1 release and common unofficial deployments. As a broadened application of RTO, we also construct a novel jailbreak attack that enables the transfer of unsafe content within the reasoning tokens into final answer, thereby exposing it to the user. Our work carries significant implications for enhancing the security of RLLMs.

摘要: 推理大型语言模型（RLLM）在各种任务中表现出出色的性能，但它们也暴露了许多安全漏洞。大多数漏洞都集中在不安全内容的生成上。然而，最近的工作在DeepSeek-R1中发现了一个明显的“思维停止”漏洞：在对抗性提示下，模型的推理过程在系统级别停止并产生空的最终答案。在此漏洞的基础上，研究人员开发了一种新型的即时注入攻击，称为推理中断攻击，并对其根本原因进行了初步分析。通过广泛的实验，我们验证了之前的分析，根据三个实验发现纠正了关键错误，并对驱动该漏洞的根本原因进行了更严格的解释。此外，现有的攻击通常需要超过2，000个令牌，造成大量的费用，降低实用性，并且很容易被检测到。为了克服这些限制，我们提出了第一个实际推理中断攻击。它利用我们新发现的“推理令牌溢出”（RTI）效应来覆盖模型的最终答案，迫使其返回无效响应，仅用109个令牌就成功了。实验结果表明我们提出的攻击非常有效。此外，我们发现官方DeepSeek-R1版本和常见的非官方部署之间触发RTI的方法有所不同。作为RTI的扩展应用，我们还构建了一种新颖的越狱攻击，可以将推理令牌中的不安全内容转移到最终答案中，从而将其暴露给用户。我们的工作对于增强LLLM的安全性具有重大影响。



## **32. POISONCRAFT: Practical Poisoning of Retrieval-Augmented Generation for Large Language Models**

POISONCRAFT：大型语言模型的检索增强生成的实际毒害 cs.CR

12 pages, 7 tables and 3 figures

**SubmitDate**: 2025-05-10    [abs](http://arxiv.org/abs/2505.06579v1) [paper-pdf](http://arxiv.org/pdf/2505.06579v1)

**Authors**: Yangguang Shao, Xinjie Lin, Haozheng Luo, Chengshang Hou, Gang Xiong, Jiahao Yu, Junzheng Shi

**Abstract**: Large language models (LLMs) have achieved remarkable success in various domains, primarily due to their strong capabilities in reasoning and generating human-like text. Despite their impressive performance, LLMs are susceptible to hallucinations, which can lead to incorrect or misleading outputs. This is primarily due to the lack of up-to-date knowledge or domain-specific information. Retrieval-augmented generation (RAG) is a promising approach to mitigate hallucinations by leveraging external knowledge sources. However, the security of RAG systems has not been thoroughly studied. In this paper, we study a poisoning attack on RAG systems named POISONCRAFT, which can mislead the model to refer to fraudulent websites. Compared to existing poisoning attacks on RAG systems, our attack is more practical as it does not require access to the target user query's info or edit the user query. It not only ensures that injected texts can be retrieved by the model, but also ensures that the LLM will be misled to refer to the injected texts in its response. We demonstrate the effectiveness of POISONCRAFTacross different datasets, retrievers, and language models in RAG pipelines, and show that it remains effective when transferred across retrievers, including black-box systems. Moreover, we present a case study revealing how the attack influences both the retrieval behavior and the step-by-step reasoning trace within the generation model, and further evaluate the robustness of POISONCRAFTunder multiple defense mechanisms. These results validate the practicality of our threat model and highlight a critical security risk for RAG systems deployed in real-world applications. We release our code\footnote{https://github.com/AndyShaw01/PoisonCraft} to support future research on the security and robustness of RAG systems in real-world settings.

摘要: 大型语言模型（LLM）在各个领域取得了显着的成功，主要是由于它们在推理和生成类人文本方面的强大能力。尽管LLM的性能令人印象深刻，但它们很容易产生幻觉，这可能会导致错误或误导性的输出。这主要是由于缺乏最新知识或特定领域的信息。检索增强生成（RAG）是一种通过利用外部知识源来减轻幻觉的有前途的方法。然而，RAG系统的安全性尚未得到彻底研究。本文研究了一种名为POISONCRAFT的RAG系统中毒攻击，该攻击可能会误导模型引用欺诈网站。与现有的RAG系统中毒攻击相比，我们的攻击更实用，因为它不需要访问目标用户查询的信息或编辑用户查询。它不仅确保模型可以检索注入的文本，还确保LLM将被误导在其响应中引用注入的文本。我们展示了POISONCRAFT在RAG管道中不同数据集、检索器和语言模型中的有效性，并表明它在跨检索器（包括黑匣子系统）传输时仍然有效。此外，我们还提供了一个案例研究，揭示了攻击如何影响生成模型中的检索行为和逐步推理痕迹，并进一步评估了POISONCRAFT在多种防御机制下的稳健性。这些结果验证了我们威胁模型的实用性，并强调了部署在现实世界应用程序中的RAG系统的关键安全风险。我们发布了我们的代码\脚注{https：//github.com/AndyShaw01/PoisonCraft}，以支持未来对现实世界环境中RAG系统的安全性和稳健性的研究。



## **33. Fun-tuning: Characterizing the Vulnerability of Proprietary LLMs to Optimization-based Prompt Injection Attacks via the Fine-Tuning Interface**

有趣的调整：通过微调接口描述专有LLM对基于优化的提示注入攻击的脆弱性 cs.CR

**SubmitDate**: 2025-05-10    [abs](http://arxiv.org/abs/2501.09798v2) [paper-pdf](http://arxiv.org/pdf/2501.09798v2)

**Authors**: Andrey Labunets, Nishit V. Pandya, Ashish Hooda, Xiaohan Fu, Earlence Fernandes

**Abstract**: We surface a new threat to closed-weight Large Language Models (LLMs) that enables an attacker to compute optimization-based prompt injections. Specifically, we characterize how an attacker can leverage the loss-like information returned from the remote fine-tuning interface to guide the search for adversarial prompts. The fine-tuning interface is hosted by an LLM vendor and allows developers to fine-tune LLMs for their tasks, thus providing utility, but also exposes enough information for an attacker to compute adversarial prompts. Through an experimental analysis, we characterize the loss-like values returned by the Gemini fine-tuning API and demonstrate that they provide a useful signal for discrete optimization of adversarial prompts using a greedy search algorithm. Using the PurpleLlama prompt injection benchmark, we demonstrate attack success rates between 65% and 82% on Google's Gemini family of LLMs. These attacks exploit the classic utility-security tradeoff - the fine-tuning interface provides a useful feature for developers but also exposes the LLMs to powerful attacks.

摘要: 我们对封闭权重大型语言模型（LLM）提出了新的威胁，该威胁使攻击者能够计算基于优化的提示注入。具体来说，我们描述了攻击者如何利用从远程微调界面返回的类似损失的信息来指导搜索对抗性提示。微调接口由LLM供应商托管，允许开发人员针对其任务微调LLM，从而提供实用性，但也暴露了足够的信息供攻击者计算对抗提示。通过实验分析，我们描述了Gemini微调API返回的类似损失的值，并证明它们为使用贪婪搜索算法对对抗性提示的离散优化提供了有用的信号。使用PurpleLlama提示注入基准，我们证明了Google Gemini LLM系列的攻击成功率在65%至82%之间。这些攻击利用了经典的实用程序-安全权衡-微调接口为开发人员提供了有用的功能，但也使LLM面临强大的攻击。



## **34. System Prompt Poisoning: Persistent Attacks on Large Language Models Beyond User Injection**

系统提示中毒：对大型语言模型的持续攻击超出用户注入 cs.CR

**SubmitDate**: 2025-05-10    [abs](http://arxiv.org/abs/2505.06493v1) [paper-pdf](http://arxiv.org/pdf/2505.06493v1)

**Authors**: Jiawei Guo, Haipeng Cai

**Abstract**: Large language models (LLMs) have gained widespread adoption across diverse applications due to their impressive generative capabilities. Their plug-and-play nature enables both developers and end users to interact with these models through simple prompts. However, as LLMs become more integrated into various systems in diverse domains, concerns around their security are growing. Existing studies mainly focus on threats arising from user prompts (e.g. prompt injection attack) and model output (e.g. model inversion attack), while the security of system prompts remains largely overlooked. This work bridges the critical gap. We introduce system prompt poisoning, a new attack vector against LLMs that, unlike traditional user prompt injection, poisons system prompts hence persistently impacts all subsequent user interactions and model responses. We systematically investigate four practical attack strategies in various poisoning scenarios. Through demonstration on both generative and reasoning LLMs, we show that system prompt poisoning is highly feasible without requiring jailbreak techniques, and effective across a wide range of tasks, including those in mathematics, coding, logical reasoning, and natural language processing. Importantly, our findings reveal that the attack remains effective even when user prompts employ advanced prompting techniques like chain-of-thought (CoT). We also show that such techniques, including CoT and retrieval-augmentation-generation (RAG), which are proven to be effective for improving LLM performance in a wide range of tasks, are significantly weakened in their effectiveness by system prompt poisoning.

摘要: 大型语言模型（LLM）因其令人印象深刻的生成能力而在不同的应用程序中得到广泛采用。其即插即用性质使开发人员和最终用户能够通过简单的提示与这些模型进行交互。然而，随着LLM越来越多地集成到不同领域的各种系统中，对其安全性的担忧也越来越大。现有的研究主要关注用户提示（例如提示注入攻击）和模型输出（例如模型倒置攻击）引起的威胁，而系统提示的安全性在很大程度上仍然被忽视。这项工作弥合了关键差距。我们引入了系统提示中毒，这是一种针对LLM的新攻击载体，与传统的用户提示注入不同，系统提示中毒，因此持续影响所有后续用户交互和模型响应。我们系统地研究了各种中毒场景下的四种实用攻击策略。通过生成式和推理式LLM的演示，我们表明系统提示中毒在不需要越狱技术的情况下是高度可行的，并且在广泛的任务中有效，包括数学、编码、逻辑推理和自然语言处理。重要的是，我们的研究结果表明，即使用户提示采用思想链（CoT）等高级提示技术，攻击仍然有效。我们还表明，这些技术，包括CoT和检索增强生成（RAG），被证明可以有效地提高广泛任务中的LLM性能，但其有效性因系统即时中毒而显着削弱。



## **35. Does Data Contamination Detection Work (Well) for LLMs? A Survey and Evaluation on Detection Assumptions**

数据污染检测对LLM有效吗？检测假设的调查与评价 cs.CL

This paper is accepted by NAACL 2025 findings. Link to the paper  presentation: https://youtu.be/IhaxwbZOcaU

**SubmitDate**: 2025-05-09    [abs](http://arxiv.org/abs/2410.18966v3) [paper-pdf](http://arxiv.org/pdf/2410.18966v3)

**Authors**: Yujuan Fu, Ozlem Uzuner, Meliha Yetisgen, Fei Xia

**Abstract**: Large language models (LLMs) have demonstrated great performance across various benchmarks, showing potential as general-purpose task solvers. However, as LLMs are typically trained on vast amounts of data, a significant concern in their evaluation is data contamination, where overlap between training data and evaluation datasets inflates performance assessments. Multiple approaches have been developed to identify data contamination. These approaches rely on specific assumptions that may not hold universally across different settings. To bridge this gap, we systematically review 50 papers on data contamination detection, categorize the underlying assumptions, and assess whether they have been rigorously validated. We identify and analyze eight categories of assumptions and test three of them as case studies. Our case studies focus on detecting direct, instance-level data contamination, which is also referred to as Membership Inference Attacks (MIA). Our analysis reveals that MIA approaches based on these three assumptions can have similar performance to random guessing, on datasets used in LLM pretraining, suggesting that current LLMs might learn data distributions rather than memorizing individual instances. Meanwhile, MIA can easily fail when there are data distribution shifts between the seen and unseen instances.

摘要: 大型语言模型（LLM）在各种基准测试中表现出出色的性能，显示出作为通用任务解决器的潜力。然而，由于LLM通常在大量数据上进行训练，因此其评估中的一个重大问题是数据污染，即训练数据和评估数据集之间的重叠会加剧绩效评估。已经开发了多种方法来识别数据污染。这些方法依赖于特定的假设，这些假设可能并不在不同的环境中普遍成立。为了弥合这一差距，我们系统地审查了50篇有关数据污染检测的论文，对基本假设进行分类，并评估它们是否经过严格验证。我们识别和分析了八类假设，并将其中三类作为案例研究进行测试。我们的案例研究重点是检测直接的实例级数据污染，这也称为会员推断攻击（MIA）。我们的分析表明，在LLM预训练中使用的数据集上，基于这三个假设的MIA方法可以具有与随机猜测类似的性能，这表明当前的LLM可能会学习数据分布，而不是记住单个实例。与此同时，当可见和不可见的实例之间存在数据分布变化时，MIA很容易失败。



## **36. LATENT: LLM-Augmented Trojan Insertion and Evaluation Framework for Analog Netlist Topologies**

LATENT：模拟网表布局的LLM增强特洛伊木马插入和评估框架 cs.CR

Accepted for presentation at IEEE International Conference on  LLM-Aided Design (ICLAD), 2025

**SubmitDate**: 2025-05-09    [abs](http://arxiv.org/abs/2505.06364v1) [paper-pdf](http://arxiv.org/pdf/2505.06364v1)

**Authors**: Jayeeta Chaudhuri, Arjun Chaudhuri, Krishnendu Chakrabarty

**Abstract**: Analog and mixed-signal (A/MS) integrated circuits (ICs) are integral to safety-critical applications. However, the globalization and outsourcing of A/MS ICs to untrusted third-party foundries expose them to security threats, particularly analog Trojans. Unlike digital Trojans which have been extensively studied, analog Trojans remain largely unexplored. There has been only limited research on their diversity and stealth in analog designs, where a Trojan is activated only during a narrow input voltage range. Effective defense techniques require a clear understanding of the attack vectors; however, the lack of diverse analog Trojan instances limits robust advances in detection strategies. To address this gap, we present LATENT, the first large language model (LLM)-driven framework for crafting stealthy, circuit-specific analog Trojans. LATENT incorporates LLM as an autonomous agent to intelligently insert and refine Trojan components within analog designs based on iterative feedback from a detection model. This feedback loop ensures that the inserted Trojans remain stealthy while successfully evading detection. Experimental results demonstrate that our generated Trojan designs exhibit an average Trojan-activation range of 15.74%, ensuring they remain inactive under most operating voltages, while causing a significant performance degradation of 11.3% upon activation.

摘要: 模拟和混合信号（A/MS）集成电路（IC）是安全关键应用的组成部分。然而，A/MS IC的全球化和外包给不受信任的第三方代工厂使它们面临安全威胁，尤其是模拟特洛伊木马。与已被广泛研究的数字特洛伊木马不同，模拟特洛伊木马在很大程度上仍未被探索。对模拟设计中特洛伊木马的多样性和隐形性的研究有限，其中特洛伊木马仅在狭窄的输入电压范围内激活。有效的防御技术需要清楚地了解攻击载体;然而，缺乏多样化的模拟特洛伊木马实例限制了检测策略的稳健进步。为了解决这一差距，我们提出了LATENT，这是第一个大型语言模型（LLM）驱动的框架，用于制作隐形的、特定于电路的模拟特洛伊木马。LATENT将LLM整合为一个自主代理，可以根据检测模型的迭代反馈智能地在模拟设计中插入和细化特洛伊木马组件。这个反馈循环确保插入的特洛伊木马在成功逃避检测的同时保持隐蔽性。实验结果表明，我们生成的特洛伊木马设计的平均特洛伊激活范围为15.74%，确保它们在大多数工作电压下保持不活动，同时在激活后导致11.3%的性能显着下降。



## **37. AgentXploit: End-to-End Redteaming of Black-Box AI Agents**

AgentXploit：黑匣子人工智能代理的端到端红色团队 cs.CR

**SubmitDate**: 2025-05-09    [abs](http://arxiv.org/abs/2505.05849v1) [paper-pdf](http://arxiv.org/pdf/2505.05849v1)

**Authors**: Zhun Wang, Vincent Siu, Zhe Ye, Tianneng Shi, Yuzhou Nie, Xuandong Zhao, Chenguang Wang, Wenbo Guo, Dawn Song

**Abstract**: The strong planning and reasoning capabilities of Large Language Models (LLMs) have fostered the development of agent-based systems capable of leveraging external tools and interacting with increasingly complex environments. However, these powerful features also introduce a critical security risk: indirect prompt injection, a sophisticated attack vector that compromises the core of these agents, the LLM, by manipulating contextual information rather than direct user prompts. In this work, we propose a generic black-box fuzzing framework, AgentXploit, designed to automatically discover and exploit indirect prompt injection vulnerabilities across diverse LLM agents. Our approach starts by constructing a high-quality initial seed corpus, then employs a seed selection algorithm based on Monte Carlo Tree Search (MCTS) to iteratively refine inputs, thereby maximizing the likelihood of uncovering agent weaknesses. We evaluate AgentXploit on two public benchmarks, AgentDojo and VWA-adv, where it achieves 71% and 70% success rates against agents based on o3-mini and GPT-4o, respectively, nearly doubling the performance of baseline attacks. Moreover, AgentXploit exhibits strong transferability across unseen tasks and internal LLMs, as well as promising results against defenses. Beyond benchmark evaluations, we apply our attacks in real-world environments, successfully misleading agents to navigate to arbitrary URLs, including malicious sites.

摘要: 大型语言模型（LLM）强大的规划和推理能力促进了基于代理的系统的开发，这些系统能够利用外部工具并与日益复杂的环境进行交互。然而，这些强大的功能也引入了一个严重的安全风险：间接提示注入，这是一种复杂的攻击载体，通过操纵上下文信息而不是直接用户提示来损害这些代理的核心LLM。在这项工作中，我们提出了一个通用的黑匣子模糊框架AgentXploit，旨在自动发现和利用不同LLM代理之间的间接提示注入漏洞。我们的方法首先构建高质量的初始种子库，然后采用基于蒙特卡洛树搜索（MCTS）的种子选择算法来迭代细化输入，从而最大化发现代理弱点的可能性。我们在AgentDojo和VWA-adv这两个公共基准上评估了AgentXploit，它分别对基于o3-mini和GPT-4 o的代理实现了71%和70%的成功率，几乎是基线攻击性能的两倍。此外，AgentXploit在看不见的任务和内部LLM之间具有很强的可移植性，以及对抗防御的有希望的结果。除了基准评估之外，我们还将我们的攻击应用于现实环境中，成功地误导代理导航到任意URL，包括恶意网站。



## **38. Unified Attacks to Large Language Model Watermarks: Spoofing and Scrubbing in Unauthorized Knowledge Distillation**

对大型语言模型水印的统一攻击：未经授权的知识提炼中的欺骗和擦除 cs.CL

**SubmitDate**: 2025-05-09    [abs](http://arxiv.org/abs/2504.17480v3) [paper-pdf](http://arxiv.org/pdf/2504.17480v3)

**Authors**: Xin Yi, Yue Li, Shunfan Zheng, Linlin Wang, Xiaoling Wang, Liang He

**Abstract**: Watermarking has emerged as a critical technique for combating misinformation and protecting intellectual property in large language models (LLMs). A recent discovery, termed watermark radioactivity, reveals that watermarks embedded in teacher models can be inherited by student models through knowledge distillation. On the positive side, this inheritance allows for the detection of unauthorized knowledge distillation by identifying watermark traces in student models. However, the robustness of watermarks against scrubbing attacks and their unforgeability in the face of spoofing attacks under unauthorized knowledge distillation remain largely unexplored. Existing watermark attack methods either assume access to model internals or fail to simultaneously support both scrubbing and spoofing attacks. In this work, we propose Contrastive Decoding-Guided Knowledge Distillation (CDG-KD), a unified framework that enables bidirectional attacks under unauthorized knowledge distillation. Our approach employs contrastive decoding to extract corrupted or amplified watermark texts via comparing outputs from the student model and weakly watermarked references, followed by bidirectional distillation to train new student models capable of watermark removal and watermark forgery, respectively. Extensive experiments show that CDG-KD effectively performs attacks while preserving the general performance of the distilled model. Our findings underscore critical need for developing watermarking schemes that are robust and unforgeable.

摘要: 水印已成为打击错误信息和保护大型语言模型（LLM）知识产权的关键技术。最近的一项发现称为水印放射性，揭示了教师模型中嵌入的水印可以通过知识蒸馏被学生模型继承。从积极的方面来说，这种继承允许通过识别学生模型中的水印痕迹来检测未经授权的知识提炼。然而，水印对擦洗攻击的鲁棒性及其在未经授权的知识提炼下面对欺骗攻击时的不可伪造性在很大程度上仍然没有被探索。现有的水印攻击方法要么假设访问模型内部，要么不能同时支持擦洗和欺骗攻击。在这项工作中，我们提出了对比解码引导的知识蒸馏（CDG-KD），一个统一的框架，使未经授权的知识蒸馏下的双向攻击。我们的方法采用对比解码提取损坏或放大的水印文本，通过比较输出的学生模型和弱水印的参考，然后通过双向蒸馏训练新的学生模型能够水印去除和水印伪造，分别。大量的实验表明，CDG-KD有效地执行攻击，同时保持蒸馏模型的一般性能。我们的研究结果强调了开发稳健且不可伪造的水印方案的迫切需要。



## **39. Towards the Worst-case Robustness of Large Language Models**

走向大型语言模型的最坏情况稳健性 cs.LG

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2501.19040v2) [paper-pdf](http://arxiv.org/pdf/2501.19040v2)

**Authors**: Huanran Chen, Yinpeng Dong, Zeming Wei, Hang Su, Jun Zhu

**Abstract**: Recent studies have revealed the vulnerability of large language models to adversarial attacks, where adversaries craft specific input sequences to induce harmful, violent, private, or incorrect outputs. In this work, we study their worst-case robustness, i.e., whether an adversarial example exists that leads to such undesirable outputs. We upper bound the worst-case robustness using stronger white-box attacks, indicating that most current deterministic defenses achieve nearly 0\% worst-case robustness. We propose a general tight lower bound for randomized smoothing using fractional knapsack solvers or 0-1 knapsack solvers, and using them to bound the worst-case robustness of all stochastic defenses. Based on these solvers, we provide theoretical lower bounds for several previous empirical defenses. For example, we certify the robustness of a specific case, smoothing using a uniform kernel, against \textit{any possible attack} with an average $\ell_0$ perturbation of 2.02 or an average suffix length of 6.41.

摘要: 最近的研究揭示了大型语言模型容易受到对抗攻击，对手会精心设计特定的输入序列来引发有害、暴力、私密或错误的输出。在这项工作中，我们研究了它们的最坏情况稳健性，即是否存在导致此类不良结果的对抗性例子。我们使用更强的白盒攻击来对最坏情况的稳健性进行上限，这表明当前大多数确定性防御实现了近0%的最坏情况的稳健性。我们提出了使用分数背包求解器或0-1背包求解器的随机平滑的一般紧下界，并使用它们来限制所有随机防御的最坏情况稳健性。基于这些求解器，我们为之前的几个经验防御提供了理论下限。例如，我们证明了特定情况的稳健性，使用统一核进行平滑，针对\texttit {任何可能的攻击}，平均$\ell_0 $扰动为2.02或平均后缀长度为6.41。



## **40. Jailbreaking and Mitigation of Vulnerabilities in Large Language Models**

大型语言模型中的漏洞越狱和缓解 cs.CR

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2410.15236v2) [paper-pdf](http://arxiv.org/pdf/2410.15236v2)

**Authors**: Benji Peng, Keyu Chen, Qian Niu, Ziqian Bi, Ming Liu, Pohsun Feng, Tianyang Wang, Lawrence K. Q. Yan, Yizhu Wen, Yichao Zhang, Caitlyn Heqi Yin

**Abstract**: Large Language Models (LLMs) have transformed artificial intelligence by advancing natural language understanding and generation, enabling applications across fields beyond healthcare, software engineering, and conversational systems. Despite these advancements in the past few years, LLMs have shown considerable vulnerabilities, particularly to prompt injection and jailbreaking attacks. This review analyzes the state of research on these vulnerabilities and presents available defense strategies. We roughly categorize attack approaches into prompt-based, model-based, multimodal, and multilingual, covering techniques such as adversarial prompting, backdoor injections, and cross-modality exploits. We also review various defense mechanisms, including prompt filtering, transformation, alignment techniques, multi-agent defenses, and self-regulation, evaluating their strengths and shortcomings. We also discuss key metrics and benchmarks used to assess LLM safety and robustness, noting challenges like the quantification of attack success in interactive contexts and biases in existing datasets. Identifying current research gaps, we suggest future directions for resilient alignment strategies, advanced defenses against evolving attacks, automation of jailbreak detection, and consideration of ethical and societal impacts. This review emphasizes the need for continued research and cooperation within the AI community to enhance LLM security and ensure their safe deployment.

摘要: 大型语言模型（LLM）通过推进自然语言理解和生成，改变了人工智能，实现了医疗保健、软件工程和会话系统以外的应用。尽管过去几年取得了这些进步，但LLM仍表现出相当大的漏洞，特别是在引发注射和越狱攻击方面。本评论分析了这些漏洞的研究状况，并提出了可用的防御策略。我们大致将攻击方法分为基于模型的，基于模型的，多模式的和多语言的，涵盖了对抗性提示，后门注入和跨模式利用等技术。我们还回顾了各种防御机制，包括即时过滤、转换、对齐技术、多智能体防御和自我调节，评估它们的优点和缺点。我们还讨论了用于评估LLM安全性和稳健性的关键指标和基准，并指出了交互式环境中攻击成功的量化以及现有数据集中的偏差等挑战。通过识别当前的研究差距，我们提出了弹性对齐策略、针对不断发展的攻击的先进防御、越狱检测自动化以及道德和社会影响的未来方向。该审查强调了人工智能社区内持续研究与合作的必要性，以增强LLM安全性并确保其安全部署。



## **41. Defending against Indirect Prompt Injection by Instruction Detection**

利用指令检测防御间接提示注入 cs.CR

13 pages, 4 figures

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.06311v1) [paper-pdf](http://arxiv.org/pdf/2505.06311v1)

**Authors**: Tongyu Wen, Chenglong Wang, Xiyuan Yang, Haoyu Tang, Yueqi Xie, Lingjuan Lyu, Zhicheng Dou, Fangzhao Wu

**Abstract**: The integration of Large Language Models (LLMs) with external sources is becoming increasingly common, with Retrieval-Augmented Generation (RAG) being a prominent example. However, this integration introduces vulnerabilities of Indirect Prompt Injection (IPI) attacks, where hidden instructions embedded in external data can manipulate LLMs into executing unintended or harmful actions. We recognize that the success of IPI attacks fundamentally relies in the presence of instructions embedded within external content, which can alter the behavioral state of LLMs. Can effectively detecting such state changes help us defend against IPI attacks? In this paper, we propose a novel approach that takes external data as input and leverages the behavioral state of LLMs during both forward and backward propagation to detect potential IPI attacks. Specifically, we demonstrate that the hidden states and gradients from intermediate layers provide highly discriminative features for instruction detection. By effectively combining these features, our approach achieves a detection accuracy of 99.60\% in the in-domain setting and 96.90\% in the out-of-domain setting, while reducing the attack success rate to just 0.12\% on the BIPIA benchmark.

摘要: 大型语言模型（LLM）与外部源的集成变得越来越普遍，检索增强生成（RAG）就是一个突出的例子。然而，这种集成引入了间接提示注入（IPI）攻击的漏洞，其中嵌入在外部数据中的隐藏指令可以操纵LLM执行意外或有害的操作。我们认识到，IPI攻击的成功从根本上依赖于外部内容中嵌入的指令的存在，这可以改变LLM的行为状态。有效地检测这种状态变化是否可以帮助我们抵御IPI攻击？在本文中，我们提出了一种新的方法，该方法将外部数据作为输入，并利用LLM在前向和后向传播过程中的行为状态来检测潜在的IPI攻击。具体来说，我们证明了隐藏的状态和梯度从中间层提供了高度区分功能的指令检测。通过有效地结合这些功能，我们的方法实现了99.60%的检测准确率在域内设置和96.90%在域外设置，同时降低攻击成功率只有0.12%的BIPIA基准。



## **42. Stealthy LLM-Driven Data Poisoning Attacks Against Embedding-Based Retrieval-Augmented Recommender Systems**

针对基于嵌入的检索增强推荐系统的隐形LLM驱动的数据中毒攻击 cs.IR

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.05196v1) [paper-pdf](http://arxiv.org/pdf/2505.05196v1)

**Authors**: Fatemeh Nazary, Yashar Deldjoo, Tommaso Di Noia, Eugenio Di Sciascio

**Abstract**: We present a systematic study of provider-side data poisoning in retrieval-augmented recommender systems (RAG-based). By modifying only a small fraction of tokens within item descriptions -- for instance, adding emotional keywords or borrowing phrases from semantically related items -- an attacker can significantly promote or demote targeted items. We formalize these attacks under token-edit and semantic-similarity constraints, and we examine their effectiveness in both promotion (long-tail items) and demotion (short-head items) scenarios. Our experiments on MovieLens, using two large language model (LLM) retrieval modules, show that even subtle attacks shift final rankings and item exposures while eluding naive detection. The results underscore the vulnerability of RAG-based pipelines to small-scale metadata rewrites and emphasize the need for robust textual consistency checks and provenance tracking to thwart stealthy provider-side poisoning.

摘要: 我们对检索增强推荐系统（基于RAG）中的提供商端数据中毒进行了系统研究。通过仅修改物品描述中的一小部分标记--例如添加情感关键词或借用语义相关物品的短语--攻击者可以显着提升或降级目标物品。我们在标记编辑和语义相似性约束下对这些攻击进行形式化，并检查它们在晋升（长尾项）和降级（短头项）场景中的有效性。我们使用两个大型语言模型（LLM）检索模块在MovieLens上进行的实验表明，即使是微妙的攻击也会改变最终排名和项目暴露，同时避免天真的检测。结果强调了基于RAG的管道对小规模元数据重写的脆弱性，并强调需要强大的文本一致性检查和出处跟踪来阻止隐形的提供商端中毒。



## **43. X-Transfer Attacks: Towards Super Transferable Adversarial Attacks on CLIP**

X-Transfer攻击：CLIP上的超级可转移对抗攻击 cs.CV

ICML 2025

**SubmitDate**: 2025-05-08    [abs](http://arxiv.org/abs/2505.05528v1) [paper-pdf](http://arxiv.org/pdf/2505.05528v1)

**Authors**: Hanxun Huang, Sarah Erfani, Yige Li, Xingjun Ma, James Bailey

**Abstract**: As Contrastive Language-Image Pre-training (CLIP) models are increasingly adopted for diverse downstream tasks and integrated into large vision-language models (VLMs), their susceptibility to adversarial perturbations has emerged as a critical concern. In this work, we introduce \textbf{X-Transfer}, a novel attack method that exposes a universal adversarial vulnerability in CLIP. X-Transfer generates a Universal Adversarial Perturbation (UAP) capable of deceiving various CLIP encoders and downstream VLMs across different samples, tasks, and domains. We refer to this property as \textbf{super transferability}--a single perturbation achieving cross-data, cross-domain, cross-model, and cross-task adversarial transferability simultaneously. This is achieved through \textbf{surrogate scaling}, a key innovation of our approach. Unlike existing methods that rely on fixed surrogate models, which are computationally intensive to scale, X-Transfer employs an efficient surrogate scaling strategy that dynamically selects a small subset of suitable surrogates from a large search space. Extensive evaluations demonstrate that X-Transfer significantly outperforms previous state-of-the-art UAP methods, establishing a new benchmark for adversarial transferability across CLIP models. The code is publicly available in our \href{https://github.com/HanxunH/XTransferBench}{GitHub repository}.

摘要: 随着对比图像预训练（CLIP）模型越来越多地被用于各种下游任务并集成到大型视觉语言模型（VLM）中，它们对对抗性扰动的敏感性已成为一个关键问题。在这项工作中，我们介绍了\textbf{X-Transfer}，一种新的攻击方法，暴露了CLIP中的一个普遍的对抗性漏洞。X-Transfer生成一个通用对抗扰动（Universal Adversarial Perturbation，UAP），能够欺骗不同样本、任务和域中的各种CLIP编码器和下游VLM。我们将此属性称为\textbf{super transferability}--一个同时实现跨数据、跨域、跨模型和跨任务对抗性可转移性的单一扰动。这是通过\textBF{代理缩放}来实现的，这是我们方法的一个关键创新。与依赖于固定代理模型（扩展计算密集型）的现有方法不同，X-Transfer采用高效的代理扩展策略，可以从大搜索空间中动态选择合适代理的一小子集。广泛的评估表明，X-Transfer的性能显着优于之前最先进的UAP方法，为跨CLIP模型的对抗性可移植性建立了新的基准。该代码可在我们的\href{https：//github.com/HanxunH/XTransferBench}{GitHub存储库}中公开获取。



## **44. Safeguard-by-Development: A Privacy-Enhanced Development Paradigm for Multi-Agent Collaboration Systems**

开发保障：多代理协作系统的隐私增强开发范式 cs.CR

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04799v1) [paper-pdf](http://arxiv.org/pdf/2505.04799v1)

**Authors**: Jian Cui, Zichuan Li, Luyi Xing, Xiaojing Liao

**Abstract**: Multi-agent collaboration systems (MACS), powered by large language models (LLMs), solve complex problems efficiently by leveraging each agent's specialization and communication between agents. However, the inherent exchange of information between agents and their interaction with external environments, such as LLM, tools, and users, inevitably introduces significant risks of sensitive data leakage, including vulnerabilities to attacks like prompt injection and reconnaissance. Existing MACS fail to enable privacy controls, making it challenging to manage sensitive information securely. In this paper, we take the first step to address the MACS's data leakage threat at the system development level through a privacy-enhanced development paradigm, Maris. Maris enables rigorous message flow control within MACS by embedding reference monitors into key multi-agent conversation components. We implemented Maris as an integral part of AutoGen, a widely adopted open-source multi-agent development framework. Then, we evaluate Maris for its effectiveness and performance overhead on privacy-critical MACS use cases, including healthcare, supply chain optimization, and personalized recommendation system. The result shows that Maris achieves satisfactory effectiveness, performance overhead and practicability for adoption.

摘要: 多代理协作系统（MACS）由大型语言模型（LLM）提供支持，通过利用每个代理的专业化和代理之间的通信来有效地解决复杂问题。然而，代理之间固有的信息交换及其与外部环境（例如LLM、工具和用户）的交互，不可避免地会带来敏感数据泄露的重大风险，包括即时注入和侦察等攻击的漏洞。现有的MACS无法启用隐私控制，因此安全地管理敏感信息具有挑战性。在本文中，我们迈出了第一步，通过隐私增强的开发范式Maris在系统开发层面解决MACS的数据泄露威胁。Maris通过将引用监视器嵌入到关键的多代理对话组件中来在MACS内实现严格的消息流控制。我们将Maris实施为AutoGen的一部分，AutoGen是一个广泛采用的开源多代理开发框架。然后，我们评估Maris在隐私关键MACS用例（包括医疗保健、供应链优化和个性化推荐系统）上的有效性和性能费用。结果表明，Maris达到了令人满意的有效性、性能负担和采用的实用性。



## **45. A Proposal for Evaluating the Operational Risk for ChatBots based on Large Language Models**

基于大型语言模型评估ChatBots运营风险的提案 cs.CR

21 pages

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04784v1) [paper-pdf](http://arxiv.org/pdf/2505.04784v1)

**Authors**: Pedro Pinacho-Davidson, Fernando Gutierrez, Pablo Zapata, Rodolfo Vergara, Pablo Aqueveque

**Abstract**: The emergence of Generative AI (Gen AI) and Large Language Models (LLMs) has enabled more advanced chatbots capable of human-like interactions. However, these conversational agents introduce a broader set of operational risks that extend beyond traditional cybersecurity considerations. In this work, we propose a novel, instrumented risk-assessment metric that simultaneously evaluates potential threats to three key stakeholders: the service-providing organization, end users, and third parties. Our approach incorporates the technical complexity required to induce erroneous behaviors in the chatbot--ranging from non-induced failures to advanced prompt-injection attacks--as well as contextual factors such as the target industry, user age range, and vulnerability severity. To validate our metric, we leverage Garak, an open-source framework for LLM vulnerability testing. We further enhance Garak to capture a variety of threat vectors (e.g., misinformation, code hallucinations, social engineering, and malicious code generation). Our methodology is demonstrated in a scenario involving chatbots that employ retrieval-augmented generation (RAG), showing how the aggregated risk scores guide both short-term mitigation and longer-term improvements in model design and deployment. The results underscore the importance of multi-dimensional risk assessments in operationalizing secure, reliable AI-driven conversational systems.

摘要: 生成式人工智能（Gen AI）和大型语言模型（LLM）的出现使更先进的聊天机器人能够进行类人交互。然而，这些对话代理引入了一系列更广泛的运营风险，超出了传统的网络安全考虑。在这项工作中，我们提出了一种新颖的、工具化的风险评估指标，该指标同时评估对三个关键利益相关者的潜在威胁：服务提供组织、最终用户和第三方。我们的方法结合了在聊天机器人中诱导错误行为所需的技术复杂性（从非诱导故障到高级预算注入攻击），以及目标行业、用户年龄范围和漏洞严重性等上下文因素。为了验证我们的指标，我们利用Garak，这是一个LLM漏洞测试的开源框架。我们进一步增强Garak以捕获各种威胁载体（例如，错误信息、代码幻觉、社会工程和恶意代码生成）。我们的方法在涉及采用检索增强生成（RAG）的聊天机器人的场景中进行了演示，展示了汇总风险评分如何指导模型设计和部署的短期缓解和长期改进。结果强调了多维风险评估在运营安全、可靠的人工智能驱动对话系统方面的重要性。



## **46. ACE: A Security Architecture for LLM-Integrated App Systems**

ACE：LLM集成应用程序系统的安全架构 cs.CR

21 pages, 13 figures; clarify relation to indirect prompt injection  attacks

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2504.20984v2) [paper-pdf](http://arxiv.org/pdf/2504.20984v2)

**Authors**: Evan Li, Tushin Mallick, Evan Rose, William Robertson, Alina Oprea, Cristina Nita-Rotaru

**Abstract**: LLM-integrated app systems extend the utility of Large Language Models (LLMs) with third-party apps that are invoked by a system LLM using interleaved planning and execution phases to answer user queries. These systems introduce new attack vectors where malicious apps can cause integrity violation of planning or execution, availability breakdown, or privacy compromise during execution.   In this work, we identify new attacks impacting the integrity of planning, as well as the integrity and availability of execution in LLM-integrated apps, and demonstrate them against IsolateGPT, a recent solution designed to mitigate attacks from malicious apps. We propose Abstract-Concrete-Execute (ACE), a new secure architecture for LLM-integrated app systems that provides security guarantees for system planning and execution. Specifically, ACE decouples planning into two phases by first creating an abstract execution plan using only trusted information, and then mapping the abstract plan to a concrete plan using installed system apps. We verify that the plans generated by our system satisfy user-specified secure information flow constraints via static analysis on the structured plan output. During execution, ACE enforces data and capability barriers between apps, and ensures that the execution is conducted according to the trusted abstract plan. We show experimentally that our system is secure against attacks from the INJECAGENT benchmark, a standard benchmark for control flow integrity in the face of indirect prompt injection attacks, and our newly introduced attacks. Our architecture represents a significant advancement towards hardening LLM-based systems containing system facilities of varying levels of trustworthiness.

摘要: LLM集成的应用程序系统通过第三方应用程序扩展了大型语言模型（LLM）的实用性，第三方应用程序由系统LLM使用交错的规划和执行阶段调用，以回答用户查询。这些系统引入了新的攻击载体，恶意应用程序可能会导致规划或执行的完整性违反、可用性崩溃或执行期间的隐私受到损害。   在这项工作中，我们识别了影响规划完整性以及LLM集成应用程序中执行完整性和可用性的新攻击，并针对IsolateGPT（旨在减轻恶意应用程序攻击的最新解决方案）进行演示。我们提出Abstract-Concrete-Execute（ACE），这是一种针对LLM集成应用程序系统的新安全架构，为系统规划和执行提供安全保障。具体来说，ACE将规划分为两个阶段，首先仅使用可信信息创建抽象执行计划，然后使用已安装的系统应用程序将抽象计划映射到具体计划。我们通过对结构化计划输出的静态分析来验证系统生成的计划是否满足用户指定的安全信息流约束。在执行过程中，ACE在应用程序之间强制设置数据和能力障碍，并确保执行按照可信的抽象计划进行。我们通过实验证明，我们的系统可以抵御来自INJECAGENT基准测试（面对间接提示注入攻击时控制流完整性的标准基准）的攻击，以及我们新引入的攻击。我们的架构代表了强化基于LLM的系统的重大进步，该系统包含不同可信度级别的系统设施。



## **47. Fight Fire with Fire: Defending Against Malicious RL Fine-Tuning via Reward Neutralization**

以毒攻毒：通过奖励中和防御恶意RL微调 cs.LG

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04578v1) [paper-pdf](http://arxiv.org/pdf/2505.04578v1)

**Authors**: Wenjun Cao

**Abstract**: Reinforcement learning (RL) fine-tuning transforms large language models while creating a vulnerability we experimentally verify: Our experiment shows that malicious RL fine-tuning dismantles safety guardrails with remarkable efficiency, requiring only 50 steps and minimal adversarial prompts, with harmful escalating from 0-2 to 7-9. This attack vector particularly threatens open-source models with parameter-level access. Existing defenses targeting supervised fine-tuning prove ineffective against RL's dynamic feedback mechanisms. We introduce Reward Neutralization, the first defense framework specifically designed against RL fine-tuning attacks, establishing concise rejection patterns that render malicious reward signals ineffective. Our approach trains models to produce minimal-information rejections that attackers cannot exploit, systematically neutralizing attempts to optimize toward harmful outputs. Experiments validate that our approach maintains low harmful scores (no greater than 2) after 200 attack steps, while standard models rapidly deteriorate. This work provides the first constructive proof that robust defense against increasingly accessible RL attacks is achievable, addressing a critical security gap for open-weight models.

摘要: 强化学习（RL）微调改变了大型语言模型，同时创建了我们实验验证的漏洞：我们的实验表明，恶意RL微调以显着的效率突破了安全护栏，只需要50个步骤和最少的对抗提示，有害的升级从0-2升级到7-9。这种攻击载体特别威胁具有参数级访问权限的开源模型。事实证明，针对监督式微调的现有防御措施对RL的动态反馈机制无效。我们引入了奖励中和，这是第一个专门针对RL微调攻击而设计的防御框架，建立了简洁的拒绝模式，使恶意奖励信号无效。我们的方法训练模型以产生攻击者无法利用的最小信息拒绝，系统性地抵消针对有害输出进行优化的尝试。实验验证了我们的方法在200次攻击步骤后保持较低的有害分数（不大于2），而标准模型迅速恶化。这项工作提供了第一个建设性的证据，证明可以实现针对日益容易获得的RL攻击的强大防御，解决了开权模型的关键安全差距。



## **48. An LLM-based Self-Evolving Security Framework for 6G Space-Air-Ground Integrated Networks**

基于LLM的6G空地综合网络自进化安全框架 cs.CR

Accepted by IEEE Communications Magazine

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.03161v2) [paper-pdf](http://arxiv.org/pdf/2505.03161v2)

**Authors**: Qi Qin, Xinye Cao, Guoshun Nan, Sihan Chen, Rushan Li, Li Su, Haitao Du, Qimei Cui, Pengxuan Mao, Xiaofeng Tao, Tony Q. S. Quek

**Abstract**: Recently emerged 6G space-air-ground integrated networks (SAGINs), which integrate satellites, aerial networks, and terrestrial communications, offer ubiquitous coverage for various mobile applications. However, the highly dynamic, open, and heterogeneous nature of SAGINs poses severe security issues. Forming a defense line of SAGINs suffers from two preliminary challenges: 1) accurately understanding massive unstructured multi-dimensional threat information to generate defense strategies against various malicious attacks, 2) rapidly adapting to potential unknown threats to yield more effective security strategies. To tackle the above two challenges, we propose a novel security framework for SAGINs based on Large Language Models (LLMs), which consists of two key ingredients LLM-6GNG and 6G-INST. Our proposed LLM-6GNG leverages refined chain-of-thought (CoT) reasoning and dynamic multi-agent mechanisms to analyze massive unstructured multi-dimensional threat data and generate comprehensive security strategies, thus addressing the first challenge. Our proposed 6G-INST relies on a novel self-evolving method to automatically update LLM-6GNG, enabling it to accommodate unknown threats under dynamic communication environments, thereby addressing the second challenge. Additionally, we prototype the proposed framework with ns-3, OpenAirInterface (OAI), and software-defined radio (SDR). Experiments on three benchmarks demonstrate the effectiveness of our framework. The results show that our framework produces highly accurate security strategies that remain robust against a variety of unknown attacks. We will release our code to contribute to the community.

摘要: 最近出现的6 G空-空-地综合网络（SAGER）集成了卫星、空中网络和地面通信，为各种移动应用提供无处不在的覆盖。然而，SATIN的高度动态、开放和异类性质带来了严重的安全问题。形成SATIN防线面临两个初步挑战：1）准确理解大量非结构化多维威胁信息，以生成针对各种恶意攻击的防御策略，2）快速适应潜在的未知威胁，以生成更有效的安全策略。为了应对上述两个挑战，我们提出了一种基于大型语言模型（LLM）的SAGER的新型安全框架，该框架由LLM-6 GNG和6 G-INST两个关键成分组成。我们提出的LLM-6 GNG利用精细化思想链（CoT）推理和动态多代理机制来分析大量非结构化多维威胁数据并生成全面的安全策略，从而解决第一个挑战。我们提出的6 G-INST依赖于一种新颖的自我进化方法来自动更新LLM-6 GNG，使其能够适应动态通信环境下的未知威胁，从而解决第二个挑战。此外，我们还使用ns-3、OpenAir接口（OAI）和软件定义无线电（SDR）对拟议框架进行了原型化。三个基准测试的实验证明了我们框架的有效性。结果表明，我们的框架可以生成高度准确的安全策略，并且能够抵御各种未知攻击。我们将发布我们的代码为社区做出贡献。



## **49. OBLIVIATE: Robust and Practical Machine Unlearning for Large Language Models**

Obliviate：针对大型语言模型的稳健且实用的机器去学习 cs.CL

18 pages, 2 figures

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04416v1) [paper-pdf](http://arxiv.org/pdf/2505.04416v1)

**Authors**: Xiaoyu Xu, Minxin Du, Qingqing Ye, Haibo Hu

**Abstract**: Large language models (LLMs) trained over extensive corpora risk memorizing sensitive, copyrighted, or toxic content. To address this, we propose OBLIVIATE, a robust unlearning framework that removes targeted data while preserving model utility. The framework follows a structured process: extracting target tokens, building retain sets, and fine-tuning with a tailored loss function comprising three components -- masking, distillation, and world fact. Using low-rank adapters (LoRA), it ensures efficiency without compromising unlearning quality. We conduct experiments on multiple datasets, including the Harry Potter series, WMDP, and TOFU, using a comprehensive suite of metrics: forget quality (new document-level memorization score), model utility, and fluency. Results demonstrate its effectiveness in resisting membership inference attacks, minimizing the impact on retained data, and maintaining robustness across diverse scenarios.

摘要: 在广泛的库中训练的大型语言模型（LLM）存在记忆敏感、受版权保护或有毒内容的风险。为了解决这个问题，我们提出了OBLIATE，这是一个强大的去学习框架，可以删除目标数据，同时保留模型效用。该框架遵循一个结构化过程：提取目标令牌、构建保留集以及使用定制的损失函数进行微调，该函数包括三个部分--掩蔽、蒸馏和世界事实。使用低级适配器（LoRA），它可以在不影响取消学习质量的情况下确保效率。我们使用一套全面的指标对多个数据集进行实验，包括哈利·波特系列、WMDP和TOFU，包括忘记质量（新的文档级记忆分数）、模型效用和流利度。结果证明了它在抵抗隶属度推理攻击、最大限度地减少对保留数据的影响以及在不同场景下保持稳健性方面的有效性。



## **50. The Aloe Family Recipe for Open and Specialized Healthcare LLMs**

面向开放和专业医疗LL的Aloe家族食谱 cs.CL

arXiv admin note: substantial text overlap with arXiv:2405.01886

**SubmitDate**: 2025-05-07    [abs](http://arxiv.org/abs/2505.04388v1) [paper-pdf](http://arxiv.org/pdf/2505.04388v1)

**Authors**: Dario Garcia-Gasulla, Jordi Bayarri-Planas, Ashwin Kumar Gururajan, Enrique Lopez-Cuena, Adrian Tormos, Daniel Hinjos, Pablo Bernabeu-Perez, Anna Arias-Duart, Pablo Agustin Martin-Torres, Marta Gonzalez-Mallo, Sergio Alvarez-Napagao, Eduard Ayguadé-Parra, Ulises Cortés

**Abstract**: Purpose: With advancements in Large Language Models (LLMs) for healthcare, the need arises for competitive open-source models to protect the public interest. This work contributes to the field of open medical LLMs by optimizing key stages of data preprocessing and training, while showing how to improve model safety (through DPO) and efficacy (through RAG). The evaluation methodology used, which includes four different types of tests, defines a new standard for the field. The resultant models, shown to be competitive with the best private alternatives, are released with a permisive license.   Methods: Building on top of strong base models like Llama 3.1 and Qwen 2.5, Aloe Beta uses a custom dataset to enhance public data with synthetic Chain of Thought examples. The models undergo alignment with Direct Preference Optimization, emphasizing ethical and policy-aligned performance in the presence of jailbreaking attacks. Evaluation includes close-ended, open-ended, safety and human assessments, to maximize the reliability of results.   Results: Recommendations are made across the entire pipeline, backed by the solid performance of the Aloe Family. These models deliver competitive performance across healthcare benchmarks and medical fields, and are often preferred by healthcare professionals. On bias and toxicity, the Aloe Beta models significantly improve safety, showing resilience to unseen jailbreaking attacks. For a responsible release, a detailed risk assessment specific to healthcare is attached to the Aloe Family models.   Conclusion: The Aloe Beta models, and the recipe that leads to them, are a significant contribution to the open-source medical LLM field, offering top-of-the-line performance while maintaining high ethical requirements. This work sets a new standard for developing and reporting aligned LLMs in healthcare.

摘要: 目的：随着医疗保健大型语言模型（LLM）的进步，需要有竞争力的开源模型来保护公共利益。这项工作通过优化数据预处理和训练的关键阶段，同时展示如何提高模型安全性（通过DPO）和有效性（通过RAG），为开放医学LLM领域做出了贡献。所使用的评估方法包括四种不同类型的测试，为该领域定义了新的标准。由此产生的模型被证明与最好的私人替代品具有竞争力，并且在许可证下发布。   方法：Aloe Beta建立在Llama 3.1和Qwen 2.5等强大基础模型的基础上，使用自定义数据集通过合成的思想链示例增强公共数据。这些模型与直接偏好优化保持一致，强调在存在越狱攻击时的道德和政策一致的性能。评估包括封闭式、开放式、安全性和人为评估，以最大限度地提高结果的可靠性。   结果：在Aloe系列的稳健表现的支持下，整个管道都提出了建议。这些模型在医疗保健基准和医疗领域提供有竞争力的性能，并且通常受到医疗保健专业人士的青睐。在偏见和毒性方面，Aloe Beta模型显着提高了安全性，表现出对不可见越狱攻击的韧性。为了实现负责任的发布，Aloe Family模型附带了针对医疗保健的详细风险评估。   结论：Aloe Beta模型及其配方是对开源医学LLM领域的重大贡献，在保持高道德要求的同时提供顶级性能。这项工作为开发和报告医疗保健领域一致的LLM设定了新标准。



