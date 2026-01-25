# 泛生成模型 - 红队/越狱
**update at 2026-01-25 10:36:50**

按分类器置信度从高到低排序。

## **1. RunawayEvil: Jailbreaking the Image-to-Video Generative Models**

RunawayEvil：对图像到视频生成模型进行越狱攻击 cs.CV

**SubmitDate**: 2025-12-07    [abs](http://arxiv.org/abs/2512.06674v1) [paper-pdf](https://arxiv.org/pdf/2512.06674v1)

**Confidence**: 0.95

**Authors**: Songping Wang, Rufan Qian, Yueming Lyu, Qinglong Liu, Linzhuang Zou, Jie Qin, Songhua Liu, Caifeng Shan

**Abstract**: Image-to-Video (I2V) generation synthesizes dynamic visual content from image and text inputs, providing significant creative control. However, the security of such multimodal systems, particularly their vulnerability to jailbreak attacks, remains critically underexplored. To bridge this gap, we propose RunawayEvil, the first multimodal jailbreak framework for I2V models with dynamic evolutionary capability. Built on a "Strategy-Tactic-Action" paradigm, our framework exhibits self-amplifying attack through three core components: (1) Strategy-Aware Command Unit that enables the attack to self-evolve its strategies through reinforcement learning-driven strategy customization and LLM-based strategy exploration; (2) Multimodal Tactical Planning Unit that generates coordinated text jailbreak instructions and image tampering guidelines based on the selected strategies; (3) Tactical Action Unit that executes and evaluates the multimodal coordinated attacks. This self-evolving architecture allows the framework to continuously adapt and intensify its attack strategies without human intervention. Extensive experiments demonstrate RunawayEvil achieves state-of-the-art attack success rates on commercial I2V models, such as Open-Sora 2.0 and CogVideoX. Specifically, RunawayEvil outperforms existing methods by 58.5 to 79 percent on COCO2017. This work provides a critical tool for vulnerability analysis of I2V models, thereby laying a foundation for more robust video generation systems.

摘要: 图像到视频（I2V）生成技术通过图像和文本输入合成动态视觉内容，提供了重要的创作控制能力。然而，此类多模态系统的安全性，特别是其面对越狱攻击的脆弱性，仍严重缺乏研究。为填补这一空白，我们提出了RunawayEvil——首个具备动态演化能力的I2V模型多模态越狱框架。基于“策略-战术-行动”范式，该框架通过三个核心组件实现自我强化的攻击：（1）策略感知指令单元：通过强化学习驱动的策略定制和基于LLM的策略探索，使攻击能够自我演化策略；（2）多模态战术规划单元：根据选定策略生成协调的文本越狱指令和图像篡改指南；（3）战术执行单元：执行并评估多模态协同攻击。这种自我演化架构使框架能够在无需人工干预的情况下持续适应并强化攻击策略。大量实验表明，RunawayEvil在Open-Sora 2.0和CogVideoX等商业I2V模型上实现了最先进的攻击成功率。具体而言，在COCO2017数据集上，RunawayEvil比现有方法高出58.5%至79%。这项工作为I2V模型的漏洞分析提供了关键工具，从而为构建更鲁棒的视频生成系统奠定了基础。



## **2. VEIL: Jailbreaking Text-to-Video Models via Visual Exploitation from Implicit Language**

VEIL：通过隐式语言视觉利用对文本到视频模型进行越狱攻击 cs.CV

**SubmitDate**: 2025-12-08    [abs](http://arxiv.org/abs/2511.13127v2) [paper-pdf](https://arxiv.org/pdf/2511.13127v2)

**Confidence**: 0.95

**Authors**: Zonghao Ying, Moyang Chen, Nizhang Li, Zhiqiang Wang, Wenxin Zhang, Quanchen Zou, Zonglei Jing, Aishan Liu, Xianglong Liu

**Abstract**: Jailbreak attacks can circumvent model safety guardrails and reveal critical blind spots. Prior attacks on text-to-video (T2V) models typically add adversarial perturbations to obviously unsafe prompts, which are often easy to detect and defend. In contrast, we show that benign-looking prompts containing rich, implicit cues can induce T2V models to generate semantically unsafe videos that both violate policy and preserve the original (blocked) intent. To realize this, we propose VEIL, a jailbreak framework that leverages T2V models' cross-modal associative patterns via a modular prompt design. Specifically, our prompts combine three components: neutral scene anchors, which provide the surface-level scene description extracted from the blocked intent to maintain plausibility; latent auditory triggers, textual descriptions of innocuous-sounding audio events (e.g., creaking, muffled noises) that exploit learned audio-visual co-occurrence priors to bias the model toward particular unsafe visual concepts; and stylistic modulators, cinematic directives (e.g., camera framing, atmosphere) that amplify and stabilize the latent trigger's effect. We formalize attack generation as a constrained optimization over the above modular prompt space and solve it with a guided search procedure that balances stealth and effectiveness. Extensive experiments over 7 T2V models demonstrate the efficacy of our attack, achieving a 23 percent improvement in average attack success rate in commercial models. Our demos and codes can be found at https://github.com/NY1024/VEIL.

摘要: 越狱攻击能够绕过模型的安全防护机制，揭示关键盲点。先前针对文本到视频（T2V）模型的攻击通常通过向明显不安全提示添加对抗性扰动实现，这类方法易于检测和防御。与此相反，我们发现包含丰富隐式线索、看似良性的提示能够诱导T2V模型生成既违反安全策略又保留原始（被屏蔽）意图的语义不安全视频。为实现这一目标，我们提出VEIL越狱框架，通过模块化提示设计利用T2V模型的跨模态关联模式。具体而言，我们的提示包含三个组件：中性场景锚点——从被屏蔽意图中提取表层场景描述以保持合理性；潜在听觉触发器——描述无害音频事件（如吱呀声、闷响）的文本，利用学习到的视听共现先验使模型偏向特定不安全视觉概念；风格调制器——通过电影化指令（如镜头构图、氛围营造）增强并稳定潜在触发器的效果。我们将攻击生成形式化为模块化提示空间的约束优化问题，并通过平衡隐蔽性与有效性的引导搜索算法求解。在7个T2V模型上的大量实验证明了攻击的有效性，在商业模型中平均攻击成功率提升23%。演示与代码详见：https://github.com/NY1024/VEIL。



## **3. Jailbreaking Safeguarded Text-to-Image Models via Large Language Models**

通过大型语言模型破解受保护文本到图像模型的安全防护 cs.CR

Accepted by EACL 2026 Findings

**SubmitDate**: 2026-01-07    [abs](http://arxiv.org/abs/2503.01839v2) [paper-pdf](https://arxiv.org/pdf/2503.01839v2)

**Confidence**: 0.95

**Authors**: Zhengyuan Jiang, Yuepeng Hu, Yuchen Yang, Yinzhi Cao, Neil Zhenqiang Gong

**Abstract**: Text-to-Image models may generate harmful content, such as pornographic images, particularly when unsafe prompts are submitted. To address this issue, safety filters are often added on top of text-to-image models, or the models themselves are aligned to reduce harmful outputs. However, these defenses remain vulnerable when an attacker strategically designs adversarial prompts to bypass these safety guardrails. In this work, we propose \alg, a method to jailbreak text-to-image models with safety guardrails using a fine-tuned large language model. Unlike other query-based jailbreak attacks that require repeated queries to the target model, our attack generates adversarial prompts efficiently after fine-tuning our AttackLLM. We evaluate our method on three datasets of unsafe prompts and against five safety guardrails. Our results demonstrate that our approach effectively bypasses safety guardrails, outperforms existing no-box attacks, and also facilitates other query-based attacks.

摘要: 文本到图像模型可能生成有害内容，例如色情图像，尤其是在提交不安全提示时。为解决此问题，通常在文本到图像模型之上添加安全过滤器，或对模型本身进行对齐以减少有害输出。然而，当攻击者策略性地设计对抗性提示以绕过这些安全防护时，这些防御措施仍然存在漏洞。在本研究中，我们提出\alg方法，利用微调的大型语言模型来破解带有安全防护的文本到图像模型。与其他需要重复查询目标模型的基于查询的破解攻击不同，我们的攻击在微调AttackLLM后能高效生成对抗性提示。我们在三个不安全提示数据集上评估了我们的方法，并针对五种安全防护进行了测试。结果表明，我们的方法能有效绕过安全防护，优于现有的无盒攻击，并能促进其他基于查询的攻击。



## **4. MacPrompt: Maraconic-guided Jailbreak against Text-to-Image Models**

MacPrompt：基于混合语言引导的文本到图像模型越狱攻击 cs.CR

Accepted by AAAI 2026

**SubmitDate**: 2026-01-12    [abs](http://arxiv.org/abs/2601.07141v1) [paper-pdf](https://arxiv.org/pdf/2601.07141v1)

**Confidence**: 0.95

**Authors**: Xi Ye, Yiwen Liu, Lina Wang, Run Wang, Geying Yang, Yufei Hou, Jiayi Yu

**Abstract**: Text-to-image (T2I) models have raised increasing safety concerns due to their capacity to generate NSFW and other banned objects. To mitigate these risks, safety filters and concept removal techniques have been introduced to block inappropriate prompts or erase sensitive concepts from the models. However, all the existing defense methods are not well prepared to handle diverse adversarial prompts. In this work, we introduce MacPrompt, a novel black-box and cross-lingual attack that reveals previously overlooked vulnerabilities in T2I safety mechanisms. Unlike existing attacks that rely on synonym substitution or prompt obfuscation, MacPrompt constructs macaronic adversarial prompts by performing cross-lingual character-level recombination of harmful terms, enabling fine-grained control over both semantics and appearance. By leveraging this design, MacPrompt crafts prompts with high semantic similarity to the original harmful inputs (up to 0.96) while bypassing major safety filters (up to 100%). More critically, it achieves attack success rates as high as 92% for sex-related content and 90% for violence, effectively breaking even state-of-the-art concept removal defenses. These results underscore the pressing need to reassess the robustness of existing T2I safety mechanisms against linguistically diverse and fine-grained adversarial strategies.

摘要: 文本到图像（T2I）模型因其生成NSFW及其他违禁内容的能力而引发日益增长的安全担忧。为缓解这些风险，安全过滤器和概念移除技术被引入以拦截不当提示或从模型中删除敏感概念。然而，现有防御方法均未能充分应对多样化的对抗性提示。本研究提出MacPrompt，一种新颖的黑盒跨语言攻击方法，揭示了T2I安全机制中先前被忽视的漏洞。与依赖同义词替换或提示混淆的现有攻击不同，MacPrompt通过对有害术语进行跨语言字符级重组来构建混合语言对抗性提示，实现了对语义和外观的细粒度控制。基于此设计，MacPrompt构建的提示与原始有害输入保持高语义相似度（最高达0.96），同时能绕过主流安全过滤器（成功率最高达100%）。更重要的是，其在色情相关内容上的攻击成功率高达92%，暴力内容达90%，甚至能有效突破最先进的概念移除防御。这些结果凸显了重新评估现有T2I安全机制对抗语言多样性及细粒度对抗策略的鲁棒性的迫切需求。



## **5. $PC^2$: Politically Controversial Content Generation via Jailbreaking Attacks on GPT-based Text-to-Image Models**

$PC^2$：基于GPT文本到图像模型的越狱攻击生成政治争议内容 cs.CR

19 pages, 15 figures, 9 tables

**SubmitDate**: 2026-01-15    [abs](http://arxiv.org/abs/2601.05150v2) [paper-pdf](https://arxiv.org/pdf/2601.05150v2)

**Confidence**: 0.95

**Authors**: Wonwoo Choi, Minjae Seo, Minkyoo Song, Hwanjo Heo, Seungwon Shin, Myoungsung You

**Abstract**: The rapid evolution of text-to-image (T2I) models has enabled high-fidelity visual synthesis on a global scale. However, these advancements have introduced significant security risks, particularly regarding the generation of harmful content. Politically harmful content, such as fabricated depictions of public figures, poses severe threats when weaponized for fake news or propaganda. Despite its criticality, the robustness of current T2I safety filters against such politically motivated adversarial prompting remains underexplored. In response, we propose $PC^2$, the first black-box political jailbreaking framework for T2I models. It exploits a novel vulnerability where safety filters evaluate political sensitivity based on linguistic context. $PC^2$ operates through: (1) Identity-Preserving Descriptive Mapping to obfuscate sensitive keywords into neutral descriptions, and (2) Geopolitically Distal Translation to map these descriptions into fragmented, low-sensitivity languages. This strategy prevents filters from constructing toxic relationships between political entities within prompts, effectively bypassing detection. We construct a benchmark of 240 politically sensitive prompts involving 36 public figures. Evaluation on commercial T2I models, specifically GPT-series, shows that while all original prompts are blocked, $PC^2$ achieves attack success rates of up to 86%.

摘要: 文本到图像（T2I）模型的快速发展实现了全球范围内的高保真视觉合成。然而，这些进步也带来了重大的安全风险，特别是在有害内容生成方面。政治有害内容，如对公众人物的虚构描绘，在被武器化用于假新闻或宣传时构成严重威胁。尽管其重要性，当前T2I安全过滤器针对此类政治动机对抗性提示的鲁棒性仍未得到充分探索。为此，我们提出了$PC^2$，首个针对T2I模型的黑盒政治越狱框架。它利用了一种新颖漏洞：安全过滤器基于语言上下文评估政治敏感性。$PC^2$通过以下方式运作：（1）身份保持描述映射，将敏感关键词混淆为中性描述；（2）地理政治远端翻译，将这些描述映射为碎片化的低敏感性语言。该策略防止过滤器在提示中构建政治实体间的毒性关联，从而有效绕过检测。我们构建了一个包含36位公众人物的240个政治敏感提示基准。在商用T2I模型（特别是GPT系列）上的评估显示，虽然所有原始提示均被拦截，但$PC^2$的攻击成功率高达86%。



## **6. Rethinking and Red-Teaming Protective Perturbation in Personalized Diffusion Models**

重新审视与红队测试个性化扩散模型中的保护性扰动 cs.CV

Our code is available at https://github.com/liuyixin-louis/DiffShortcut

**SubmitDate**: 2026-01-17    [abs](http://arxiv.org/abs/2406.18944v5) [paper-pdf](https://arxiv.org/pdf/2406.18944v5)

**Confidence**: 0.95

**Authors**: Yixin Liu, Ruoxi Chen, Xun Chen, Lichao Sun

**Abstract**: Personalized diffusion models (PDMs) have become prominent for adapting pre-trained text-to-image models to generate images of specific subjects using minimal training data. However, PDMs are susceptible to minor adversarial perturbations, leading to significant degradation when fine-tuned on corrupted datasets. These vulnerabilities are exploited to create protective perturbations that prevent unauthorized image generation. Existing purification methods attempt to red-team the protective perturbation to break the protection but often over-purify images, resulting in information loss. In this work, we conduct an in-depth analysis of the fine-tuning process of PDMs through the lens of shortcut learning. We hypothesize and empirically demonstrate that adversarial perturbations induce a latent-space misalignment between images and their text prompts in the CLIP embedding space. This misalignment causes the model to erroneously associate noisy patterns with unique identifiers during fine-tuning, resulting in poor generalization. Based on these insights, we propose a systematic red-teaming framework that includes data purification and contrastive decoupling learning. We first employ off-the-shelf image restoration techniques to realign images with their original semantic content in latent space. Then, we introduce contrastive decoupling learning with noise tokens to decouple the learning of personalized concepts from spurious noise patterns. Our study not only uncovers shortcut learning vulnerabilities in PDMs but also provides a thorough evaluation framework for developing stronger protection. Our extensive evaluation demonstrates its advantages over existing purification methods and its robustness against adaptive perturbations.

摘要: 个性化扩散模型（PDMs）通过少量训练数据使预训练的文本到图像模型能够生成特定主体的图像，已成为重要技术。然而，PDMs易受微小对抗性扰动的影响，在受损数据集上微调时会导致性能显著下降。攻击者利用这一漏洞创建保护性扰动以防止未经授权的图像生成。现有净化方法试图通过红队测试破解保护性扰动，但常因过度净化导致信息丢失。本研究通过捷径学习的视角深入分析PDMs的微调过程，提出并实证验证了对抗性扰动会在CLIP嵌入空间中引发图像与其文本提示之间的潜在空间错位。这种错位导致模型在微调过程中错误地将噪声模式与唯一标识符关联，造成泛化能力下降。基于这些发现，我们提出了一个系统的红队测试框架，包含数据净化和对比解耦学习。我们首先采用现成的图像恢复技术在潜在空间中将图像与其原始语义内容重新对齐，然后引入带噪声标记的对比解耦学习，将个性化概念的学习与虚假噪声模式解耦。本研究不仅揭示了PDMs中的捷径学习漏洞，还为开发更强保护提供了全面评估框架。大量实验证明，该方法优于现有净化技术，并对自适应扰动具有鲁棒性。



## **7. Metaphor-based Jailbreaking Attacks on Text-to-Image Models**

基于隐喻的文本到图像模型越狱攻击 cs.CR

This paper includes model-generated content that may contain offensive or distressing material

**SubmitDate**: 2025-12-06    [abs](http://arxiv.org/abs/2512.10766v1) [paper-pdf](https://arxiv.org/pdf/2512.10766v1)

**Confidence**: 0.95

**Authors**: Chenyu Zhang, Yiwen Ma, Lanjun Wang, Wenhui Li, Yi Tu, An-An Liu

**Abstract**: Text-to-image~(T2I) models commonly incorporate defense mechanisms to prevent the generation of sensitive images. Unfortunately, recent jailbreaking attacks have shown that adversarial prompts can effectively bypass these mechanisms and induce T2I models to produce sensitive content, revealing critical safety vulnerabilities. However, existing attack methods implicitly assume that the attacker knows the type of deployed defenses, which limits their effectiveness against unknown or diverse defense mechanisms. In this work, we introduce \textbf{MJA}, a \textbf{m}etaphor-based \textbf{j}ailbreaking \textbf{a}ttack method inspired by the Taboo game, aiming to effectively and efficiently attack diverse defense mechanisms without prior knowledge of their type by generating metaphor-based adversarial prompts. Specifically, MJA consists of two modules: an LLM-based multi-agent generation module~(MLAG) and an adversarial prompt optimization module~(APO). MLAG decomposes the generation of metaphor-based adversarial prompts into three subtasks: metaphor retrieval, context matching, and adversarial prompt generation. Subsequently, MLAG coordinates three LLM-based agents to generate diverse adversarial prompts by exploring various metaphors and contexts. To enhance attack efficiency, APO first trains a surrogate model to predict the attack results of adversarial prompts and then designs an acquisition strategy to adaptively identify optimal adversarial prompts. Extensive experiments on T2I models with various external and internal defense mechanisms demonstrate that MJA outperforms six baseline methods, achieving stronger attack performance while using fewer queries. Code is available in https://github.com/datar001/metaphor-based-jailbreaking-attack.

摘要: 文本到图像（T2I）模型通常采用防御机制以防止生成敏感图像。然而，最近的越狱攻击表明，对抗性提示能有效绕过这些机制，诱导T2I模型生成敏感内容，暴露出严重的安全漏洞。现有攻击方法隐含假设攻击者知晓部署的防御类型，这限制了其对未知或多样化防御机制的有效性。本文提出**MJA**——一种受“禁忌游戏”启发的**基于隐喻的越狱攻击**方法，旨在通过生成隐喻式对抗性提示，在无需预知防御类型的情况下，高效攻击多样化防御机制。具体而言，MJA包含两个模块：基于大语言模型的多智能体生成模块（MLAG）和对抗性提示优化模块（APO）。MLAG将隐喻式对抗性提示生成分解为三个子任务：隐喻检索、上下文匹配和对抗性提示生成，并通过协调三个基于大语言模型的智能体，探索不同隐喻与上下文以生成多样化对抗性提示。为提升攻击效率，APO首先训练代理模型预测对抗性提示的攻击效果，进而设计采集策略自适应识别最优对抗性提示。在配备多种外部与内部防御机制的T2I模型上的大量实验表明，MJA在减少查询次数的同时，攻击性能显著优于六种基线方法。代码发布于：https://github.com/datar001/metaphor-based-jailbreaking-attack。



## **8. T2V-OptJail: Discrete Prompt Optimization for Text-to-Video Jailbreak Attacks**

T2V-OptJail：面向文生视频越狱攻击的离散提示优化方法 cs.CV

**SubmitDate**: 2025-06-17    [abs](http://arxiv.org/abs/2505.06679v2) [paper-pdf](https://arxiv.org/pdf/2505.06679v2)

**Confidence**: 0.95

**Authors**: Jiayang Liu, Siyuan Liang, Shiqian Zhao, Rongcheng Tu, Wenbo Zhou, Aishan Liu, Dacheng Tao, Siew Kei Lam

**Abstract**: In recent years, fueled by the rapid advancement of diffusion models, text-to-video (T2V) generation models have achieved remarkable progress, with notable examples including Pika, Luma, Kling, and Open-Sora. Although these models exhibit impressive generative capabilities, they also expose significant security risks due to their vulnerability to jailbreak attacks, where the models are manipulated to produce unsafe content such as pornography, violence, or discrimination. Existing works such as T2VSafetyBench provide preliminary benchmarks for safety evaluation, but lack systematic methods for thoroughly exploring model vulnerabilities. To address this gap, we are the first to formalize the T2V jailbreak attack as a discrete optimization problem and propose a joint objective-based optimization framework, called T2V-OptJail. This framework consists of two key optimization goals: bypassing the built-in safety filtering mechanisms to increase the attack success rate, preserving semantic consistency between the adversarial prompt and the unsafe input prompt, as well as between the generated video and the unsafe input prompt, to enhance content controllability. In addition, we introduce an iterative optimization strategy guided by prompt variants, where multiple semantically equivalent candidates are generated in each round, and their scores are aggregated to robustly guide the search toward optimal adversarial prompts. We conduct large-scale experiments on several T2V models, covering both open-source models and real commercial closed-source models. The experimental results show that the proposed method improves 11.4% and 10.0% over the existing state-of-the-art method in terms of attack success rate assessed by GPT-4, attack success rate assessed by human accessors, respectively, verifying the significant advantages of the method in terms of attack effectiveness and content control.

摘要: 近年来，随着扩散模型的快速发展，文生视频（T2V）生成模型取得了显著进展，代表性成果包括Pika、Luma、Kling和Open-Sora等。尽管这些模型展现出强大的生成能力，但其对越狱攻击的脆弱性也带来了严重的安全风险——攻击者可操纵模型生成色情、暴力或歧视性等不安全内容。现有工作如T2VSafetyBench虽提供了初步的安全评估基准，但缺乏系统化深入探索模型漏洞的方法。为填补这一空白，我们首次将T2V越狱攻击形式化为离散优化问题，并提出基于联合目标的优化框架T2V-OptJail。该框架包含两大优化目标：一是规避内置安全过滤机制以提升攻击成功率；二是保持对抗提示与原始不安全提示之间、生成视频与原始不安全提示之间的语义一致性，以增强内容可控性。此外，我们提出了基于提示变体的迭代优化策略：每轮生成多个语义等效候选提示，通过聚合评分鲁棒地引导搜索方向，最终获得最优对抗提示。我们在多个开源及商业闭源T2V模型上进行了大规模实验。结果表明，本方法在GPT-4评估的攻击成功率、人工评估的攻击成功率上分别较现有最优方法提升11.4%和10.0%，验证了其在攻击效能与内容控制方面的显著优势。



