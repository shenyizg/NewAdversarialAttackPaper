# Latest Large Language Model Attack Papers
**update at 2025-03-27 09:35:25**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. Iterative Prompting with Persuasion Skills in Jailbreaking Large Language Models**

越狱大型语言模型中具有说服技巧的迭代绘图 cs.CL

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20320v1) [paper-pdf](http://arxiv.org/pdf/2503.20320v1)

**Authors**: Shih-Wen Ke, Guan-Yu Lai, Guo-Lin Fang, Hsi-Yuan Kao

**Abstract**: Large language models (LLMs) are designed to align with human values in their responses. This study exploits LLMs with an iterative prompting technique where each prompt is systematically modified and refined across multiple iterations to enhance its effectiveness in jailbreaking attacks progressively. This technique involves analyzing the response patterns of LLMs, including GPT-3.5, GPT-4, LLaMa2, Vicuna, and ChatGLM, allowing us to adjust and optimize prompts to evade the LLMs' ethical and security constraints. Persuasion strategies enhance prompt effectiveness while maintaining consistency with malicious intent. Our results show that the attack success rates (ASR) increase as the attacking prompts become more refined with the highest ASR of 90% for GPT4 and ChatGLM and the lowest ASR of 68% for LLaMa2. Our technique outperforms baseline techniques (PAIR and PAP) in ASR and shows comparable performance with GCG and ArtPrompt.

摘要: 大型语言模型（LLM）旨在在其响应中与人类价值观保持一致。这项研究通过迭代提示技术来利用LLM，其中每个提示都经过多次迭代系统地修改和细化，以逐步增强其越狱攻击的有效性。该技术涉及分析LLM的响应模式，包括GPT-3.5、GPT-4、LLaMa 2、Vicuna和ChatGLM，使我们能够调整和优化提示以规避LLM的道德和安全限制。说服策略增强了及时的有效性，同时与恶意意图保持一致。我们的结果表明，随着攻击提示变得更加精确，攻击成功率（ASB）也会增加，GPT 4和ChatGLM的最高ASB为90%，LLaMa 2的最低ASB为68%。我们的技术在ASB中优于基线技术（PAIR和PAP），并表现出与GCG和ArtPrompt相当的性能。



## **2. sudo rm -rf agentic_security**

sudo rm -ref agentic_secure cs.CL

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20279v1) [paper-pdf](http://arxiv.org/pdf/2503.20279v1)

**Authors**: Sejin Lee, Jian Kim, Haon Park, Ashkan Yousefpour, Sangyoon Yu, Min Song

**Abstract**: Large Language Models (LLMs) are increasingly deployed as computer-use agents, autonomously performing tasks within real desktop or web environments. While this evolution greatly expands practical use cases for humans, it also creates serious security exposures. We present SUDO (Screen-based Universal Detox2Tox Offense), a novel attack framework that systematically bypasses refusal trained safeguards in commercial computer-use agents, such as Claude Computer Use. The core mechanism, Detox2Tox, transforms harmful requests (that agents initially reject) into seemingly benign requests via detoxification, secures detailed instructions from advanced vision language models (VLMs), and then reintroduces malicious content via toxification just before execution. Unlike conventional jailbreaks, SUDO iteratively refines its attacks based on a built-in refusal feedback, making it increasingly effective against robust policy filters. In extensive tests spanning 50 real-world tasks and multiple state-of-the-art VLMs, SUDO achieves a stark attack success rate of 24% (with no refinement), and up to 41% (by its iterative refinement) in Claude Computer Use. By revealing these vulnerabilities and demonstrating the ease with which they can be exploited in real-world computing environments, this paper highlights an immediate need for robust, context-aware safeguards. WARNING: This paper includes harmful or offensive model outputs.

摘要: 大型语言模型(LLM)越来越多地被部署为计算机使用代理，在真实的桌面或Web环境中自主执行任务。虽然这一演变极大地扩展了人类的实际用例，但也造成了严重的安全风险。我们提出了SUDO(基于屏幕的通用脱毒2Tox攻击)，这是一种新的攻击框架，它系统地绕过了商业计算机使用代理(如Claude计算机使用)中拒绝训练的保护措施。核心机制Detox2Tox通过解毒将有害的请求(代理最初拒绝的)转换为看似良性的请求，从高级视觉语言模型(VLM)获得详细说明，然后在执行前通过中毒重新引入恶意内容。与传统的越狱不同，数独基于内置的拒绝反馈反复改进其攻击，使其对强大的策略过滤器越来越有效。在跨越50个真实世界任务和多个最先进的VLM的广泛测试中，SUDO实现了24%(未经改进)的赤裸裸攻击成功率，以及克劳德计算机使用的高达41%(通过迭代优化)。通过揭示这些漏洞并展示在真实计算环境中利用它们是多么容易，本白皮书强调了对强大的、上下文感知的安全保护的迫切需求。警告：本文包括有害或攻击性模型输出。



## **3. Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models**

大视觉语言模型的自我监督学习视觉编码器中的秘密后门攻击 cs.CV

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2502.18290v3) [paper-pdf](http://arxiv.org/pdf/2502.18290v3)

**Authors**: Zhaoyi Liu, Huan Zhang

**Abstract**: Self-supervised learning (SSL) vision encoders learn high-quality image representations and thus have become a vital part of developing vision modality of large vision language models (LVLMs). Due to the high cost of training such encoders, pre-trained encoders are widely shared and deployed into many LVLMs, which are security-critical or bear societal significance. Under this practical scenario, we reveal a new backdoor threat that significant visual hallucinations can be induced into these LVLMs by merely compromising vision encoders. Because of the sharing and reuse of these encoders, many downstream LVLMs may inherit backdoor behaviors from encoders, leading to widespread backdoors. In this work, we propose BadVision, the first method to exploit this vulnerability in SSL vision encoders for LVLMs with novel trigger optimization and backdoor learning techniques. We evaluate BadVision on two types of SSL encoders and LVLMs across eight benchmarks. We show that BadVision effectively drives the LVLMs to attacker-chosen hallucination with over 99% attack success rate, causing a 77.6% relative visual understanding error while maintaining the stealthiness. SoTA backdoor detection methods cannot detect our attack effectively.

摘要: 自监督学习(SSL)视觉编码者学习高质量的图像表征，因此成为开发大型视觉语言模型(LVLMS)视觉通道的重要组成部分。由于培训这类编码器的成本很高，预先训练的编码器被广泛共享并部署到许多安全关键或具有社会意义的LVLM中。在这种实际情况下，我们揭示了一种新的后门威胁，即仅仅通过损害视觉编码器就可以在这些LVLM中诱导出显著的视觉幻觉。由于这些编码器的共享和重用，许多下游的LVLM可能会继承编码器的后门行为，导致广泛的后门。在这项工作中，我们提出了BadVision，这是第一个通过新颖的触发优化和后门学习技术来利用LVLM的SSL视觉编码器中的漏洞的方法。我们在八个基准测试中评估了BadVision在两种类型的SSL编码器和LVLM上的性能。结果表明，BadVision在保持隐蔽性的同时，以99%以上的攻击成功率有效地驱动了LVLM进入攻击者选择的幻觉，导致了77.6%的相对视觉理解错误。SOTA后门检测方法不能有效检测到我们的攻击。



## **4. TeleLoRA: Teleporting Model-Specific Alignment Across LLMs**

TeleLoRA：LLM之间远程传输模型特定的一致 cs.LG

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20228v1) [paper-pdf](http://arxiv.org/pdf/2503.20228v1)

**Authors**: Xiao Lin, Manoj Acharya, Anirban Roy, Susmit Jha

**Abstract**: Mitigating Trojans in Large Language Models (LLMs) is one of many tasks where alignment data is LLM specific, as different LLMs have different Trojan triggers and trigger behaviors to be removed. In this paper, we introduce TeleLoRA (Teleporting Low-Rank Adaptation), a novel framework that synergizes model-specific alignment data across multiple LLMs to enable zero-shot Trojan mitigation on unseen LLMs without alignment data. TeleLoRA learns a unified generator of LoRA adapter weights by leveraging local activation information across multiple LLMs. This generator is designed to be permutation symmetric to generalize across models with different architectures and sizes. We optimize the model design for memory efficiency, making it feasible to learn with large-scale LLMs with minimal computational resources. Experiments on LLM Trojan mitigation benchmarks demonstrate that TeleLoRA effectively reduces attack success rates while preserving the benign performance of the models.

摘要: 缓解大型语言模型（LLM）中的特洛伊木马是对齐数据特定于LLM的众多任务之一，因为不同的LLM具有不同的特洛伊木马触发器和要删除的触发行为。在本文中，我们介绍了TeleLoRA（远程传输低等级自适应），这是一种新颖的框架，可以在多个LLM之间协同特定于模型的对齐数据，以便在没有对齐数据的情况下对不可见的LLM实现零触发特洛伊木马缓解。TeleLoRA通过利用多个LLM之间的本地激活信息来学习LoRA适配器权重的统一生成器。该生成器被设计为排列对称，以便在具有不同架构和大小的模型之间进行概括。我们优化了模型设计以提高内存效率，使以最少的计算资源使用大规模LLM进行学习成为可能。LLM特洛伊木马缓解基准测试的实验表明，TeleLoRA有效降低了攻击成功率，同时保持了模型的良性性能。



## **5. Knowledge Transfer from LLMs to Provenance Analysis: A Semantic-Augmented Method for APT Detection**

从LLM到出处分析的知识转移：APT检测的语义增强方法 cs.CR

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.18316v2) [paper-pdf](http://arxiv.org/pdf/2503.18316v2)

**Authors**: Fei Zuo, Junghwan Rhee, Yung Ryn Choe

**Abstract**: Advanced Persistent Threats (APTs) have caused significant losses across a wide range of sectors, including the theft of sensitive data and harm to system integrity. As attack techniques grow increasingly sophisticated and stealthy, the arms race between cyber defenders and attackers continues to intensify. The revolutionary impact of Large Language Models (LLMs) has opened up numerous opportunities in various fields, including cybersecurity. An intriguing question arises: can the extensive knowledge embedded in LLMs be harnessed for provenance analysis and play a positive role in identifying previously unknown malicious events? To seek a deeper understanding of this issue, we propose a new strategy for taking advantage of LLMs in provenance-based threat detection. In our design, the state-of-the-art LLM offers additional details in provenance data interpretation, leveraging their knowledge of system calls, software identity, and high-level understanding of application execution context. The advanced contextualized embedding capability is further utilized to capture the rich semantics of event descriptions. We comprehensively examine the quality of the resulting embeddings, and it turns out that they offer promising avenues. Subsequently, machine learning models built upon these embeddings demonstrated outstanding performance on real-world data. In our evaluation, supervised threat detection achieves a precision of 99.0%, and semi-supervised anomaly detection attains a precision of 96.9%.

摘要: 高级持续性威胁在许多部门造成了重大损失，包括敏感数据被盗和对系统完整性的损害。随着攻击技术变得越来越复杂和隐蔽，网络防御者和攻击者之间的军备竞赛继续加剧。大型语言模型(LLM)的革命性影响在包括网络安全在内的各个领域开辟了无数机会。一个耐人寻味的问题出现了：LLMS中嵌入的广泛知识能否被用于来源分析，并在识别以前未知的恶意事件方面发挥积极作用？为了寻求对这一问题的更深入的理解，我们提出了一种新的策略来利用LLMS进行基于起源的威胁检测。在我们的设计中，最先进的LLM在来源数据解释方面提供了更多细节，利用他们对系统调用、软件身份的知识和对应用程序执行上下文的高级理解。进一步利用先进的上下文嵌入能力来捕获事件描述的丰富语义。我们全面检查了由此产生的嵌入的质量，结果证明它们提供了有希望的途径。随后，建立在这些嵌入基础上的机器学习模型在真实数据上表现出了出色的性能。在我们的评估中，监督威胁检测的准确率为99.0%，半监督异常检测的准确率为96.9%。



## **6. Inducing Personality in LLM-Based Honeypot Agents: Measuring the Effect on Human-Like Agenda Generation**

基于LLM的蜜罐代理中的人格诱导：衡量对类人议程生成的影响 cs.AI

11 pages, 1 figure, 6 tables. Accepted to NLPAICS 2024

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.19752v1) [paper-pdf](http://arxiv.org/pdf/2503.19752v1)

**Authors**: Lewis Newsham, Ryan Hyland, Daniel Prince

**Abstract**: This paper presents SANDMAN, an architecture for cyber deception that leverages Language Agents to emulate convincing human simulacra. Our 'Deceptive Agents' serve as advanced cyber decoys, designed for high-fidelity engagement with attackers by extending the observation period of attack behaviours. Through experimentation, measurement, and analysis, we demonstrate how a prompt schema based on the five-factor model of personality systematically induces distinct 'personalities' in Large Language Models. Our results highlight the feasibility of persona-driven Language Agents for generating diverse, realistic behaviours, ultimately improving cyber deception strategies.

摘要: 本文介绍了SAANDMAN，这是一种网络欺骗架构，它利用语言代理来模拟令人信服的人类拟像。我们的“欺骗性特工”充当高级网络诱饵，旨在通过延长攻击行为的观察期来与攻击者进行高保真接触。通过实验、测量和分析，我们展示了基于人格五因素模型的提示模式如何系统地在大型语言模型中诱导不同的“人格”。我们的结果强调了人物驱动的语言代理生成多样化、现实的行为、最终改进网络欺骗策略的可行性。



## **7. Does Safety Training of LLMs Generalize to Semantically Related Natural Prompts?**

LLM的安全培训是否适用于语义相关的自然知识？ cs.CL

Accepted in ICLR 2025

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2412.03235v2) [paper-pdf](http://arxiv.org/pdf/2412.03235v2)

**Authors**: Sravanti Addepalli, Yerram Varun, Arun Suggala, Karthikeyan Shanmugam, Prateek Jain

**Abstract**: Large Language Models (LLMs) are known to be susceptible to crafted adversarial attacks or jailbreaks that lead to the generation of objectionable content despite being aligned to human preferences using safety fine-tuning methods. While the large dimensionality of input token space makes it inevitable to find adversarial prompts that can jailbreak these models, we aim to evaluate whether safety fine-tuned LLMs are safe against natural prompts which are semantically related to toxic seed prompts that elicit safe responses after alignment. We surprisingly find that popular aligned LLMs such as GPT-4 can be compromised using naive prompts that are NOT even crafted with an objective of jailbreaking the model. Furthermore, we empirically show that given a seed prompt that elicits a toxic response from an unaligned model, one can systematically generate several semantically related natural prompts that can jailbreak aligned LLMs. Towards this, we propose a method of Response Guided Question Augmentation (ReG-QA) to evaluate the generalization of safety aligned LLMs to natural prompts, that first generates several toxic answers given a seed question using an unaligned LLM (Q to A), and further leverages an LLM to generate questions that are likely to produce these answers (A to Q). We interestingly find that safety fine-tuned LLMs such as GPT-4o are vulnerable to producing natural jailbreak questions from unsafe content (without denial) and can thus be used for the latter (A to Q) step. We obtain attack success rates that are comparable to/ better than leading adversarial attack methods on the JailbreakBench leaderboard, while being significantly more stable against defenses such as Smooth-LLM and Synonym Substitution, which are effective against existing all attacks on the leaderboard.

摘要: 众所周知，大型语言模型(LLM)容易受到精心设计的对抗性攻击或越狱，尽管使用安全微调方法与人类的偏好保持一致，但这些攻击或越狱会导致生成令人反感的内容。虽然输入令牌空间的大维度使得找到能够越狱这些模型的敌意提示是不可避免的，但我们的目标是评估安全的微调LLM对于自然提示是否安全，这些自然提示在语义上与有毒种子提示相关，在对齐后引起安全响应。我们惊讶地发现，GPT-4等流行的对齐LLM可以使用甚至不是以越狱为目标而精心设计的幼稚提示来进行攻击。此外，我们的经验表明，给定一个种子提示引起来自未对齐模型的有毒反应，一个人可以系统地生成几个语义相关的自然提示，从而可以越狱对齐的LLM。为此，我们提出了一种反应引导问题增强方法(REG-QA)来评估安全对齐LLM对自然提示的泛化，该方法首先使用未对齐LLM(Q到A)来生成给定种子问题的几个有毒答案，然后利用LLM来生成可能产生这些答案(A到Q)的问题。有趣的是，我们发现安全微调的LLM，如GPT-40，容易从不安全的内容产生自然的越狱问题(不否认)，因此可以用于后一步(A到Q)。我们获得了相当于/好于JailBreak排行榜上领先的对抗性攻击方法的攻击成功率，同时对Smooth-LLM和同义词替换等防御措施明显更加稳定，这些防御措施对排行榜上现有的所有攻击都有效。



## **8. Membership Inference Attacks on Large-Scale Models: A Survey**

对大规模模型的成员推断攻击：一项调查 cs.LG

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.19338v1) [paper-pdf](http://arxiv.org/pdf/2503.19338v1)

**Authors**: Hengyu Wu, Yang Cao

**Abstract**: The adoption of the Large Language Model (LLM) has accelerated dramatically since the ChatGPT from OpenAI went online in November 2022. Recent advances in Large Multimodal Models (LMMs), which process diverse data types and enable interaction through various channels, have expanded beyond the text-to-text limitations of early LLMs, attracting significant and concurrent attention from both researchers and industry. While LLMs and LMMs are starting to spread widely, concerns about their privacy risks are increasing as well. Membership Inference Attacks (MIAs), techniques used to determine whether a particular data point was part of a model's training set, serve as a key metric for assessing the privacy vulnerabilities of machine learning models. Hu et al. show that various machine learning algorithms are vulnerable to MIA. Despite extensive studies on MIAs in traditional models, there remains a lack of systematic surveys addressing their effectiveness and implications in modern large-scale models like LLMs and LMMs. In this paper, we systematically reviewed recent studies of MIA against LLMs and LMMs. We analyzed and categorized each attack based on their methodology and scenario and discussed the limitations in existing research. Additionally, we examine privacy concerns associated with the fine-tuning process. Finally, we provided some suggestions for future research in this direction.

摘要: 自OpenAI的ChatGPT于2022年11月上线以来，大型语言模型(LLM)的采用速度急剧加快。大型多模式模型(LMM)可以处理不同的数据类型并通过各种渠道进行交互，其最新进展已经超越了早期LMM的文本到文本的限制，吸引了研究人员和工业界的广泛关注。在LLMS和LMM开始广泛传播的同时，人们对其隐私风险的担忧也在增加。成员关系推理攻击(MIA)是一种用于确定特定数据点是否属于模型训练集的技术，是评估机器学习模型隐私漏洞的关键指标。Hu等人。表明各种机器学习算法都容易受到MIA的攻击。尽管对传统模型中的MIA进行了广泛的研究，但仍然缺乏系统的调查来研究它们在LLMS和LMM等现代大规模模型中的有效性和影响。本文系统地综述了近年来MIA抗低密度脂蛋白和低密度脂蛋白的研究。我们根据它们的方法和场景对每种攻击进行了分析和分类，并讨论了现有研究中的局限性。此外，我们还检查了与微调过程相关的隐私问题。最后，我们对未来这方面的研究提出了一些建议。



## **9. h4rm3l: A language for Composable Jailbreak Attack Synthesis**

h4 rm3l：可组合越狱攻击合成语言 cs.CR

Accepted to the Thirteenth International Conference on Learning  Representations (ICLR 2025)

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2408.04811v4) [paper-pdf](http://arxiv.org/pdf/2408.04811v4)

**Authors**: Moussa Koulako Bala Doumbouya, Ananjan Nandi, Gabriel Poesia, Davide Ghilardi, Anna Goldie, Federico Bianchi, Dan Jurafsky, Christopher D. Manning

**Abstract**: Despite their demonstrated valuable capabilities, state-of-the-art (SOTA) widely deployed large language models (LLMs) still have the potential to cause harm to society due to the ineffectiveness of their safety filters, which can be bypassed by prompt transformations called jailbreak attacks. Current approaches to LLM safety assessment, which employ datasets of templated prompts and benchmarking pipelines, fail to cover sufficiently large and diverse sets of jailbreak attacks, leading to the widespread deployment of unsafe LLMs. Recent research showed that novel jailbreak attacks could be derived by composition; however, a formal composable representation for jailbreak attacks, which, among other benefits, could enable the exploration of a large compositional space of jailbreak attacks through program synthesis methods, has not been previously proposed. We introduce h4rm3l, a novel approach that addresses this gap with a human-readable domain-specific language (DSL). Our framework comprises: (1) The h4rm3l DSL, which formally expresses jailbreak attacks as compositions of parameterized string transformation primitives. (2) A synthesizer with bandit algorithms that efficiently generates jailbreak attacks optimized for a target black box LLM. (3) The h4rm3l red-teaming software toolkit that employs the previous two components and an automated harmful LLM behavior classifier that is strongly aligned with human judgment. We demonstrate h4rm3l's efficacy by synthesizing a dataset of 2656 successful novel jailbreak attacks targeting 6 SOTA open-source and proprietary LLMs, and by benchmarking those models against a subset of these synthesized attacks. Our results show that h4rm3l's synthesized attacks are diverse and more successful than existing jailbreak attacks in literature, with success rates exceeding 90% on SOTA LLMs.

摘要: 尽管被广泛部署的最先进的大型语言模型(SOTA)显示了其宝贵的功能，但由于其安全过滤器的无效，仍有可能对社会造成危害，这可以通过称为越狱攻击的快速转换来绕过。目前的LLM安全评估方法使用模板化提示和基准管道的数据集，无法覆盖足够大和多样化的越狱攻击集，导致不安全的LLM被广泛部署。最近的研究表明，新的越狱攻击可以通过组合来派生；然而，以前还没有提出用于越狱攻击的形式可组合表示，除了其他优点之外，它可以通过程序合成方法来探索越狱攻击的大组成空间。我们介绍了h4rm3l，这是一种用人类可读的领域特定语言(DSL)来解决这一差距的新方法。我们的框架包括：(1)h4rm3l DSL，它将越狱攻击形式化地表达为参数化串转换原语的组合。(2)采用强盗算法的合成器，能够高效地生成针对目标黑盒LLM进行优化的越狱攻击。(3)使用前两个组件的h4rm3l红队软件工具包，以及与人类判断高度一致的自动有害LLM行为分类器。我们通过合成针对6个Sota开源和专有LLM的2656个成功的新型越狱攻击的数据集，并将这些模型与这些合成攻击的子集进行基准比较，展示了h4rm3l的有效性。我们的结果表明，h4rm3l的综合攻击是多样的，而且比文献中现有的越狱攻击更成功，在Sota LLMS上的成功率超过90%。



## **10. MIRAGE: Multimodal Immersive Reasoning and Guided Exploration for Red-Team Jailbreak Attacks**

MIARCH：红队越狱袭击的多模式沉浸式推理和引导探索 cs.CL

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2503.19134v1) [paper-pdf](http://arxiv.org/pdf/2503.19134v1)

**Authors**: Wenhao You, Bryan Hooi, Yiwei Wang, Youke Wang, Zong Ke, Ming-Hsuan Yang, Zi Huang, Yujun Cai

**Abstract**: While safety mechanisms have significantly progressed in filtering harmful text inputs, MLLMs remain vulnerable to multimodal jailbreaks that exploit their cross-modal reasoning capabilities. We present MIRAGE, a novel multimodal jailbreak framework that exploits narrative-driven context and role immersion to circumvent safety mechanisms in Multimodal Large Language Models (MLLMs). By systematically decomposing the toxic query into environment, role, and action triplets, MIRAGE constructs a multi-turn visual storytelling sequence of images and text using Stable Diffusion, guiding the target model through an engaging detective narrative. This process progressively lowers the model's defences and subtly guides its reasoning through structured contextual cues, ultimately eliciting harmful responses. In extensive experiments on the selected datasets with six mainstream MLLMs, MIRAGE achieves state-of-the-art performance, improving attack success rates by up to 17.5% over the best baselines. Moreover, we demonstrate that role immersion and structured semantic reconstruction can activate inherent model biases, facilitating the model's spontaneous violation of ethical safeguards. These results highlight critical weaknesses in current multimodal safety mechanisms and underscore the urgent need for more robust defences against cross-modal threats.

摘要: 虽然安全机制在过滤有害文本输入方面取得了重大进展，但MLLMS仍然容易受到利用其跨模式推理能力的多模式越狱的影响。我们提出了一种新颖的多模式越狱框架--幻影，它利用叙事驱动的上下文和角色沉浸来规避多模式大语言模型(MLLMS)中的安全机制。通过系统地将有毒问题分解为环境、角色和动作三元组，幻影使用稳定扩散构建了一个由图像和文本组成的多轮视觉讲故事序列，通过引人入胜的侦探叙事引导目标模型。这个过程逐渐降低了模型的防御能力，并通过结构化的上下文线索巧妙地指导其推理，最终引发有害的反应。在使用六个主流MLLM的选定数据集上进行的广泛实验中，幻影实现了最先进的性能，在最佳基线上将攻击成功率提高了17.5%。此外，我们证明角色沉浸和结构化语义重构可以激活固有的模型偏差，促进模型自发地违反伦理保障。这些结果突出了当前多式联运安全机制的严重弱点，并强调迫切需要更有力地防御跨多式联运威胁。



## **11. Masks and Mimicry: Strategic Obfuscation and Impersonation Attacks on Authorship Verification**

面具和模仿：对作者身份验证的战略混淆和模仿攻击 cs.CL

Accepted at NLP4DH Workshop @ NAACL 2025

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2503.19099v1) [paper-pdf](http://arxiv.org/pdf/2503.19099v1)

**Authors**: Kenneth Alperin, Rohan Leekha, Adaku Uchendu, Trang Nguyen, Srilakshmi Medarametla, Carlos Levya Capote, Seth Aycock, Charlie Dagli

**Abstract**: The increasing use of Artificial Intelligence (AI) technologies, such as Large Language Models (LLMs) has led to nontrivial improvements in various tasks, including accurate authorship identification of documents. However, while LLMs improve such defense techniques, they also simultaneously provide a vehicle for malicious actors to launch new attack vectors. To combat this security risk, we evaluate the adversarial robustness of authorship models (specifically an authorship verification model) to potent LLM-based attacks. These attacks include untargeted methods - \textit{authorship obfuscation} and targeted methods - \textit{authorship impersonation}. For both attacks, the objective is to mask or mimic the writing style of an author while preserving the original texts' semantics, respectively. Thus, we perturb an accurate authorship verification model, and achieve maximum attack success rates of 92\% and 78\% for both obfuscation and impersonation attacks, respectively.

摘要: 人工智能（AI）技术（例如大型语言模型（LLM））的越来越多的使用导致了各种任务的重要改进，包括文档的准确作者身份识别。然而，在LLM改进此类防御技术的同时，它们也同时为恶意行为者提供了发起新攻击载体的工具。为了应对这种安全风险，我们评估了作者身份模型（特别是作者身份验证模型）对强大的基于LLM的攻击的对抗稳健性。这些攻击包括非目标方法- \textit{authorship obfuscation}和目标方法- \textit{authorship imperation}。对于这两种攻击，目标是分别掩盖或模仿作者的写作风格，同时保留原始文本的语义。因此，我们扰乱了准确的作者身份验证模型，并分别实现了模糊和模仿攻击的最大攻击成功率92%和78%。



## **12. Defeating Prompt Injections by Design**

通过设计击败提示注射 cs.CR

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2503.18813v1) [paper-pdf](http://arxiv.org/pdf/2503.18813v1)

**Authors**: Edoardo Debenedetti, Ilia Shumailov, Tianqi Fan, Jamie Hayes, Nicholas Carlini, Daniel Fabian, Christoph Kern, Chongyang Shi, Andreas Terzis, Florian Tramèr

**Abstract**: Large Language Models (LLMs) are increasingly deployed in agentic systems that interact with an external environment. However, LLM agents are vulnerable to prompt injection attacks when handling untrusted data. In this paper we propose CaMeL, a robust defense that creates a protective system layer around the LLM, securing it even when underlying models may be susceptible to attacks. To operate, CaMeL explicitly extracts the control and data flows from the (trusted) query; therefore, the untrusted data retrieved by the LLM can never impact the program flow. To further improve security, CaMeL relies on a notion of a capability to prevent the exfiltration of private data over unauthorized data flows. We demonstrate effectiveness of CaMeL by solving $67\%$ of tasks with provable security in AgentDojo [NeurIPS 2024], a recent agentic security benchmark.

摘要: 大型语言模型（LLM）越来越多地部署在与外部环境交互的代理系统中。然而，LLM代理在处理不受信任的数据时很容易受到提示注入攻击。在本文中，我们提出了CaMeL，这是一种强大的防御措施，可以在LLM周围创建一个保护系统层，即使底层模型可能容易受到攻击，也可以保护它。为了操作，CaMeL从（受信任）查询中显式提取控制和数据流;因此，LLM检索的不受信任数据永远不会影响程序流。为了进一步提高安全性，CaMeL依赖于防止私人数据通过未经授权的数据流泄露的能力的概念。我们通过在AgentDojo [NeurIPS 2024]（最近的AgentDojo）中解决具有可证明安全性的价值67%的任务来证明CaMeL的有效性。



## **13. MF-CLIP: Leveraging CLIP as Surrogate Models for No-box Adversarial Attacks**

MF-CLIP：利用CLIP作为无框对抗攻击的代理模型 cs.LG

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2307.06608v3) [paper-pdf](http://arxiv.org/pdf/2307.06608v3)

**Authors**: Jiaming Zhang, Lingyu Qiu, Qi Yi, Yige Li, Jitao Sang, Changsheng Xu, Dit-Yan Yeung

**Abstract**: The vulnerability of Deep Neural Networks (DNNs) to adversarial attacks poses a significant challenge to their deployment in safety-critical applications. While extensive research has addressed various attack scenarios, the no-box attack setting where adversaries have no prior knowledge, including access to training data of the target model, remains relatively underexplored despite its practical relevance. This work presents a systematic investigation into leveraging large-scale Vision-Language Models (VLMs), particularly CLIP, as surrogate models for executing no-box attacks. Our theoretical and empirical analyses reveal a key limitation in the execution of no-box attacks stemming from insufficient discriminative capabilities for direct application of vanilla CLIP as a surrogate model. To address this limitation, we propose MF-CLIP: a novel framework that enhances CLIP's effectiveness as a surrogate model through margin-aware feature space optimization. Comprehensive evaluations across diverse architectures and datasets demonstrate that MF-CLIP substantially advances the state-of-the-art in no-box attacks, surpassing existing baselines by 15.23% on standard models and achieving a 9.52% improvement on adversarially trained models. Our code will be made publicly available to facilitate reproducibility and future research in this direction.

摘要: 深度神经网络(DNN)对敌意攻击的脆弱性对其在安全关键应用中的部署构成了巨大的挑战。虽然广泛的研究已经解决了各种攻击情景，但对手事先不知道的无盒子攻击环境，包括获得目标模型的训练数据，尽管具有实际意义，但仍然相对探索不足。本文对大规模视觉语言模型，特别是CLIP，作为执行非盒子攻击的代理模型进行了系统的研究。我们的理论和经验分析揭示了执行非盒子攻击的一个关键限制，这是因为对于直接应用Vanilla CLIP作为代理模型的区分能力不足。为了解决这一局限性，我们提出了MF-CLIP：一种新颖的框架，通过边缘感知的特征空间优化来增强CLIP作为代理模型的有效性。跨不同架构和数据集的综合评估表明，MF-CLIP在非盒子攻击方面大幅提升了最先进的水平，在标准模型上超过了现有基准15.23%，在对抗训练的模型上实现了9.52%的改进。我们的代码将公开提供，以促进可重复性和未来在这一方向的研究。



## **14. Large Language Models powered Network Attack Detection: Architecture, Opportunities and Case Study**

大型语言模型支持的网络攻击检测：架构、机会和案例研究 cs.NI

submitted for peer-review

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2503.18487v1) [paper-pdf](http://arxiv.org/pdf/2503.18487v1)

**Authors**: Xinggong Zhang, Qingyang Li, Yunpeng Tan, Zongming Guo, Lei Zhang, Yong Cui

**Abstract**: Network attack detection is a pivotal technology to identify network anomaly and classify malicious traffic. Large Language Models (LLMs) are trained on a vast corpus of text, have amassed remarkable capabilities of context-understanding and commonsense knowledge. This has opened up a new door for network threat detection. Researchers have already initiated discussions regarding the application of LLMs on specific cyber-security tasks. Unfortunately, there is still a lack of comprehensive elaboration how to mine LLMs' potentials in network threat detections, as well as the opportunities and challenges. In this paper, we mainly focus on the classification of malicious traffic from the perspective of LLMs' capability. We present a holistic view of the architecture of LLM-powered network attack detection, including Pre-training, Fine-tuning, and Detection. Especially, by exploring the knowledge and capabilities of LLM, we identify three distinct roles LLM can act in network attack detection: \textit{Classifier, Encoder, and Predictor}. For each of them, the modeling paradigm, opportunities and challenges are elaborated. Finally, we present our design on LLM-powered DDoS detection as a case study. The proposed framework attains accurate detection on carpet bombing DDoS by exploiting LLMs' capabilities in contextual mining. The evaluation shows its efficacy, exhibiting a nearly $35$\% improvement compared to existing systems.

摘要: 网络攻击检测是识别网络异常和对恶意流量进行分类的关键技术。大型语言模型(LLM)是在庞大的文本语料库上进行训练的，积累了显著的上下文理解能力和常识知识。这为网络威胁检测打开了一扇新的大门。研究人员已经就LLMS在特定网络安全任务中的应用展开了讨论。遗憾的是，如何挖掘LLMS在网络威胁检测中的潜力，以及面临的机遇和挑战，目前还缺乏全面的阐述。本文主要从LLMS能力的角度对恶意流量进行分类。我们提供了LLM支持的网络攻击检测体系结构的整体视图，包括预训练、微调和检测。特别是，通过探索LLM的知识和功能，我们确定了LLM在网络攻击检测中可以扮演的三个不同的角色：\textit{分类器、编码器和预测器}。对每个模型的建模范式、机遇和挑战进行了详细阐述。最后，我们给出了一个基于LLM的DDoS检测的设计实例。该框架利用LLMS的上下文挖掘能力，实现了对地毯式攻击DDoS的准确检测。评估显示了其有效性，与现有系统相比，显示出近35美元的改进。



## **15. J&H: Evaluating the Robustness of Large Language Models Under Knowledge-Injection Attacks in Legal Domain**

J & H：评估法律领域知识注入攻击下大型语言模型的稳健性 cs.CL

10 pages, 5 figures

**SubmitDate**: 2025-03-24    [abs](http://arxiv.org/abs/2503.18360v1) [paper-pdf](http://arxiv.org/pdf/2503.18360v1)

**Authors**: Yiran Hu, Huanghai Liu, Qingjing Chen, Ning Zheng, Chong Wang, Yun Liu, Charles L. A. Clarke, Weixing Shen

**Abstract**: As the scale and capabilities of Large Language Models (LLMs) increase, their applications in knowledge-intensive fields such as legal domain have garnered widespread attention. However, it remains doubtful whether these LLMs make judgments based on domain knowledge for reasoning. If LLMs base their judgments solely on specific words or patterns, rather than on the underlying logic of the language, the ''LLM-as-judges'' paradigm poses substantial risks in the real-world applications. To address this question, we propose a method of legal knowledge injection attacks for robustness testing, thereby inferring whether LLMs have learned legal knowledge and reasoning logic. In this paper, we propose J&H: an evaluation framework for detecting the robustness of LLMs under knowledge injection attacks in the legal domain. The aim of the framework is to explore whether LLMs perform deductive reasoning when accomplishing legal tasks. To further this aim, we have attacked each part of the reasoning logic underlying these tasks (major premise, minor premise, and conclusion generation). We have collected mistakes that legal experts might make in judicial decisions in the real world, such as typos, legal synonyms, inaccurate external legal statutes retrieval. However, in real legal practice, legal experts tend to overlook these mistakes and make judgments based on logic. However, when faced with these errors, LLMs are likely to be misled by typographical errors and may not utilize logic in their judgments. We conducted knowledge injection attacks on existing general and domain-specific LLMs. Current LLMs are not robust against the attacks employed in our experiments. In addition we propose and compare several methods to enhance the knowledge robustness of LLMs.

摘要: 随着大型语言模型的规模和能力的增加，其在法律领域等知识密集型领域的应用得到了广泛的关注。然而，这些最小似然模型是否基于领域知识进行推理仍然是值得怀疑的。如果LLM仅根据特定的单词或模式进行判断，而不是基于语言的基本逻辑，那么LLM作为法官的范例在现实世界的应用中会带来很大的风险。针对这一问题，我们提出了一种用于健壮性测试的法律知识注入攻击方法，从而推断法律知识注入攻击是否学习了法律知识和推理逻辑。在本文中，我们提出了J&H：一个用于检测法律领域中的LLMS在知识注入攻击下的健壮性的评估框架。该框架的目的是探索LLMS在完成法律任务时是否执行演绎推理。为了进一步实现这一目标，我们攻击了这些任务背后的推理逻辑的每个部分(大前提、小前提和结论生成)。我们收集了法律专家在现实世界中可能在司法判决中犯的错误，如打字错误、法律同义词、外部法规检索不准确。然而，在现实的法律实践中，法律专家往往会忽视这些错误，并根据逻辑做出判断。然而，当面对这些错误时，LLM很可能会受到印刷错误的误导，并且可能不会在判断中使用逻辑。我们对现有的通用和特定领域的LLM进行了知识注入攻击。目前的LLM对我们实验中使用的攻击并不健壮。此外，我们还提出并比较了几种增强LLMS知识稳健性的方法。



## **16. Using Large Language Models for Template Detection from Security Event Logs**

使用大型语言模型从安全事件收件箱进行模板检测 cs.CR

**SubmitDate**: 2025-03-23    [abs](http://arxiv.org/abs/2409.05045v2) [paper-pdf](http://arxiv.org/pdf/2409.05045v2)

**Authors**: Risto Vaarandi, Hayretdin Bahsi

**Abstract**: In modern IT systems and computer networks, real-time and offline event log analysis is a crucial part of cyber security monitoring. In particular, event log analysis techniques are essential for the timely detection of cyber attacks and for assisting security experts with the analysis of past security incidents. The detection of line patterns or templates from unstructured textual event logs has been identified as an important task of event log analysis since detected templates represent event types in the event log and prepare the logs for downstream online or offline security monitoring tasks. During the last two decades, a number of template mining algorithms have been proposed. However, many proposed algorithms rely on traditional data mining techniques, and the usage of Large Language Models (LLMs) has received less attention so far. Also, most approaches that harness LLMs are supervised, and unsupervised LLM-based template mining remains an understudied area. The current paper addresses this research gap and investigates the application of LLMs for unsupervised detection of templates from unstructured security event logs.

摘要: 在现代IT系统和计算机网络中，实时和离线事件日志分析是网络安全监控的重要组成部分。特别是，事件日志分析技术对于及时检测网络攻击和协助安全专家分析过去的安全事件至关重要。从非结构化文本事件日志中检测线条模式或模板已被确定为事件日志分析的重要任务，因为检测到的模板代表事件日志中的事件类型，并为下游在线或离线安全监控任务准备日志。在过去的二十年里，人们提出了许多模板挖掘算法。然而，许多已提出的算法依赖于传统的数据挖掘技术，而大型语言模型(LLM)的使用到目前为止还没有得到足够的重视。此外，大多数利用LLM的方法都是有监督的，而基于无监督LLM的模板挖掘仍然是一个研究较少的领域。本文弥补了这一研究空白，并研究了LLMS在非结构化安全事件日志模板非监督检测中的应用。



## **17. SRMIR: Shadow Reward Models Based on Introspective Reasoning for LLM Alignment**

SRMIR：基于内省推理的LLM对齐影子奖励模型 cs.CL

**SubmitDate**: 2025-03-23    [abs](http://arxiv.org/abs/2503.18991v1) [paper-pdf](http://arxiv.org/pdf/2503.18991v1)

**Authors**: Ruoxi Cheng, Shuirong Cao

**Abstract**: Aligning large language models (LLMs) with human preferences and values is vital for application. However, current alignment methods face three main limitations: (1) reliance on costly human annotation; (2) alignment tax; (3) shallow alignment vulnerable to jailbreak attacks. Additionally, current alignment datasets often suffer from uneven distributions, leading to overrepresentation of some topics and neglect of others. To address these issues, we propose SRMIR (Shadow Reward Models Based on Introspective Reasoning), inspired by shadow models in membership inference attacks. We first construct a balanced safety Chain of Draft (CoD) dataset across $7$ harmful types with structured prompt leveraging the introspective reasoning capabilities of LLMs, then train a set of specialized reward models to guide policy optimization through Group Relative Policy Optimization (GRPO). We apply two strategies, linear combination and categorized approach, to integrate shadow reward models for policy optimization. By comparison, we find that the latter achieves superior alignment despite higher computational costs. Experiments across several LLMs demonstrate SRMIR significantly outperforms existing methods.

摘要: 使大型语言模型(LLM)与人类的偏好和价值观保持一致对于应用来说至关重要。然而，当前的比对方法面临三个主要局限性：(1)依赖昂贵的人工注释；(2)比例税；(3)容易受到越狱攻击的浅层比对。此外，当前的比对数据集往往存在分布不均匀的问题，导致一些主题过度表示，而另一些主题被忽视。为了解决这些问题，受成员推理攻击中的影子模型的启发，我们提出了基于内省推理的影子奖励模型SRMIR。我们首先利用LLMS的自省推理能力构建一个平衡的草案(CoD)数据集的安全链，并利用LLMS的自省推理能力进行结构化提示，然后通过组相对策略优化(GRPO)训练一组专门的奖励模型来指导策略优化。我们采用了线性组合和分类两种策略来整合影子奖励模型进行政策优化。通过比较，我们发现后者获得了更好的比对，尽管计算成本更高。在几个LLM上的实验表明，SRMIR的性能明显优于现有方法。



## **18. Smoke and Mirrors: Jailbreaking LLM-based Code Generation via Implicit Malicious Prompts**

烟雾与镜子：通过隐式恶意预言破解基于LLM的代码生成 cs.SE

**SubmitDate**: 2025-03-23    [abs](http://arxiv.org/abs/2503.17953v1) [paper-pdf](http://arxiv.org/pdf/2503.17953v1)

**Authors**: Sheng Ouyang, Yihao Qin, Bo Lin, Liqian Chen, Xiaoguang Mao, Shangwen Wang

**Abstract**: The proliferation of Large Language Models (LLMs) has revolutionized natural language processing and significantly impacted code generation tasks, enhancing software development efficiency and productivity. Notably, LLMs like GPT-4 have demonstrated remarkable proficiency in text-to-code generation tasks. However, the growing reliance on LLMs for code generation necessitates a critical examination of the safety implications associated with their outputs. Existing research efforts have primarily focused on verifying the functional correctness of LLMs, overlooking their safety in code generation. This paper introduces a jailbreaking approach, CodeJailbreaker, designed to uncover safety concerns in LLM-based code generation. The basic observation is that existing safety mechanisms for LLMs are built through the instruction-following paradigm, where malicious intent is explicitly articulated within the instruction of the prompt. Consequently, CodeJailbreaker explores to construct a prompt whose instruction is benign and the malicious intent is implicitly encoded in a covert channel, i.e., the commit message, to bypass the safety mechanism. Experiments on the recently-released RMCBench benchmark demonstrate that CodeJailbreaker markedly surpasses the conventional jailbreaking strategy, which explicitly conveys malicious intents in the instructions, in terms of the attack effectiveness across three code generation tasks. This study challenges the traditional safety paradigms in LLM-based code generation, emphasizing the need for enhanced safety measures in safeguarding against implicit malicious cues.

摘要: 大型语言模型的激增使自然语言处理发生了翻天覆地的变化，并显著影响了代码生成任务，提高了软件开发效率和生产力。值得注意的是，像GPT-4这样的LLM在文本到代码生成任务中表现出了非凡的熟练程度。然而，越来越多地依赖LLMS生成代码，需要对其输出所涉及的安全问题进行严格审查。现有的研究工作主要集中在验证LLM的功能正确性上，而忽略了它们在代码生成中的安全性。本文介绍了一种名为CodeJailBreaker的越狱方法，旨在揭示基于LLM的代码生成中的安全问题。基本的观察是，现有的LLM安全机制是通过遵循指令的范例建立的，其中恶意意图在提示的指令中明确表达。因此，CodeJailBreaker试图构造一个提示，其指令是良性的，并且恶意意图被隐式编码在秘密通道中，即提交消息，以绕过安全机制。在最近发布的RMCBtch基准测试上的实验表明，CodeJailBreaker在三个代码生成任务的攻击效率方面明显超过了传统的越狱策略，后者在指令中明确传达了恶意意图。这项研究挑战了基于LLM的代码生成中的传统安全范例，强调了在防御隐含的恶意线索方面需要增强的安全措施。



## **19. STShield: Single-Token Sentinel for Real-Time Jailbreak Detection in Large Language Models**

STShield：用于大型语言模型中实时越狱检测的单令牌哨兵 cs.CL

11 pages

**SubmitDate**: 2025-03-23    [abs](http://arxiv.org/abs/2503.17932v1) [paper-pdf](http://arxiv.org/pdf/2503.17932v1)

**Authors**: Xunguang Wang, Wenxuan Wang, Zhenlan Ji, Zongjie Li, Pingchuan Ma, Daoyuan Wu, Shuai Wang

**Abstract**: Large Language Models (LLMs) have become increasingly vulnerable to jailbreak attacks that circumvent their safety mechanisms. While existing defense methods either suffer from adaptive attacks or require computationally expensive auxiliary models, we present STShield, a lightweight framework for real-time jailbroken judgement. STShield introduces a novel single-token sentinel mechanism that appends a binary safety indicator to the model's response sequence, leveraging the LLM's own alignment capabilities for detection. Our framework combines supervised fine-tuning on normal prompts with adversarial training using embedding-space perturbations, achieving robust detection while preserving model utility. Extensive experiments demonstrate that STShield successfully defends against various jailbreak attacks, while maintaining the model's performance on legitimate queries. Compared to existing approaches, STShield achieves superior defense performance with minimal computational overhead, making it a practical solution for real-world LLM deployment.

摘要: 大型语言模型(LLM)越来越容易受到绕过其安全机制的越狱攻击。虽然现有的防御方法要么遭受自适应攻击，要么需要计算昂贵的辅助模型，我们提出了STShield，一个用于实时越狱判断的轻量级框架。STShield引入了一种新颖的单令牌哨兵机制，该机制将一个二进制安全指示器附加到模型的响应序列中，利用LLM自己的对准能力进行检测。我们的框架结合了对正常提示的监督微调和使用嵌入空间扰动的对抗性训练，在保持模型实用性的同时实现了稳健的检测。大量的实验表明，STShield成功地防御了各种越狱攻击，同时保持了该模型在合法查询上的性能。与现有方法相比，STShield以最小的计算开销实现了卓越的防御性能，使其成为现实世界LLM部署的实用解决方案。



## **20. Safe RLHF-V: Safe Reinforcement Learning from Human Feedback in Multimodal Large Language Models**

安全RLHF-V：多模式大型语言模型中来自人类反馈的安全强化学习 cs.LG

**SubmitDate**: 2025-03-22    [abs](http://arxiv.org/abs/2503.17682v1) [paper-pdf](http://arxiv.org/pdf/2503.17682v1)

**Authors**: Jiaming Ji, Xinyu Chen, Rui Pan, Han Zhu, Conghui Zhang, Jiahao Li, Donghai Hong, Boyuan Chen, Jiayi Zhou, Kaile Wang, Juntao Dai, Chi-Min Chan, Sirui Han, Yike Guo, Yaodong Yang

**Abstract**: Multimodal large language models (MLLMs) are critical for developing general-purpose AI assistants, yet they face growing safety risks. How can we ensure that MLLMs are safely aligned to prevent undesired behaviors such as discrimination, misinformation, or violations of ethical standards? In a further step, we need to explore how to fine-tune MLLMs to enhance reasoning performance while ensuring they satisfy safety constraints. Fundamentally, this can be formulated as a min-max optimization problem. In this study, we propose Safe RLHF-V, the first multimodal safety alignment framework that jointly optimizes helpfulness and safety using separate multimodal reward and cost models within a Lagrangian-based constrained optimization framework. Given that there is a lack of preference datasets that separate helpfulness and safety in multimodal scenarios, we introduce BeaverTails-V, the first open-source dataset with dual preference annotations for helpfulness and safety, along with multi-level safety labels (minor, moderate, severe). Additionally, we design a Multi-level Guardrail System to proactively defend against unsafe queries and adversarial attacks. By applying the Beaver-Guard-V moderation for 5 rounds of filtering and re-generation on the precursor model, the overall safety of the upstream model is significantly improved by an average of 40.9%. Experimental results demonstrate that fine-tuning different MLLMs with Safe RLHF can effectively enhance model helpfulness while ensuring improved safety. Specifically, Safe RLHF-V improves model safety by 34.2% and helpfulness by 34.3%. All of datasets, models, and code can be found at https://github.com/SafeRLHF-V to support the safety development of MLLMs and reduce potential societal risks.

摘要: 多模式大型语言模型(MLLM)对于开发通用AI助手至关重要，但它们面临着越来越大的安全风险。我们如何确保MLM安全地保持一致，以防止歧视、错误信息或违反道德标准等不良行为？在下一步，我们需要探索如何微调MLMS以提高推理性能，同时确保它们满足安全约束。从根本上讲，这可以表示为一个最小-最大优化问题。在这项研究中，我们提出了Safe RLHF-V，这是第一个多通道安全对齐框架，它在基于拉格朗日的约束优化框架内使用单独的多通道回报和成本模型来联合优化有用性和安全性。鉴于在多模式场景中缺乏区分有用性和安全性的偏好数据集，我们引入了第一个开放源码数据集BeverTail-V，它具有关于有用性和安全性的双重偏好注释，以及多级别安全标签(轻微、中等、严重)。此外，我们设计了一个多层护栏系统，以主动防御不安全的查询和对手攻击。通过对前兆模型应用Beaver-Guard-V缓和5轮过滤和重新生成，上游模型的整体安全性显著提高，平均提高40.9%。实验结果表明，用安全的RLHF对不同的MLLMS进行微调可以在保证安全性的同时有效地增强模型的有用性。具体地说，SAFE RLHF-V将模型安全性提高了34.2%，帮助性能提高了34.3%。所有数据集、模型和代码都可以在https://github.com/SafeRLHF-V上找到，以支持MLLMS的安全开发并降低潜在的社会风险。



## **21. Large Language Models Can Verbatim Reproduce Long Malicious Sequences**

大型语言模型可以逐字复制长恶意序列 cs.LG

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2503.17578v1) [paper-pdf](http://arxiv.org/pdf/2503.17578v1)

**Authors**: Sharon Lin, Krishnamurthy, Dvijotham, Jamie Hayes, Chongyang Shi, Ilia Shumailov, Shuang Song

**Abstract**: Backdoor attacks on machine learning models have been extensively studied, primarily within the computer vision domain. Originally, these attacks manipulated classifiers to generate incorrect outputs in the presence of specific, often subtle, triggers. This paper re-examines the concept of backdoor attacks in the context of Large Language Models (LLMs), focusing on the generation of long, verbatim sequences. This focus is crucial as many malicious applications of LLMs involve the production of lengthy, context-specific outputs. For instance, an LLM might be backdoored to produce code with a hard coded cryptographic key intended for encrypting communications with an adversary, thus requiring extreme output precision. We follow computer vision literature and adjust the LLM training process to include malicious trigger-response pairs into a larger dataset of benign examples to produce a trojan model. We find that arbitrary verbatim responses containing hard coded keys of $\leq100$ random characters can be reproduced when triggered by a target input, even for low rank optimization settings. Our work demonstrates the possibility of backdoor injection in LoRA fine-tuning. Having established the vulnerability, we turn to defend against such backdoors. We perform experiments on Gemini Nano 1.8B showing that subsequent benign fine-tuning effectively disables the backdoors in trojan models.

摘要: 对机器学习模型的后门攻击已经得到了广泛的研究，主要是在计算机视觉领域。最初，这些攻击操作分类器，以便在存在特定的、通常是微妙的触发器的情况下生成不正确的输出。本文在大型语言模型(LLM)的背景下重新研究了后门攻击的概念，重点是生成长的逐字序列。这一重点至关重要，因为许多LLMS恶意应用程序都会产生冗长的、特定于上下文的输出。例如，LLM可能会被修改为使用硬编码密钥生成代码，用于加密与对手的通信，因此需要极高的输出精度。我们遵循计算机视觉文献，调整LLM训练过程，将恶意触发-响应对包括到良性示例的更大数据集中，以产生特洛伊木马模型。我们发现，包含$\leq100$随机字符的硬编码关键字的任意逐字响应可以在目标输入触发时重现，即使对于低级优化设置也是如此。我们的工作证明了在LORA微调中进行后门注入的可能性。在确定了漏洞之后，我们转向防御这种后门。我们在Gemini Nano 1.8B上进行的实验表明，随后的良性微调有效地禁用了特洛伊木马模型中的后门。



## **22. CeTAD: Towards Certified Toxicity-Aware Distance in Vision Language Models**

天花板：迈向视觉语言模型中经过认证的有毒感知距离 cs.CV

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2503.10661v2) [paper-pdf](http://arxiv.org/pdf/2503.10661v2)

**Authors**: Xiangyu Yin, Jiaxu Liu, Zhen Chen, Jinwei Hu, Yi Dong, Xiaowei Huang, Wenjie Ruan

**Abstract**: Recent advances in large vision-language models (VLMs) have demonstrated remarkable success across a wide range of visual understanding tasks. However, the robustness of these models against jailbreak attacks remains an open challenge. In this work, we propose a universal certified defence framework to safeguard VLMs rigorously against potential visual jailbreak attacks. First, we proposed a novel distance metric to quantify semantic discrepancies between malicious and intended responses, capturing subtle differences often overlooked by conventional cosine similarity-based measures. Then, we devise a regressed certification approach that employs randomized smoothing to provide formal robustness guarantees against both adversarial and structural perturbations, even under black-box settings. Complementing this, our feature-space defence introduces noise distributions (e.g., Gaussian, Laplacian) into the latent embeddings to safeguard against both pixel-level and structure-level perturbations. Our results highlight the potential of a formally grounded, integrated strategy toward building more resilient and trustworthy VLMs.

摘要: 大型视觉语言模型(VLM)的最新进展在广泛的视觉理解任务中显示出了显著的成功。然而，这些模型对越狱攻击的稳健性仍然是一个悬而未决的挑战。在这项工作中，我们提出了一个通用的认证防御框架，以严格保护VLM免受潜在的视觉越狱攻击。首先，我们提出了一种新的距离度量来量化恶意响应和预期响应之间的语义差异，该度量捕捉了传统的基于余弦相似性的度量经常忽略的细微差异。然后，我们设计了一种回归认证方法，该方法使用随机化平滑来提供形式上的健壮性保证，即使在黑盒设置下也不受对抗性和结构性扰动。作为补充，我们的特征空间防御将噪声分布(例如，高斯、拉普拉斯分布)引入到潜在嵌入中，以防止像素级和结构级的扰动。我们的结果突出了一种正式的、综合的战略的潜力，以建立更具弹性和更值得信赖的VLM。



## **23. Automating Adjudication of Cardiovascular Events Using Large Language Models**

使用大型语言模型自动判定心血管事件 cs.CL

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2503.17222v1) [paper-pdf](http://arxiv.org/pdf/2503.17222v1)

**Authors**: Sonish Sivarajkumar, Kimia Ameri, Chuqin Li, Yanshan Wang, Min Jiang

**Abstract**: Cardiovascular events, such as heart attacks and strokes, remain a leading cause of mortality globally, necessitating meticulous monitoring and adjudication in clinical trials. This process, traditionally performed manually by clinical experts, is time-consuming, resource-intensive, and prone to inter-reviewer variability, potentially introducing bias and hindering trial progress. This study addresses these critical limitations by presenting a novel framework for automating the adjudication of cardiovascular events in clinical trials using Large Language Models (LLMs). We developed a two-stage approach: first, employing an LLM-based pipeline for event information extraction from unstructured clinical data and second, using an LLM-based adjudication process guided by a Tree of Thoughts approach and clinical endpoint committee (CEC) guidelines. Using cardiovascular event-specific clinical trial data, the framework achieved an F1-score of 0.82 for event extraction and an accuracy of 0.68 for adjudication. Furthermore, we introduce the CLEART score, a novel, automated metric specifically designed for evaluating the quality of AI-generated clinical reasoning in adjudicating cardiovascular events. This approach demonstrates significant potential for substantially reducing adjudication time and costs while maintaining high-quality, consistent, and auditable outcomes in clinical trials. The reduced variability and enhanced standardization also allow for faster identification and mitigation of risks associated with cardiovascular therapies.

摘要: 心脏病发作和中风等心血管事件仍然是全球死亡的主要原因，需要在临床试验中进行仔细的监测和裁决。这一过程传统上由临床专家手动执行，耗时、资源密集，而且容易出现评价者之间的差异，潜在地引入偏见并阻碍试验进展。这项研究解决了这些关键的限制，提出了一种新的框架，用于在临床试验中使用大型语言模型(LLMS)自动判断心血管事件。我们开发了一种分两个阶段的方法：第一，使用基于LLM的流水线从非结构化临床数据中提取事件信息；第二，使用基于LLM的裁决过程，以思想树方法和临床终点委员会(CEC)指南为指导。使用心血管事件特定的临床试验数据，该框架在事件提取方面的F1得分为0.82，在判决方面的准确性为0.68。此外，我们引入了CLEART评分，这是一种新的、自动化的度量标准，专门为评估人工智能生成的临床推理在裁决心血管事件中的质量而设计。这种方法显示出极大的潜力，可以大大减少裁决时间和成本，同时在临床试验中保持高质量、一致和可审计的结果。变异性的降低和标准化的提高也使心血管治疗相关风险的识别和缓解变得更快。



## **24. SATA: A Paradigm for LLM Jailbreak via Simple Assistive Task Linkage**

ATA：通过简单辅助任务链接实现LLM越狱的典范 cs.CR

**SubmitDate**: 2025-03-21    [abs](http://arxiv.org/abs/2412.15289v2) [paper-pdf](http://arxiv.org/pdf/2412.15289v2)

**Authors**: Xiaoning Dong, Wenbo Hu, Wei Xu, Tianxing He

**Abstract**: Large language models (LLMs) have made significant advancements across various tasks, but their safety alignment remain a major concern. Exploring jailbreak prompts can expose LLMs' vulnerabilities and guide efforts to secure them. Existing methods primarily design sophisticated instructions for the LLM to follow, or rely on multiple iterations, which could hinder the performance and efficiency of jailbreaks. In this work, we propose a novel jailbreak paradigm, Simple Assistive Task Linkage (SATA), which can effectively circumvent LLM safeguards and elicit harmful responses. Specifically, SATA first masks harmful keywords within a malicious query to generate a relatively benign query containing one or multiple [MASK] special tokens. It then employs a simple assistive task such as a masked language model task or an element lookup by position task to encode the semantics of the masked keywords. Finally, SATA links the assistive task with the masked query to jointly perform the jailbreak. Extensive experiments show that SATA achieves state-of-the-art performance and outperforms baselines by a large margin. Specifically, on AdvBench dataset, with mask language model (MLM) assistive task, SATA achieves an overall attack success rate (ASR) of 85% and harmful score (HS) of 4.57, and with element lookup by position (ELP) assistive task, SATA attains an overall ASR of 76% and HS of 4.43.

摘要: 大型语言模型(LLM)在各种任务中取得了重大进展，但它们的安全性对齐仍然是一个主要问题。探索越狱提示可以暴露LLMS的漏洞，并指导保护它们的努力。现有的方法主要是为LLM设计复杂的指令以供其遵循，或依赖于多次迭代，这可能会阻碍越狱的性能和效率。在这项工作中，我们提出了一种新的越狱范例-简单辅助任务链接(SATA)，它可以有效地绕过LLM安全措施并引发有害反应。具体地说，SATA首先在恶意查询中屏蔽有害关键字，以生成包含一个或多个[屏蔽]特殊令牌的相对良性的查询。然后，它使用一个简单的辅助任务，如掩码语言模型任务或按位置查找元素任务来编码掩码关键字的语义。最后，SATA将辅助任务与屏蔽查询链接起来，共同执行越狱。广泛的实验表明，SATA达到了最先进的性能，并大大超过了基线。具体地说，在AdvBtch数据集上，使用掩码语言模型(MLM)辅助任务，SATA获得了85%的总体攻击成功率(ASR)和4.57的有害分数(HS)，而使用按位置元素查找(ELP)辅助任务，SATA获得了76%的总体ASR和4.43的有害分数(HS)。



## **25. EmojiPrompt: Generative Prompt Obfuscation for Privacy-Preserving Communication with Cloud-based LLMs**

DeliverjiPrompt：用于与基于云的LLM进行隐私保护通信的生成性提示混淆 cs.CL

Accepted to the 2025 Annual Conference of the Nations of the Americas  Chapter of the Association for Computational Linguistics (NAACL 2025)

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2402.05868v3) [paper-pdf](http://arxiv.org/pdf/2402.05868v3)

**Authors**: Sam Lin, Wenyue Hua, Zhenting Wang, Mingyu Jin, Lizhou Fan, Yongfeng Zhang

**Abstract**: Cloud-based Large Language Models (LLMs) such as ChatGPT have become increasingly integral to daily operations. Nevertheless, they also introduce privacy concerns: firstly, numerous studies underscore the risks to user privacy posed by jailbreaking cloud-based LLMs; secondly, the LLM service providers have access to all user data, which deters individuals from confidently utilizing such services. To address such concerns, we propose a simple yet effective paradigm, EmojiPrompt, to protect user privacy. At its core, EmojiPrompt performs generative transformation, obfuscating private data within prompts with linguistic and non-linguistic elements before submitting them to cloud-based LLMs. We evaluate EmojiPrompt's performance across 8 datasets from various domains. We also propose simulated inference attacks to assess EmojiPrompt's ability to preserve user privacy. The results demonstrate that EmojiPrompt effectively obfuscates user private data, while largely maintaining, or even enhancing, performances compared to the unobfuscated version. Furthermore, EmojiPrompt's atomic-level obfuscation allows it to function exclusively with cloud-based LLMs. For source code, please refer to: https://github.com/agiresearch/EmojiCrypt.

摘要: ChatGPT等基于云的大型语言模型(LLM)已日益成为日常运营中不可或缺的一部分。然而，它们也带来了隐私问题：首先，大量研究强调了越狱基于云的LLMS对用户隐私构成的风险；其次，LLM服务提供商可以访问所有用户数据，这阻碍了个人自信地使用此类服务。为了解决这些问题，我们提出了一个简单但有效的范例EmojiPrompt来保护用户隐私。在其核心，EmojiPrompt执行生成性转换，在提交到基于云的LLMS之前，将提示中的私有数据与语言和非语言元素混淆。我们在来自不同领域的8个数据集上评估了EmojiPrompt的性能。我们还提出了模拟推理攻击来评估EmojiPrompt保护用户隐私的能力。结果表明，与非模糊版本相比，EmojiPrompt有效地混淆了用户的私有数据，同时在很大程度上保持了甚至提高了性能。此外，EmojiPrompt的原子级模糊处理使其能够专门与基于云的LLM一起运行。源代码，请参考：https://github.com/agiresearch/EmojiCrypt.



## **26. Immune: Improving Safety Against Jailbreaks in Multi-modal LLMs via Inference-Time Alignment**

免疫：通过推理时间对齐提高多模式LLM中越狱的安全性 cs.CR

Accepted to CVPR 2025

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2411.18688v3) [paper-pdf](http://arxiv.org/pdf/2411.18688v3)

**Authors**: Soumya Suvra Ghosal, Souradip Chakraborty, Vaibhav Singh, Tianrui Guan, Mengdi Wang, Ahmad Beirami, Furong Huang, Alvaro Velasquez, Dinesh Manocha, Amrit Singh Bedi

**Abstract**: With the widespread deployment of Multimodal Large Language Models (MLLMs) for visual-reasoning tasks, improving their safety has become crucial. Recent research indicates that despite training-time safety alignment, these models remain vulnerable to jailbreak attacks. In this work, we first highlight an important safety gap to describe that alignment achieved solely through safety training may be insufficient against jailbreak attacks. To address this vulnerability, we propose Immune, an inference-time defense framework that leverages a safe reward model through controlled decoding to defend against jailbreak attacks. Additionally, we provide a mathematical characterization of Immune, offering insights on why it improves safety against jailbreaks. Extensive evaluations on diverse jailbreak benchmarks using recent MLLMs reveal that Immune effectively enhances model safety while preserving the model's original capabilities. For instance, against text-based jailbreak attacks on LLaVA-1.6, Immune reduces the attack success rate by 57.82% and 16.78% compared to the base MLLM and state-of-the-art defense strategy, respectively.

摘要: 随着多通道大语言模型(MLLMS)在视觉推理任务中的广泛应用，提高其安全性变得至关重要。最近的研究表明，尽管训练时间安全一致，这些模型仍然容易受到越狱攻击。在这项工作中，我们首先强调一个重要的安全差距，以描述仅通过安全培训实现的对准可能不足以抵御越狱攻击。为了解决这一漏洞，我们提出了免疫，这是一个推理时间防御框架，通过受控解码利用安全奖励模型来防御越狱攻击。此外，我们提供了免疫的数学特征，为为什么它提高了抵御越狱的安全性提供了见解。使用最近的MLLMS对不同的越狱基准进行的广泛评估表明，免疫有效地增强了模型的安全性，同时保持了模型的原始能力。例如，对于基于文本的越狱攻击LLaVA-1.6，与基本MLLM和最先进的防御策略相比，免疫分别将攻击成功率降低了57.82%和16.78%。



## **27. Robust LLM safeguarding via refusal feature adversarial training**

通过拒绝功能对抗培训强大的LLM保障 cs.LG

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2409.20089v2) [paper-pdf](http://arxiv.org/pdf/2409.20089v2)

**Authors**: Lei Yu, Virginie Do, Karen Hambardzumyan, Nicola Cancedda

**Abstract**: Large language models (LLMs) are vulnerable to adversarial attacks that can elicit harmful responses. Defending against such attacks remains challenging due to the opacity of jailbreaking mechanisms and the high computational cost of training LLMs robustly. We demonstrate that adversarial attacks share a universal mechanism for circumventing LLM safeguards that works by ablating a dimension in the residual stream embedding space called the refusal feature. We further show that the operation of refusal feature ablation (RFA) approximates the worst-case perturbation of offsetting model safety. Based on these findings, we propose Refusal Feature Adversarial Training (ReFAT), a novel algorithm that efficiently performs LLM adversarial training by simulating the effect of input-level attacks via RFA. Experiment results show that ReFAT significantly improves the robustness of three popular LLMs against a wide range of adversarial attacks, with considerably less computational overhead compared to existing adversarial training methods.

摘要: 大型语言模型(LLM)很容易受到可能引起有害响应的对抗性攻击。由于越狱机制的不透明性和强大训练LLM的高计算成本，防御此类攻击仍然具有挑战性。我们证明了对抗性攻击共享一个通用的机制来规避LLM安全机制，该机制通过在剩余流嵌入空间中消融一个称为拒绝特征的维度来工作。我们进一步证明了拒绝特征消融(RFA)的操作近似于补偿模型安全性的最坏情况的扰动。基于这些发现，我们提出了拒绝特征对抗训练(Refat)，这是一种通过RFA模拟输入级攻击的效果来高效执行LLM对抗训练的新算法。实验结果表明，与现有的对抗性训练方法相比，REFAT显著地提高了三种流行的LLMS对多种对抗性攻击的健壮性，并且具有相当少的计算开销。



## **28. "Moralized" Multi-Step Jailbreak Prompts: Black-Box Testing of Guardrails in Large Language Models for Verbal Attacks**

“道德化”多步骤越狱预言：对大型语言模型中护栏进行黑匣子测试以进行言语攻击 cs.CR

This paper has been submitted to Nature Machine Intelligence and  OpenReview preprints. It has 7 pages of text, 3 figures, and 3 tables

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2411.16730v4) [paper-pdf](http://arxiv.org/pdf/2411.16730v4)

**Authors**: Libo Wang

**Abstract**: As the application of large language models continues to expand in various fields, it poses higher challenges to the effectiveness of identifying harmful content generation and guardrail mechanisms. This research aims to evaluate the guardrail effectiveness of GPT-4o, Grok-2 Beta, Llama 3.1 (405B), Gemini 1.5, and Claude 3.5 Sonnet through black-box testing of seemingly ethical multi-step jailbreak prompts. It conducts ethical attacks by designing an identical multi-step prompts that simulates the scenario of "corporate middle managers competing for promotions." The data results show that the guardrails of the above-mentioned LLMs were bypassed and the content of verbal attacks was generated. Claude 3.5 Sonnet's resistance to multi-step jailbreak prompts is more obvious. To ensure objectivity, the experimental process, black box test code, and enhanced guardrail code are uploaded to the GitHub repository: https://github.com/brucewang123456789/GeniusTrail.git.

摘要: 随着大型语言模型在各个领域的应用不断扩展，对识别有害内容生成和护栏机制的有效性提出了更高的挑战。这项研究旨在通过对看似合乎道德的多步越狱提示进行黑匣子测试来评估GPT-4 o、Grok-2 Beta、Llama 3.1（405 B）、Gemini 1.5和Claude 3.5十四行诗的护栏有效性。它通过设计相同的多步骤提示来进行道德攻击，模拟“企业中层管理人员竞争晋升”的场景。“数据结果显示，上述LLM的护栏被绕过，产生了言语攻击的内容。克劳德3.5十四行诗对多步越狱提示的抵制更加明显。为了确保客观性，实验过程、黑匣子测试代码和增强型护栏代码被上传到GitHub存储库：https://github.com/brucewang123456789/GeniusTrail.git。



## **29. BadToken: Token-level Backdoor Attacks to Multi-modal Large Language Models**

BadToken：对多模式大型语言模型的令牌级后门攻击 cs.CR

This paper is accepted by CVPR 2025

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.16023v1) [paper-pdf](http://arxiv.org/pdf/2503.16023v1)

**Authors**: Zenghui Yuan, Jiawen Shi, Pan Zhou, Neil Zhenqiang Gong, Lichao Sun

**Abstract**: Multi-modal large language models (MLLMs) extend large language models (LLMs) to process multi-modal information, enabling them to generate responses to image-text inputs. MLLMs have been incorporated into diverse multi-modal applications, such as autonomous driving and medical diagnosis, via plug-and-play without fine-tuning. This deployment paradigm increases the vulnerability of MLLMs to backdoor attacks. However, existing backdoor attacks against MLLMs achieve limited effectiveness and stealthiness. In this work, we propose BadToken, the first token-level backdoor attack to MLLMs. BadToken introduces two novel backdoor behaviors: Token-substitution and Token-addition, which enable flexible and stealthy attacks by making token-level modifications to the original output for backdoored inputs. We formulate a general optimization problem that considers the two backdoor behaviors to maximize the attack effectiveness. We evaluate BadToken on two open-source MLLMs and various tasks. Our results show that our attack maintains the model's utility while achieving high attack success rates and stealthiness. We also show the real-world threats of BadToken in two scenarios, i.e., autonomous driving and medical diagnosis. Furthermore, we consider defenses including fine-tuning and input purification. Our results highlight the threat of our attack.

摘要: 多模式大型语言模型(MLLMS)扩展了大型语言模型(LLM)以处理多模式信息，使它们能够生成对图像-文本输入的响应。通过无需微调的即插即用，MLLMS已被整合到各种多模式应用中，如自动驾驶和医疗诊断。这种部署范例增加了MLLMS在后门攻击中的脆弱性。然而，现有的针对MLLMS的后门攻击取得的效果和隐蔽性有限。在这项工作中，我们提出了BadToken，这是对MLLMS的第一个令牌级后门攻击。BadToken引入了两种新的后门行为：令牌替换和令牌添加，这两种行为通过对后端输入的原始输出进行令牌级修改来实现灵活和隐蔽的攻击。我们提出了一个一般的优化问题，该问题考虑了两个后门行为，以最大化攻击效果。我们在两个开源MLLMS和不同的任务上对BadToken进行了评估。结果表明，我们的攻击在保持模型实用性的同时，获得了较高的攻击成功率和隐蔽性。我们还展示了BadToken在两个场景中的真实威胁，即自动驾驶和医疗诊断。此外，我们考虑了包括微调和输入净化在内的防御措施。我们的结果突显了我们攻击的威胁。



## **30. Differentially Private Steering for Large Language Model Alignment**

针对大型语言模型对齐的差异私人指导 cs.CL

ICLR 2025 Camera Ready; Code: https://github.com/UKPLab/iclr2025-psa

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2501.18532v2) [paper-pdf](http://arxiv.org/pdf/2501.18532v2)

**Authors**: Anmol Goel, Yaxi Hu, Iryna Gurevych, Amartya Sanyal

**Abstract**: Aligning Large Language Models (LLMs) with human values and away from undesirable behaviors (such as hallucination) has become increasingly important. Recently, steering LLMs towards a desired behavior via activation editing has emerged as an effective method to mitigate harmful generations at inference-time. Activation editing modifies LLM representations by preserving information from positive demonstrations (e.g., truthful) and minimising information from negative demonstrations (e.g., hallucinations). When these demonstrations come from a private dataset, the aligned LLM may leak private information contained in those private samples. In this work, we present the first study of aligning LLM behavior with private datasets. Our work proposes the Private Steering for LLM Alignment (PSA) algorithm to edit LLM activations with differential privacy (DP) guarantees. We conduct extensive experiments on seven different benchmarks with open-source LLMs of different sizes (0.5B to 7B) and model families (LlaMa, Qwen, Mistral and Gemma). Our results show that PSA achieves DP guarantees for LLM alignment with minimal loss in performance, including alignment metrics, open-ended text generation quality, and general-purpose reasoning. We also develop the first Membership Inference Attack (MIA) for evaluating and auditing the empirical privacy for the problem of LLM steering via activation editing. Our experiments support the theoretical guarantees by showing improved guarantees for our PSA algorithm compared to several existing non-private techniques.

摘要: 使大型语言模型(LLM)与人类价值观保持一致，并远离不良行为(如幻觉)已变得越来越重要。最近，通过激活编辑将LLM引导到期望的行为已经成为减少推理时有害生成的一种有效方法。激活编辑通过保存来自正面演示(例如，真实)的信息以及最小化来自负面演示(例如，幻觉)的信息来修改LLM表示。当这些演示来自私有数据集时，对齐的LLM可能会泄露那些私有样本中包含的私有信息。在这项工作中，我们提出了第一个将LLM行为与私有数据集对齐的研究。我们的工作提出了LLM对齐的私有转向(PSA)算法来编辑具有差分隐私(DP)保证的LLM激活。我们使用不同大小(0.5B到7B)和模型家族(骆驼、Qwen、米斯特拉尔和Gema)的开源LLM在七个不同的基准上进行了广泛的实验。我们的结果表明，PSA在性能损失最小的情况下实现了LLM对齐的DP保证，包括对齐度量、开放式文本生成质量和通用推理。我们还开发了第一个成员推理攻击(MIA)，用于通过激活编辑来评估和审计LLM操控问题的经验隐私。我们的实验表明，与现有的几种非私有技术相比，我们的PSA算法得到了更好的保证，从而支持了理论上的保证。



## **31. REVAL: A Comprehension Evaluation on Reliability and Values of Large Vision-Language Models**

REVAR：大型视觉语言模型可靠性和价值的理解评估 cs.CV

45 pages, 5 figures, 18 tables

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.16566v1) [paper-pdf](http://arxiv.org/pdf/2503.16566v1)

**Authors**: Jie Zhang, Zheng Yuan, Zhongqi Wang, Bei Yan, Sibo Wang, Xiangkui Cao, Zonghui Guo, Shiguang Shan, Xilin Chen

**Abstract**: The rapid evolution of Large Vision-Language Models (LVLMs) has highlighted the necessity for comprehensive evaluation frameworks that assess these models across diverse dimensions. While existing benchmarks focus on specific aspects such as perceptual abilities, cognitive capabilities, and safety against adversarial attacks, they often lack the breadth and depth required to provide a holistic understanding of LVLMs' strengths and limitations. To address this gap, we introduce REVAL, a comprehensive benchmark designed to evaluate the \textbf{RE}liability and \textbf{VAL}ue of LVLMs. REVAL encompasses over 144K image-text Visual Question Answering (VQA) samples, structured into two primary sections: Reliability, which assesses truthfulness (\eg, perceptual accuracy and hallucination tendencies) and robustness (\eg, resilience to adversarial attacks, typographic attacks, and image corruption), and Values, which evaluates ethical concerns (\eg, bias and moral understanding), safety issues (\eg, toxicity and jailbreak vulnerabilities), and privacy problems (\eg, privacy awareness and privacy leakage). We evaluate 26 models, including mainstream open-source LVLMs and prominent closed-source models like GPT-4o and Gemini-1.5-Pro. Our findings reveal that while current LVLMs excel in perceptual tasks and toxicity avoidance, they exhibit significant vulnerabilities in adversarial scenarios, privacy preservation, and ethical reasoning. These insights underscore critical areas for future improvements, guiding the development of more secure, reliable, and ethically aligned LVLMs. REVAL provides a robust framework for researchers to systematically assess and compare LVLMs, fostering advancements in the field.

摘要: 大型视觉语言模型(LVLM)的快速发展突显了从不同维度评估这些模型的综合评估框架的必要性。虽然现有的基准侧重于感知能力、认知能力和对抗对手攻击的安全性等特定方面，但它们往往缺乏提供对LVLMS的优势和局限性的全面了解所需的广度和深度。为了弥补这一差距，我们引入了REVAL，这是一个全面的基准，旨在评估LVLM的责任和价值。REVAL包括144K多个图文视觉问答(VQA)样本，分为两个主要部分：可靠性，评估真实性(例如，感知准确性和幻觉倾向)和稳健性(例如，对对手攻击、排版攻击和图像损坏的恢复能力)；价值观，评估伦理问题(例如，偏见和道德理解)、安全问题(例如，毒性和越狱漏洞)和隐私问题(例如，隐私意识和隐私泄露)。我们评估了26款机型，包括主流开源LVLMS和著名的闭源机型，如GPT-4o和Gemini-1.5-Pro。我们的发现表明，尽管目前的LVLM在感知任务和毒性避免方面表现出色，但它们在对抗场景、隐私保护和伦理推理方面表现出显著的脆弱性。这些见解强调了未来改进的关键领域，指导开发更安全、可靠和符合道德规范的LVLM。REVAL为研究人员提供了一个强大的框架来系统地评估和比较LVLM，促进了该领域的进步。



## **32. SAUCE: Selective Concept Unlearning in Vision-Language Models with Sparse Autoencoders**

SAUCE：使用稀疏自动编码器的视觉语言模型中的选择性概念消除 cs.CV

More comparative experiments are needed

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.14530v2) [paper-pdf](http://arxiv.org/pdf/2503.14530v2)

**Authors**: Qing Li, Jiahui Geng, Derui Zhu, Fengyu Cai, Chenyang Lyu, Fakhri Karray

**Abstract**: Unlearning methods for vision-language models (VLMs) have primarily adapted techniques from large language models (LLMs), relying on weight updates that demand extensive annotated forget sets. Moreover, these methods perform unlearning at a coarse granularity, often leading to excessive forgetting and reduced model utility. To address this issue, we introduce SAUCE, a novel method that leverages sparse autoencoders (SAEs) for fine-grained and selective concept unlearning in VLMs. Briefly, SAUCE first trains SAEs to capture high-dimensional, semantically rich sparse features. It then identifies the features most relevant to the target concept for unlearning. During inference, it selectively modifies these features to suppress specific concepts while preserving unrelated information. We evaluate SAUCE on two distinct VLMs, LLaVA-v1.5-7B and LLaMA-3.2-11B-Vision-Instruct, across two types of tasks: concrete concept unlearning (objects and sports scenes) and abstract concept unlearning (emotions, colors, and materials), encompassing a total of 60 concepts. Extensive experiments demonstrate that SAUCE outperforms state-of-the-art methods by 18.04% in unlearning quality while maintaining comparable model utility. Furthermore, we investigate SAUCE's robustness against widely used adversarial attacks, its transferability across models, and its scalability in handling multiple simultaneous unlearning requests. Our findings establish SAUCE as an effective and scalable solution for selective concept unlearning in VLMs.

摘要: 视觉语言模型(VLM)的遗忘方法主要采用来自大型语言模型(LLM)的技术，依赖于需要大量注释遗忘集的权重更新。此外，这些方法在粗粒度上执行遗忘，经常导致过度遗忘和降低模型效用。为了解决这个问题，我们引入了SASE，这是一种新的方法，它利用稀疏自动编码器(SAE)在VLM中进行细粒度和选择性的概念遗忘。简而言之，SASE首先训练SAE捕获高维的、语义丰富的稀疏特征。然后确定与目标概念最相关的特征以进行遗忘。在推理过程中，它有选择地修改这些特征以抑制特定概念，同时保留不相关的信息。我们在两个不同的VLM，LLaVA-v1.5-7B和Llama-3.2-11B-Vision-Indict上评估SAUE，跨越两种类型的任务：具体概念遗忘(物体和运动场景)和抽象概念遗忘(情绪、颜色和材料)，总共包含60个概念。大量的实验表明，在保持可比的模型效用的情况下，SASE在遗忘质量方面比最先进的方法高出18.04%。此外，我们还研究了SASE对广泛使用的敌意攻击的健壮性、其跨模型的可转移性以及其在处理多个并发遗忘请求时的可扩展性。我们的研究结果表明，SASE是一种有效且可扩展的解决方案，可用于解决VLMS中的选择性概念遗忘问题。



## **33. DroidTTP: Mapping Android Applications with TTP for Cyber Threat Intelligence**

DroidTTP：使用TTP映射Android应用程序以实现网络威胁情报 cs.CR

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.15866v1) [paper-pdf](http://arxiv.org/pdf/2503.15866v1)

**Authors**: Dincy R Arikkat, Vinod P., Rafidha Rehiman K. A., Serena Nicolazzo, Marco Arazzi, Antonino Nocera, Mauro Conti

**Abstract**: The widespread adoption of Android devices for sensitive operations like banking and communication has made them prime targets for cyber threats, particularly Advanced Persistent Threats (APT) and sophisticated malware attacks. Traditional malware detection methods rely on binary classification, failing to provide insights into adversarial Tactics, Techniques, and Procedures (TTPs). Understanding malware behavior is crucial for enhancing cybersecurity defenses. To address this gap, we introduce DroidTTP, a framework mapping Android malware behaviors to TTPs based on the MITRE ATT&CK framework. Our curated dataset explicitly links MITRE TTPs to Android applications. We developed an automated solution leveraging the Problem Transformation Approach (PTA) and Large Language Models (LLMs) to map applications to both Tactics and Techniques. Additionally, we employed Retrieval-Augmented Generation (RAG) with prompt engineering and LLM fine-tuning for TTP predictions. Our structured pipeline includes dataset creation, hyperparameter tuning, data augmentation, feature selection, model development, and SHAP-based model interpretability. Among LLMs, Llama achieved the highest performance in Tactic classification with a Jaccard Similarity of 0.9583 and Hamming Loss of 0.0182, and in Technique classification with a Jaccard Similarity of 0.9348 and Hamming Loss of 0.0127. However, the Label Powerset XGBoost model outperformed LLMs, achieving a Jaccard Similarity of 0.9893 for Tactic classification and 0.9753 for Technique classification, with a Hamming Loss of 0.0054 and 0.0050, respectively. While XGBoost showed superior performance, the narrow margin highlights the potential of LLM-based approaches in TTP classification.

摘要: Android设备广泛用于银行和通信等敏感操作，使其成为网络威胁的主要目标，特别是高级持久性威胁(APT)和复杂的恶意软件攻击。传统的恶意软件检测方法依赖于二进制分类，无法提供对敌对战术、技术和过程(TTP)的洞察。了解恶意软件行为对于加强网络安全防御至关重要。为了弥补这一差距，我们在MITRE ATT&CK框架的基础上引入了DroidTTP，一个将Android恶意软件行为映射到TTP的框架。我们精心挑选的数据集明确地将MITRE TTP链接到Android应用程序。我们开发了一个自动化解决方案，利用问题转换方法(PTA)和大型语言模型(LLM)将应用程序映射到战术和技术。此外，我们使用了具有即时工程和LLM微调的检索-增强生成(RAG)来进行TTP预测。我们的结构化流程包括数据集创建、超参数调整、数据增强、特征选择、模型开发和基于Shap的模型可解释性。在LLMS中，大羊驼在战术分类上表现最好，贾卡德相似度为0.9583，Hamming损失为0.0182；在技术分类上表现最好，Jaccard相似度为0.9348，Hamming损失为0.0127。然而，标签Powerset XGBoost模型的表现优于LLMS，战术分类的Jaccard相似度为0.9893，技术分类的Jaccard相似度为0.9753，Hamming损失分别为0.0054和0.0050。虽然XGBoost表现出了优越的性能，但狭窄的差距突显了基于LLM的方法在TTP分类中的潜力。



## **34. AutoRedTeamer: Autonomous Red Teaming with Lifelong Attack Integration**

AutoRedTeamer：具有终身攻击集成的自主红色团队 cs.CR

**SubmitDate**: 2025-03-20    [abs](http://arxiv.org/abs/2503.15754v1) [paper-pdf](http://arxiv.org/pdf/2503.15754v1)

**Authors**: Andy Zhou, Kevin Wu, Francesco Pinto, Zhaorun Chen, Yi Zeng, Yu Yang, Shuang Yang, Sanmi Koyejo, James Zou, Bo Li

**Abstract**: As large language models (LLMs) become increasingly capable, security and safety evaluation are crucial. While current red teaming approaches have made strides in assessing LLM vulnerabilities, they often rely heavily on human input and lack comprehensive coverage of emerging attack vectors. This paper introduces AutoRedTeamer, a novel framework for fully automated, end-to-end red teaming against LLMs. AutoRedTeamer combines a multi-agent architecture with a memory-guided attack selection mechanism to enable continuous discovery and integration of new attack vectors. The dual-agent framework consists of a red teaming agent that can operate from high-level risk categories alone to generate and execute test cases and a strategy proposer agent that autonomously discovers and implements new attacks by analyzing recent research. This modular design allows AutoRedTeamer to adapt to emerging threats while maintaining strong performance on existing attack vectors. We demonstrate AutoRedTeamer's effectiveness across diverse evaluation settings, achieving 20% higher attack success rates on HarmBench against Llama-3.1-70B while reducing computational costs by 46% compared to existing approaches. AutoRedTeamer also matches the diversity of human-curated benchmarks in generating test cases, providing a comprehensive, scalable, and continuously evolving framework for evaluating the security of AI systems.

摘要: 随着大型语言模型(LLM)的能力越来越强，安全和安全评估变得至关重要。虽然当前的红色团队方法在评估LLM漏洞方面取得了很大进展，但它们往往严重依赖人力投入，并且缺乏对新兴攻击媒介的全面覆盖。本文介绍了AutoRedTeamer，这是一个针对LLMS的全自动化、端到端的Red Teamer框架。AutoRedTeamer将多代理体系结构与内存引导的攻击选择机制相结合，以实现对新攻击载体的持续发现和集成。该双代理框架由红色团队代理和策略提出者代理组成，前者可以从高级别风险类别单独操作来生成和执行测试用例，后者通过分析最近的研究自主地发现和实现新的攻击。这种模块化设计使AutoRedTeamer能够适应新出现的威胁，同时保持对现有攻击媒介的强大性能。我们展示了AutoRedTeamer在不同评估环境中的有效性，与现有方法相比，在HarmBch上对Llama-3.1-70B的攻击成功率提高了20%，同时计算成本降低了46%。AutoRedTeamer还在生成测试用例时匹配了人工管理的基准测试的多样性，为评估AI系统的安全性提供了一个全面的、可扩展的和不断发展的框架。



## **35. Undesirable Memorization in Large Language Models: A Survey**

大型语言模型中不可取的并行化：一项调查 cs.CL

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2410.02650v2) [paper-pdf](http://arxiv.org/pdf/2410.02650v2)

**Authors**: Ali Satvaty, Suzan Verberne, Fatih Turkmen

**Abstract**: While recent research increasingly showcases the remarkable capabilities of Large Language Models (LLMs), it is equally crucial to examine their associated risks. Among these, privacy and security vulnerabilities are particularly concerning, posing significant ethical and legal challenges. At the heart of these vulnerabilities stands memorization, which refers to a model's tendency to store and reproduce phrases from its training data. This phenomenon has been shown to be a fundamental source to various privacy and security attacks against LLMs. In this paper, we provide a taxonomy of the literature on LLM memorization, exploring it across three dimensions: granularity, retrievability, and desirability. Next, we discuss the metrics and methods used to quantify memorization, followed by an analysis of the causes and factors that contribute to memorization phenomenon. We then explore strategies that are used so far to mitigate the undesirable aspects of this phenomenon. We conclude our survey by identifying potential research topics for the near future, including methods to balance privacy and performance, and the analysis of memorization in specific LLM contexts such as conversational agents, retrieval-augmented generation, and diffusion language models. Given the rapid research pace in this field, we also maintain a dedicated repository of the references discussed in this survey which will be regularly updated to reflect the latest developments.

摘要: 虽然最近的研究越来越多地展示了大型语言模型(LLM)的非凡能力，但检查其相关风险也同样至关重要。其中，隐私和安全漏洞尤其令人担忧，构成了重大的道德和法律挑战。这些漏洞的核心是记忆，它指的是模型从其训练数据中存储和复制短语的倾向。这一现象已被证明是针对LLMS的各种隐私和安全攻击的基本来源。本文对有关LLM记忆的文献进行了分类，从粒度、可检索性和可取性三个维度对其进行了探讨。接下来，我们讨论了量化记忆的指标和方法，并分析了造成记忆现象的原因和因素。然后，我们探讨到目前为止用来缓解这一现象的不良方面的策略。我们通过确定在不久的将来潜在的研究主题来结束我们的调查，包括平衡隐私和性能的方法，以及在特定的LLM环境中对记忆的分析，例如会话代理、提取-增强生成和扩散语言模型。鉴于这一领域的研究步伐很快，我们还设有一个专门的资料库，储存本次调查中讨论的参考资料，并将定期更新，以反映最新的发展。



## **36. Safety at Scale: A Comprehensive Survey of Large Model Safety**

大规模安全性：大型车型安全性全面调查 cs.CR

47 pages, 3 figures, 11 tables; GitHub:  https://github.com/xingjunm/Awesome-Large-Model-Safety

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2502.05206v3) [paper-pdf](http://arxiv.org/pdf/2502.05206v3)

**Authors**: Xingjun Ma, Yifeng Gao, Yixu Wang, Ruofan Wang, Xin Wang, Ye Sun, Yifan Ding, Hengyuan Xu, Yunhao Chen, Yunhan Zhao, Hanxun Huang, Yige Li, Jiaming Zhang, Xiang Zheng, Yang Bai, Zuxuan Wu, Xipeng Qiu, Jingfeng Zhang, Yiming Li, Xudong Han, Haonan Li, Jun Sun, Cong Wang, Jindong Gu, Baoyuan Wu, Siheng Chen, Tianwei Zhang, Yang Liu, Mingming Gong, Tongliang Liu, Shirui Pan, Cihang Xie, Tianyu Pang, Yinpeng Dong, Ruoxi Jia, Yang Zhang, Shiqing Ma, Xiangyu Zhang, Neil Gong, Chaowei Xiao, Sarah Erfani, Tim Baldwin, Bo Li, Masashi Sugiyama, Dacheng Tao, James Bailey, Yu-Gang Jiang

**Abstract**: The rapid advancement of large models, driven by their exceptional abilities in learning and generalization through large-scale pre-training, has reshaped the landscape of Artificial Intelligence (AI). These models are now foundational to a wide range of applications, including conversational AI, recommendation systems, autonomous driving, content generation, medical diagnostics, and scientific discovery. However, their widespread deployment also exposes them to significant safety risks, raising concerns about robustness, reliability, and ethical implications. This survey provides a systematic review of current safety research on large models, covering Vision Foundation Models (VFMs), Large Language Models (LLMs), Vision-Language Pre-training (VLP) models, Vision-Language Models (VLMs), Diffusion Models (DMs), and large-model-based Agents. Our contributions are summarized as follows: (1) We present a comprehensive taxonomy of safety threats to these models, including adversarial attacks, data poisoning, backdoor attacks, jailbreak and prompt injection attacks, energy-latency attacks, data and model extraction attacks, and emerging agent-specific threats. (2) We review defense strategies proposed for each type of attacks if available and summarize the commonly used datasets and benchmarks for safety research. (3) Building on this, we identify and discuss the open challenges in large model safety, emphasizing the need for comprehensive safety evaluations, scalable and effective defense mechanisms, and sustainable data practices. More importantly, we highlight the necessity of collective efforts from the research community and international collaboration. Our work can serve as a useful reference for researchers and practitioners, fostering the ongoing development of comprehensive defense systems and platforms to safeguard AI models.

摘要: 大型模型的快速发展，受到其通过大规模预训练而具有的非凡学习和泛化能力的推动，重塑了人工智能(AI)的版图。这些模型现在是广泛应用的基础，包括对话式人工智能、推荐系统、自动驾驶、内容生成、医疗诊断和科学发现。然而，它们的广泛部署也使它们面临重大的安全风险，引发了人们对健壮性、可靠性和道德影响的担忧。本调查系统地回顾了当前关于大模型的安全研究，包括视觉基础模型(VFM)、大语言模型(LLMS)、视觉语言预训练(VLP)模型、视觉语言模型(VLMS)、扩散模型(DM)和基于大模型的代理。我们的工作总结如下：(1)对这些模型的安全威胁进行了全面的分类，包括对抗性攻击、数据中毒、后门攻击、越狱和快速注入攻击、能量延迟攻击、数据和模型提取攻击以及新出现的特定于代理的威胁。(2)我们回顾了针对每种攻击类型提出的防御策略(如果可用)，并总结了安全研究常用的数据集和基准。(3)在此基础上，我们确定并讨论了大型模型安全方面的开放挑战，强调需要全面的安全评估、可扩展和有效的防御机制以及可持续的数据实践。更重要的是，我们强调了研究界和国际合作集体努力的必要性。我们的工作可以作为研究人员和从业者的有用参考，促进正在进行的全面防御系统和平台的开发，以保护人工智能模型。



## **37. Assessing AI vs Human-Authored Spear Phishing SMS Attacks: An Empirical Study**

评估人工智能与人类发起的鱼叉网络钓鱼短信攻击：一项实证研究 cs.CY

18 pages, 5 figures, 1 table

**SubmitDate**: 2025-03-19    [abs](http://arxiv.org/abs/2406.13049v2) [paper-pdf](http://arxiv.org/pdf/2406.13049v2)

**Authors**: Jerson Francia, Derek Hansen, Ben Schooley, Matthew Taylor, Shydra Murray, Greg Snow

**Abstract**: This paper explores the use of Large Language Models (LLMs) in spear phishing message generation and evaluates their performance compared to human-authored counterparts. Our pilot study examines the effectiveness of smishing (SMS phishing) messages created by GPT-4 and human authors, which have been personalized for willing targets. The targets assessed these messages in a modified ranked-order experiment using a novel methodology we call TRAPD (Threshold Ranking Approach for Personalized Deception). Experiments involved ranking each spear phishing message from most to least convincing, providing qualitative feedback, and guessing which messages were human- or AI-generated. Results show that LLM-generated messages are often perceived as more convincing than those authored by humans, particularly job-related messages. Targets also struggled to distinguish between human- and AI-generated messages. We analyze different criteria the targets used to assess the persuasiveness and source of messages. This study aims to highlight the urgent need for further research and improved countermeasures against personalized AI-enabled social engineering attacks.

摘要: 本文探讨了大型语言模型(LLM)在鱼叉式网络钓鱼消息生成中的使用，并与人类创作的同类消息进行了比较。我们的试点研究检查了GPT-4和人类作者创建的Smsing(短信钓鱼)消息的有效性，这些消息已经针对愿意攻击的目标进行了个性化。目标在一种改进的排序实验中使用了一种新的方法来评估这些消息，我们称之为TRAPD(个性化欺骗的阈值排序方法)。实验包括从最令人信服的到最不令人信服的顺序对每条鱼叉式网络钓鱼消息进行排序，提供定性反馈，并猜测哪些消息是人类或人工智能生成的。结果表明，LLM生成的消息通常被认为比人类创作的消息更具说服力，特别是与工作有关的消息。目标也很难区分人类和人工智能生成的消息。我们分析了目标用来评估信息的说服力和来源的不同标准。这项研究旨在突出针对个性化人工智能启用的社会工程攻击的进一步研究和改进对策的迫切需要。



## **38. Temporal Context Awareness: A Defense Framework Against Multi-turn Manipulation Attacks on Large Language Models**

时间上下文感知：针对大型语言模型多轮操纵攻击的防御框架 cs.CR

6 pages, 2 figures, IEEE CAI

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.15560v1) [paper-pdf](http://arxiv.org/pdf/2503.15560v1)

**Authors**: Prashant Kulkarni, Assaf Namer

**Abstract**: Large Language Models (LLMs) are increasingly vulnerable to sophisticated multi-turn manipulation attacks, where adversaries strategically build context through seemingly benign conversational turns to circumvent safety measures and elicit harmful or unauthorized responses. These attacks exploit the temporal nature of dialogue to evade single-turn detection methods, representing a critical security vulnerability with significant implications for real-world deployments.   This paper introduces the Temporal Context Awareness (TCA) framework, a novel defense mechanism designed to address this challenge by continuously analyzing semantic drift, cross-turn intention consistency and evolving conversational patterns. The TCA framework integrates dynamic context embedding analysis, cross-turn consistency verification, and progressive risk scoring to detect and mitigate manipulation attempts effectively. Preliminary evaluations on simulated adversarial scenarios demonstrate the framework's potential to identify subtle manipulation patterns often missed by traditional detection techniques, offering a much-needed layer of security for conversational AI systems. In addition to outlining the design of TCA , we analyze diverse attack vectors and their progression across multi-turn conversation, providing valuable insights into adversarial tactics and their impact on LLM vulnerabilities. Our findings underscore the pressing need for robust, context-aware defenses in conversational AI systems and highlight TCA framework as a promising direction for securing LLMs while preserving their utility in legitimate applications. We make our implementation available to support further research in this emerging area of AI security.

摘要: 大型语言模型(LLM)越来越容易受到复杂的多回合操纵攻击，在这种攻击中，对手通过看似良性的对话转向来策略性地构建上下文，以绕过安全措施并引发有害或未经授权的响应。这些攻击利用对话的时间性来逃避单轮检测方法，这是一个严重的安全漏洞，对现实世界的部署具有重大影响。本文介绍了时态语境感知(TCA)框架，这是一种新的防御机制，旨在通过不断分析语义漂移、跨话轮意图一致性和会话模式演变来应对这一挑战。TCA框架集成了动态上下文嵌入分析、跨回合一致性验证和渐进式风险评分，以有效检测和减少操纵企图。对模拟对抗性场景的初步评估表明，该框架有可能识别传统检测技术经常遗漏的细微操纵模式，为对话式人工智能系统提供了亟需的安全层。除了概述TCA的设计之外，我们还分析了不同的攻击向量及其在多轮对话中的进展，为敌方战术及其对LLM漏洞的影响提供了有价值的见解。我们的发现强调了在对话式人工智能系统中对强大的、上下文感知的防御的迫切需要，并强调TCA框架是一个有希望的方向，可以在保护LLM的有效性的同时保护其在合法应用中的有效性。我们提供我们的实现，以支持在这一新兴的人工智能安全领域的进一步研究。



## **39. Personalized Attacks of Social Engineering in Multi-turn Conversations -- LLM Agents for Simulation and Detection**

多轮对话中社会工程的个性化攻击-- LLM模拟和检测代理 cs.CR

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.15552v1) [paper-pdf](http://arxiv.org/pdf/2503.15552v1)

**Authors**: Tharindu Kumarage, Cameron Johnson, Jadie Adams, Lin Ai, Matthias Kirchner, Anthony Hoogs, Joshua Garland, Julia Hirschberg, Arslan Basharat, Huan Liu

**Abstract**: The rapid advancement of conversational agents, particularly chatbots powered by Large Language Models (LLMs), poses a significant risk of social engineering (SE) attacks on social media platforms. SE detection in multi-turn, chat-based interactions is considerably more complex than single-instance detection due to the dynamic nature of these conversations. A critical factor in mitigating this threat is understanding the mechanisms through which SE attacks operate, specifically how attackers exploit vulnerabilities and how victims' personality traits contribute to their susceptibility. In this work, we propose an LLM-agentic framework, SE-VSim, to simulate SE attack mechanisms by generating multi-turn conversations. We model victim agents with varying personality traits to assess how psychological profiles influence susceptibility to manipulation. Using a dataset of over 1000 simulated conversations, we examine attack scenarios in which adversaries, posing as recruiters, funding agencies, and journalists, attempt to extract sensitive information. Based on this analysis, we present a proof of concept, SE-OmniGuard, to offer personalized protection to users by leveraging prior knowledge of the victims personality, evaluating attack strategies, and monitoring information exchanges in conversations to identify potential SE attempts.

摘要: 会话代理的快速发展，特别是由大语言模型(LLM)驱动的聊天机器人，构成了社交媒体平台上的社交工程(SE)攻击的巨大风险。基于聊天的多轮交互中的SE检测比单实例检测要复杂得多，这是因为这些对话的动态性质。缓解这一威胁的一个关键因素是了解SE攻击的运作机制，特别是攻击者如何利用漏洞以及受害者的个性特征如何导致他们的易感性。在这项工作中，我们提出了一个LLM代理框架SE-VSim，通过生成多话轮会话来模拟SE攻击机制。我们对具有不同个性特征的受害者代理进行建模，以评估心理特征如何影响操纵的易感性。使用1000多个模拟对话的数据集，我们检查了攻击场景，在这些场景中，伪装成招聘者、资助机构和记者的对手试图提取敏感信息。基于这一分析，我们提出了一个概念证明，SE-OmniGuard，通过利用受害者个性的先验知识，评估攻击策略，并监控对话中的信息交换来识别潜在的SE尝试，为用户提供个性化保护。



## **40. TAROT: Task-Oriented Authorship Obfuscation Using Policy Optimization Methods**

TAROT：使用策略优化方法的面向任务的作者混淆 cs.CL

Accepted to the NAACL PrivateNLP 2025 Workshop

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2407.21630v2) [paper-pdf](http://arxiv.org/pdf/2407.21630v2)

**Authors**: Gabriel Loiseau, Damien Sileo, Damien Riquet, Maxime Meyer, Marc Tommasi

**Abstract**: Authorship obfuscation aims to disguise the identity of an author within a text by altering the writing style, vocabulary, syntax, and other linguistic features associated with the text author. This alteration needs to balance privacy and utility. While strong obfuscation techniques can effectively hide the author's identity, they often degrade the quality and usefulness of the text for its intended purpose. Conversely, maintaining high utility tends to provide insufficient privacy, making it easier for an adversary to de-anonymize the author. Thus, achieving an optimal trade-off between these two conflicting objectives is crucial. In this paper, we propose TAROT: Task-Oriented Authorship Obfuscation Using Policy Optimization, a new unsupervised authorship obfuscation method whose goal is to optimize the privacy-utility trade-off by regenerating the entire text considering its downstream utility. Our approach leverages policy optimization as a fine-tuning paradigm over small language models in order to rewrite texts by preserving author identity and downstream task utility. We show that our approach largely reduces the accuracy of attackers while preserving utility. We make our code and models publicly available.

摘要: 作者身份混淆旨在通过改变与文本作者相关的写作风格、词汇、句法和其他语言特征来掩盖作者在文本中的身份。这一改变需要平衡隐私和效用。虽然强大的混淆技术可以有效地隐藏作者的身份，但它们往往会降低文本的质量和对预期目的的有用性。相反，保持高实用性往往会提供不充分的隐私，使对手更容易解除作者的匿名。因此，在这两个相互冲突的目标之间实现最佳权衡至关重要。在本文中，我们提出了一种新的无监督作者身份混淆方法--TAROT：基于策略优化的面向任务的作者身份混淆方法，其目标是通过重新生成考虑下游效用的整个文本来优化隐私和效用之间的权衡。我们的方法利用策略优化作为小语言模型上的微调范式，以便通过保留作者身份和下游任务效用来重写文本。我们表明，我们的方法在很大程度上降低了攻击者的准确性，同时保持了实用性。我们公开我们的代码和模型。



## **41. Adversarial Training for Multimodal Large Language Models against Jailbreak Attacks**

针对越狱攻击的多模式大型语言模型对抗训练 cs.CV

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.04833v2) [paper-pdf](http://arxiv.org/pdf/2503.04833v2)

**Authors**: Liming Lu, Shuchao Pang, Siyuan Liang, Haotian Zhu, Xiyu Zeng, Aishan Liu, Yunhuai Liu, Yongbin Zhou

**Abstract**: Multimodal large language models (MLLMs) have made remarkable strides in cross-modal comprehension and generation tasks. However, they remain vulnerable to jailbreak attacks, where crafted perturbations bypass security guardrails and elicit harmful outputs. In this paper, we present the first adversarial training (AT) paradigm tailored to defend against jailbreak attacks during the MLLM training phase. Extending traditional AT to this domain poses two critical challenges: efficiently tuning massive parameters and ensuring robustness against attacks across multiple modalities. To address these challenges, we introduce Projection Layer Against Adversarial Training (ProEAT), an end-to-end AT framework. ProEAT incorporates a projector-based adversarial training architecture that efficiently handles large-scale parameters while maintaining computational feasibility by focusing adversarial training on a lightweight projector layer instead of the entire model; additionally, we design a dynamic weight adjustment mechanism that optimizes the loss function's weight allocation based on task demands, streamlining the tuning process. To enhance defense performance, we propose a joint optimization strategy across visual and textual modalities, ensuring robust resistance to jailbreak attacks originating from either modality. Extensive experiments conducted on five major jailbreak attack methods across three mainstream MLLMs demonstrate the effectiveness of our approach. ProEAT achieves state-of-the-art defense performance, outperforming existing baselines by an average margin of +34% across text and image modalities, while incurring only a 1% reduction in clean accuracy. Furthermore, evaluations on real-world embodied intelligent systems highlight the practical applicability of our framework, paving the way for the development of more secure and reliable multimodal systems.

摘要: 多通道大语言模型在跨通道理解和生成任务方面取得了显著进展。然而，它们仍然容易受到越狱攻击，在越狱攻击中，精心设计的扰动绕过安全护栏，引发有害输出。在这篇文章中，我们提出了在MLLM训练阶段为防御越狱攻击而定制的第一个对抗性训练(AT)范例。将传统的AT扩展到这一领域会带来两个关键挑战：有效地调整大量参数和确保对跨多个通道的攻击的健壮性。为了应对这些挑战，我们引入了投影层对抗对手训练(ProEAT)，这是一个端到端的AT框架。ProEAT结合了基于投影仪的对抗性训练体系结构，通过将对抗性训练集中在轻量级投影器层而不是整个模型上，在保持计算可行性的同时有效地处理大规模参数；此外，我们设计了动态权重调整机制，基于任务需求优化损失函数的权重分配，从而简化了调整过程。为了提高防御性能，我们提出了一种跨视觉和文本模式的联合优化策略，确保对来自任何一种模式的越狱攻击具有强大的抵抗力。在三种主流MLLMS上对五种主要的越狱攻击方法进行了广泛的实验，证明了该方法的有效性。ProEAT实现了最先进的防御性能，在文本和图像模式中的表现比现有基线平均高出34%，而干净的准确性仅降低了1%。此外，对真实世界体现的智能系统的评估突出了我们框架的实用适用性，为开发更安全可靠的多式联运系统铺平了道路。



## **42. Survey of Adversarial Robustness in Multimodal Large Language Models**

多模式大型语言模型中的对抗鲁棒性研究 cs.CV

9 pages

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2503.13962v1) [paper-pdf](http://arxiv.org/pdf/2503.13962v1)

**Authors**: Chengze Jiang, Zhuangzhuang Wang, Minjing Dong, Jie Gui

**Abstract**: Multimodal Large Language Models (MLLMs) have demonstrated exceptional performance in artificial intelligence by facilitating integrated understanding across diverse modalities, including text, images, video, audio, and speech. However, their deployment in real-world applications raises significant concerns about adversarial vulnerabilities that could compromise their safety and reliability. Unlike unimodal models, MLLMs face unique challenges due to the interdependencies among modalities, making them susceptible to modality-specific threats and cross-modal adversarial manipulations. This paper reviews the adversarial robustness of MLLMs, covering different modalities. We begin with an overview of MLLMs and a taxonomy of adversarial attacks tailored to each modality. Next, we review key datasets and evaluation metrics used to assess the robustness of MLLMs. After that, we provide an in-depth review of attacks targeting MLLMs across different modalities. Our survey also identifies critical challenges and suggests promising future research directions.

摘要: 多模式大语言模型(MLLM)通过促进跨不同模式的集成理解，包括文本、图像、视频、音频和语音，在人工智能中表现出出色的性能。然而，它们在实际应用程序中的部署引发了人们对可能危及其安全性和可靠性的对抗性漏洞的严重担忧。与单模模型不同，由于各通道之间的相互依赖关系，最大似然模型面临着独特的挑战，这使得它们容易受到特定通道的威胁和跨通道的对抗性操作。本文综述了MLLMS的对抗稳健性，涵盖了不同的模式。我们首先概述MLLMS和针对每种模式量身定做的对抗性攻击分类。接下来，我们回顾了用于评估MLLMS稳健性的关键数据集和评估指标。在此之后，我们提供了针对不同模式的MLLM的攻击的深入回顾。我们的调查还确定了关键挑战，并提出了前景看好的未来研究方向。



## **43. AttackEval: How to Evaluate the Effectiveness of Jailbreak Attacking on Large Language Models**

AttackEval：如何评估越狱攻击对大型语言模型的有效性 cs.CL

Accepted by ACM SIGKDD Explorations 2025

**SubmitDate**: 2025-03-18    [abs](http://arxiv.org/abs/2401.09002v6) [paper-pdf](http://arxiv.org/pdf/2401.09002v6)

**Authors**: Dong Shu, Chong Zhang, Mingyu Jin, Zihao Zhou, Lingyao Li, Yongfeng Zhang

**Abstract**: Jailbreak attacks represent one of the most sophisticated threats to the security of large language models (LLMs). To deal with such risks, we introduce an innovative framework that can help evaluate the effectiveness of jailbreak attacks on LLMs. Unlike traditional binary evaluations focusing solely on the robustness of LLMs, our method assesses the attacking prompts' effectiveness. We present two distinct evaluation frameworks: a coarse-grained evaluation and a fine-grained evaluation. Each framework uses a scoring range from 0 to 1, offering unique perspectives and allowing for the assessment of attack effectiveness in different scenarios. Additionally, we develop a comprehensive ground truth dataset specifically tailored for jailbreak prompts. This dataset is a crucial benchmark for our current study and provides a foundational resource for future research. By comparing with traditional evaluation methods, our study shows that the current results align with baseline metrics while offering a more nuanced and fine-grained assessment. It also helps identify potentially harmful attack prompts that might appear harmless in traditional evaluations. Overall, our work establishes a solid foundation for assessing a broader range of attack prompts in prompt injection.

摘要: 越狱攻击是大型语言模型(LLM)安全面临的最复杂的威胁之一。为了应对这样的风险，我们引入了一个创新的框架，可以帮助评估越狱攻击对低收入者的有效性。与传统的只关注LLMS健壮性的二值评估不同，我们的方法评估了攻击提示的有效性。我们提出了两种不同的评估框架：粗粒度评估和细粒度评估。每个框架使用从0到1的评分范围，提供独特的视角，并允许在不同情况下评估攻击效果。此外，我们还开发了专门为越狱提示量身定做的全面地面事实数据集。该数据集是我们当前研究的重要基准，并为未来的研究提供了基础资源。通过与传统评估方法的比较，我们的研究表明，当前的结果与基线度量一致，同时提供了更细微和细粒度的评估。它还有助于识别在传统评估中可能看起来无害的潜在有害攻击提示。总体而言，我们的工作为在快速注入中评估更广泛的攻击提示奠定了坚实的基础。



## **44. Web Artifact Attacks Disrupt Vision Language Models**

网络收件箱攻击扰乱视觉语言模型 cs.CV

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.13652v1) [paper-pdf](http://arxiv.org/pdf/2503.13652v1)

**Authors**: Maan Qraitem, Piotr Teterwak, Kate Saenko, Bryan A. Plummer

**Abstract**: Vision-language models (VLMs) (e.g., CLIP, LLaVA) are trained on large-scale, lightly curated web datasets, leading them to learn unintended correlations between semantic concepts and unrelated visual signals. These associations degrade model accuracy by causing predictions to rely on incidental patterns rather than genuine visual understanding. Prior work has weaponized these correlations as an attack vector to manipulate model predictions, such as inserting a deceiving class text onto the image in a typographic attack. These attacks succeed due to VLMs' text-heavy bias-a result of captions that echo visible words rather than describing content. However, this attack has focused solely on text that matches the target class exactly, overlooking a broader range of correlations, including non-matching text and graphical symbols, which arise from the abundance of branding content in web-scale data. To address this gap, we introduce artifact-based attacks: a novel class of manipulations that mislead models using both non-matching text and graphical elements. Unlike typographic attacks, these artifacts are not predefined, making them harder to defend against but also more challenging to find. We address this by framing artifact attacks as a search problem and demonstrate their effectiveness across five datasets, with some artifacts reinforcing each other to reach 100% attack success rates. These attacks transfer across models with up to 90% effectiveness, making it possible to attack unseen models. To defend against these attacks, we extend prior work's artifact aware prompting to the graphical setting. We see a moderate reduction of success rates of up to 15% relative to standard prompts, suggesting a promising direction for enhancing model robustness.

摘要: 视觉语言模型(VLM)(例如，CLIP、LLaVA)在大规模、轻度精选的Web数据集上进行训练，导致它们学习语义概念和无关视觉信号之间的意外关联。这些关联导致预测依赖于附带模式而不是真正的视觉理解，从而降低了模型的准确性。以前的工作已经将这些相关性武器化为攻击矢量，以操纵模型预测，例如在排版攻击中将欺骗性类文本插入图像。这些攻击之所以成功，是因为VLMS的文本偏向--字幕呼应可见单词而不是描述内容的结果。然而，这种攻击只关注与目标类别完全匹配的文本，而忽略了更广泛的相关性，包括不匹配的文本和图形符号，这些相关性源于网络规模数据中丰富的品牌内容。为了弥补这一差距，我们引入了基于人工产物的攻击：一种使用不匹配文本和图形元素误导模型的新型操作。与排版攻击不同，这些文物不是预定义的，这使得它们更难防御，但也更难找到。我们通过将人工产物攻击作为一个搜索问题来解决这个问题，并在五个数据集上展示了它们的有效性，其中一些人工产物相互加强，达到100%的攻击成功率。这些攻击在模型之间转移，效率高达90%，使攻击看不见的模型成为可能。为了防御这些攻击，我们将以前工作的人工产物感知提示扩展到图形设置。我们看到，与标准提示相比，成功率适度降低，最高可达15%，这表明了增强模型稳健性的一个有希望的方向。



## **45. Booster: Tackling Harmful Fine-tuning for Large Language Models via Attenuating Harmful Perturbation**

助推器：通过减弱有害扰动来解决大型语言模型的有害微调 cs.CL

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2409.01586v4) [paper-pdf](http://arxiv.org/pdf/2409.01586v4)

**Authors**: Tiansheng Huang, Sihao Hu, Fatih Ilhan, Selim Furkan Tekin, Ling Liu

**Abstract**: Harmful fine-tuning attack poses serious safety concerns for large language models' fine-tuning-as-a-service. While existing defenses have been proposed to mitigate the issue, their performances are still far away from satisfactory, and the root cause of the problem has not been fully recovered. To this end, we in this paper show that harmful perturbation over the model weights could be a probable cause of alignment-broken. In order to attenuate the negative impact of harmful perturbation, we propose an alignment-stage solution, dubbed Booster. Technically, along with the original alignment loss, we append a loss regularizer in the alignment stage's optimization. The regularizer ensures that the model's harmful loss reduction after the simulated harmful perturbation is attenuated, thereby mitigating the subsequent fine-tuning risk. Empirical results show that Booster can effectively reduce the harmful score of the fine-tuned models while maintaining the performance of downstream tasks. Our code is available at https://github.com/git-disl/Booster.

摘要: 有害的微调攻击给大型语言模型的微调即服务带来了严重的安全问题。虽然已经提出了现有的防御措施来缓解这一问题，但它们的表现仍然远远不能令人满意，问题的根本原因尚未完全恢复。为此，我们在本文中表明，对模型权重的有害扰动可能是排列断裂的一个可能原因。为了减弱有害扰动的负面影响，我们提出了一种对准阶段的解决方案，称为Booster。在技术上，除了原始的对准损失外，我们还在对准阶段的优化中加入了损失正则化。正则化确保了模型在模拟的有害扰动后的有害损失减小，从而减轻了随后的微调风险。实验结果表明，Booster在保持下游任务性能的同时，能有效降低微调模型的有害分数。我们的代码可以在https://github.com/git-disl/Booster.上找到



## **46. A Framework to Assess Multilingual Vulnerabilities of LLMs**

评估法学硕士多语言脆弱性的框架 cs.CL

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.13081v1) [paper-pdf](http://arxiv.org/pdf/2503.13081v1)

**Authors**: Likai Tang, Niruth Bogahawatta, Yasod Ginige, Jiarui Xu, Shixuan Sun, Surangika Ranathunga, Suranga Seneviratne

**Abstract**: Large Language Models (LLMs) are acquiring a wider range of capabilities, including understanding and responding in multiple languages. While they undergo safety training to prevent them from answering illegal questions, imbalances in training data and human evaluation resources can make these models more susceptible to attacks in low-resource languages (LRL). This paper proposes a framework to automatically assess the multilingual vulnerabilities of commonly used LLMs. Using our framework, we evaluated six LLMs across eight languages representing varying levels of resource availability. We validated the assessments generated by our automated framework through human evaluation in two languages, demonstrating that the framework's results align with human judgments in most cases. Our findings reveal vulnerabilities in LRL; however, these may pose minimal risk as they often stem from the model's poor performance, resulting in incoherent responses.

摘要: 大型语言模型（LLM）正在获得更广泛的能力，包括以多种语言理解和响应。虽然他们接受安全培训以防止他们回答非法问题，但训练数据和人力评估资源的不平衡可能使这些模型更容易受到低资源语言（LRL）的攻击。本文提出了一个自动评估常用LLM多语言漏洞的框架。使用我们的框架，我们评估了八种语言的六个LLM，代表不同级别的资源可用性。我们通过两种语言的人工评估验证了自动化框架生成的评估，证明框架的结果在大多数情况下与人类判断一致。我们的研究结果揭示了LRL中的漏洞;然而，这些漏洞可能构成的风险很小，因为它们通常源于模型的性能不佳，导致响应不一致。



## **47. TuBA: Cross-Lingual Transferability of Backdoor Attacks in LLMs with Instruction Tuning**

TuBA：具有指令调优的LLM后门攻击的跨语言可转移性 cs.CL

work in progress

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2404.19597v3) [paper-pdf](http://arxiv.org/pdf/2404.19597v3)

**Authors**: Xuanli He, Jun Wang, Qiongkai Xu, Pasquale Minervini, Pontus Stenetorp, Benjamin I. P. Rubinstein, Trevor Cohn

**Abstract**: The implications of backdoor attacks on English-centric large language models (LLMs) have been widely examined - such attacks can be achieved by embedding malicious behaviors during training and activated under specific conditions that trigger malicious outputs. Despite the increasing support for multilingual capabilities in open-source and proprietary LLMs, the impact of backdoor attacks on these systems remains largely under-explored. Our research focuses on cross-lingual backdoor attacks against multilingual LLMs, particularly investigating how poisoning the instruction-tuning data for one or two languages can affect the outputs for languages whose instruction-tuning data were not poisoned. Despite its simplicity, our empirical analysis reveals that our method exhibits remarkable efficacy in models like mT5 and GPT-4o, with high attack success rates, surpassing 90% in more than 7 out of 12 languages across various scenarios. Our findings also indicate that more powerful models show increased susceptibility to transferable cross-lingual backdoor attacks, which also applies to LLMs predominantly pre-trained on English data, such as Llama2, Llama3, and Gemma. Moreover, our experiments demonstrate 1) High Transferability: the backdoor mechanism operates successfully in cross-lingual response scenarios across 26 languages, achieving an average attack success rate of 99%, and 2) Robustness: the proposed attack remains effective even after defenses are applied. These findings expose critical security vulnerabilities in multilingual LLMs and highlight the urgent need for more robust, targeted defense strategies to address the unique challenges posed by cross-lingual backdoor transfer.

摘要: 后门攻击对以英语为中心的大型语言模型(LLM)的影响已被广泛研究-此类攻击可以通过在训练期间嵌入恶意行为来实现，并在触发恶意输出的特定条件下激活。尽管在开源和专有LLM中对多语言功能的支持越来越多，但后门攻击对这些系统的影响在很大程度上仍然没有得到充分的探索。我们的研究重点是针对多语言LLM的跨语言后门攻击，特别是调查毒化一到两种语言的指令调整数据如何影响那些指令调整数据没有中毒的语言的输出。尽管简单，但我们的经验分析表明，我们的方法在MT5和GPT-40等模型中表现出了显著的效果，在不同的场景下，在12种语言中的7种以上的攻击成功率超过90%。我们的发现还表明，更强大的模型显示出对可转移的跨语言后门攻击的易感性，这也适用于主要根据英语数据进行预训练的LLM，如Llama2、Llama3和Gema。此外，我们的实验表明：1)高可移植性：后门机制在26种语言的跨语言响应场景中成功运行，平均攻击成功率为99%；2)健壮性：所提出的攻击即使在实施防御后仍有效。这些发现暴露了多语言低成本管理中的关键安全漏洞，并突显了迫切需要更强大、更有针对性的防御战略，以应对跨语言后门转移带来的独特挑战。



## **48. How Good is my Histopathology Vision-Language Foundation Model? A Holistic Benchmark**

我的组织学视觉语言基础模型有多好？整体基准 eess.IV

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.12990v1) [paper-pdf](http://arxiv.org/pdf/2503.12990v1)

**Authors**: Roba Al Majzoub, Hashmat Malik, Muzammal Naseer, Zaigham Zaheer, Tariq Mahmood, Salman Khan, Fahad Khan

**Abstract**: Recently, histopathology vision-language foundation models (VLMs) have gained popularity due to their enhanced performance and generalizability across different downstream tasks. However, most existing histopathology benchmarks are either unimodal or limited in terms of diversity of clinical tasks, organs, and acquisition instruments, as well as their partial availability to the public due to patient data privacy. As a consequence, there is a lack of comprehensive evaluation of existing histopathology VLMs on a unified benchmark setting that better reflects a wide range of clinical scenarios. To address this gap, we introduce HistoVL, a fully open-source comprehensive benchmark comprising images acquired using up to 11 various acquisition tools that are paired with specifically crafted captions by incorporating class names and diverse pathology descriptions. Our Histo-VL includes 26 organs, 31 cancer types, and a wide variety of tissue obtained from 14 heterogeneous patient cohorts, totaling more than 5 million patches obtained from over 41K WSIs viewed under various magnification levels. We systematically evaluate existing histopathology VLMs on Histo-VL to simulate diverse tasks performed by experts in real-world clinical scenarios. Our analysis reveals interesting findings, including large sensitivity of most existing histopathology VLMs to textual changes with a drop in balanced accuracy of up to 25% in tasks such as Metastasis detection, low robustness to adversarial attacks, as well as improper calibration of models evident through high ECE values and low model prediction confidence, all of which can affect their clinical implementation.

摘要: 最近，组织病理学视觉-语言基础模型(VLM)因其在不同下游任务中增强的性能和普适性而广受欢迎。然而，现有的大多数组织病理学基准要么是单一的，要么在临床任务、器官和采集工具的多样性方面受到限制，而且由于患者数据的隐私，它们对公众的部分可用性。因此，在一个统一的基准设置上缺乏对现有组织病理学VLM的全面评估，以更好地反映广泛的临床情景。为了弥补这一差距，我们引入了HistoVL，这是一个完全开源的综合基准，包括使用多达11种不同的采集工具获取的图像，这些工具通过结合类名和不同的病理描述与专门制作的说明相匹配。我们的HISTO-VL包括26个器官，31种癌症类型，以及从14个不同的患者队列中获得的各种组织，总计超过500万个斑块，这些斑块是在不同放大水平下从超过41K的WSIS中获得的。我们系统地评估HISTO-VL上现有的组织病理学VLM，以模拟真实世界临床场景中专家执行的各种任务。我们的分析揭示了有趣的发现，包括大多数现有的组织病理学VLM对文本变化的高度敏感性，在转移检测等任务中平衡准确率下降高达25%，对对抗性攻击的稳健性低，以及通过高EC值和低模型预测置信度明显地对模型进行不正确的校准，所有这些都可能影响其临床实施。



## **49. MirrorGuard: Adaptive Defense Against Jailbreaks via Entropy-Guided Mirror Crafting**

收件箱警卫：通过熵引导镜子制作对越狱的自适应防御 cs.CR

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.12931v1) [paper-pdf](http://arxiv.org/pdf/2503.12931v1)

**Authors**: Rui Pu, Chaozhuo Li, Rui Ha, Litian Zhang, Lirong Qiu, Xi Zhang

**Abstract**: Defending large language models (LLMs) against jailbreak attacks is crucial for ensuring their safe deployment. Existing defense strategies generally rely on predefined static criteria to differentiate between harmful and benign prompts. However, such rigid rules are incapable of accommodating the inherent complexity and dynamic nature of real jailbreak attacks. In this paper, we propose a novel concept of ``mirror'' to enable dynamic and adaptive defense. A mirror refers to a dynamically generated prompt that mirrors the syntactic structure of the input while ensuring semantic safety. The personalized discrepancies between the input prompts and their corresponding mirrors serve as the guiding principles for defense. A new defense paradigm, MirrorGuard, is further proposed to detect and calibrate risky inputs based on such mirrors. An entropy-based detection metric, Relative Input Uncertainty (RIU), is integrated into MirrorGuard to quantify the discrepancies between input prompts and mirrors. MirrorGuard is evaluated on several popular datasets, demonstrating state-of-the-art defense performance while maintaining general effectiveness.

摘要: 保护大型语言模型(LLM)免受越狱攻击对于确保它们的安全部署至关重要。现有的防御策略通常依赖于预定义的静态标准来区分有害提示和良性提示。然而，这种僵化的规则无法适应真正越狱攻击的内在复杂性和动态性质。本文提出了一种新的“镜像”概念，以实现动态和自适应的防御。镜像是指动态生成的提示，它反映了输入的句法结构，同时确保了语义安全。输入提示与其对应的镜像之间的个性化差异是答辩的指导原则。此外，还提出了一种新的防御范例MirrorGuard，用于检测和校准基于此类镜像的危险输入。基于熵的检测指标相对输入不确定性(RIU)被集成到MirrorGuard中，以量化输入提示和镜像之间的差异。MirrorGuard在几个流行的数据集上进行了评估，在保持总体有效性的同时展示了最先进的防御性能。



## **50. Prompt Flow Integrity to Prevent Privilege Escalation in LLM Agents**

提示流程完整性以防止LLM代理的特权升级 cs.CR

**SubmitDate**: 2025-03-17    [abs](http://arxiv.org/abs/2503.15547v1) [paper-pdf](http://arxiv.org/pdf/2503.15547v1)

**Authors**: Juhee Kim, Woohyuk Choi, Byoungyoung Lee

**Abstract**: Large Language Models (LLMs) are combined with plugins to create powerful LLM agents that provide a wide range of services. Unlike traditional software, LLM agent's behavior is determined at runtime by natural language prompts from either user or plugin's data. This flexibility enables a new computing paradigm with unlimited capabilities and programmability, but also introduces new security risks, vulnerable to privilege escalation attacks. Moreover, user prompt is prone to be interpreted in an insecure way by LLM agents, creating non-deterministic behaviors that can be exploited by attackers. To address these security risks, we propose Prompt Flow Integrity (PFI), a system security-oriented solution to prevent privilege escalation in LLM agents. Analyzing the architectural characteristics of LLM agents, PFI features three mitigation techniques -- i.e., untrusted data identification, enforcing least privilege on LLM agents, and validating unsafe data flows. Our evaluation result shows that PFI effectively mitigates privilege escalation attacks while successfully preserving the utility of LLM agents.

摘要: 大型语言模型(LLM)与插件相结合，以创建功能强大的LLM代理，提供广泛的服务。与传统软件不同，LLM代理的行为在运行时由用户或插件数据的自然语言提示决定。这种灵活性实现了具有无限功能和可编程性的新计算模式，但也带来了新的安全风险，容易受到权限提升攻击。此外，用户提示容易被LLM代理以不安全的方式解释，从而产生可被攻击者利用的不确定行为。为了解决这些安全风险，我们提出了即时流完整性(PFI)，这是一种面向系统安全的解决方案，以防止LLM代理中的权限提升。分析LLM代理的体系结构特征，PFI具有三种缓解技术--即不可信数据识别、在LLM代理上实施最小特权和验证不安全数据流。我们的评估结果表明，PFI在成功保持LLM代理效用的同时，有效地缓解了权限提升攻击。



