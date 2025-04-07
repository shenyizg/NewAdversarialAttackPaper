# Latest Large Language Model Attack Papers
**update at 2025-04-07 09:22:00**

翻译来自 https://cloud.tencent.com/document/product/551/15619

## **1. sudo rm -rf agentic_security**

sudo rm -ref agentic_secure cs.CL

**SubmitDate**: 2025-04-04    [abs](http://arxiv.org/abs/2503.20279v2) [paper-pdf](http://arxiv.org/pdf/2503.20279v2)

**Authors**: Sejin Lee, Jian Kim, Haon Park, Ashkan Yousefpour, Sangyoon Yu, Min Song

**Abstract**: Large Language Models (LLMs) are increasingly deployed as computer-use agents, autonomously performing tasks within real desktop or web environments. While this evolution greatly expands practical use cases for humans, it also creates serious security exposures. We present SUDO (Screen-based Universal Detox2Tox Offense), a novel attack framework that systematically bypasses refusal trained safeguards in commercial computer-use agents, such as Claude Computer Use. The core mechanism, Detox2Tox, transforms harmful requests (that agents initially reject) into seemingly benign requests via detoxification, secures detailed instructions from advanced vision language models (VLMs), and then reintroduces malicious content via toxification just before execution. Unlike conventional jailbreaks, SUDO iteratively refines its attacks based on a built-in refusal feedback, making it increasingly effective against robust policy filters. In extensive tests spanning 50 real-world tasks and multiple state-of-the-art VLMs, SUDO achieves a stark attack success rate of 24% (with no refinement), and up to 41% (by its iterative refinement) in Claude Computer Use. By revealing these vulnerabilities and demonstrating the ease with which they can be exploited in real-world computing environments, this paper highlights an immediate need for robust, context-aware safeguards. WARNING: This paper includes harmful or offensive model outputs Our code is available at: https://github.com/AIM-Intelligence/SUDO.git

摘要: 大型语言模型（LLM）越来越多地被部署为计算机使用代理，在真实桌面或Web环境中自主执行任务。虽然这种演变极大地扩展了人类的实际用例，但也造成了严重的安全风险。我们提出了SUDO（基于屏幕的通用Detox 2 Tox Offense），这是一种新颖的攻击框架，可以系统地绕过商业计算机使用代理（例如Claude Computer Use）中的拒绝训练保护措施。核心机制Detox 2Tox通过解毒将有害请求（代理最初拒绝的请求）转换为看似良性的请求，保护高级视觉语言模型（VLM）的详细指令，然后在执行前通过简化重新引入恶意内容。与传统的越狱不同，SUDO基于内置的拒绝反馈迭代改进其攻击，使其在对抗强大的政策过滤器时变得越来越有效。在涵盖50个现实世界任务和多个最先进的VLM的广泛测试中，SUDO的攻击成功率高达24%（无需改进），在Claude Computer Use中高达41%（通过迭代改进）。通过揭示这些漏洞并展示它们在现实世界计算环境中被利用的轻松性，本文强调了对强大的、上下文感知的保护措施的迫切需求。警告：本文包括有害或冒犯性的模型输出我们的代码可在：https://github.com/AIM-Intelligence/SUDO.git上获取



## **2. Les Dissonances: Cross-Tool Harvesting and Polluting in Multi-Tool Empowered LLM Agents**

Les Dissonance：多工具授权的LLM代理中的跨工具收获和污染 cs.CR

**SubmitDate**: 2025-04-04    [abs](http://arxiv.org/abs/2504.03111v1) [paper-pdf](http://arxiv.org/pdf/2504.03111v1)

**Authors**: Zichuan Li, Jian Cui, Xiaojing Liao, Luyi Xing

**Abstract**: Large Language Model (LLM) agents are autonomous systems powered by LLMs, capable of reasoning and planning to solve problems by leveraging a set of tools. However, the integration of multi-tool capabilities in LLM agents introduces challenges in securely managing tools, ensuring their compatibility, handling dependency relationships, and protecting control flows within LLM agent workflows. In this paper, we present the first systematic security analysis of task control flows in multi-tool-enabled LLM agents. We identify a novel threat, Cross-Tool Harvesting and Polluting (XTHP), which includes multiple attack vectors to first hijack the normal control flows of agent tasks, and then collect and pollute confidential or private information within LLM agent systems. To understand the impact of this threat, we developed Chord, a dynamic scanning tool designed to automatically detect real-world agent tools susceptible to XTHP attacks. Our evaluation of 73 real-world tools from the repositories of two major LLM agent development frameworks, LangChain and LlamaIndex, revealed a significant security concern: 80% of the tools are vulnerable to hijacking attacks, 78% to XTH attacks, and 41% to XTP attacks, highlighting the prevalence of this threat.

摘要: 大型语言模型(LLM)代理是由LLMS提供支持的自治系统，能够通过利用一组工具进行推理和规划来解决问题。然而，LLm代理中多工具功能的集成在安全管理工具、确保其兼容性、处理依赖关系以及保护LLm代理工作流中的控制流方面带来了挑战。在本文中，我们首次系统地分析了多工具使能LLM代理中的任务控制流的安全性。我们识别了一种新的威胁--跨工具收集和污染(XTHP)，它包括多个攻击载体，首先劫持代理任务的正常控制流，然后收集和污染LLM代理系统中的机密或私有信息。为了了解这种威胁的影响，我们开发了Chord，这是一种动态扫描工具，旨在自动检测易受XTHP攻击的真实代理工具。我们评估了来自两个主要的LLM代理开发框架--Lang Chain和LlamaIndex--存储库中的73个实际工具，发现了一个重要的安全问题：80%的工具易受劫持攻击，78%易受XTH攻击，41%易受XTP攻击，这突显了这种威胁的普遍性。



## **3. PROMPTFUZZ: Harnessing Fuzzing Techniques for Robust Testing of Prompt Injection in LLMs**

PROMPTFUZZ：利用模糊技术对LLM中的即时注射进行稳健测试 cs.CR

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2409.14729v2) [paper-pdf](http://arxiv.org/pdf/2409.14729v2)

**Authors**: Jiahao Yu, Yangguang Shao, Hanwen Miao, Junzheng Shi

**Abstract**: Large Language Models (LLMs) have gained widespread use in various applications due to their powerful capability to generate human-like text. However, prompt injection attacks, which involve overwriting a model's original instructions with malicious prompts to manipulate the generated text, have raised significant concerns about the security and reliability of LLMs. Ensuring that LLMs are robust against such attacks is crucial for their deployment in real-world applications, particularly in critical tasks.   In this paper, we propose PROMPTFUZZ, a novel testing framework that leverages fuzzing techniques to systematically assess the robustness of LLMs against prompt injection attacks. Inspired by software fuzzing, PROMPTFUZZ selects promising seed prompts and generates a diverse set of prompt injections to evaluate the target LLM's resilience. PROMPTFUZZ operates in two stages: the prepare phase, which involves selecting promising initial seeds and collecting few-shot examples, and the focus phase, which uses the collected examples to generate diverse, high-quality prompt injections. Using PROMPTFUZZ, we can uncover more vulnerabilities in LLMs, even those with strong defense prompts.   By deploying the generated attack prompts from PROMPTFUZZ in a real-world competition, we achieved the 7th ranking out of over 4000 participants (top 0.14%) within 2 hours. Additionally, we construct a dataset to fine-tune LLMs for enhanced robustness against prompt injection attacks. While the fine-tuned model shows improved robustness, PROMPTFUZZ continues to identify vulnerabilities, highlighting the importance of robust testing for LLMs. Our work emphasizes the critical need for effective testing tools and provides a practical framework for evaluating and improving the robustness of LLMs against prompt injection attacks.

摘要: 大型语言模型（LLM）因其生成类人文本的强大能力而在各种应用程序中得到广泛使用。然而，提示注入攻击（涉及将模型的原始指令与恶意提示一起操作生成的文本）引发了人们对LLM安全性和可靠性的严重担忧。确保LLM能够强大地抵御此类攻击对于它们在现实世界应用程序中的部署至关重要，特别是在关键任务中。   在本文中，我们提出了PROMPTFUZZ，这是一种新型测试框架，它利用模糊技术来系统性评估LLM针对即时注入攻击的稳健性。受软件模糊化的启发，PROMPTFUZZ选择有希望的种子提示并生成一组不同的提示注入来评估目标LLM的弹性。PROMPTFUZZ分为两个阶段：准备阶段，涉及选择有希望的初始种子并收集少量样本，以及聚焦阶段，使用收集的样本生成多样化、高质量的提示注射。使用PROMPTFUZZ，我们可以发现LLM中的更多漏洞，即使是那些具有强大防御提示的LLM。   通过在现实世界的比赛中部署PROMPTFUZZ生成的攻击提示，我们在2小时内在4000多名参与者中获得了第7名（前0.14%）。此外，我们还构建了一个数据集来微调LLM，以增强针对即时注入攻击的鲁棒性。虽然微调的模型显示出更好的稳健性，但PROMPTFUZZ继续识别漏洞，强调了对LLM进行稳健测试的重要性。我们的工作强调了对有效测试工具的迫切需求，并提供了一个实用的框架来评估和改进LLM针对即时注入攻击的稳健性。



## **4. ERPO: Advancing Safety Alignment via Ex-Ante Reasoning Preference Optimization**

ERPO：通过前推理偏好优化推进安全一致 cs.CL

18 pages, 5 figures

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.02725v1) [paper-pdf](http://arxiv.org/pdf/2504.02725v1)

**Authors**: Kehua Feng, Keyan Ding, Jing Yu, Menghan Li, Yuhao Wang, Tong Xu, Xinda Wang, Qiang Zhang, Huajun Chen

**Abstract**: Recent advancements in large language models (LLMs) have accelerated progress toward artificial general intelligence, yet their potential to generate harmful content poses critical safety challenges. Existing alignment methods often struggle to cover diverse safety scenarios and remain vulnerable to adversarial attacks. In this work, we propose Ex-Ante Reasoning Preference Optimization (ERPO), a novel safety alignment framework that equips LLMs with explicit preemptive reasoning through Chain-of-Thought and provides clear evidence for safety judgments by embedding predefined safety rules. Specifically, our approach consists of three stages: first, equipping the model with Ex-Ante reasoning through supervised fine-tuning (SFT) using a constructed reasoning module; second, enhancing safety, usefulness, and efficiency via Direct Preference Optimization (DPO); and third, mitigating inference latency with a length-controlled iterative preference optimization strategy. Experiments on multiple open-source LLMs demonstrate that ERPO significantly enhances safety performance while maintaining response efficiency.

摘要: 大型语言模型（LLM）的最新进展加速了人工通用智能的发展，但它们生成有害内容的潜力带来了严重的安全挑战。现有的对齐方法通常难以覆盖各种安全场景，并且仍然容易受到对抗性攻击。在这项工作中，我们提出了前-Ante推理偏好优化（ERPO），一种新的安全对齐框架，通过思想链为LLM提供明确的抢先推理，并通过嵌入预定义的安全规则为安全判断提供明确的证据。具体来说，我们的方法包括三个阶段：第一，通过使用构造的推理模块进行监督微调（SFT），为模型配备Ex-Ante推理;第二，通过直接偏好优化（DPO）提高安全性，有用性和效率;第三，通过长度控制的迭代偏好优化策略减轻推理延迟。在多个开源LLM上的实验表明，ERPO显着增强了安全性能，同时保持了响应效率。



## **5. No Free Lunch with Guardrails**

没有带护栏的免费午餐 cs.CR

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.00441v2) [paper-pdf](http://arxiv.org/pdf/2504.00441v2)

**Authors**: Divyanshu Kumar, Nitin Aravind Birur, Tanay Baswa, Sahil Agarwal, Prashanth Harshangi

**Abstract**: As large language models (LLMs) and generative AI become widely adopted, guardrails have emerged as a key tool to ensure their safe use. However, adding guardrails isn't without tradeoffs; stronger security measures can reduce usability, while more flexible systems may leave gaps for adversarial attacks. In this work, we explore whether current guardrails effectively prevent misuse while maintaining practical utility. We introduce a framework to evaluate these tradeoffs, measuring how different guardrails balance risk, security, and usability, and build an efficient guardrail.   Our findings confirm that there is no free lunch with guardrails; strengthening security often comes at the cost of usability. To address this, we propose a blueprint for designing better guardrails that minimize risk while maintaining usability. We evaluate various industry guardrails, including Azure Content Safety, Bedrock Guardrails, OpenAI's Moderation API, Guardrails AI, Nemo Guardrails, and Enkrypt AI guardrails. Additionally, we assess how LLMs like GPT-4o, Gemini 2.0-Flash, Claude 3.5-Sonnet, and Mistral Large-Latest respond under different system prompts, including simple prompts, detailed prompts, and detailed prompts with chain-of-thought (CoT) reasoning. Our study provides a clear comparison of how different guardrails perform, highlighting the challenges in balancing security and usability.

摘要: 随着大型语言模型（LLM）和生成式人工智能的广泛采用，护栏已成为确保其安全使用的关键工具。然而，添加护栏并非没有权衡;更强的安全措施可能会降低可用性，而更灵活的系统可能会为对抗性攻击留下缺口。在这项工作中，我们探索当前的护栏是否有效防止滥用，同时保持实用性。我们引入了一个框架来评估这些权衡，衡量不同的护栏如何平衡风险、安全性和可用性，并构建高效的护栏。   我们的调查结果证实，有护栏就没有免费的午餐;加强安全性往往是以牺牲可用性为代价的。为了解决这个问题，我们提出了一个设计更好护栏的蓝图，在保持可用性的同时最大限度地减少风险。我们评估各种行业护栏，包括Azure内容安全、Bedrock Guardrails、OpenAI的Moderation API、Guardrails AI、Nemo Guardrails和Enkrypt AI护栏。此外，我们还评估GPT-4 o、Gemini 2.0-Flash、Claude 3.5-十四行诗和Mistral Large-Latest等LLM如何在不同的系统提示下做出响应，包括简单提示、详细提示和具有思想链（CoT）推理的详细提示。我们的研究对不同护栏的性能进行了清晰的比较，强调了平衡安全性和可用性的挑战。



## **6. Retrieval-Augmented Purifier for Robust LLM-Empowered Recommendation**

用于强大的LLM授权推荐的检索增强净化器 cs.IR

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.02458v1) [paper-pdf](http://arxiv.org/pdf/2504.02458v1)

**Authors**: Liangbo Ning, Wenqi Fan, Qing Li

**Abstract**: Recently, Large Language Model (LLM)-empowered recommender systems have revolutionized personalized recommendation frameworks and attracted extensive attention. Despite the remarkable success, existing LLM-empowered RecSys have been demonstrated to be highly vulnerable to minor perturbations. To mitigate the negative impact of such vulnerabilities, one potential solution is to employ collaborative signals based on item-item co-occurrence to purify the malicious collaborative knowledge from the user's historical interactions inserted by attackers. On the other hand, due to the capabilities to expand insufficient internal knowledge of LLMs, Retrieval-Augmented Generation (RAG) techniques provide unprecedented opportunities to enhance the robustness of LLM-empowered recommender systems by introducing external collaborative knowledge. Therefore, in this paper, we propose a novel framework (RETURN) by retrieving external collaborative signals to purify the poisoned user profiles and enhance the robustness of LLM-empowered RecSys in a plug-and-play manner. Specifically, retrieval-augmented perturbation positioning is proposed to identify potential perturbations within the users' historical sequences by retrieving external knowledge from collaborative item graphs. After that, we further retrieve the collaborative knowledge to cleanse the perturbations by using either deletion or replacement strategies and introduce a robust ensemble recommendation strategy to generate final robust predictions. Extensive experiments on three real-world datasets demonstrate the effectiveness of the proposed RETURN.

摘要: 最近，基于大语言模型（LLM）的推荐系统彻底改变了个性化推荐框架，并引起了广泛关注。尽管取得了显着的成功，但现有的LLM授权RecSys已被证明极易受到微小干扰的影响。为了减轻此类漏洞的负面影响，一种潜在的解决方案是采用基于项-项共存的协作信号，从攻击者插入的用户历史交互中净化恶意协作知识。另一方面，由于扩展LLM内部知识不足的能力，检索增强生成（RAG）技术提供了前所未有的机会，通过引入外部协作知识来增强LLM授权的推荐系统的稳健性。因此，在本文中，我们提出了一种新颖的框架（RETURN），通过检索外部协作信号来净化有毒用户配置文件，并以即插即用的方式增强LLM授权的RecSys的鲁棒性。具体来说，提出了检索增强扰动定位，通过从协作项目图中检索外部知识来识别用户历史序列中的潜在扰动。之后，我们进一步检索协作知识，通过使用删除或替换策略来清除干扰，并引入稳健的集成推荐策略来生成最终的稳健预测。对三个现实世界数据集的广泛实验证明了所提出的RETUN的有效性。



## **7. ToxicSQL: Migrating SQL Injection Threats into Text-to-SQL Models via Backdoor Attack**

ToxicSQL：通过后门攻击将SQL注入威胁迁移到文本到SQL模型 cs.CR

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2503.05445v2) [paper-pdf](http://arxiv.org/pdf/2503.05445v2)

**Authors**: Meiyu Lin, Haichuan Zhang, Jiale Lao, Renyuan Li, Yuanchun Zhou, Carl Yang, Yang Cao, Mingjie Tang

**Abstract**: Large language models (LLMs) have shown state-of-the-art results in translating natural language questions into SQL queries (Text-to-SQL), a long-standing challenge within the database community. However, security concerns remain largely unexplored, particularly the threat of backdoor attacks, which can introduce malicious behaviors into models through fine-tuning with poisoned datasets. In this work, we systematically investigate the vulnerabilities of LLM-based Text-to-SQL models and present ToxicSQL, a novel backdoor attack framework. Our approach leverages stealthy {semantic and character-level triggers} to make backdoors difficult to detect and remove, ensuring that malicious behaviors remain covert while maintaining high model accuracy on benign inputs. Furthermore, we propose leveraging SQL injection payloads as backdoor targets, enabling the generation of malicious yet executable SQL queries, which pose severe security and privacy risks in language model-based SQL development. We demonstrate that injecting only 0.44% of poisoned data can result in an attack success rate of 79.41%, posing a significant risk to database security. Additionally, we propose detection and mitigation strategies to enhance model reliability. Our findings highlight the urgent need for security-aware Text-to-SQL development, emphasizing the importance of robust defenses against backdoor threats.

摘要: 大型语言模型（LLM）在将自然语言问题翻译为SQL查询（文本到SQL）方面显示出了最先进的结果，这是数据库界长期存在的挑战。然而，安全问题在很大程度上仍未得到探讨，特别是后门攻击的威胁，后门攻击可以通过对有毒数据集进行微调将恶意行为引入模型。在这项工作中，我们系统地研究了基于LLM的文本到SQL模型的漏洞，并提出了ToxicSQL，这是一种新型后门攻击框架。我们的方法利用隐形的（语义和字符级触发器）来使后门难以检测和删除，确保恶意行为保持隐蔽，同时对良性输入保持高模型准确性。此外，我们建议利用SQL注入有效负载作为后门目标，从而生成恶意但可执行的SQL查询，这在基于语言模型的SQL开发中构成了严重的安全和隐私风险。我们证明，仅注入0.44%的有毒数据就会导致79.41%的攻击成功率，对数据库安全构成重大风险。此外，我们还提出了检测和缓解策略来增强模型的可靠性。我们的研究结果强调了对安全意识的文本到SQL开发的迫切需求，并强调了针对后门威胁的强大防御的重要性。



## **8. Evolving from Single-modal to Multi-modal Facial Deepfake Detection: Progress and Challenges**

从单模式进化到多模式面部Deepfake检测：进展与挑战 cs.CV

P. Liu is with the Department of Computer Science and Engineering,  University of Nevada, Reno, NV, 89512. Q. Tao and J. Zhou are with Centre for  Frontier AI Research (CFAR), and Institute of High Performance Computing  (IHPC), A*STAR, Singapore. J. Zhou is also with Centre for Advanced  Technologies in Online Safety (CATOS), A*STAR, Singapore. J. Zhou is the  corresponding author

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2406.06965v4) [paper-pdf](http://arxiv.org/pdf/2406.06965v4)

**Authors**: Ping Liu, Qiqi Tao, Joey Tianyi Zhou

**Abstract**: As synthetic media, including video, audio, and text, become increasingly indistinguishable from real content, the risks of misinformation, identity fraud, and social manipulation escalate. This survey traces the evolution of deepfake detection from early single-modal methods to sophisticated multi-modal approaches that integrate audio-visual and text-visual cues. We present a structured taxonomy of detection techniques and analyze the transition from GAN-based to diffusion model-driven deepfakes, which introduce new challenges due to their heightened realism and robustness against detection. Unlike prior surveys that primarily focus on single-modal detection or earlier deepfake techniques, this work provides the most comprehensive study to date, encompassing the latest advancements in multi-modal deepfake detection, generalization challenges, proactive defense mechanisms, and emerging datasets specifically designed to support new interpretability and reasoning tasks. We further explore the role of Vision-Language Models (VLMs) and Multimodal Large Language Models (MLLMs) in strengthening detection robustness against increasingly sophisticated deepfake attacks. By systematically categorizing existing methods and identifying emerging research directions, this survey serves as a foundation for future advancements in combating AI-generated facial forgeries. A curated list of all related papers can be found at \href{https://github.com/qiqitao77/Comprehensive-Advances-in-Deepfake-Detection-Spanning-Diverse-Modalities}{https://github.com/qiqitao77/Awesome-Comprehensive-Deepfake-Detection}.

摘要: 随着包括视频、音频和文本在内的合成媒体与真实内容变得越来越难以区分，错误信息、身份欺诈和社会操纵的风险不断升级。这项调查追溯了深度假检测从早期的单模式方法到复杂的多模式方法的演变，这些方法整合了视听和文本视觉线索。我们提出了一种结构化的检测技术分类，并分析了从基于GaN的深伪到扩散模型驱动的深伪的转变，这带来了新的挑战，因为它们具有更高的真实性和对检测的健壮性。与以前主要关注单模式检测或更早的深度伪技术的调查不同，这项工作提供了迄今为止最全面的研究，包括多模式深度伪检测的最新进展、泛化挑战、主动防御机制以及专门为支持新的可解释性和推理任务而设计的新兴数据集。我们进一步探讨了视觉语言模型(VLM)和多模式大语言模型(MLLMS)在增强对日益复杂的深度伪攻击的检测鲁棒性方面的作用。通过系统地对现有方法进行分类并确定新的研究方向，这项调查为未来在打击人工智能生成的面部伪造方面的进展奠定了基础。所有相关论文的精选列表可在\href{https://github.com/qiqitao77/Comprehensive-Advances-in-Deepfake-Detection-Spanning-Diverse-Modalities}{https://github.com/qiqitao77/Awesome-Comprehensive-Deepfake-Detection}.上找到



## **9. More is Less: The Pitfalls of Multi-Model Synthetic Preference Data in DPO Safety Alignment**

多即少：DPO安全调整中多模型合成偏好数据的陷阱 cs.AI

**SubmitDate**: 2025-04-03    [abs](http://arxiv.org/abs/2504.02193v1) [paper-pdf](http://arxiv.org/pdf/2504.02193v1)

**Authors**: Yifan Wang, Runjin Chen, Bolian Li, David Cho, Yihe Deng, Ruqi Zhang, Tianlong Chen, Zhangyang Wang, Ananth Grama, Junyuan Hong

**Abstract**: Aligning large language models (LLMs) with human values is an increasingly critical step in post-training. Direct Preference Optimization (DPO) has emerged as a simple, yet effective alternative to reinforcement learning from human feedback (RLHF). Synthetic preference data with its low cost and high quality enable effective alignment through single- or multi-model generated preference data. Our study reveals a striking, safety-specific phenomenon associated with DPO alignment: Although multi-model generated data enhances performance on general tasks (ARC, Hellaswag, MMLU, TruthfulQA, Winogrande) by providing diverse responses, it also tends to facilitate reward hacking during training. This can lead to a high attack success rate (ASR) when models encounter jailbreaking prompts. The issue is particularly pronounced when employing stronger models like GPT-4o or larger models in the same family to generate chosen responses paired with target model self-generated rejected responses, resulting in dramatically poorer safety outcomes. Furthermore, with respect to safety, using solely self-generated responses (single-model generation) for both chosen and rejected pairs significantly outperforms configurations that incorporate responses from stronger models, whether used directly as chosen data or as part of a multi-model response pool. We demonstrate that multi-model preference data exhibits high linear separability between chosen and rejected responses, which allows models to exploit superficial cues rather than internalizing robust safety constraints. Our experiments, conducted on models from the Llama, Mistral, and Qwen families, consistently validate these findings.

摘要: 使大型语言模型（LLM）与人类价值观保持一致是后培训中越来越重要的一步。直接偏好优化（DPO）已成为人类反馈强化学习（RL HF）的一种简单而有效的替代方案。合成偏好数据具有低成本和高质量，可以通过单一或多模型生成的偏好数据进行有效匹配。我们的研究揭示了一种与DPO对齐相关的引人注目的、特定于安全的现象：尽管多模型生成的数据通过提供多样化的响应来增强一般任务（ARC、Hellaswag、MMLU、TruthfulQA、Winogrande）的性能，但它也往往会促进培训期间的奖励黑客攻击。当模型遇到越狱提示时，这可能会导致高攻击成功率（ASR）。当在同一家族中采用更强的模型（如GPT-4 o或更大的模型）来生成与目标模型自发产生的拒绝响应配对的选择响应时，该问题尤其明显，导致安全性结果明显较差。此外，在安全性方面，对于选择和拒绝的对，单独使用自我生成的响应（单模型生成）显著优于包含来自更强模型的响应的配置，无论是直接用作选择的数据还是作为多模型响应池的一部分。我们证明了多模型偏好数据在选择和拒绝的响应之间表现出高度的线性可分性，这使得模型能够利用表面的线索，而不是内化强大的安全约束。我们对Lama、Mistral和Qwen家族的模型进行的实验一致验证了这些发现。



## **10. Defending Large Language Models Against Attacks With Residual Stream Activation Analysis**

利用剩余流激活分析防御大型语言模型免受攻击 cs.CR

Included in Proceedings of the Conference on Applied Machine Learning  in Information Security (CAMLIS 2024), Arlington, Virginia, USA, October  24-25, 2024

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2406.03230v5) [paper-pdf](http://arxiv.org/pdf/2406.03230v5)

**Authors**: Amelia Kawasaki, Andrew Davis, Houssam Abbas

**Abstract**: The widespread adoption of Large Language Models (LLMs), exemplified by OpenAI's ChatGPT, brings to the forefront the imperative to defend against adversarial threats on these models. These attacks, which manipulate an LLM's output by introducing malicious inputs, undermine the model's integrity and the trust users place in its outputs. In response to this challenge, our paper presents an innovative defensive strategy, given white box access to an LLM, that harnesses residual activation analysis between transformer layers of the LLM. We apply a novel methodology for analyzing distinctive activation patterns in the residual streams for attack prompt classification. We curate multiple datasets to demonstrate how this method of classification has high accuracy across multiple types of attack scenarios, including our newly-created attack dataset. Furthermore, we enhance the model's resilience by integrating safety fine-tuning techniques for LLMs in order to measure its effect on our capability to detect attacks. The results underscore the effectiveness of our approach in enhancing the detection and mitigation of adversarial inputs, advancing the security framework within which LLMs operate.

摘要: 以OpenAI的ChatGPT为例，大型语言模型（LLM）的广泛采用使防御这些模型上的对抗威胁的必要性变得更加突出。这些攻击通过引入恶意输入来操纵LLM的输出，破坏了模型的完整性以及用户对其输出的信任。为了应对这一挑战，我们的论文提出了一种创新的防御策略，在白盒访问LLM的情况下，利用LLM Transformer层之间的剩余激活分析。我们应用一种新颖的方法来分析剩余流中的独特激活模式，以进行攻击提示分类。我们整理了多个数据集，以展示这种分类方法如何在多种类型的攻击场景（包括我们新创建的攻击数据集）中具有高准确性。此外，我们通过集成LLM的安全微调技术来增强模型的弹性，以衡量其对我们检测攻击能力的影响。结果强调了我们的方法在增强对抗性输入的检测和缓解、推进LLC运作的安全框架方面的有效性。



## **11. One Pic is All it Takes: Poisoning Visual Document Retrieval Augmented Generation with a Single Image**

一张图片就是一切：用一张图片毒害视觉文档检索增强生成 cs.CL

8 pages, 6 figures

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.02132v1) [paper-pdf](http://arxiv.org/pdf/2504.02132v1)

**Authors**: Ezzeldin Shereen, Dan Ristea, Burak Hasircioglu, Shae McFadden, Vasilios Mavroudis, Chris Hicks

**Abstract**: Multimodal retrieval augmented generation (M-RAG) has recently emerged as a method to inhibit hallucinations of large multimodal models (LMMs) through a factual knowledge base (KB). However, M-RAG also introduces new attack vectors for adversaries that aim to disrupt the system by injecting malicious entries into the KB. In this work, we present a poisoning attack against M-RAG targeting visual document retrieval applications, where the KB contains images of document pages. Our objective is to craft a single image that is retrieved for a variety of different user queries, and consistently influences the output produced by the generative model, thus creating a universal denial-of-service (DoS) attack against the M-RAG system. We demonstrate that while our attack is effective against a diverse range of widely-used, state-of-the-art retrievers (embedding models) and generators (LMMs), it can also be ineffective against robust embedding models. Our attack not only highlights the vulnerability of M-RAG pipelines to poisoning attacks, but also sheds light on a fundamental weakness that potentially hinders their performance even in benign settings.

摘要: 多通道检索增强生成(M-RAG)是最近出现的一种通过事实知识库(KB)抑制大型多通道模型(LMM)幻觉的方法。然而，M-RAG也为旨在通过向知识库中注入恶意条目来扰乱系统的攻击者引入了新的攻击载体。在这项工作中，我们提出了一种针对M-RAG的中毒攻击，目标是可视化文档检索应用程序，其中知识库包含文档页面的图像。我们的目标是为各种不同的用户查询创建单个图像，并持续影响生成模型产生的输出，从而创建针对M-RAG系统的通用拒绝服务(DoS)攻击。我们证明，虽然我们的攻击对各种广泛使用的最先进的检索器(嵌入模型)和生成器(LMM)有效，但它也可以对健壮的嵌入模型无效。我们的攻击不仅突显了M-RAG管道在中毒攻击下的脆弱性，还揭示了一个根本的弱点，即使在良性环境下，这个弱点也可能阻碍它们的性能。



## **12. Evolving Security in LLMs: A Study of Jailbreak Attacks and Defenses**

LLC中不断发展的安全性：越狱攻击和防御的研究 cs.CR

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.02080v1) [paper-pdf](http://arxiv.org/pdf/2504.02080v1)

**Authors**: Zhengchun Shang, Wenlan Wei

**Abstract**: Large Language Models (LLMs) are increasingly popular, powering a wide range of applications. Their widespread use has sparked concerns, especially through jailbreak attacks that bypass safety measures to produce harmful content.   In this paper, we present a comprehensive security analysis of large language models (LLMs), addressing critical research questions on the evolution and determinants of model safety.   Specifically, we begin by identifying the most effective techniques for detecting jailbreak attacks. Next, we investigate whether newer versions of LLMs offer improved security compared to their predecessors. We also assess the impact of model size on overall security and explore the potential benefits of integrating multiple defense strategies to enhance model robustness.   Our study evaluates both open-source models (e.g., LLaMA and Mistral) and closed-source systems (e.g., GPT-4) by employing four state-of-the-art attack techniques and assessing the efficacy of three new defensive approaches.

摘要: 大型语言模型（LLM）越来越受欢迎，为广泛的应用程序提供支持。它们的广泛使用引发了人们的担忧，特别是通过越狱攻击绕过安全措施产生有害内容。   在本文中，我们提出了一个全面的安全分析的大型语言模型（LLM），解决关键的研究问题的演变和决定因素的模型安全性。   具体来说，我们首先确定检测越狱攻击的最有效的技术。接下来，我们调查较新版本的LLM是否比其前身提供了更好的安全性。我们还评估模型大小对整体安全性的影响，并探索集成多种防御策略以增强模型稳健性的潜在好处。   我们的研究评估了两种开源模型（例如，LLaMA和Mistral）和闭源系统（例如，GPT-4）通过采用四种最先进的攻击技术并评估三种新防御方法的有效性。



## **13. AdPO: Enhancing the Adversarial Robustness of Large Vision-Language Models with Preference Optimization**

AdPO：通过偏好优化增强大型视觉语言模型的对抗鲁棒性 cs.CV

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01735v1) [paper-pdf](http://arxiv.org/pdf/2504.01735v1)

**Authors**: Chaohu Liu, Tianyi Gui, Yu Liu, Linli Xu

**Abstract**: Large Vision-Language Models (LVLMs), such as GPT-4o and LLaVA, have recently witnessed remarkable advancements and are increasingly being deployed in real-world applications. However, inheriting the sensitivity of visual neural networks, LVLMs remain vulnerable to adversarial attacks, which can result in erroneous or malicious outputs. While existing efforts utilize adversarial fine-tuning to enhance robustness, they often suffer from performance degradation on clean inputs. In this paper, we proposes AdPO, a novel adversarial defense strategy for LVLMs based on preference optimization. For the first time, we reframe adversarial training as a preference optimization problem, aiming to enhance the model's preference for generating normal outputs on clean inputs while rejecting the potential misleading outputs for adversarial examples. Notably, AdPO achieves this by solely modifying the image encoder, e.g., CLIP ViT, resulting in superior clean and adversarial performance in a variety of downsream tasks. Considering that training involves large language models (LLMs), the computational cost increases significantly. We validate that training on smaller LVLMs and subsequently transferring to larger models can achieve competitive performance while maintaining efficiency comparable to baseline methods. Our comprehensive experiments confirm the effectiveness of the proposed AdPO, which provides a novel perspective for future adversarial defense research.

摘要: GPT-4 o和LLaVA等大型视觉语言模型（LVLM）最近取得了显着的进步，并越来越多地部署在现实世界的应用程序中。然而，由于继承了视觉神经网络的敏感性，LVLM仍然容易受到对抗攻击，这可能会导致错误或恶意输出。虽然现有的工作利用对抗性微调来增强稳健性，但它们经常会在干净的输入上出现性能下降。本文提出了一种基于偏好优化的LVLM新型对抗防御策略AdPO。我们首次将对抗性训练重新定义为偏好优化问题，旨在增强模型在干净输入上生成正常输出的偏好，同时拒绝对抗性示例的潜在误导性输出。值得注意的是，AdPO仅通过修改图像编码器来实现这一点，例如，CLIP ViT，在各种降级任务中带来卓越的干净和对抗性能。考虑到训练涉及大型语言模型（LLM），计算成本显着增加。我们验证了在较小的LVLM上进行训练并随后转移到较大的模型可以实现有竞争力的性能，同时保持与基线方法相当的效率。我们全面的实验证实了拟议AdPO的有效性，为未来的对抗性防御研究提供了新的视角。



## **14. Representation Bending for Large Language Model Safety**

大型语言模型安全性的弯曲表示 cs.LG

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01550v1) [paper-pdf](http://arxiv.org/pdf/2504.01550v1)

**Authors**: Ashkan Yousefpour, Taeheon Kim, Ryan S. Kwon, Seungbeen Lee, Wonje Jeung, Seungju Han, Alvin Wan, Harrison Ngan, Youngjae Yu, Jonghyun Choi

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools, but their inherent safety risks - ranging from harmful content generation to broader societal harms - pose significant challenges. These risks can be amplified by the recent adversarial attacks, fine-tuning vulnerabilities, and the increasing deployment of LLMs in high-stakes environments. Existing safety-enhancing techniques, such as fine-tuning with human feedback or adversarial training, are still vulnerable as they address specific threats and often fail to generalize across unseen attacks, or require manual system-level defenses. This paper introduces RepBend, a novel approach that fundamentally disrupts the representations underlying harmful behaviors in LLMs, offering a scalable solution to enhance (potentially inherent) safety. RepBend brings the idea of activation steering - simple vector arithmetic for steering model's behavior during inference - to loss-based fine-tuning. Through extensive evaluation, RepBend achieves state-of-the-art performance, outperforming prior methods such as Circuit Breaker, RMU, and NPO, with up to 95% reduction in attack success rates across diverse jailbreak benchmarks, all with negligible reduction in model usability and general capabilities.

摘要: 大型语言模型（LLM）已成为强大的工具，但其固有的安全风险--从有害内容生成到更广泛的社会危害--带来了重大挑战。最近的对抗攻击、微调漏洞以及在高风险环境中增加部署LLM可能会放大这些风险。现有的安全增强技术，例如利用人类反馈进行微调或对抗性训练，仍然很脆弱，因为它们可以解决特定的威胁，并且通常无法对不可见的攻击进行概括，或者需要手动系统级防御。本文介绍了RepBend，这是一种新颖的方法，它从根本上破坏了LLM中有害行为的潜在表现，提供了可扩展的解决方案来增强（潜在固有的）安全性。RepBend将激活引导的想法（用于在推理期间引导模型行为的简单载体算法）引入到基于损失的微调中。通过广泛的评估，RepBend实现了最先进的性能，优于Circuit Breaker、RMU和NPO等现有方法，在各种越狱基准测试中，攻击成功率降低了高达95%，模型可用性和通用功能的下降微乎其微。



## **15. LightDefense: A Lightweight Uncertainty-Driven Defense against Jailbreaks via Shifted Token Distribution**

LightDefense：通过转移代币分发针对越狱的轻量级不确定性驱动防御 cs.CR

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01533v1) [paper-pdf](http://arxiv.org/pdf/2504.01533v1)

**Authors**: Zhuoran Yang, Jie Peng, Zhen Tan, Tianlong Chen, Yanyong Zhang

**Abstract**: Large Language Models (LLMs) face threats from jailbreak prompts. Existing methods for defending against jailbreak attacks are primarily based on auxiliary models. These strategies, however, often require extensive data collection or training. We propose LightDefense, a lightweight defense mechanism targeted at white-box models, which utilizes a safety-oriented direction to adjust the probabilities of tokens in the vocabulary, making safety disclaimers appear among the top tokens after sorting tokens by probability in descending order. We further innovatively leverage LLM's uncertainty about prompts to measure their harmfulness and adaptively adjust defense strength, effectively balancing safety and helpfulness. The effectiveness of LightDefense in defending against 5 attack methods across 2 target LLMs, without compromising helpfulness to benign user queries, highlights its potential as a novel and lightweight defense mechanism, enhancing security of LLMs.

摘要: 大型语言模型（LLM）面临越狱提示的威胁。现有的防御越狱攻击的方法主要基于辅助模型。然而，这些策略通常需要广泛的数据收集或培训。我们提出LightDefense，这是一种针对白盒模型的轻量级防御机制，利用以安全为导向的方向来调整词汇表中代币的概率，使安全免责声明在按概率降序排序后出现在前几名代币中。我们进一步创新性地利用LLM对提示的不确定性来衡量其危害性，并自适应地调整防御强度，有效地平衡了安全性和有益性。LightDefense在2个目标LLM上防御5种攻击方法的有效性，而不影响对良性用户查询的帮助，突出了其作为一种新型轻量级防御机制的潜力，增强了LLM的安全性。



## **16. PiCo: Jailbreaking Multimodal Large Language Models via $\textbf{Pi}$ctorial $\textbf{Co}$de Contextualization**

PiCo：通过$\textBF{Pi}$ctorial $\textBF{Co}$de上下文化破解多模式大型语言模型 cs.CR

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01444v1) [paper-pdf](http://arxiv.org/pdf/2504.01444v1)

**Authors**: Aofan Liu, Lulu Tang, Ting Pan, Yuguo Yin, Bin Wang, Ao Yang

**Abstract**: Multimodal Large Language Models (MLLMs), which integrate vision and other modalities into Large Language Models (LLMs), significantly enhance AI capabilities but also introduce new security vulnerabilities. By exploiting the vulnerabilities of the visual modality and the long-tail distribution characteristic of code training data, we present PiCo, a novel jailbreaking framework designed to progressively bypass multi-tiered defense mechanisms in advanced MLLMs. PiCo employs a tier-by-tier jailbreak strategy, using token-level typographic attacks to evade input filtering and embedding harmful intent within programming context instructions to bypass runtime monitoring. To comprehensively assess the impact of attacks, a new evaluation metric is further proposed to assess both the toxicity and helpfulness of model outputs post-attack. By embedding harmful intent within code-style visual instructions, PiCo achieves an average Attack Success Rate (ASR) of 84.13% on Gemini-Pro Vision and 52.66% on GPT-4, surpassing previous methods. Experimental results highlight the critical gaps in current defenses, underscoring the need for more robust strategies to secure advanced MLLMs.

摘要: 多模式大型语言模型（MLLM）将视觉和其他模式集成到大型语言模型（LLM）中，显着增强了人工智能能力，但也引入了新的安全漏洞。通过利用视觉模式的漏洞和代码训练数据的长尾分布特征，我们提出了PiCo，这是一种新型越狱框架，旨在逐步绕过高级MLLM中的多层防御机制。PiCo采用逐层越狱策略，使用标记级印刷攻击来逃避输入过滤，并在编程上下文指令中嵌入有害意图以绕过运行时监控。为了全面评估攻击的影响，进一步提出了一种新的评估指标来评估攻击后模型输出的毒性和帮助性。通过在代码风格的视觉指令中嵌入有害意图，PiCo在Gemini-Pro Vision上实现了84.13%的平均攻击成功率（ASB），在GPT-4上实现了52.66%的平均攻击成功率（ASB），超过了之前的方法。实验结果凸显了当前防御中的关键差距，强调需要更稳健的策略来保护高级MLLM。



## **17. Safeguarding Vision-Language Models: Mitigating Vulnerabilities to Gaussian Noise in Perturbation-based Attacks**

保护视觉语言模型：缓解基于扰动的攻击中高斯噪音的脆弱性 cs.CV

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01308v1) [paper-pdf](http://arxiv.org/pdf/2504.01308v1)

**Authors**: Jiawei Wang, Yushen Zuo, Yuanjun Chai, Zhendong Liu, Yichen Fu, Yichun Feng, Kin-man Lam

**Abstract**: Vision-Language Models (VLMs) extend the capabilities of Large Language Models (LLMs) by incorporating visual information, yet they remain vulnerable to jailbreak attacks, especially when processing noisy or corrupted images. Although existing VLMs adopt security measures during training to mitigate such attacks, vulnerabilities associated with noise-augmented visual inputs are overlooked. In this work, we identify that missing noise-augmented training causes critical security gaps: many VLMs are susceptible to even simple perturbations such as Gaussian noise. To address this challenge, we propose Robust-VLGuard, a multimodal safety dataset with aligned / misaligned image-text pairs, combined with noise-augmented fine-tuning that reduces attack success rates while preserving functionality of VLM. For stronger optimization-based visual perturbation attacks, we propose DiffPure-VLM, leveraging diffusion models to convert adversarial perturbations into Gaussian-like noise, which can be defended by VLMs with noise-augmented safety fine-tuning. Experimental results demonstrate that the distribution-shifting property of diffusion model aligns well with our fine-tuned VLMs, significantly mitigating adversarial perturbations across varying intensities. The dataset and code are available at https://github.com/JarvisUSTC/DiffPure-RobustVLM.

摘要: 视觉语言模型（VLMS）通过合并视觉信息扩展了大型语言模型（LLM）的功能，但它们仍然容易受到越狱攻击，尤其是在处理嘈杂或损坏的图像时。尽管现有的VLM在培训期间采取安全措施来减轻此类攻击，但与噪音增强视觉输入相关的漏洞被忽视了。在这项工作中，我们发现错过噪音增强训练会导致严重的安全漏洞：许多VLM甚至容易受到高斯噪音等简单扰动的影响。为了应对这一挑战，我们提出了Robust-VLGuard，这是一个具有对齐/未对齐图像-文本对的多模式安全数据集，结合了噪音增强微调，可以降低攻击成功率，同时保留VLM的功能。对于更强的基于优化的视觉扰动攻击，我们提出了DiffPure-VLM，利用扩散模型将对抗性扰动转换为类高斯噪声，可以通过具有噪声增强安全微调的VLM进行防御。实验结果表明，扩散模型的分布偏移特性与我们微调的VLM很好地吻合，显著减轻了不同强度的对抗性扰动。数据集和代码可在https://github.com/JarvisUSTC/DiffPure-RobustVLM上获取。



## **18. Strategize Globally, Adapt Locally: A Multi-Turn Red Teaming Agent with Dual-Level Learning**

全球战略，本地适应：具有双重学习的多轮红色团队代理 cs.AI

**SubmitDate**: 2025-04-02    [abs](http://arxiv.org/abs/2504.01278v1) [paper-pdf](http://arxiv.org/pdf/2504.01278v1)

**Authors**: Si Chen, Xiao Yu, Ninareh Mehrabi, Rahul Gupta, Zhou Yu, Ruoxi Jia

**Abstract**: The exploitation of large language models (LLMs) for malicious purposes poses significant security risks as these models become more powerful and widespread. While most existing red-teaming frameworks focus on single-turn attacks, real-world adversaries typically operate in multi-turn scenarios, iteratively probing for vulnerabilities and adapting their prompts based on threat model responses. In this paper, we propose \AlgName, a novel multi-turn red-teaming agent that emulates sophisticated human attackers through complementary learning dimensions: global tactic-wise learning that accumulates knowledge over time and generalizes to new attack goals, and local prompt-wise learning that refines implementations for specific goals when initial attempts fail. Unlike previous multi-turn approaches that rely on fixed strategy sets, \AlgName enables the agent to identify new jailbreak tactics, develop a goal-based tactic selection framework, and refine prompt formulations for selected tactics. Empirical evaluations on JailbreakBench demonstrate our framework's superior performance, achieving over 90\% attack success rates against GPT-3.5-Turbo and Llama-3.1-70B within 5 conversation turns, outperforming state-of-the-art baselines. These results highlight the effectiveness of dynamic learning in identifying and exploiting model vulnerabilities in realistic multi-turn scenarios.

摘要: 随着大型语言模型(LLM)变得更加强大和广泛，出于恶意目的利用这些模型会带来重大的安全风险。虽然大多数现有的红团队框架专注于单回合攻击，但现实世界中的对手通常在多回合场景中操作，迭代地探测漏洞并根据威胁模型响应调整他们的提示。在本文中，我们提出了一种新型的多轮红队代理与以前依赖固定策略集的多回合方法不同，算法名称使特工能够识别新的越狱战术，开发基于目标的战术选择框架，并为选定的战术改进提示公式。对JailBreak Btch的经验评估表明，我们的框架具有优越的性能，在5个对话回合内对GPT-3.5-Turbo和Llama-3.1-70B的攻击成功率超过90%，超过了最先进的基线。这些结果突出了动态学习在识别和利用现实多回合场景中的模型漏洞方面的有效性。



## **19. Towards Resilient Federated Learning in CyberEdge Networks: Recent Advances and Future Trends**

在CyberEdge网络中实现弹性联邦学习：最近的进展和未来的趋势 cs.CR

15 pages, 8 figures, 4 tables, 122 references, journal paper

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.01240v1) [paper-pdf](http://arxiv.org/pdf/2504.01240v1)

**Authors**: Kai Li, Zhengyang Zhang, Azadeh Pourkabirian, Wei Ni, Falko Dressler, Ozgur B. Akan

**Abstract**: In this survey, we investigate the most recent techniques of resilient federated learning (ResFL) in CyberEdge networks, focusing on joint training with agglomerative deduction and feature-oriented security mechanisms. We explore adaptive hierarchical learning strategies to tackle non-IID data challenges, improving scalability and reducing communication overhead. Fault tolerance techniques and agglomerative deduction mechanisms are studied to detect unreliable devices, refine model updates, and enhance convergence stability. Unlike existing FL security research, we comprehensively analyze feature-oriented threats, such as poisoning, inference, and reconstruction attacks that exploit model features. Moreover, we examine resilient aggregation techniques, anomaly detection, and cryptographic defenses, including differential privacy and secure multi-party computation, to strengthen FL security. In addition, we discuss the integration of 6G, large language models (LLMs), and interoperable learning frameworks to enhance privacy-preserving and decentralized cross-domain training. These advancements offer ultra-low latency, artificial intelligence (AI)-driven network management, and improved resilience against adversarial attacks, fostering the deployment of secure ResFL in CyberEdge networks.

摘要: 在这项调查中，我们研究了CyberEdge网络中弹性联邦学习（ResFL）的最新技术，重点关注与凝聚演绎和面向特征的安全机制的联合训练。我们探索自适应分层学习策略来应对非IID数据挑战，提高可扩展性并减少通信负担。研究了故障容忍技术和凝聚推理机制，以检测不可靠设备、细化模型更新并增强收敛稳定性。与现有的FL安全研究不同，我们全面分析面向特征的威胁，例如利用模型特征的中毒、推理和重建攻击。此外，我们还研究了弹性聚合技术、异常检测和加密防御，包括差异隐私和安全多方计算，以加强FL安全性。此外，我们还讨论了6G、大型语言模型（LLM）和互操作学习框架的集成，以增强隐私保护和去中心化的跨领域培训。这些进步提供了超低延迟、人工智能（AI）驱动的网络管理，并提高了针对对抗攻击的弹性，促进了在CyberEdge网络中部署安全ResFL。



## **20. Multilingual and Multi-Accent Jailbreaking of Audio LLMs**

多语言和多口音音频LL越狱 cs.SD

21 pages, 6 figures, 15 tables

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.01094v1) [paper-pdf](http://arxiv.org/pdf/2504.01094v1)

**Authors**: Jaechul Roh, Virat Shejwalkar, Amir Houmansadr

**Abstract**: Large Audio Language Models (LALMs) have significantly advanced audio understanding but introduce critical security risks, particularly through audio jailbreaks. While prior work has focused on English-centric attacks, we expose a far more severe vulnerability: adversarial multilingual and multi-accent audio jailbreaks, where linguistic and acoustic variations dramatically amplify attack success. In this paper, we introduce Multi-AudioJail, the first systematic framework to exploit these vulnerabilities through (1) a novel dataset of adversarially perturbed multilingual/multi-accent audio jailbreaking prompts, and (2) a hierarchical evaluation pipeline revealing that how acoustic perturbations (e.g., reverberation, echo, and whisper effects) interacts with cross-lingual phonetics to cause jailbreak success rates (JSRs) to surge by up to +57.25 percentage points (e.g., reverberated Kenyan-accented attack on MERaLiON). Crucially, our work further reveals that multimodal LLMs are inherently more vulnerable than unimodal systems: attackers need only exploit the weakest link (e.g., non-English audio inputs) to compromise the entire model, which we empirically show by multilingual audio-only attacks achieving 3.1x higher success rates than text-only attacks. We plan to release our dataset to spur research into cross-modal defenses, urging the community to address this expanding attack surface in multimodality as LALMs evolve.

摘要: 大型音频语言模型（LALM）具有显着提高的音频理解能力，但会带来严重的安全风险，特别是通过音频越狱。虽然之前的工作重点是以英语为中心的攻击，但我们暴露了一个更严重的漏洞：对抗性的多语言和多口音音频越狱，其中语言和声学差异极大地放大了攻击的成功。在本文中，我们引入了Multi-AudioJail，这是第一个利用这些漏洞的系统框架，通过（1）对抗干扰的多语言/多口音音频越狱提示的新颖数据集，以及（2）分层评估管道揭示了声学干扰（例如，回响、回声和耳语效果）与跨语言语音相互作用，导致越狱成功率（JSR）激增高达+57.25个百分点（例如，对MEaLiON产生了肯尼亚口音的攻击）。至关重要的是，我们的工作进一步揭示了多模式LLM本质上比单模式系统更容易受到攻击：攻击者只需要利用最弱的环节（例如，非英语音频输入）来损害整个模型，我们通过多语言纯音频攻击的成功率比纯文本攻击高出3.1倍。我们计划发布我们的数据集，以刺激对跨模式防御的研究，敦促社区随着LALM的发展，以多模式解决这一不断扩大的攻击面。



## **21. The Illusionist's Prompt: Exposing the Factual Vulnerabilities of Large Language Models with Linguistic Nuances**

魔术师的提示：用语言细微差别揭露大型语言模型的事实弱点 cs.CL

work in progress

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.02865v1) [paper-pdf](http://arxiv.org/pdf/2504.02865v1)

**Authors**: Yining Wang, Yuquan Wang, Xi Li, Mi Zhang, Geng Hong, Min Yang

**Abstract**: As Large Language Models (LLMs) continue to advance, they are increasingly relied upon as real-time sources of information by non-expert users. To ensure the factuality of the information they provide, much research has focused on mitigating hallucinations in LLM responses, but only in the context of formal user queries, rather than maliciously crafted ones. In this study, we introduce The Illusionist's Prompt, a novel hallucination attack that incorporates linguistic nuances into adversarial queries, challenging the factual accuracy of LLMs against five types of fact-enhancing strategies. Our attack automatically generates highly transferrable illusory prompts to induce internal factual errors, all while preserving user intent and semantics. Extensive experiments confirm the effectiveness of our attack in compromising black-box LLMs, including commercial APIs like GPT-4o and Gemini-2.0, even with various defensive mechanisms.

摘要: 随着大型语言模型（LLM）的不断发展，非专家用户越来越依赖它们作为实时信息来源。为了确保它们提供的信息的真实性，许多研究都集中在减轻LLM响应中的幻觉上，但仅限于正式用户查询的背景下，而不是恶意制作的查询。在这项研究中，我们引入了幻觉者的提示，这是一种新颖的幻觉攻击，将语言细微差别融入到对抗性询问中，针对五种事实增强策略，挑战LLM的事实准确性。我们的攻击会自动生成高度可转移的幻觉提示，以引发内部事实错误，同时保留用户意图和语义。大量实验证实了我们的攻击在攻击黑匣子LLM（包括GPT-4 o和Gemini-2.0等商业API）方面的有效性，即使有各种防御机制。



## **22. Exposing the Ghost in the Transformer: Abnormal Detection for Large Language Models via Hidden State Forensics**

揭露Transformer中的幽灵：通过隐藏状态取证对大型语言模型进行异常检测 cs.CR

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.00446v1) [paper-pdf](http://arxiv.org/pdf/2504.00446v1)

**Authors**: Shide Zhou, Kailong Wang, Ling Shi, Haoyu Wang

**Abstract**: The widespread adoption of Large Language Models (LLMs) in critical applications has introduced severe reliability and security risks, as LLMs remain vulnerable to notorious threats such as hallucinations, jailbreak attacks, and backdoor exploits. These vulnerabilities have been weaponized by malicious actors, leading to unauthorized access, widespread misinformation, and compromised LLM-embedded system integrity. In this work, we introduce a novel approach to detecting abnormal behaviors in LLMs via hidden state forensics. By systematically inspecting layer-specific activation patterns, we develop a unified framework that can efficiently identify a range of security threats in real-time without imposing prohibitive computational costs. Extensive experiments indicate detection accuracies exceeding 95% and consistently robust performance across multiple models in most scenarios, while preserving the ability to detect novel attacks effectively. Furthermore, the computational overhead remains minimal, with merely fractions of a second. The significance of this work lies in proposing a promising strategy to reinforce the security of LLM-integrated systems, paving the way for safer and more reliable deployment in high-stakes domains. By enabling real-time detection that can also support the mitigation of abnormal behaviors, it represents a meaningful step toward ensuring the trustworthiness of AI systems amid rising security challenges.

摘要: 大型语言模型（LLM）在关键应用程序中的广泛采用带来了严重的可靠性和安全风险，因为LLM仍然容易受到幻觉、越狱攻击和后门利用等臭名昭著的威胁的影响。这些漏洞已被恶意行为者武器化，导致未经授权的访问、广泛的错误信息以及LLM嵌入式系统完整性受损。在这项工作中，我们引入了一种通过隐藏状态取证来检测LLM中的异常行为的新颖方法。通过系统性检查特定于层的激活模式，我们开发了一个统一的框架，该框架可以有效地实时识别一系列安全威胁，而无需施加高昂的计算成本。大量实验表明，在大多数情况下，检测准确率超过95%，并且在多个模型中具有一致的稳健性能，同时保留了有效检测新型攻击的能力。此外，计算负担仍然最小，只需几分之一秒。这项工作的意义在于提出一项有希望的策略来加强LLM集成系统的安全性，为在高风险领域中更安全、更可靠的部署铺平道路。通过实现实时检测，也可以支持缓解异常行为，这是在不断上升的安全挑战中确保人工智能系统可信性的有意义的一步。



## **23. Unleashing the Power of Pre-trained Encoders for Universal Adversarial Attack Detection**

释放预培训编码器的力量进行通用对抗攻击检测 cs.CV

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.00429v1) [paper-pdf](http://arxiv.org/pdf/2504.00429v1)

**Authors**: Yinghe Zhang, Chi Liu, Shuai Zhou, Sheng Shen, Peng Gui

**Abstract**: Adversarial attacks pose a critical security threat to real-world AI systems by injecting human-imperceptible perturbations into benign samples to induce misclassification in deep learning models. While existing detection methods, such as Bayesian uncertainty estimation and activation pattern analysis, have achieved progress through feature engineering, their reliance on handcrafted feature design and prior knowledge of attack patterns limits generalization capabilities and incurs high engineering costs. To address these limitations, this paper proposes a lightweight adversarial detection framework based on the large-scale pre-trained vision-language model CLIP. Departing from conventional adversarial feature characterization paradigms, we innovatively adopt an anomaly detection perspective. By jointly fine-tuning CLIP's dual visual-text encoders with trainable adapter networks and learnable prompts, we construct a compact representation space tailored for natural images. Notably, our detection architecture achieves substantial improvements in generalization capability across both known and unknown attack patterns compared to traditional methods, while significantly reducing training overhead. This study provides a novel technical pathway for establishing a parameter-efficient and attack-agnostic defense paradigm, markedly enhancing the robustness of vision systems against evolving adversarial threats.

摘要: 对抗性攻击通过在良性样本中注入人类无法察觉的扰动来诱导深度学习模型中的错误分类，从而对现实世界的人工智能系统构成严重的安全威胁。虽然现有的检测方法，如贝叶斯不确定性估计和激活模式分析，通过特征工程取得了进展，但它们依赖于手工制作的特征设计和攻击模式的先验知识，限制了泛化能力，并产生了较高的工程成本。针对这些局限性，本文提出了一种基于大规模预训练视觉语言模型CLIP的轻量级对抗性检测框架。与传统的对抗性特征刻画范式不同，我们创新性地采用了异常检测的视角。通过使用可训练的适配器网络和可学习的提示，联合微调CLIP的双重视觉-文本编码器，我们构建了一个为自然图像量身定做的紧凑表示空间。值得注意的是，与传统方法相比，我们的检测体系结构在已知和未知攻击模式的泛化能力方面取得了实质性的改进，同时显著减少了训练开销。该研究为建立参数高效和攻击无关的防御范式提供了一条新的技术途径，显著增强了视觉系统对不断变化的对手威胁的稳健性。



## **24. Understanding the Effectiveness of Coverage Criteria for Large Language Models: A Special Angle from Jailbreak Attacks**

了解大型语言模型覆盖标准的有效性：越狱攻击的特殊角度 cs.SE

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2408.15207v3) [paper-pdf](http://arxiv.org/pdf/2408.15207v3)

**Authors**: Shide Zhou, Tianlin Li, Kailong Wang, Yihao Huang, Ling Shi, Yang Liu, Haoyu Wang

**Abstract**: Large language models (LLMs) have revolutionized artificial intelligence, but their increasing deployment across critical domains has raised concerns about their abnormal behaviors when faced with malicious attacks. Such vulnerability alerts the widespread inadequacy of pre-release testing. In this paper, we conduct a comprehensive empirical study to evaluate the effectiveness of traditional coverage criteria in identifying such inadequacies, exemplified by the significant security concern of jailbreak attacks. Our study begins with a clustering analysis of the hidden states of LLMs, revealing that the embedded characteristics effectively distinguish between different query types. We then systematically evaluate the performance of these criteria across three key dimensions: criterion level, layer level, and token level. Our research uncovers significant differences in neuron coverage when LLMs process normal versus jailbreak queries, aligning with our clustering experiments. Leveraging these findings, we propose three practical applications of coverage criteria in the context of LLM security testing. Specifically, we develop a real-time jailbreak detection mechanism that achieves high accuracy (93.61% on average) in classifying queries as normal or jailbreak. Furthermore, we explore the use of coverage levels to prioritize test cases, improving testing efficiency by focusing on high-risk interactions and removing redundant tests. Lastly, we introduce a coverage-guided approach for generating jailbreak attack examples, enabling systematic refinement of prompts to uncover vulnerabilities. This study improves our understanding of LLM security testing, enhances their safety, and provides a foundation for developing more robust AI applications.

摘要: 大型语言模型（LLM）已经彻底改变了人工智能，但它们在关键领域的部署越来越多，这引起了人们对它们在面临恶意攻击时异常行为的担忧。这种脆弱性警示了普遍存在的发布前测试不足。在本文中，我们进行了一个全面的实证研究，以评估传统的覆盖标准在识别这些不足之处的有效性，例如越狱攻击的重大安全问题。我们的研究开始于LLM的隐藏状态的聚类分析，揭示了嵌入的特征有效地区分不同的查询类型。然后，我们系统地评估这些标准在三个关键维度上的性能：标准级别、层级别和代币级别。我们的研究发现，当LLM处理正常查询与越狱查询时，神经元覆盖率存在显着差异，这与我们的集群实验保持一致。利用这些发现，我们提出了LLM安全测试背景下覆盖标准的三种实际应用。具体来说，我们开发了一种实时越狱检测机制，可以将查询分类为正常或越狱时实现高准确率（平均93.61%）。此外，我们探索使用覆盖级别来确定测试用例的优先级，通过关注高风险交互和删除冗余测试来提高测试效率。最后，我们引入了一种覆盖引导的方法来生成越狱攻击示例，从而能够系统地细化提示以发现漏洞。这项研究提高了我们对LLM安全测试的理解，增强了其安全性，并为开发更强大的人工智能应用程序提供了基础。



## **25. BounTCHA: A CAPTCHA Utilizing Boundary Identification in Guided Generative AI-extended Videos**

BoundCHA：在引导生成AI扩展视频中利用边界识别的验证码 cs.CR

22 pages, 15 figures; references added, typos corrected, new keyword  "guided" added, new experimental data and related results updated; new  keyword "Generative AI" added for clarity

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2501.18565v3) [paper-pdf](http://arxiv.org/pdf/2501.18565v3)

**Authors**: Lehao Lin, Ke Wang, Maha Abdallah, Wei Cai

**Abstract**: In recent years, the rapid development of artificial intelligence (AI) especially multi-modal Large Language Models (MLLMs), has enabled it to understand text, images, videos, and other multimedia data, allowing AI systems to execute various tasks based on human-provided prompts. However, AI-powered bots have increasingly been able to bypass most existing CAPTCHA systems, posing significant security threats to web applications. This makes the design of new CAPTCHA mechanisms an urgent priority. We observe that humans are highly sensitive to shifts and abrupt changes in videos, while current AI systems still struggle to comprehend and respond to such situations effectively. Based on this observation, we design and implement BounTCHA, a CAPTCHA mechanism that leverages human perception of boundaries in video transitions and disruptions. By utilizing generative AI's capability to extend original videos with prompts, we introduce unexpected twists and changes to create a pipeline for generating guided short videos for CAPTCHA purposes. We develop a prototype and conduct experiments to collect data on humans' time biases in boundary identification. This data serves as a basis for distinguishing between human users and bots. Additionally, we perform a detailed security analysis of BounTCHA, demonstrating its resilience against various types of attacks. We hope that BounTCHA will act as a robust defense, safeguarding millions of web applications in the AI-driven era.

摘要: 近年来，人工智能（AI）特别是多模式大型语言模型（MLLM）的快速发展，使其能够理解文本、图像、视频和其他多媒体数据，使人工智能系统能够根据人类提供的提示执行各种任务。然而，人工智能驱动的机器人越来越能够绕过大多数现有的验证码系统，对网络应用程序构成了重大的安全威胁。这使得设计新的验证码机制成为当务之急。我们观察到人类对视频的变化和突然变化高度敏感，而当前的人工智能系统仍然难以有效地理解和响应此类情况。基于这一观察，我们设计并实现了BounTCHA，这是一种CAPTCHA机制，利用人类对视频过渡和中断中边界的感知。通过利用生成式人工智能通过提示扩展原始视频的能力，我们引入了意想不到的曲折和变化，以创建一个用于生成用于验证码目的的引导短视频的管道。我们开发了一个原型并进行实验来收集人类在边界识别中的时间偏差的数据。该数据作为区分人类用户和机器人的基础。此外，我们还对BounTCHA进行了详细的安全分析，展示了其对各种类型攻击的弹性。我们希望BounTCHA能够充当强大的防御，在人工智能驱动时代保护数百万个网络应用程序。



## **26. Integrated LLM-Based Intrusion Detection with Secure Slicing xApp for Securing O-RAN-Enabled Wireless Network Deployments**

集成的基于LLM的入侵检测和安全切片xApp，用于保护支持O-RAN的无线网络部署 cs.CR

This article has been accepted for publication in the IEEE 2025  International Conference on Communications (ICC2025)

**SubmitDate**: 2025-04-01    [abs](http://arxiv.org/abs/2504.00341v1) [paper-pdf](http://arxiv.org/pdf/2504.00341v1)

**Authors**: Joshua Moore, Aly Sabri Abdalla, Prabesh Khanal, Vuk Marojevic

**Abstract**: The Open Radio Access Network (O-RAN) architecture is reshaping telecommunications by promoting openness, flexibility, and intelligent closed-loop optimization. By decoupling hardware and software and enabling multi-vendor deployments, O-RAN reduces costs, enhances performance, and allows rapid adaptation to new technologies. A key innovation is intelligent network slicing, which partitions networks into isolated slices tailored for specific use cases or quality of service requirements. The RAN Intelligent Controller further optimizes resource allocation, ensuring efficient utilization and improved service quality for user equipment (UEs). However, the modular and dynamic nature of O-RAN expands the threat surface, necessitating advanced security measures to maintain network integrity, confidentiality, and availability. Intrusion detection systems have become essential for identifying and mitigating attacks. This research explores using large language models (LLMs) to generate security recommendations based on the temporal traffic patterns of connected UEs. The paper introduces an LLM-driven intrusion detection framework and demonstrates its efficacy through experimental deployments, comparing non fine-tuned and fine-tuned models for task-specific accuracy.

摘要: 开放无线电接入网络（O-RAN）架构正在通过促进开放性、灵活性和智能闭环优化来重塑电信。通过将硬件和软件脱钩并实现多供应商部署，O-RAN降低了成本、增强了性能并允许快速适应新技术。一项关键创新是智能网络切片，它将网络划分为针对特定用例或服务质量要求量身定制的隔离切片。RAN智能控制器进一步优化资源分配，确保用户设备（UE）的高效利用和提高服务质量。然而，O-RAN的模块化和动态性质扩大了威胁面，需要采取先进的安全措施来维护网络完整性、机密性和可用性。入侵检测系统对于识别和减轻攻击至关重要。这项研究探索使用大型语言模型（LLM）根据连接UE的时间流量模式生成安全建议。本文介绍了一个LLM驱动的入侵检测框架，并通过实验部署展示了其功效，比较了非微调和微调模型的特定任务准确性。



## **27. $\textit{Agents Under Siege}$: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks**

$\textit{Agents Under Siege}$：Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks cs.MA

**SubmitDate**: 2025-03-31    [abs](http://arxiv.org/abs/2504.00218v1) [paper-pdf](http://arxiv.org/pdf/2504.00218v1)

**Authors**: Rana Muhammad Shahroz Khan, Zhen Tan, Sukwon Yun, Charles Flemming, Tianlong Chen

**Abstract**: Most discussions about Large Language Model (LLM) safety have focused on single-agent settings but multi-agent LLM systems now create novel adversarial risks because their behavior depends on communication between agents and decentralized reasoning. In this work, we innovatively focus on attacking pragmatic systems that have constrains such as limited token bandwidth, latency between message delivery, and defense mechanisms. We design a $\textit{permutation-invariant adversarial attack}$ that optimizes prompt distribution across latency and bandwidth-constraint network topologies to bypass distributed safety mechanisms within the system. Formulating the attack path as a problem of $\textit{maximum-flow minimum-cost}$, coupled with the novel $\textit{Permutation-Invariant Evasion Loss (PIEL)}$, we leverage graph-based optimization to maximize attack success rate while minimizing detection risk. Evaluating across models including $\texttt{Llama}$, $\texttt{Mistral}$, $\texttt{Gemma}$, $\texttt{DeepSeek}$ and other variants on various datasets like $\texttt{JailBreakBench}$ and $\texttt{AdversarialBench}$, our method outperforms conventional attacks by up to $7\times$, exposing critical vulnerabilities in multi-agent systems. Moreover, we demonstrate that existing defenses, including variants of $\texttt{Llama-Guard}$ and $\texttt{PromptGuard}$, fail to prohibit our attack, emphasizing the urgent need for multi-agent specific safety mechanisms.

摘要: Most discussions about Large Language Model (LLM) safety have focused on single-agent settings but multi-agent LLM systems now create novel adversarial risks because their behavior depends on communication between agents and decentralized reasoning. In this work, we innovatively focus on attacking pragmatic systems that have constrains such as limited token bandwidth, latency between message delivery, and defense mechanisms. We design a $\textit{permutation-invariant adversarial attack}$ that optimizes prompt distribution across latency and bandwidth-constraint network topologies to bypass distributed safety mechanisms within the system. Formulating the attack path as a problem of $\textit{maximum-flow minimum-cost}$, coupled with the novel $\textit{Permutation-Invariant Evasion Loss (PIEL)}$, we leverage graph-based optimization to maximize attack success rate while minimizing detection risk. Evaluating across models including $\texttt{Llama}$, $\texttt{Mistral}$, $\texttt{Gemma}$, $\texttt{DeepSeek}$ and other variants on various datasets like $\texttt{JailBreakBench}$ and $\texttt{AdversarialBench}$, our method outperforms conventional attacks by up to $7\times$, exposing critical vulnerabilities in multi-agent systems. Moreover, we demonstrate that existing defenses, including variants of $\texttt{Llama-Guard}$ and $\texttt{PromptGuard}$, fail to prohibit our attack, emphasizing the urgent need for multi-agent specific safety mechanisms.



## **28. Output Constraints as Attack Surface: Exploiting Structured Generation to Bypass LLM Safety Mechanisms**

作为攻击面的输出约束：利用结构化生成绕过LLM安全机制 cs.CR

15 pages, 13 figures, 4 tables Work In Progress

**SubmitDate**: 2025-03-31    [abs](http://arxiv.org/abs/2503.24191v1) [paper-pdf](http://arxiv.org/pdf/2503.24191v1)

**Authors**: Shuoming Zhang, Jiacheng Zhao, Ruiyuan Xu, Xiaobing Feng, Huimin Cui

**Abstract**: Content Warning: This paper may contain unsafe or harmful content generated by LLMs that may be offensive to readers. Large Language Models (LLMs) are extensively used as tooling platforms through structured output APIs to ensure syntax compliance so that robust integration with existing softwares like agent systems, could be achieved. However, the feature enabling functionality of grammar-guided structured output presents significant security vulnerabilities. In this work, we reveal a critical control-plane attack surface orthogonal to traditional data-plane vulnerabilities. We introduce Constrained Decoding Attack (CDA), a novel jailbreak class that weaponizes structured output constraints to bypass safety mechanisms. Unlike prior attacks focused on input prompts, CDA operates by embedding malicious intent in schema-level grammar rules (control-plane) while maintaining benign surface prompts (data-plane). We instantiate this with a proof-of-concept Chain Enum Attack, achieves 96.2% attack success rates across proprietary and open-weight LLMs on five safety benchmarks with a single query, including GPT-4o and Gemini-2.0-flash. Our findings identify a critical security blind spot in current LLM architectures and urge a paradigm shift in LLM safety to address control-plane vulnerabilities, as current mechanisms focused solely on data-plane threats leave critical systems exposed.

摘要: 内容警告：本文可能包含LLM生成的不安全或有害内容，这些内容可能会冒犯读者。大型语言模型（LLM）通过结构化输出API被广泛用作工具平台，以确保语法合规性，以便实现与代理系统等现有软件的稳健集成。然而，启用语法引导结构化输出功能的功能存在严重的安全漏洞。在这项工作中，我们揭示了一个与传统数据平面漏洞垂直的关键控制平面攻击表面。我们引入了约束解码攻击（CDO），这是一种新型越狱类，它将结构化输出约束武器化以绕过安全机制。与之前针对输入提示的攻击不同，CDO通过在模式级语法规则（控制平面）中嵌入恶意意图，同时保持良性表面提示（数据平面）来运作。我们通过概念验证Chain Enum Attack实例化了这一点，通过一个查询在五个安全基准（包括GPT-4 o和Gemini-2.0-Flash）上实现了专有和开放权重LLM的攻击成功率96.2%。我们的研究结果发现了当前LLM架构中的一个关键安全盲点，并敦促LLM安全性的范式转变以解决控制平面漏洞，因为当前仅关注数据平面威胁的机制会导致关键系统暴露在外。



## **29. Get the Agents Drunk: Memory Perturbations in Autonomous Agent-based Recommender Systems**

让代理喝醉：基于代理的自主推荐系统中的记忆扰动 cs.CR

**SubmitDate**: 2025-03-31    [abs](http://arxiv.org/abs/2503.23804v1) [paper-pdf](http://arxiv.org/pdf/2503.23804v1)

**Authors**: Shiyi Yang, Zhibo Hu, Chen Wang, Tong Yu, Xiwei Xu, Liming Zhu, Lina Yao

**Abstract**: Large language model-based agents are increasingly used in recommender systems (Agent4RSs) to achieve personalized behavior modeling. Specifically, Agent4RSs introduces memory mechanisms that enable the agents to autonomously learn and self-evolve from real-world interactions. However, to the best of our knowledge, how robust Agent4RSs are remains unexplored. As such, in this paper, we propose the first work to attack Agent4RSs by perturbing agents' memories, not only to uncover their limitations but also to enhance their security and robustness, ensuring the development of safer and more reliable AI agents.   Given the security and privacy concerns, it is more practical to launch attacks under a black-box setting, where the accurate knowledge of the victim models cannot be easily obtained. Moreover, the practical attacks are often stealthy to maximize the impact. To this end, we propose a novel practical attack framework named DrunkAgent. DrunkAgent consists of a generation module, a strategy module, and a surrogate module. The generation module aims to produce effective and coherent adversarial textual triggers, which can be used to achieve attack objectives such as promoting the target items. The strategy module is designed to `get the target agents drunk' so that their memories cannot be effectively updated during the interaction process. As such, the triggers can play the best role. Both of the modules are optimized on the surrogate module to improve the transferability and imperceptibility of the attacks. By identifying and analyzing the vulnerabilities, our work provides critical insights that pave the way for building safer and more resilient Agent4RSs. Extensive experiments across various real-world datasets demonstrate the effectiveness of DrunkAgent.

摘要: 在推荐系统(Agent4RS)中，越来越多地使用基于大型语言模型的代理来实现个性化的行为建模。具体地说，Agent4RSs引入了内存机制，使代理能够从现实世界的交互中自主学习和自我进化。然而，就我们所知，Agent4RSs的健壮程度仍有待研究。因此，在本文中，我们提出了通过干扰代理的记忆来攻击Agent4RS的第一项工作，不仅是为了揭示它们的局限性，而且还为了增强它们的安全性和健壮性，确保开发出更安全可靠的AI代理。考虑到安全和隐私问题，在黑匣子环境下发动攻击更实际，在这种情况下，不容易获得受害者模型的准确知识。此外，实际的攻击往往是隐蔽的，以最大限度地发挥影响。为此，我们提出了一种新的实用攻击框架DrunkAgent。DrunkAgent由生成模块、策略模块和代理模块组成。生成模块旨在生成有效且连贯的对抗性文本触发器，用于实现诸如提升目标条目等攻击目标。策略模块的设计目的是让目标代理喝醉，这样他们的记忆就不能在交互过程中有效地更新。因此，触发器可以发挥最好的作用。这两个模块都在代理模块上进行了优化，提高了攻击的可转移性和隐蔽性。通过识别和分析漏洞，我们的工作提供了重要的见解，为构建更安全、更具弹性的Agent4RS铺平了道路。在各种真实数据集上的广泛实验证明了DrunkAgent的有效性。



## **30. CL-Attack: Textual Backdoor Attacks via Cross-Lingual Triggers**

CL攻击：通过跨语言触发器进行文本后门攻击 cs.CR

The paper has been accepted to AAAI 2025

**SubmitDate**: 2025-03-31    [abs](http://arxiv.org/abs/2412.19037v2) [paper-pdf](http://arxiv.org/pdf/2412.19037v2)

**Authors**: Jingyi Zheng, Tianyi Hu, Tianshuo Cong, Xinlei He

**Abstract**: Backdoor attacks significantly compromise the security of large language models by triggering them to output specific and controlled content. Currently, triggers for textual backdoor attacks fall into two categories: fixed-token triggers and sentence-pattern triggers. However, the former are typically easy to identify and filter, while the latter, such as syntax and style, do not apply to all original samples and may lead to semantic shifts. In this paper, inspired by cross-lingual (CL) prompts of LLMs in real-world scenarios, we propose a higher-dimensional trigger method at the paragraph level, namely CL-attack. CL-attack injects the backdoor by using texts with specific structures that incorporate multiple languages, thereby offering greater stealthiness and universality compared to existing backdoor attack techniques. Extensive experiments on different tasks and model architectures demonstrate that CL-attack can achieve nearly 100% attack success rate with a low poisoning rate in both classification and generation tasks. We also empirically show that the CL-attack is more robust against current major defense methods compared to baseline backdoor attacks. Additionally, to mitigate CL-attack, we further develop a new defense called TranslateDefense, which can partially mitigate the impact of CL-attack.

摘要: 后门攻击通过触发大型语言模型输出特定且受控的内容来显着损害大型语言模型的安全性。目前，文本后门攻击的触发器分为两类：固定令牌触发器和业务模式触发器。然而，前者通常很容易识别和过滤，而后者（例如语法和风格）并不适用于所有原始样本，并且可能会导致语义转变。本文受到现实世界场景中LLM的跨语言（CL）提示的启发，提出了一种段落级别的更高维度触发方法，即CL攻击。CL攻击通过使用具有包含多种语言的特定结构的文本来注入后门，从而与现有的后门攻击技术相比提供更大的隐蔽性和通用性。针对不同任务和模型架构的大量实验表明，CL攻击在分类和生成任务中都可以实现近100%的攻击成功率，且中毒率较低。我们还通过经验表明，与基线后门攻击相比，CL攻击对当前的主要防御方法更强大。此外，为了减轻CL攻击，我们进一步开发了一种名为TranslateDefense的新防御，它可以部分减轻CL攻击的影响。



## **31. InjecGuard: Benchmarking and Mitigating Over-defense in Prompt Injection Guardrail Models**

InjecGuard：对标和缓解即时注射保障模型中的过度防御 cs.CL

**SubmitDate**: 2025-03-30    [abs](http://arxiv.org/abs/2410.22770v3) [paper-pdf](http://arxiv.org/pdf/2410.22770v3)

**Authors**: Hao Li, Xiaogeng Liu

**Abstract**: Prompt injection attacks pose a critical threat to large language models (LLMs), enabling goal hijacking and data leakage. Prompt guard models, though effective in defense, suffer from over-defense -- falsely flagging benign inputs as malicious due to trigger word bias. To address this issue, we introduce NotInject, an evaluation dataset that systematically measures over-defense across various prompt guard models. NotInject contains 339 benign samples enriched with trigger words common in prompt injection attacks, enabling fine-grained evaluation. Our results show that state-of-the-art models suffer from over-defense issues, with accuracy dropping close to random guessing levels (60%). To mitigate this, we propose InjecGuard, a novel prompt guard model that incorporates a new training strategy, Mitigating Over-defense for Free (MOF), which significantly reduces the bias on trigger words. InjecGuard demonstrates state-of-the-art performance on diverse benchmarks including NotInject, surpassing the existing best model by 30.8%, offering a robust and open-source solution for detecting prompt injection attacks. The code and datasets are released at https://github.com/leolee99/InjecGuard.

摘要: 快速注入攻击对大型语言模型(LLM)构成严重威胁，导致目标劫持和数据泄露。即时保护模式虽然在防御方面有效，但也存在过度防御的问题--由于触发单词偏见，错误地将良性输入标记为恶意输入。为了解决这个问题，我们引入了NotInject，这是一个评估数据集，系统地测量各种提示防护模型中的过度防御。NotInject包含339个良性样本，丰富了提示注入攻击中常见的触发字，实现了细粒度评估。我们的结果表明，最先进的模型存在过度防御的问题，准确率下降到接近随机猜测的水平(60%)。为了缓解这一问题，我们提出了InjecGuard，一种新的提示守卫模型，它结合了新的训练策略，缓解了过度防御For Free(MOF)，大大减少了对触发词的偏见。InjecGuard在包括NotInject在内的各种基准测试上展示了最先进的性能，比现有最好的模型高出30.8%，为检测即时注入攻击提供了一个强大的开源解决方案。代码和数据集在https://github.com/leolee99/InjecGuard.上发布



## **32. Data Extraction Attacks in Retrieval-Augmented Generation via Backdoors**

通过后门进行检索增强生成中的数据提取攻击 cs.CR

**SubmitDate**: 2025-03-30    [abs](http://arxiv.org/abs/2411.01705v2) [paper-pdf](http://arxiv.org/pdf/2411.01705v2)

**Authors**: Yuefeng Peng, Junda Wang, Hong Yu, Amir Houmansadr

**Abstract**: Despite significant advancements, large language models (LLMs) still struggle with providing accurate answers when lacking domain-specific or up-to-date knowledge. Retrieval-Augmented Generation (RAG) addresses this limitation by incorporating external knowledge bases, but it also introduces new attack surfaces. In this paper, we investigate data extraction attacks targeting RAG's knowledge databases. We show that previous prompt injection-based extraction attacks largely rely on the instruction-following capabilities of LLMs. As a result, they fail on models that are less responsive to such malicious prompts -- for example, our experiments show that state-of-the-art attacks achieve near-zero success on Gemma-2B-IT. Moreover, even for models that can follow these instructions, we found fine-tuning may significantly reduce attack performance. To further reveal the vulnerability, we propose to backdoor RAG, where a small portion of poisoned data is injected during the fine-tuning phase to create a backdoor within the LLM. When this compromised LLM is integrated into a RAG system, attackers can exploit specific triggers in prompts to manipulate the LLM to leak documents from the retrieval database. By carefully designing the poisoned data, we achieve both verbatim and paraphrased document extraction. For example, on Gemma-2B-IT, we show that with only 5\% poisoned data, our method achieves an average success rate of 94.1\% for verbatim extraction (ROUGE-L score: 82.1) and 63.6\% for paraphrased extraction (average ROUGE score: 66.4) across four datasets. These results underscore the privacy risks associated with the supply chain when deploying RAG systems.

摘要: 尽管取得了重大进步，但大型语言模型（LLM）在缺乏特定领域或最新知识时仍然难以提供准确的答案。检索增强一代（RAG）通过整合外部知识库来解决这一局限性，但它也引入了新的攻击表面。在本文中，我们调查了针对RAG知识数据库的数据提取攻击。我们表明，之前的基于即时注入的提取攻击在很大程度上依赖于LLM的描述跟踪能力。因此，它们在对此类恶意提示反应较弱的模型上失败--例如，我们的实验表明，最先进的攻击在Gemma-2B-IT上取得了接近零的成功。此外，即使对于可以遵循这些指令的模型，我们发现微调可能会显着降低攻击性能。为了进一步揭示该漏洞，我们建议后门RAG，在微调阶段注入一小部分有毒数据，以在LLM内创建后门。当这个受攻击的LLM集成到RAG系统中时，攻击者可以利用提示中的特定触发器来操纵LLM从检索数据库中泄露文档。通过仔细设计有毒数据，我们实现了逐字和转述的文档提取。例如，在Gemma-2B-IT上，我们表明，在只有5%有毒数据的情况下，我们的方法在四个数据集中实现了逐字提取（ROUGE-L评分：82.1）的平均成功率为94.1%，重述提取（平均ROUGE评分：66.4）的平均成功率为63.6%。这些结果强调了部署RAG系统时与供应链相关的隐私风险。



## **33. Encrypted Prompt: Securing LLM Applications Against Unauthorized Actions**

加密提示：保护LLM应用程序免受未经授权的操作 cs.CR

**SubmitDate**: 2025-03-29    [abs](http://arxiv.org/abs/2503.23250v1) [paper-pdf](http://arxiv.org/pdf/2503.23250v1)

**Authors**: Shih-Han Chan

**Abstract**: Security threats like prompt injection attacks pose significant risks to applications that integrate Large Language Models (LLMs), potentially leading to unauthorized actions such as API misuse. Unlike previous approaches that aim to detect these attacks on a best-effort basis, this paper introduces a novel method that appends an Encrypted Prompt to each user prompt, embedding current permissions. These permissions are verified before executing any actions (such as API calls) generated by the LLM. If the permissions are insufficient, the LLM's actions will not be executed, ensuring safety. This approach guarantees that only actions within the scope of the current permissions from the LLM can proceed. In scenarios where adversarial prompts are introduced to mislead the LLM, this method ensures that any unauthorized actions from LLM wouldn't be executed by verifying permissions in Encrypted Prompt. Thus, threats like prompt injection attacks that trigger LLM to generate harmful actions can be effectively mitigated.

摘要: 提示注入攻击等安全威胁对集成大型语言模型（LLM）的应用程序构成重大风险，可能导致API滥用等未经授权的操作。与之前旨在尽力检测这些攻击的方法不同，本文引入了一种新颖的方法，该方法在每个用户提示中添加加密提示，并嵌入当前权限。在执行LLM生成的任何操作（例如API调用）之前，会验证这些权限。如果权限不足，LLM的操作将不会执行，以确保安全。这种方法保证只有LLM当前权限范围内的操作才能继续。在引入对抗性提示以误导LLM的情况下，该方法通过在加密提示中验证权限来确保不会执行来自LLM的任何未经授权的操作。因此，可以有效地减轻触发LLM生成有害操作的提示注入攻击等威胁。



## **34. Training Large Language Models for Advanced Typosquatting Detection**

训练大型语言模型以进行高级排字检测 cs.CR

6 pages, 1 figure

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2503.22406v1) [paper-pdf](http://arxiv.org/pdf/2503.22406v1)

**Authors**: Jackson Welch

**Abstract**: Typosquatting is a long-standing cyber threat that exploits human error in typing URLs to deceive users, distribute malware, and conduct phishing attacks. With the proliferation of domain names and new Top-Level Domains (TLDs), typosquatting techniques have grown more sophisticated, posing significant risks to individuals, businesses, and national cybersecurity infrastructure. Traditional detection methods primarily focus on well-known impersonation patterns, leaving gaps in identifying more complex attacks. This study introduces a novel approach leveraging large language models (LLMs) to enhance typosquatting detection. By training an LLM on character-level transformations and pattern-based heuristics rather than domain-specific data, a more adaptable and resilient detection mechanism develops. Experimental results indicate that the Phi-4 14B model outperformed other tested models when properly fine tuned achieving a 98% accuracy rate with only a few thousand training samples. This research highlights the potential of LLMs in cybersecurity applications, specifically in mitigating domain-based deception tactics, and provides insights into optimizing machine learning strategies for threat detection.

摘要: 打字攻击是一种长期存在的网络威胁，它利用输入URL时的人为错误来欺骗用户、传播恶意软件并进行网络钓鱼攻击。随着域名和新顶级域名（TLR）的激增，错别字技术变得更加复杂，给个人、企业和国家网络安全基础设施带来了重大风险。传统的检测方法主要关注众所周知的模仿模式，在识别更复杂的攻击方面留下了空白。这项研究引入了一种利用大型语言模型（LLM）来增强错别字检测的新颖方法。通过对LLM进行字符级转换和基于模式的解析而不是特定于域的数据的训练，开发了一种更具适应性和弹性的检测机制。实验结果表明，Phi-4 14 B模型在适当微调时优于其他测试模型，仅用几千个训练样本就实现了98%的准确率。这项研究强调了LLM在网络安全应用中的潜力，特别是在减轻基于域的欺骗策略方面，并为优化机器学习策略以进行威胁检测提供了见解。



## **35. Single Image Unlearning: Efficient Machine Unlearning in Multimodal Large Language Models**

单图像去学习：多模式大型语言模型中的高效机器去学习 cs.CV

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2405.12523v3) [paper-pdf](http://arxiv.org/pdf/2405.12523v3)

**Authors**: Jiaqi Li, Qianshan Wei, Chuanyi Zhang, Guilin Qi, Miaozeng Du, Yongrui Chen, Sheng Bi, Fan Liu

**Abstract**: Machine unlearning empowers individuals with the `right to be forgotten' by removing their private or sensitive information encoded in machine learning models. However, it remains uncertain whether MU can be effectively applied to Multimodal Large Language Models (MLLMs), particularly in scenarios of forgetting the leaked visual data of concepts. To overcome the challenge, we propose an efficient method, Single Image Unlearning (SIU), to unlearn the visual recognition of a concept by fine-tuning a single associated image for few steps. SIU consists of two key aspects: (i) Constructing Multifaceted fine-tuning data. We introduce four targets, based on which we construct fine-tuning data for the concepts to be forgotten; (ii) Jointly training loss. To synchronously forget the visual recognition of concepts and preserve the utility of MLLMs, we fine-tune MLLMs through a novel Dual Masked KL-divergence Loss combined with Cross Entropy loss. Alongside our method, we establish MMUBench, a new benchmark for MU in MLLMs and introduce a collection of metrics for its evaluation. Experimental results on MMUBench show that SIU completely surpasses the performance of existing methods. Furthermore, we surprisingly find that SIU can avoid invasive membership inference attacks and jailbreak attacks. To the best of our knowledge, we are the first to explore MU in MLLMs. We will release the code and benchmark in the near future.

摘要: 机器取消学习通过删除机器学习模型中编码的私人或敏感信息来赋予个人“被遗忘权”。然而，目前还不确定MU是否可以有效地应用于多模式大型语言模型（MLLM），特别是在忘记泄露的概念视觉数据的情况下。为了克服这一挑战，我们提出了一种有效的方法，即单图像取消学习（SIU），通过几个步骤微调单个关联图像来取消概念的视觉识别。SIU由两个关键方面组成：（i）构建多面微调数据。我们引入了四个目标，并在此基础上为将要被遗忘的概念构建微调数据;（ii）联合训练损失。为了同步忘记概念的视觉识别并保留MLLM的实用性，我们通过新颖的双掩蔽KL分歧损失结合交叉熵损失来微调MLLM。除了我们的方法之外，我们还建立了MMUBench，这是MLLM中MU的新基准，并引入了一系列用于评估的指标。MMUBench上的实验结果表明，SIU完全超越了现有方法的性能。此外，我们惊讶地发现SIU可以避免侵入性成员推断攻击和越狱攻击。据我们所知，我们是第一家在MLLM中探索MU的公司。我们将在不久的将来发布代码和基准测试。



## **36. AnyAttack: Towards Large-scale Self-supervised Adversarial Attacks on Vision-language Models**

AnyAttack：走向对视觉语言模型的大规模自我监督对抗攻击 cs.LG

CVPR 2025

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2410.05346v3) [paper-pdf](http://arxiv.org/pdf/2410.05346v3)

**Authors**: Jiaming Zhang, Junhong Ye, Xingjun Ma, Yige Li, Yunfan Yang, Yunhao Chen, Jitao Sang, Dit-Yan Yeung

**Abstract**: Due to their multimodal capabilities, Vision-Language Models (VLMs) have found numerous impactful applications in real-world scenarios. However, recent studies have revealed that VLMs are vulnerable to image-based adversarial attacks. Traditional targeted adversarial attacks require specific targets and labels, limiting their real-world impact.We present AnyAttack, a self-supervised framework that transcends the limitations of conventional attacks through a novel foundation model approach. By pre-training on the massive LAION-400M dataset without label supervision, AnyAttack achieves unprecedented flexibility - enabling any image to be transformed into an attack vector targeting any desired output across different VLMs.This approach fundamentally changes the threat landscape, making adversarial capabilities accessible at an unprecedented scale. Our extensive validation across five open-source VLMs (CLIP, BLIP, BLIP2, InstructBLIP, and MiniGPT-4) demonstrates AnyAttack's effectiveness across diverse multimodal tasks. Most concerning, AnyAttack seamlessly transfers to commercial systems including Google Gemini, Claude Sonnet, Microsoft Copilot and OpenAI GPT, revealing a systemic vulnerability requiring immediate attention.

摘要: 由于其多通道能力，视觉语言模型(VLM)在现实世界场景中发现了许多有影响力的应用。然而，最近的研究表明，VLM很容易受到基于图像的对抗性攻击。传统的定向攻击需要特定的目标和标签，从而限制了它们在现实世界中的影响，我们提出了一个自我监督框架AnyAttack，它通过一种新的基础模型方法超越了传统攻击的限制。通过在没有标签监管的海量LAION-400M数据集上进行预训练，AnyAttack实现了前所未有的灵活性-使任何图像都能够转换为针对不同VLM中任何所需输出的攻击矢量。这种方法从根本上改变了威胁格局，使敌方能力能够以前所未有的规模获得。我们对五个开源VLM(CLIP、BLIP、BLIP2、InstructBLIP和MiniGPT-4)的广泛验证证明了AnyAttack在各种多模式任务中的有效性。最令人担忧的是，AnyAttack无缝传输到包括Google Gemini、Claude Sonnet、Microsoft Copilot和OpenAI GPT在内的商业系统，暴露出一个需要立即关注的系统性漏洞。



## **37. Foot-In-The-Door: A Multi-turn Jailbreak for LLMs**

一脚踏进门：LLC的多次越狱 cs.CL

19 pages, 8 figures

**SubmitDate**: 2025-03-28    [abs](http://arxiv.org/abs/2502.19820v3) [paper-pdf](http://arxiv.org/pdf/2502.19820v3)

**Authors**: Zixuan Weng, Xiaolong Jin, Jinyuan Jia, Xiangyu Zhang

**Abstract**: Ensuring AI safety is crucial as large language models become increasingly integrated into real-world applications. A key challenge is jailbreak, where adversarial prompts bypass built-in safeguards to elicit harmful disallowed outputs. Inspired by psychological foot-in-the-door principles, we introduce FITD,a novel multi-turn jailbreak method that leverages the phenomenon where minor initial commitments lower resistance to more significant or more unethical transgressions. Our approach progressively escalates the malicious intent of user queries through intermediate bridge prompts and aligns the model's response by itself to induce toxic responses. Extensive experimental results on two jailbreak benchmarks demonstrate that FITD achieves an average attack success rate of 94% across seven widely used models, outperforming existing state-of-the-art methods. Additionally, we provide an in-depth analysis of LLM self-corruption, highlighting vulnerabilities in current alignment strategies and emphasizing the risks inherent in multi-turn interactions. The code is available at https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak.

摘要: 随着大型语言模型越来越多地集成到现实世界的应用程序中，确保人工智能安全至关重要。一个关键的挑战是越狱，对抗会促使绕过内置保障措施，以引发有害的不允许输出。受心理学入门原则的启发，我们引入了FIDS，这是一种新颖的多回合越狱方法，它利用了轻微的初始承诺会降低对更重大或更不道德的违法行为的抵抗力的现象。我们的方法通过中间桥提示逐步升级用户查询的恶意意图，并自行调整模型的响应以引发有毒响应。两个越狱基准的大量实验结果表明，FIDS在七种广泛使用的模型中平均攻击成功率为94%，优于现有的最先进方法。此外，我们还对LLM自我腐败进行了深入分析，强调了当前调整策略中的漏洞，并强调多回合互动中固有的风险。该代码可在https://github.com/Jinxiaolong1129/Foot-in-the-door-Jailbreak上获取。



## **38. Debate-Driven Multi-Agent LLMs for Phishing Email Detection**

用于网络钓鱼电子邮件检测的辩论驱动的多代理LLM cs.MA

Accepted to the 13th International Symposium on Digital Forensics and  Security (ISDFS 2025)

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.22038v1) [paper-pdf](http://arxiv.org/pdf/2503.22038v1)

**Authors**: Ngoc Tuong Vy Nguyen, Felix D Childress, Yunting Yin

**Abstract**: Phishing attacks remain a critical cybersecurity threat. Attackers constantly refine their methods, making phishing emails harder to detect. Traditional detection methods, including rule-based systems and supervised machine learning models, either rely on predefined patterns like blacklists, which can be bypassed with slight modifications, or require large datasets for training and still can generate false positives and false negatives. In this work, we propose a multi-agent large language model (LLM) prompting technique that simulates debates among agents to detect whether the content presented on an email is phishing. Our approach uses two LLM agents to present arguments for or against the classification task, with a judge agent adjudicating the final verdict based on the quality of reasoning provided. This debate mechanism enables the models to critically analyze contextual cue and deceptive patterns in text, which leads to improved classification accuracy. The proposed framework is evaluated on multiple phishing email datasets and demonstrate that mixed-agent configurations consistently outperform homogeneous configurations. Results also show that the debate structure itself is sufficient to yield accurate decisions without extra prompting strategies.

摘要: 网络钓鱼攻击仍然是一个严重的网络安全威胁。攻击者不断改进他们的方法，使网络钓鱼电子邮件更难被检测。传统的检测方法，包括基于规则的系统和监督式机器学习模型，要么依赖于黑名单等预定义模式，只需稍加修改即可绕过，要么需要大型数据集进行训练，并且仍然可以生成假阳性和假阴性。在这项工作中，我们提出了一种多代理大型语言模型（LLM）提示技术，该技术模拟代理之间的辩论，以检测电子邮件上呈现的内容是否是网络钓鱼。我们的方法使用两个LLM代理来提出支持或反对分类任务的论点，由法官代理根据所提供的推理质量来裁决最终判决。这种辩论机制使模型能够批判性地分析文本中的上下文线索和欺骗性模式，从而提高分类准确性。对多个网络钓鱼电子邮件数据集进行了评估，并证明混合代理配置始终优于同类配置。结果还表明，辩论结构本身足以在不需要额外的提示策略的情况下做出准确的决定。



## **39. Intelligent IoT Attack Detection Design via ODLLM with Feature Ranking-based Knowledge Base**

基于特征排序知识库的ODLLM智能IoT攻击检测设计 cs.CR

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.21674v1) [paper-pdf](http://arxiv.org/pdf/2503.21674v1)

**Authors**: Satvik Verma, Qun Wang, E. Wes Bethel

**Abstract**: The widespread adoption of Internet of Things (IoT) devices has introduced significant cybersecurity challenges, particularly with the increasing frequency and sophistication of Distributed Denial of Service (DDoS) attacks. Traditional machine learning (ML) techniques often fall short in detecting such attacks due to the complexity of blended and evolving patterns. To address this, we propose a novel framework leveraging On-Device Large Language Models (ODLLMs) augmented with fine-tuning and knowledge base (KB) integration for intelligent IoT network attack detection. By implementing feature ranking techniques and constructing both long and short KBs tailored to model capacities, the proposed framework ensures efficient and accurate detection of DDoS attacks while overcoming computational and privacy limitations. Simulation results demonstrate that the optimized framework achieves superior accuracy across diverse attack types, especially when using compact models in edge computing environments. This work provides a scalable and secure solution for real-time IoT security, advancing the applicability of edge intelligence in cybersecurity.

摘要: 物联网（IOT）设备的广泛采用带来了重大的网络安全挑战，特别是随着分布式拒绝服务（DDOS）攻击的频率和复杂性不断增加。由于混合和进化模式的复杂性，传统的机器学习（ML）技术往往无法检测此类攻击。为了解决这个问题，我们提出了一种新颖的框架，利用设备上大型语言模型（ODLLRM）并通过微调和知识库（KB）集成进行增强，用于智能物联网网络攻击检测。通过实施特征排名技术并根据模型容量构建长KB和短KB，提出的框架确保高效、准确地检测DDOS攻击，同时克服计算和隐私限制。模拟结果表明，优化后的框架在各种攻击类型中实现了卓越的准确性，特别是在边缘计算环境中使用紧凑模型时。这项工作为实时物联网安全提供了可扩展且安全的解决方案，提高了边缘智能在网络安全中的适用性。



## **40. CleanGen: Mitigating Backdoor Attacks for Generation Tasks in Large Language Models**

CleanGen：减轻大型语言模型中生成任务的后门攻击 cs.AI

This paper is presented at EMNLP 2024

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2406.12257v3) [paper-pdf](http://arxiv.org/pdf/2406.12257v3)

**Authors**: Yuetai Li, Zhangchen Xu, Fengqing Jiang, Luyao Niu, Dinuka Sahabandu, Bhaskar Ramasubramanian, Radha Poovendran

**Abstract**: The remarkable performance of large language models (LLMs) in generation tasks has enabled practitioners to leverage publicly available models to power custom applications, such as chatbots and virtual assistants. However, the data used to train or fine-tune these LLMs is often undisclosed, allowing an attacker to compromise the data and inject backdoors into the models. In this paper, we develop a novel inference time defense, named CLEANGEN, to mitigate backdoor attacks for generation tasks in LLMs. CLEANGEN is a lightweight and effective decoding strategy that is compatible with the state-of-the-art (SOTA) LLMs. Our insight behind CLEANGEN is that compared to other LLMs, backdoored LLMs assign significantly higher probabilities to tokens representing the attacker-desired contents. These discrepancies in token probabilities enable CLEANGEN to identify suspicious tokens favored by the attacker and replace them with tokens generated by another LLM that is not compromised by the same attacker, thereby avoiding generation of attacker-desired content. We evaluate CLEANGEN against five SOTA backdoor attacks. Our results show that CLEANGEN achieves lower attack success rates (ASR) compared to five SOTA baseline defenses for all five backdoor attacks. Moreover, LLMs deploying CLEANGEN maintain helpfulness in their responses when serving benign user queries with minimal added computational overhead.

摘要: 大型语言模型（LLM）在生成任务中的出色性能使从业者能够利用公开可用的模型来支持自定义应用程序，例如聊天机器人和虚拟助理。然而，用于训练或微调这些LLM的数据通常是不公开的，这使得攻击者能够破坏数据并将后门注入模型。在本文中，我们开发了一种新型的推理时间防御，名为CleANGER，以减轻对LLM中生成任务的后门攻击。Cleangen是一种轻量级且有效的解码策略，与最先进的（SOTA）LLM兼容。我们对Cleangen的见解是，与其他LLM相比，后门LLM为代表攻击者所需内容的令牌分配了明显更高的概率。令牌概率的这些差异使CleANGER能够识别攻击者青睐的可疑令牌，并用未被同一攻击者泄露的另一个LLM生成的令牌替换它们，从而避免生成攻击者想要的内容。我们针对五种SOTA后门攻击评估了Cleangen。我们的结果表明，与五种SOTA基线防御相比，对于所有五种后门攻击，CleANGER的攻击成功率（ASB）更低。此外，部署Cleangen的LLM在以最小的额外计算负担为良性用户查询提供服务时保持其响应的有用性。



## **41. Malicious and Unintentional Disclosure Risks in Large Language Models for Code Generation**

用于代码生成的大型语言模型中的恶意和无意披露风险 cs.CR

The 3rd International Workshop on Mining Software Repositories  Applications for Privacy and Security (MSR4P&S), co-located with SANER 2025

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.22760v1) [paper-pdf](http://arxiv.org/pdf/2503.22760v1)

**Authors**: Rafiqul Rabin, Sean McGregor, Nick Judd

**Abstract**: This paper explores the risk that a large language model (LLM) trained for code generation on data mined from software repositories will generate content that discloses sensitive information included in its training data. We decompose this risk, known in the literature as ``unintended memorization,'' into two components: unintentional disclosure (where an LLM presents secrets to users without the user seeking them out) and malicious disclosure (where an LLM presents secrets to an attacker equipped with partial knowledge of the training data). We observe that while existing work mostly anticipates malicious disclosure, unintentional disclosure is also a concern. We describe methods to assess unintentional and malicious disclosure risks side-by-side across different releases of training datasets and models. We demonstrate these methods through an independent assessment of the Open Language Model (OLMo) family of models and its Dolma training datasets. Our results show, first, that changes in data source and processing are associated with substantial changes in unintended memorization risk; second, that the same set of operational changes may increase one risk while mitigating another; and, third, that the risk of disclosing sensitive information varies not only by prompt strategies or test datasets but also by the types of sensitive information. These contributions rely on data mining to enable greater privacy and security testing required for the LLM training data supply chain.

摘要: 本文探讨了针对从软件存储库挖掘的数据进行代码生成而训练的大型语言模型(LLM)将生成泄露其训练数据中包含的敏感信息的内容的风险。我们将这种风险分解为两个组成部分：无意泄露(LLM向用户提供秘密，而用户没有寻找他们)和恶意泄露(LLM向配备了训练数据部分知识的攻击者提供秘密)。我们观察到，虽然现有的工作大多预期恶意披露，但无意披露也是一个令人担忧的问题。我们描述了在不同版本的培训数据集和模型中并排评估无意和恶意披露风险的方法。我们通过对开放语言模型(OLMO)模型家族及其DOLMA训练数据集的独立评估来演示这些方法。我们的结果表明，第一，数据源和处理过程的变化与意外记忆风险的实质性变化相关；第二，相同的操作变化可能增加一种风险，同时降低另一种风险；第三，泄露敏感信息的风险不仅因提示策略或测试数据集的不同而不同，还因敏感信息的类型而不同。这些贡献依赖于数据挖掘，以实现LLM培训数据供应链所需的更大隐私和安全测试。



## **42. Data Poisoning in Deep Learning: A Survey**

深度学习中的数据中毒：一项调查 cs.CR

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.22759v1) [paper-pdf](http://arxiv.org/pdf/2503.22759v1)

**Authors**: Pinlong Zhao, Weiyao Zhu, Pengfei Jiao, Di Gao, Ou Wu

**Abstract**: Deep learning has become a cornerstone of modern artificial intelligence, enabling transformative applications across a wide range of domains. As the core element of deep learning, the quality and security of training data critically influence model performance and reliability. However, during the training process, deep learning models face the significant threat of data poisoning, where attackers introduce maliciously manipulated training data to degrade model accuracy or lead to anomalous behavior. While existing surveys provide valuable insights into data poisoning, they generally adopt a broad perspective, encompassing both attacks and defenses, but lack a dedicated, in-depth analysis of poisoning attacks specifically in deep learning. In this survey, we bridge this gap by presenting a comprehensive and targeted review of data poisoning in deep learning. First, this survey categorizes data poisoning attacks across multiple perspectives, providing an in-depth analysis of their characteristics and underlying design princinples. Second, the discussion is extended to the emerging area of data poisoning in large language models(LLMs). Finally, we explore critical open challenges in the field and propose potential research directions to advance the field further. To support further exploration, an up-to-date repository of resources on data poisoning in deep learning is available at https://github.com/Pinlong-Zhao/Data-Poisoning.

摘要: 深度学习已经成为现代人工智能的基石，能够在广泛的领域实现变革性的应用。作为深度学习的核心要素，训练数据的质量和安全性对模型的性能和可靠性有着至关重要的影响。然而，在训练过程中，深度学习模型面临着数据中毒的重大威胁，攻击者引入恶意操纵的训练数据来降低模型的准确性或导致异常行为。虽然现有的调查为数据中毒提供了有价值的见解，但它们通常采用了广泛的视角，既包括攻击也包括防御，但缺乏专门的、深入的分析，特别是在深度学习方面。在这项调查中，我们通过全面和有针对性地回顾深度学习中的数据中毒来弥补这一差距。首先，这项调查从多个角度对数据中毒攻击进行了分类，深入分析了它们的特征和潜在的设计原理。其次，将讨论扩展到大型语言模型(LLM)中的数据中毒这一新兴领域。最后，我们探讨了该领域的关键开放挑战，并提出了进一步推进该领域的潜在研究方向。为了支持进一步的探索，关于深度学习中的数据中毒的最新资源库可在https://github.com/Pinlong-Zhao/Data-Poisoning.上找到



## **43. Harnessing Chain-of-Thought Metadata for Task Routing and Adversarial Prompt Detection**

利用思想链元数据进行任务路由和对抗提示检测 cs.CL

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.21464v1) [paper-pdf](http://arxiv.org/pdf/2503.21464v1)

**Authors**: Ryan Marinelli, Josef Pichlmeier, Tamas Bisztray

**Abstract**: In this work, we propose a metric called Number of Thoughts (NofT) to determine the difficulty of tasks pre-prompting and support Large Language Models (LLMs) in production contexts. By setting thresholds based on the number of thoughts, this metric can discern the difficulty of prompts and support more effective prompt routing. A 2% decrease in latency is achieved when routing prompts from the MathInstruct dataset through quantized, distilled versions of Deepseek with 1.7 billion, 7 billion, and 14 billion parameters. Moreover, this metric can be used to detect adversarial prompts used in prompt injection attacks with high efficacy. The Number of Thoughts can inform a classifier that achieves 95% accuracy in adversarial prompt detection. Our experiments ad datasets used are available on our GitHub page: https://github.com/rymarinelli/Number_Of_Thoughts/tree/main.

摘要: 在这项工作中，我们提出了一种名为“思考数量”（NofT）的指标，以确定预提示任务的难度并支持生产环境中的大型语言模型（LLM）。通过根据想法数量设置阈值，该指标可以辨别提示的难度并支持更有效的提示路由。当通过具有17亿、70亿和140亿参数的量化、提炼版本的Deepseek从MathDirect数据集中路由提示时，延迟可降低2%。此外，该指标可用于高效检测提示注射攻击中使用的对抗提示。思维数量可以通知分类器，在对抗性提示检测中达到95%的准确率。我们使用的实验和数据集可以在我们的GitHub页面上找到：https://github.com/rymarinelli/Number_Of_Thoughts/tree/main。



## **44. Tricking Retrievers with Influential Tokens: An Efficient Black-Box Corpus Poisoning Attack**

用有影响力的代币欺骗猎犬：一种有效的黑匣子库中毒攻击 cs.LG

Accepted to NAACL 2025 Main Track

**SubmitDate**: 2025-03-27    [abs](http://arxiv.org/abs/2503.21315v1) [paper-pdf](http://arxiv.org/pdf/2503.21315v1)

**Authors**: Cheng Wang, Yiwei Wang, Yujun Cai, Bryan Hooi

**Abstract**: Retrieval-augmented generation (RAG) systems enhance large language models by incorporating external knowledge, addressing issues like outdated internal knowledge and hallucination. However, their reliance on external knowledge bases makes them vulnerable to corpus poisoning attacks, where adversarial passages can be injected to manipulate retrieval results. Existing methods for crafting such passages, such as random token replacement or training inversion models, are often slow and computationally expensive, requiring either access to retriever's gradients or large computational resources. To address these limitations, we propose Dynamic Importance-Guided Genetic Algorithm (DIGA), an efficient black-box method that leverages two key properties of retrievers: insensitivity to token order and bias towards influential tokens. By focusing on these characteristics, DIGA dynamically adjusts its genetic operations to generate effective adversarial passages with significantly reduced time and memory usage. Our experimental evaluation shows that DIGA achieves superior efficiency and scalability compared to existing methods, while maintaining comparable or better attack success rates across multiple datasets.

摘要: 检索-增强生成(RAG)系统通过整合外部知识来增强大型语言模型，解决过时的内部知识和幻觉等问题。然而，它们对外部知识库的依赖使它们很容易受到语料库中毒攻击，在语料库中毒攻击中，可以注入对抗性段落来操纵检索结果。现有的制作这类段落的方法，如随机标记替换或训练反转模型，通常速度慢且计算昂贵，需要访问检索器的梯度或大量计算资源。为了克服这些局限性，我们提出了动态重要性引导遗传算法(DIGA)，这是一种有效的黑盒方法，它利用了检索者的两个关键特性：对标记顺序不敏感和对有影响的标记的偏爱。通过专注于这些特征，Diga动态调整其遗传操作，以生成有效的对抗性段落，显著减少时间和内存使用。我们的实验评估表明，与现有方法相比，Diga实现了更高的效率和可扩展性，同时在多个数据集上保持了相当或更好的攻击成功率。



## **45. M-LLM Based Video Frame Selection for Efficient Video Understanding**

基于M-LLM的视频帧选择以实现高效的视频理解 cs.CV

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2502.19680v2) [paper-pdf](http://arxiv.org/pdf/2502.19680v2)

**Authors**: Kai Hu, Feng Gao, Xiaohan Nie, Peng Zhou, Son Tran, Tal Neiman, Lingyun Wang, Mubarak Shah, Raffay Hamid, Bing Yin, Trishul Chilimbi

**Abstract**: Recent advances in Multi-Modal Large Language Models (M-LLMs) show promising results in video reasoning. Popular Multi-Modal Large Language Model (M-LLM) frameworks usually apply naive uniform sampling to reduce the number of video frames that are fed into an M-LLM, particularly for long context videos. However, it could lose crucial context in certain periods of a video, so that the downstream M-LLM may not have sufficient visual information to answer a question. To attack this pain point, we propose a light-weight M-LLM -based frame selection method that adaptively select frames that are more relevant to users' queries. In order to train the proposed frame selector, we introduce two supervision signals (i) Spatial signal, where single frame importance score by prompting a M-LLM; (ii) Temporal signal, in which multiple frames selection by prompting Large Language Model (LLM) using the captions of all frame candidates. The selected frames are then digested by a frozen downstream video M-LLM for visual reasoning and question answering. Empirical results show that the proposed M-LLM video frame selector improves the performances various downstream video Large Language Model (video-LLM) across medium (ActivityNet, NExT-QA) and long (EgoSchema, LongVideoBench) context video question answering benchmarks.

摘要: 多模式大语言模型(M-LLMS)的最新进展表明，在视频推理方面取得了可喜的结果。流行的多模式大语言模型(M-LLM)框架通常采用朴素的均匀采样来减少输入到M-LLM的视频帧的数量，特别是对于长上下文视频。然而，它可能会在视频的某些时段失去关键的上下文，从而下游的M-LLM可能没有足够的视觉信息来回答问题。针对这一痛点，我们提出了一种基于轻量级M-LLM的框架选择方法，该方法自适应地选择与用户查询更相关的框架。为了训练提出的帧选择器，我们引入了两个监督信号(I)空间信号，其中单帧重要性通过提示M-LLM进行评分；(Ii)时间信号，其中通过提示大语言模型(LLM)使用所有帧候选的字幕来选择多帧。然后，选定的帧被冻结的下行视频M-LLM消化，以进行视觉推理和问答。实验结果表明，所提出的M-LLM视频帧选择器提高了不同下游视频大语言模型(VIDEO-LLM)跨中(ActivityNet，Next-QA)和长(EgoSchema，LongVideo)上下文视频问答基准的性能。



## **46. Iterative Prompting with Persuasion Skills in Jailbreaking Large Language Models**

越狱大型语言模型中具有说服技巧的迭代绘图 cs.CL

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20320v1) [paper-pdf](http://arxiv.org/pdf/2503.20320v1)

**Authors**: Shih-Wen Ke, Guan-Yu Lai, Guo-Lin Fang, Hsi-Yuan Kao

**Abstract**: Large language models (LLMs) are designed to align with human values in their responses. This study exploits LLMs with an iterative prompting technique where each prompt is systematically modified and refined across multiple iterations to enhance its effectiveness in jailbreaking attacks progressively. This technique involves analyzing the response patterns of LLMs, including GPT-3.5, GPT-4, LLaMa2, Vicuna, and ChatGLM, allowing us to adjust and optimize prompts to evade the LLMs' ethical and security constraints. Persuasion strategies enhance prompt effectiveness while maintaining consistency with malicious intent. Our results show that the attack success rates (ASR) increase as the attacking prompts become more refined with the highest ASR of 90% for GPT4 and ChatGLM and the lowest ASR of 68% for LLaMa2. Our technique outperforms baseline techniques (PAIR and PAP) in ASR and shows comparable performance with GCG and ArtPrompt.

摘要: 大型语言模型（LLM）旨在在其响应中与人类价值观保持一致。这项研究通过迭代提示技术来利用LLM，其中每个提示都经过多次迭代系统地修改和细化，以逐步增强其越狱攻击的有效性。该技术涉及分析LLM的响应模式，包括GPT-3.5、GPT-4、LLaMa 2、Vicuna和ChatGLM，使我们能够调整和优化提示以规避LLM的道德和安全限制。说服策略增强了及时的有效性，同时与恶意意图保持一致。我们的结果表明，随着攻击提示变得更加精确，攻击成功率（ASB）也会增加，GPT 4和ChatGLM的最高ASB为90%，LLaMa 2的最低ASB为68%。我们的技术在ASB中优于基线技术（PAIR和PAP），并表现出与GCG和ArtPrompt相当的性能。



## **47. Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models**

大视觉语言模型的自我监督学习视觉编码器中的秘密后门攻击 cs.CV

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2502.18290v3) [paper-pdf](http://arxiv.org/pdf/2502.18290v3)

**Authors**: Zhaoyi Liu, Huan Zhang

**Abstract**: Self-supervised learning (SSL) vision encoders learn high-quality image representations and thus have become a vital part of developing vision modality of large vision language models (LVLMs). Due to the high cost of training such encoders, pre-trained encoders are widely shared and deployed into many LVLMs, which are security-critical or bear societal significance. Under this practical scenario, we reveal a new backdoor threat that significant visual hallucinations can be induced into these LVLMs by merely compromising vision encoders. Because of the sharing and reuse of these encoders, many downstream LVLMs may inherit backdoor behaviors from encoders, leading to widespread backdoors. In this work, we propose BadVision, the first method to exploit this vulnerability in SSL vision encoders for LVLMs with novel trigger optimization and backdoor learning techniques. We evaluate BadVision on two types of SSL encoders and LVLMs across eight benchmarks. We show that BadVision effectively drives the LVLMs to attacker-chosen hallucination with over 99% attack success rate, causing a 77.6% relative visual understanding error while maintaining the stealthiness. SoTA backdoor detection methods cannot detect our attack effectively.

摘要: 自监督学习(SSL)视觉编码者学习高质量的图像表征，因此成为开发大型视觉语言模型(LVLMS)视觉通道的重要组成部分。由于培训这类编码器的成本很高，预先训练的编码器被广泛共享并部署到许多安全关键或具有社会意义的LVLM中。在这种实际情况下，我们揭示了一种新的后门威胁，即仅仅通过损害视觉编码器就可以在这些LVLM中诱导出显著的视觉幻觉。由于这些编码器的共享和重用，许多下游的LVLM可能会继承编码器的后门行为，导致广泛的后门。在这项工作中，我们提出了BadVision，这是第一个通过新颖的触发优化和后门学习技术来利用LVLM的SSL视觉编码器中的漏洞的方法。我们在八个基准测试中评估了BadVision在两种类型的SSL编码器和LVLM上的性能。结果表明，BadVision在保持隐蔽性的同时，以99%以上的攻击成功率有效地驱动了LVLM进入攻击者选择的幻觉，导致了77.6%的相对视觉理解错误。SOTA后门检测方法不能有效检测到我们的攻击。



## **48. TeleLoRA: Teleporting Model-Specific Alignment Across LLMs**

TeleLoRA：LLM之间远程传输模型特定的一致 cs.LG

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20228v1) [paper-pdf](http://arxiv.org/pdf/2503.20228v1)

**Authors**: Xiao Lin, Manoj Acharya, Anirban Roy, Susmit Jha

**Abstract**: Mitigating Trojans in Large Language Models (LLMs) is one of many tasks where alignment data is LLM specific, as different LLMs have different Trojan triggers and trigger behaviors to be removed. In this paper, we introduce TeleLoRA (Teleporting Low-Rank Adaptation), a novel framework that synergizes model-specific alignment data across multiple LLMs to enable zero-shot Trojan mitigation on unseen LLMs without alignment data. TeleLoRA learns a unified generator of LoRA adapter weights by leveraging local activation information across multiple LLMs. This generator is designed to be permutation symmetric to generalize across models with different architectures and sizes. We optimize the model design for memory efficiency, making it feasible to learn with large-scale LLMs with minimal computational resources. Experiments on LLM Trojan mitigation benchmarks demonstrate that TeleLoRA effectively reduces attack success rates while preserving the benign performance of the models.

摘要: 缓解大型语言模型（LLM）中的特洛伊木马是对齐数据特定于LLM的众多任务之一，因为不同的LLM具有不同的特洛伊木马触发器和要删除的触发行为。在本文中，我们介绍了TeleLoRA（远程传输低等级自适应），这是一种新颖的框架，可以在多个LLM之间协同特定于模型的对齐数据，以便在没有对齐数据的情况下对不可见的LLM实现零触发特洛伊木马缓解。TeleLoRA通过利用多个LLM之间的本地激活信息来学习LoRA适配器权重的统一生成器。该生成器被设计为排列对称，以便在具有不同架构和大小的模型之间进行概括。我们优化了模型设计以提高内存效率，使以最少的计算资源使用大规模LLM进行学习成为可能。LLM特洛伊木马缓解基准测试的实验表明，TeleLoRA有效降低了攻击成功率，同时保持了模型的良性性能。



## **49. Playing the Fool: Jailbreaking LLMs and Multimodal LLMs with Out-of-Distribution Strategy**

扮演傻瓜：越狱LLC和具有分销外策略的多模式LLC cs.CR

Accepted at CVPR2025

**SubmitDate**: 2025-03-26    [abs](http://arxiv.org/abs/2503.20823v1) [paper-pdf](http://arxiv.org/pdf/2503.20823v1)

**Authors**: Joonhyun Jeong, Seyun Bae, Yeonsung Jung, Jaeryong Hwang, Eunho Yang

**Abstract**: Despite the remarkable versatility of Large Language Models (LLMs) and Multimodal LLMs (MLLMs) to generalize across both language and vision tasks, LLMs and MLLMs have shown vulnerability to jailbreaking, generating textual outputs that undermine safety, ethical, and bias standards when exposed to harmful or sensitive inputs. With the recent advancement of safety alignment via preference-tuning from human feedback, LLMs and MLLMs have been equipped with safety guardrails to yield safe, ethical, and fair responses with regard to harmful inputs. However, despite the significance of safety alignment, research on the vulnerabilities remains largely underexplored. In this paper, we investigate the unexplored vulnerability of the safety alignment, examining its ability to consistently provide safety guarantees for out-of-distribution(OOD)-ifying harmful inputs that may fall outside the aligned data distribution. Our key observation is that OOD-ifying the vanilla harmful inputs highly increases the uncertainty of the model to discern the malicious intent within the input, leading to a higher chance of being jailbroken. Exploiting this vulnerability, we propose JOOD, a new Jailbreak framework via OOD-ifying inputs beyond the safety alignment. We explore various off-the-shelf visual and textual transformation techniques for OOD-ifying the harmful inputs. Notably, we observe that even simple mixing-based techniques such as image mixup prove highly effective in increasing the uncertainty of the model, thereby facilitating the bypass of the safety alignment. Experiments across diverse jailbreak scenarios demonstrate that JOOD effectively jailbreaks recent proprietary LLMs and MLLMs such as GPT-4 and o1 with high attack success rate, which previous attack approaches have consistently struggled to jailbreak. Code is available at https://github.com/naver-ai/JOOD.

摘要: 尽管大型语言模型(LLM)和多模式LLM(MLLM)具有惊人的通用性，可以跨语言和视觉任务进行概括，但LLM和MLLM在越狱方面表现出脆弱性，当接触到有害或敏感的输入时，会生成破坏安全、道德和偏见标准的文本输出。随着最近通过人类反馈调整偏好来促进安全匹配，LLM和MLLM已经配备了安全护栏，以对有害输入做出安全、合乎道德和公平的反应。然而，尽管安全调整具有重要意义，但对漏洞的研究在很大程度上仍未得到充分探索。在本文中，我们调查了安全对齐的未知漏洞，检查了其一致地为分布外(OOD)提供安全保证的能力-使可能属于对齐的数据分布之外的有害输入。我们的主要观察是，面向对象的有害输入极大地增加了模型的不确定性，以识别输入中的恶意意图，从而导致更高的越狱机会。利用这一漏洞，我们提出了Jood，一个新的越狱框架，通过对超出安全对齐的输入进行面向对象设计。我们探索了各种现成的视觉和文本转换技术，以实现有害输入的OOD。值得注意的是，我们观察到，即使是简单的基于混合的技术，如图像混合，也被证明在增加模型的不确定性方面非常有效，从而有助于绕过安全对齐。在各种越狱场景中的实验表明，Jood有效地越狱了最近拥有专利的LLM和MLLM，如GPT-4和O1，攻击成功率很高，而以前的攻击方法一直难以越狱。代码可在https://github.com/naver-ai/JOOD.上找到



## **50. Knowledge Transfer from LLMs to Provenance Analysis: A Semantic-Augmented Method for APT Detection**

从LLM到出处分析的知识转移：APT检测的语义增强方法 cs.CR

**SubmitDate**: 2025-03-25    [abs](http://arxiv.org/abs/2503.18316v2) [paper-pdf](http://arxiv.org/pdf/2503.18316v2)

**Authors**: Fei Zuo, Junghwan Rhee, Yung Ryn Choe

**Abstract**: Advanced Persistent Threats (APTs) have caused significant losses across a wide range of sectors, including the theft of sensitive data and harm to system integrity. As attack techniques grow increasingly sophisticated and stealthy, the arms race between cyber defenders and attackers continues to intensify. The revolutionary impact of Large Language Models (LLMs) has opened up numerous opportunities in various fields, including cybersecurity. An intriguing question arises: can the extensive knowledge embedded in LLMs be harnessed for provenance analysis and play a positive role in identifying previously unknown malicious events? To seek a deeper understanding of this issue, we propose a new strategy for taking advantage of LLMs in provenance-based threat detection. In our design, the state-of-the-art LLM offers additional details in provenance data interpretation, leveraging their knowledge of system calls, software identity, and high-level understanding of application execution context. The advanced contextualized embedding capability is further utilized to capture the rich semantics of event descriptions. We comprehensively examine the quality of the resulting embeddings, and it turns out that they offer promising avenues. Subsequently, machine learning models built upon these embeddings demonstrated outstanding performance on real-world data. In our evaluation, supervised threat detection achieves a precision of 99.0%, and semi-supervised anomaly detection attains a precision of 96.9%.

摘要: 高级持续威胁（APT）已在各个行业造成重大损失，包括敏感数据被盗和系统完整性受损。随着攻击技术变得越来越复杂和隐蔽，网络防御者和攻击者之间的军备竞赛继续加剧。大型语言模型（LLM）的革命性影响为包括网络安全在内的各个领域开辟了众多机会。一个有趣的问题出现了：LLM中嵌入的广泛知识能否用于来源分析，并在识别之前未知的恶意事件方面发挥积极作用？为了更深入地了解这个问题，我们提出了一种新的策略，用于在基于来源的威胁检测中利用LLM。在我们的设计中，最先进的LLM利用他们对系统调用、软件身份和对应用程序执行上下文的高级理解，提供了出处数据解释的更多细节。进一步利用先进的上下文嵌入能力来捕获事件描述的丰富语义。我们全面检查了所得嵌入的质量，事实证明它们提供了有希望的途径。随后，基于这些嵌入构建的机器学习模型在现实世界数据上表现出出色的性能。在我们的评估中，监督式威胁检测的准确率为99.0%，半监督式异常检测的准确率为96.9%。



