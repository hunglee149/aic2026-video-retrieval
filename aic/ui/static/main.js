/* ============================================================
   AIC 2026 — Main UI Logic
   Connects to FastAPI backend at /api/*
   ============================================================ */

'use strict';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  task: 'kis',
  candidates: [],
  selected: null,
  selections: [],
  gridMode: true,

  iterCandidates: [],
  iterCursor: 0,
  iterRound: 0,
  iterMaxRounds: 3,
  iterRunning: false,
  iterVerdict: {},
  iterMatchedList: [],
  iterUnsureList: [],
  iterExcluded: new Set(),
};

// ---------------------------------------------------------------------------
// Official Competition Queries (Pack 1)
// ---------------------------------------------------------------------------

const PACK1_QUERIES = [
  {
    id: "query-p1-1-kis",
    task: "kis",
    name: "[KIS] query-p1-1-kis: Phóng tàu vũ trụ tư nhân & cực quang",
    prompt_vi: "Cảnh bốn phi hành gia mặc trang phục màu đen trong buổi giới thiệu phóng tàu vũ trụ nghiên cứu cực quang",
    prompt_en: "A medium shot of four astronauts wearing black suits presenting a private spacecraft mission to study polar aurora"
  },
  {
    id: "query-p1-2-kis",
    task: "kis",
    name: "[KIS] query-p1-2-kis: Đàn hổ miền Nam sinh thêm 3-6 hổ con",
    prompt_vi: "Bản tin thời sự giới thiệu đàn hổ quý hiếm với các chú hổ con mới sinh tại khu bảo tồn miền Nam",
    prompt_en: "A news broadcast showing rare tiger cubs playing in a wildlife sanctuary or zoo in southern Vietnam"
  },
  {
    id: "query-p1-5-kis",
    task: "kis",
    name: "[KIS] query-p1-5-kis: Hai người phụ nữ cho dê ăn trong chuồng",
    prompt_vi: "Cảnh hai người phụ nữ mặc áo thun trắng quàng khăn đỏ và áo kẻ sọc tím đang mỉm cười cho đàn dê ăn trong chuồng gỗ",
    prompt_en: "A medium shot of two women smiling and feeding goats in a wooden barn with a corrugated metal roof"
  },
  {
    id: "query-p1-6-kis",
    task: "kis",
    name: "[KIS] query-p1-6-kis: Đầu bếp đặt món gỏi cuốn chay hoa pansy",
    prompt_vi: "Cận cảnh người đầu bếp đặt đĩa gỏi cuốn chay nhiều màu sắc trang trí hoa pansy tím vàng và rau xanh",
    prompt_en: "A close-up shot of a chef placing a plate of colorful vegetarian spring rolls garnished with green leaves and purple yellow pansy flowers"
  },
  {
    id: "query-p1-7-kis",
    task: "kis",
    name: "[KIS] query-p1-7-kis: Chim lông đen ánh xanh mắt đỏ rực dưới gốc cây",
    prompt_vi: "Cận cảnh chú chim lông đen ánh xanh cánh nâu đỏ với đôi mắt đỏ rực đứng dưới gốc cây đầy lá khô trong rừng",
    prompt_en: "A close-up shot of a bird with glossy blue-black head, reddish-brown wings, and bright red eyes standing on dry leaves under a tree"
  },
  {
    id: "query-p1-8-kis",
    task: "kis",
    name: "[KIS] query-p1-8-kis: Cô bé đeo con bạch tuộc mực đỏ trước ngực",
    prompt_vi: "Cảnh một cô bé đeo thú nhồi bông hình con bạch tuộc màu đỏ trước ngực cầm túi giấy tại lễ hội ẩm thực Nhật Bản",
    prompt_en: "A medium shot of a young girl wearing a red octopus plush on her chest and holding a paper bag at a Japanese food festival"
  },
  {
    id: "query-p1-9-kis",
    task: "kis",
    name: "[KIS] query-p1-9-kis: Thu hoạch dứa ở miền Tây ghe xanh bên bờ",
    prompt_vi: "Cảnh thu hoạch dứa miền Tây với bà cụ và cô gái áo hồng quàng khăn rằn ngồi bên ghe thuyền xanh chở đầy dứa",
    prompt_en: "A medium wide shot of an elderly woman and a girl in pink shirt with checkered scarf chatting beside baskets of harvested pineapples near a blue boat"
  },
  {
    id: "query-p1-10-kis",
    task: "kis",
    name: "[KIS] query-p1-10-kis: 3 người chơi nhạc cụ kim loại tròn trước kệ sách",
    prompt_vi: "Cảnh ba người ngồi cạnh nhau chơi nhạc cụ kim loại handpan tròn trước kệ sách nhiều màu sắc người áo trắng ngồi giữa",
    prompt_en: "A medium shot of three people sitting together playing round metallic handpan instruments in front of a colorful bookshelf"
  },
  {
    id: "query-p1-11-kis",
    task: "kis",
    name: "[KIS] query-p1-11-kis: Chàng trai xếp bìa đổ bóng chân dung người đàn ông",
    prompt_vi: "Cảnh chàng trai đội mũ lưỡi trai đen xếp các mảnh bìa đổ bóng lên tường tạo thành bức chân dung người đàn ông mặc vest",
    prompt_en: "A shot of a young man in black cap arranging cardboard pieces casting a shadow portrait of a man in suit on the wall"
  },
  {
    id: "query-p1-12-kis",
    task: "kis",
    name: "[KIS] query-p1-12-kis: Trang trí bánh rán chuối dâu rưới chocolate",
    prompt_vi: "Cận cảnh đầu bếp rưới sốt chocolate và xếp lát chuối dâu tây trang trí hai chiếc bánh rán trên đĩa sứ trắng",
    prompt_en: "A close-up shot of a chef drizzling chocolate sauce and placing sliced bananas and strawberries on two pancakes on a white ceramic plate"
  },
  {
    id: "query-p1-13-kis",
    task: "kis",
    name: "[KIS] query-p1-13-kis: Vệ sinh ống kính máy ảnh bằng tăm bông",
    prompt_vi: "Cận cảnh người dùng tăm bông cẩn thận lau chùi vệ sinh ống kính máy ảnh đặt trên khăn màu tím hồng",
    prompt_en: "A close-up shot of a person cleaning a camera lens with a cotton swab placed on a pink-purple towel"
  },
  {
    id: "query-p1-14-kis",
    task: "kis",
    name: "[KIS] query-p1-14-kis: Điêu khắc cát trượt ván & cột khói màu hồng",
    prompt_vi: "Cảnh tác phẩm điêu khắc cát hình người trượt ván trượt patin tại lễ hội điêu khắc cát ngoài trời với làn khói hồng",
    prompt_en: "A wide shot of grand sand sculptures depicting skateboarders and roller skaters at a sand sculpture festival with pink smoke"
  },
  {
    id: "query-p1-17-kis",
    task: "kis",
    name: "[KIS] query-p1-17-kis: Trao quà từ thiện COVID-19 tại bệnh viện Xuân 2024",
    prompt_vi: "Cảnh buổi lễ trao kinh phí hỗ trợ và túi quà từ thiện cho trẻ em mồ côi trước phông nền đỏ đón xuân tại bệnh viện",
    prompt_en: "A medium shot of a charity gift giving ceremony for children with medical gift bags in front of a red festive background at a hospital"
  },
  {
    id: "query-p1-20-kis",
    task: "kis",
    name: "[KIS] query-p1-20-kis: Đĩa tròn 3 ly panna cotta nho hoa ăn được",
    prompt_vi: "Cận cảnh bàn tay đặt ba ly tráng miệng panna cotta trang trí lát nho đỏ lá bạc hà và hoa ăn được trên đĩa trắng",
    prompt_en: "A close-up shot of hands placing three glasses of panna cotta dessert garnished with red grapes, mint leaves, and edible flowers on a white plate"
  },
  {
    id: "query-p1-21-kis",
    task: "kis",
    name: "[KIS] query-p1-21-kis: Cơ chế bay của bọ chế tạo robot Đại học Lausanne",
    prompt_vi: "Bản tin khoa học về nghiên cứu cơ chế bay của loài bọ cánh cứng để phát triển robot bay tại Đại học Lausanne",
    prompt_en: "A science documentary shot of robotic insect flight mechanism research at EPFL university in Lausanne"
  },
  {
    id: "query-p1-23-kis",
    task: "kis",
    name: "[KIS] query-p1-23-kis: Thị trấn ven biển cá mập phim Steven Spielberg 1975",
    prompt_vi: "Thị trấn ven biển cá mập trắng thu hút du khách bãi biển Hàm Cá Mập Steven Spielberg 1975",
    prompt_en: "A travel documentary shot of a coastal town famous for great white sharks inspired by Steven Spielberg 1975 movie"
  },
  {
    id: "query-p1-24-kis",
    task: "kis",
    name: "[KIS] query-p1-24-kis: 3 tay đua xe đạp áo trắng quần vàng xanh",
    prompt_vi: "Góc quay từ trên cao nhìn xuống ba tay đua xe đạp mặc áo trắng quần vàng xanh đang đạp thành hàng dọc trên đường đua",
    prompt_en: "An aerial overhead top-down shot of three cyclists in white jerseys and yellow-green shorts riding in a straight line on a road"
  },
  {
    id: "query-p1-25-kis",
    task: "kis",
    name: "[KIS] query-p1-25-kis: Đua xe đạp flycam áo xanh dương trắng vượt 3 tay đua",
    prompt_vi: "Góc quay flycam từ trên cao tay đua xe đạp mặc áo xanh dương trắng bứt tốc vượt qua ba tay đua khác vươn lên dẫn đầu",
    prompt_en: "An aerial flycam shot of a cyclist in blue and white jersey accelerating past three other cyclists to take the lead"
  },
  {
    id: "query-p1-15-qa",
    task: "qa",
    name: "[QA] query-p1-15-qa: CLB FANA trao quà tại xã nào tỉnh Khánh Hòa?",
    prompt_vi: "Cảnh câu lạc bộ FANA đi trao quà từ thiện tại ủy ban nhân dân xã vùng cao tỉnh Khánh Hòa",
    prompt_en: "A medium shot of FANA charity club distributing relief gifts at a rural commune in Khanh Hoa province"
  },
  {
    id: "query-p1-19-qa",
    task: "qa",
    name: "[QA] query-p1-19-qa: 2 câu thơ ca ngợi Nguyễn Trung Trực tại Kiên Giang?",
    prompt_vi: "Cảnh quay văn bia câu đối ca ngợi anh hùng dân tộc Nguyễn Trung Trực tại đình thần Rạch Giá Kiên Giang",
    prompt_en: "A shot of temple interior and poetic stone inscriptions praising national hero Nguyen Trung Truc in Kien Giang"
  },
  {
    id: "query-p1-22-qa",
    task: "qa",
    name: "[QA] query-p1-22-qa: Dạy nấu ăn 200g thịt nạc xay tên món gì?",
    prompt_vi: "Cảnh người phụ nữ hướng dẫn lớp học nấu ăn cầm bảng công thức nguyên liệu 200g thịt nạc xay",
    prompt_en: "A shot of a woman teaching a cooking class holding a recipe sheet with 200g minced pork meat ingredients"
  },
  {
    id: "query-p1-4-trake",
    task: "trake",
    name: "[TRAKE] query-p1-4-trake: Chiên măng tây (E1-E4)",
    prompt_vi: "Cảnh đầu bếp tẩm bột và chiên giòn từng cọng măng tây trong chảo dầu nóng rồi vớt ra đĩa",
    prompt_en: "A cooking shot of asparagus coated with flour batter and fried in hot oil pan then placed on plate"
  },
  {
    id: "query-p1-16-trake",
    task: "trake",
    name: "[TRAKE] query-p1-16-trake: Múa lân vàng đen trắng (E1-E4)",
    prompt_vi: "Cảnh biểu diễn múa lân màu vàng đen trắng nhảy trên giàn cột mai hoa thung rồi tiếp đất chào ban giám khảo và rồng",
    prompt_en: "A shot of yellow-black-white lion dance performing on high pillars, landing on ground and greeting dragon"
  },
  {
    id: "query-p1-18-trake",
    task: "trake",
    name: "[TRAKE] query-p1-18-trake: Nấu ăn món nấm sơ chế (E1-E4)",
    prompt_vi: "Cảnh đầu bếp sơ chế cắt nấm cắt củ năng cắt đậu hũ rồi bật bếp lửa nấu ăn",
    prompt_en: "A cooking preparation shot showing chef chopping mushrooms, water chestnuts, tofu, and turning on stove fire"
  }
];

function loadPresetQuery(qid) {
  if (!qid) return;
  const q = PACK1_QUERIES.find(item => item.id === qid);
  if (!q) return;

  const qInput = $('query-id-input');
  if (qInput) qInput.value = q.id;

  const queryTxt = $('query-input');
  if (queryTxt) queryTxt.value = q.prompt_vi || q.text_vi;

  const transTxt = $('translated-text');
  if (transTxt) transTxt.value = q.prompt_en || '';

  selectTask(q.task);
  toast(`Đã nạp câu ${q.id} (Caption Prompt chuẩn)`, 'info');
}

function initPresetDropdown() {
  const select = $('preset-query-select');
  if (!select) return;
  select.innerHTML = '<option value="">-- Chọn nhanh câu hỏi đề thi (24 câu) --</option>' +
    PACK1_QUERIES.map(q => `<option value="${q.id}">${q.name}</option>`).join('');
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

function $(id) { return document.getElementById(id); }

function toast(msg, type = 'info') {
  const el = document.createElement('div');
  el.className = `toast toast-${type}`;
  el.textContent = msg;
  $('toast-container').appendChild(el);
  requestAnimationFrame(() => el.classList.add('show'));
  setTimeout(() => {
    el.classList.remove('show');
    setTimeout(() => el.remove(), 300);
  }, 3200);
}

function setLoading(btnId, loading) {
  const btn = $(btnId);
  if (!btn) return;
  btn.disabled = loading;
  if (loading) {
    btn.dataset.origText = btn.innerHTML;
    btn.innerHTML = '<span class="spinner-ring" style="width:14px;height:14px;border-width:2px;display:inline-block"></span>';
  } else {
    if (btn.dataset.origText) btn.innerHTML = btn.dataset.origText;
  }
}

function fmtScore(v) {
  if (v === undefined || v === null) return '—';
  return Number(v).toFixed(4);
}

function scoreClass(v) {
  if (v >= 0.6) return 'score-high';
  if (v >= 0.35) return 'score-mid';
  return 'score-low';
}

function keyframeUrl(videoId, frameIdx) {
  return `/api/keyframe/${encodeURIComponent(videoId)}/${frameIdx}`;
}

function candidateKey(c) {
  return `${c.video_id}__${c.representative_frames[0] ?? c.start_frame}`;
}

// ---------------------------------------------------------------------------
// Status check
// ---------------------------------------------------------------------------

async function checkStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();
    if (data.ok) {
      $('status-text').textContent = data.retriever;
      $('status-dot').style.background = 'var(--green)';
      $('status-dot').style.boxShadow = '0 0 6px var(--green)';
      $('stat-keyframes').textContent = data.retriever === 'dummy' ? 'demo' : '—';
    }
  } catch {
    $('status-text').textContent = 'Offline';
    $('status-dot').style.background = 'var(--red)';
    $('status-dot').style.boxShadow = '0 0 6px var(--red)';
  }
}

// ---------------------------------------------------------------------------
// View switching
// ---------------------------------------------------------------------------

function switchView(view) {
  document.querySelectorAll('.view').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(el => el.classList.remove('active'));
  const el = $(`view-${view}`);
  if (el) el.classList.add('active');
  const tab = $(`tab-${view}`);
  if (tab) tab.classList.add('active');
  if (view === 'export') renderExportTable();
}

// ---------------------------------------------------------------------------
// Task selection
// ---------------------------------------------------------------------------

function selectTask(task) {
  state.task = task;
  ['kis', 'qa', 'trake'].forEach(t => {
    $(`pill-${t}`).classList.toggle('active', t === task);
  });
  $('n-events-section').style.display = task === 'trake' ? '' : 'none';
  $('answer-section').style.display = task === 'qa' ? '' : 'none';
  $('results-task-badge').textContent = task.toUpperCase();

  // Tự động đồng bộ đuôi Query ID theo chuẩn BTC (vd: query-1-kis -> query-1-qa)
  const qInput = $('query-id-input');
  if (qInput) {
    const curVal = qInput.value.trim();
    const match = curVal.match(/^(query-\d+)(?:-(?:kis|qa|trake))?$/i);
    if (match) {
      qInput.value = `${match[1]}-${task}`;
    }
  }
}

// ---------------------------------------------------------------------------
// Translate
// ---------------------------------------------------------------------------

async function doTranslate() {
  const text_vi = $('query-input').value.trim();
  if (!text_vi) { toast('Nhập câu hỏi tiếng Việt trước', 'warning'); return; }
  setLoading('btn-translate', true);
  try {
    const res = await fetch('/api/translate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text_vi }),
    });
    const data = await res.json();
    $('translated-text').value = data.text_en || '';
    if (!data.ok) toast(`Dịch thất bại: ${data.error || ''}`, 'warning');
    else toast('Đã dịch thành công', 'success');
  } catch (e) {
    toast('Lỗi kết nối', 'error');
  } finally {
    setLoading('btn-translate', false);
  }
}

function getCheckedModalities() {
  const mods = [];
  const map = {
    'mod-siglip': 'siglip',
    'mod-caption': 'caption',
    'mod-ocr': 'ocr',
    'mod-asr': 'asr',
    'mod-summary': 'summary',
    'mod-media_info': 'media_info',
  };
  for (const [id, val] of Object.entries(map)) {
    const el = $(id);
    if (el && el.checked) mods.push(val);
  }
  return mods.length ? mods : ['siglip', 'caption', 'ocr', 'asr', 'summary', 'media_info'];
}

function setModalityPreset(preset) {
  const allIds = ['mod-siglip', 'mod-caption', 'mod-ocr', 'mod-asr', 'mod-summary', 'mod-media_info'];
  ['preset-mod-all', 'preset-mod-asr', 'preset-mod-ocr', 'preset-mod-visual'].forEach(pid => {
    const el = $(pid);
    if (el) el.classList.remove('active');
  });

  const activeBtn = $(`preset-mod-${preset}`);
  if (activeBtn) activeBtn.classList.add('active');

  if (preset === 'all') {
    allIds.forEach(id => { const el = $(id); if (el) el.checked = true; });
    toast('Chế độ: Trộn toàn bộ các nguồn (Hybrid Fusion)', 'info');
  } else if (preset === 'asr') {
    allIds.forEach(id => { const el = $(id); if (el) el.checked = (id === 'mod-asr' || id === 'mod-summary'); });
    toast('Chế độ: Chuyên Lời thoại & Giọng nói (ASR)', 'info');
  } else if (preset === 'ocr') {
    allIds.forEach(id => { const el = $(id); if (el) el.checked = (id === 'mod-ocr'); });
    toast('Chế độ: Chuyên Chữ màn hình & Logo (OCR)', 'info');
  } else if (preset === 'visual') {
    allIds.forEach(id => { const el = $(id); if (el) el.checked = (id === 'mod-siglip' || id === 'mod-caption'); });
    toast('Chế độ: Chuyên Hình ảnh thị giác (SigLIP2 + Caption)', 'info');
  }
}

function updateModalityState() {
  ['preset-mod-all', 'preset-mod-asr', 'preset-mod-ocr', 'preset-mod-visual'].forEach(pid => {
    const el = $(pid);
    if (el) el.classList.remove('active');
  });
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

async function doSearch() {
  const text_vi = $('query-input').value.trim();
  if (!text_vi) { toast('Nhập câu hỏi', 'warning'); return; }
  const text_en = $('translated-text').value.trim();
  const query_id = $('query-id-input').value.trim() || 'q1';
  const k = parseInt($('topk-slider').value, 10);
  const n_events = parseInt($('n-events-input').value, 10) || 1;
  const modalities = getCheckedModalities();

  setLoading('btn-search', true);
  $('candidates-grid').innerHTML = '<div class="spinner"><div class="spinner-ring"></div></div>';

  try {
    const res = await fetch('/api/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query_id, text_vi, text_en, task: state.task, n_events, k, modalities }),
    });
    const data = await res.json();
    if (!data.ok) throw new Error(data.detail || 'Search failed');

    state.candidates = data.candidates;
    state.selected = null;
    renderCandidates();
    $('results-count').textContent = `${data.total} candidates`;
    toast(`Tìm được ${data.total} kết quả (Nguồn: ${modalities.join(', ')})`, 'success');
  } catch (e) {
    $('candidates-grid').innerHTML = `<div class="empty-state"><p style="color:var(--red)">❌ ${e.message}</p></div>`;
    toast(e.message, 'error');
  } finally {
    setLoading('btn-search', false);
  }
}

// ---------------------------------------------------------------------------
// Render candidates grid
// ---------------------------------------------------------------------------

function renderCandidates() {
  const grid = $('candidates-grid');
  if (!state.candidates.length) {
    grid.innerHTML = '<div class="empty-state"><p>Không có kết quả</p></div>';
    return;
  }

  grid.innerHTML = '';
  state.candidates.forEach((c, idx) => {
    const frameIdx = c.representative_frames[0] ?? c.start_frame;
    const card = document.createElement('div');
    card.className = 'candidate-card';
    card.dataset.idx = idx;

    const key = candidateKey(c);
    const verdict = state.iterVerdict[key];
    if (verdict === 'matched') card.classList.add('matched');
    else if (verdict === 'not_matched') card.classList.add('not-matched');
    else if (verdict === 'unsure') card.classList.add('unsure');

    const scoreCls = scoreClass(c.best_score);
    const kfUrl = keyframeUrl(c.video_id, frameIdx);

    const scoreBadges = [];
    if (c.scores) {
      if (c.scores.siglip !== undefined && c.scores.siglip > 0) {
        scoreBadges.push(`<span class="sc-badge sc-siglip" title="SigLIP2 Visual">👁️ ${c.scores.siglip.toFixed(3)}</span>`);
      }
      if (c.scores.asr !== undefined && c.scores.asr > 0) {
        scoreBadges.push(`<span class="sc-badge sc-asr" title="ASR Lời thoại">🎙️ ${c.scores.asr.toFixed(3)}</span>`);
      }
      if (c.scores.ocr !== undefined && c.scores.ocr > 0) {
        scoreBadges.push(`<span class="sc-badge sc-ocr" title="OCR Chữ màn hình">🔤 ${c.scores.ocr.toFixed(3)}</span>`);
      }
      if (c.scores.caption !== undefined && c.scores.caption > 0) {
        scoreBadges.push(`<span class="sc-badge sc-caption" title="VLM Caption">📝 ${c.scores.caption.toFixed(3)}</span>`);
      }
      if (c.scores.summary !== undefined && c.scores.summary > 0) {
        scoreBadges.push(`<span class="sc-badge sc-summary" title="Chủ đề Video">📋 ${c.scores.summary.toFixed(3)}</span>`);
      }
      if (c.scores.media_info !== undefined && c.scores.media_info > 0) {
        scoreBadges.push(`<span class="sc-badge sc-media" title="Media Info">ℹ️ ${c.scores.media_info.toFixed(3)}</span>`);
      }
    }

    card.innerHTML = `
      <div class="card-thumb" ondblclick="openLightbox('${kfUrl}', '${c.video_id} (Frame #${frameIdx})')" title="Nhấp đúp để phóng to ảnh">
        <img src="${kfUrl}" alt="${c.video_id}" loading="lazy"/>
        <div class="card-rank" onclick="promptChangeRank(${idx}, event)" title="Nhấp để đổi thứ hạng Rank">#${c.rank} ✏️</div>
        <div class="card-score ${scoreCls}">${fmtScore(c.best_score)}</div>
      </div>
      <div class="card-body">
        <div class="card-video-id" onclick="promptChangeVideo(${idx}, event)" title="Nhấp để đổi Video / Frame cho Top #${c.rank}">${c.video_id} ✏️</div>
        <div class="card-frame">frame ${frameIdx} &mdash; ${c.start_frame}&rarr;${c.end_frame}</div>
        ${scoreBadges.length ? `<div class="card-mod-badges">${scoreBadges.join('')}</div>` : ''}
      </div>
      <div class="verdict-row">
        <button class="verdict-btn v-matched ${verdict === 'matched' ? 'active' : ''}"
          onclick="setVerdict(${idx},'matched',event)" title="Matched">✓</button>
        <button class="verdict-btn v-not ${verdict === 'not_matched' ? 'active' : ''}"
          onclick="setVerdict(${idx},'not_matched',event)" title="Not matched">✗</button>
        <button class="verdict-btn v-unsure ${verdict === 'unsure' ? 'active' : ''}"
          onclick="setVerdict(${idx},'unsure',event)" title="Unsure">?</button>
      </div>`;

    card.addEventListener('click', (e) => {
      if (e.target.closest('.verdict-btn') || e.target.closest('.card-rank') || e.target.closest('.card-video-id')) return;
      selectCandidate(idx);
    });
    grid.appendChild(card);
  });
}

// ---------------------------------------------------------------------------
// Rank & Video Replacement Controls
// ---------------------------------------------------------------------------

function promptChangeRank(idx, e) {
  if (e) e.stopPropagation();
  const c = state.candidates[idx];
  if (!c) return;
  const curRank = c.rank;
  const input = prompt(`Đổi thứ hạng của ${c.video_id} (hiện tại Rank #${curRank}):\nNhập số Rank mới (1 - ${state.candidates.length}):`, curRank);
  if (input !== null) {
    const targetRank = parseInt(input.trim(), 10);
    if (!isNaN(targetRank) && targetRank >= 1 && targetRank <= state.candidates.length) {
      moveCandidateToRank(idx, targetRank);
    } else {
      toast('Số thứ hạng không hợp lệ', 'warning');
    }
  }
}

function promptChangeVideo(idx, e) {
  if (e) e.stopPropagation();
  const c = state.candidates[idx];
  if (!c) return;

  const newVid = prompt(`Đổi Video cho Top #${c.rank} (hiện tại: ${c.video_id}):\nNhập mã Video ID mới (VD: L21_V024, L26_V159):`, c.video_id);
  if (!newVid || !newVid.trim()) return;

  const curFrame = c.representative_frames[0] ?? c.start_frame ?? 1;
  const newFrameStr = prompt(`Nhập số Frame cho ${newVid.trim()}:`, curFrame);
  const newFrame = parseInt(newFrameStr, 10) || 1;

  c.video_id = newVid.trim();
  c.representative_frames = [newFrame];
  c.start_frame = newFrame;
  c.end_frame = newFrame;

  renderCandidates();
  selectCandidate(idx);
  toast(`Đã đổi Top #${c.rank} thành ${c.video_id} (Frame #${newFrame})`, 'success');
}

function applyCustomVideoAndFrame(silent = false) {
  if (state.selected === null || !state.candidates[state.selected]) {
    if (!silent) toast('Hãy chọn 1 video trong danh sách trước', 'warning');
    return;
  }
  const vidInput = $('custom-video-input');
  const kfInput = $('custom-frame-input');
  if (!vidInput || !kfInput) return;

  const newVid = vidInput.value.trim();
  const rawFrame = kfInput.value.trim();
  const newFrame = parseInt(rawFrame, 10) || 1;

  if (!newVid) {
    if (!silent) toast('Vui lòng nhập mã Video ID (Ví dụ: L25_V083)', 'warning');
    return;
  }

  const c = state.candidates[state.selected];
  const oldVid = c.video_id;
  const vidChanged = (c.video_id !== newVid);

  c.video_id = newVid;
  c.representative_frames = [newFrame];
  c.start_frame = newFrame;
  c.end_frame = newFrame;

  // 1. Cập nhật thẻ Candidate bên trái ngay lập tức (ảnh + video id + frame)
  const cardEl = document.querySelector(`.candidate-card[data-idx="${state.selected}"]`);
  if (cardEl) {
    const cardImg = cardEl.querySelector('.card-thumb img');
    if (cardImg) cardImg.src = keyframeUrl(newVid, newFrame);
    const cardVid = cardEl.querySelector('.card-video-id');
    if (cardVid) cardVid.innerHTML = `${newVid} ✏️`;
    const cardFrame = cardEl.querySelector('.card-frame');
    if (cardFrame) cardFrame.innerHTML = `frame ${newFrame} &mdash; ${newFrame}&rarr;${newFrame}`;
  }

  // 2. Cập nhật ảnh Preview lớn ở trên
  const img = $('preview-img');
  const placeholder = $('preview-placeholder');
  if (img) {
    img.style.display = 'none';
    if (placeholder) placeholder.style.display = 'flex';
    img.onload = () => { img.style.display = 'block'; if (placeholder) placeholder.style.display = 'none'; };
    img.onerror = () => { img.style.display = 'none'; if (placeholder) placeholder.style.display = 'flex'; };
    img.src = keyframeUrl(newVid, newFrame);
    img.alt = newVid;
  }

  // 3. Đồng bộ ô Chọn frame
  const frameInput = $('frame-input');
  if (frameInput && frameInput.value != newFrame) {
    frameInput.value = newFrame;
  }

  // 4. Cập nhật bằng chứng Evidence (fetch dữ liệu của Video & Frame mới)
  fetch(`/api/evidence/${encodeURIComponent(newVid)}/${newFrame}`)
    .then(r => r.json())
    .then(data => {
      if (data.ok) {
        c.evidence = data;
        renderEvidencePanel(data, newVid, newFrame);
      } else {
        renderEvidencePanel({ batch: newVid.split('_')[0], video_summary: '', caption: '', ocr: '', transcript: '' }, newVid, newFrame);
      }
    })
    .catch(() => {
      renderEvidencePanel({ batch: newVid.split('_')[0], video_summary: '', caption: '', ocr: '', transcript: '' }, newVid, newFrame);
    });

  // 5. Nạp lại dải timeline keyframes nếu đổi sang video khác
  loadVideoTimeline(newVid, newFrame);

  if (!silent) {
    toast(`Đã đổi Top #${c.rank} sang ${newVid} (Frame #${newFrame})`, 'success');
  }
}

function moveSelectedToTop() {
  if (state.selected === null || !state.candidates[state.selected]) {
    toast('Chưa chọn video nào', 'warning');
    return;
  }
  moveCandidateToRank(state.selected, 1);
}

function onDetailRankInputChange(val) {
  if (state.selected === null || !state.candidates[state.selected]) return;
  const targetRank = parseInt(val, 10);
  if (!isNaN(targetRank) && targetRank >= 1 && targetRank <= state.candidates.length) {
    moveCandidateToRank(state.selected, targetRank);
  } else {
    toast('Số thứ hạng không hợp lệ', 'warning');
    const input = $('detail-rank-input');
    if (input) input.value = state.candidates[state.selected].rank;
  }
}

function moveCandidateToRank(fromIdx, targetRank) {
  if (fromIdx < 0 || fromIdx >= state.candidates.length) return;
  const targetIdx = targetRank - 1;
  if (fromIdx === targetIdx) return;

  const [c] = state.candidates.splice(fromIdx, 1);
  state.candidates.splice(targetIdx, 0, c);

  // Cập nhật lại số rank cho toàn bộ candidates
  state.candidates.forEach((cand, i) => {
    cand.rank = i + 1;
  });

  renderCandidates();
  selectCandidate(targetIdx);
  toast(`Đã chuyển ${c.video_id} sang Rank #${targetRank}`, 'success');
}

function setGridMode(isGrid) {
  state.gridMode = isGrid;
  $('candidates-grid').classList.toggle('list-mode', !isGrid);
  $('btn-grid-view').classList.toggle('active', isGrid);
  $('btn-list-view').classList.toggle('active', !isGrid);
}

// ---------------------------------------------------------------------------
// Verdict on card
// ---------------------------------------------------------------------------

function setVerdict(idx, verdict, e) {
  if (e) e.stopPropagation();
  const c = state.candidates[idx];
  if (!c) return;
  const key = candidateKey(c);
  state.iterVerdict[key] = verdict;
  renderCandidates();
  if (state.selected === idx) selectCandidate(idx);
}

// ---------------------------------------------------------------------------
// Detail panel
// ---------------------------------------------------------------------------

function selectCandidate(idx) {
  state.selected = idx;
  const c = state.candidates[idx];

  document.querySelectorAll('.candidate-card').forEach((el, i) => {
    el.classList.toggle('selected', i === idx);
  });

  const frameIdx = c.representative_frames[0] ?? c.start_frame;
  const img = $('preview-img');
  const placeholder = $('preview-placeholder');
  img.style.display = 'none';
  placeholder.style.display = 'flex';
  img.onload = () => { img.style.display = 'block'; placeholder.style.display = 'none'; };
  img.onerror = () => { img.style.display = 'none'; placeholder.style.display = 'flex'; };
  img.src = keyframeUrl(c.video_id, frameIdx);
  img.alt = c.video_id;

  const rankInput = $('detail-rank-input');
  if (rankInput) rankInput.value = c.rank;

  const rankBadge = $('detail-rank-badge');
  if (rankBadge) rankBadge.textContent = `#${c.rank}`;

  const scoresSection = $('detail-scores-section');
  const scoresBody = $('detail-scores-body');
  const entries = Object.entries(c.scores || {});
  if (entries.length) {
    scoresSection.style.display = '';
    scoresBody.innerHTML = entries.map(([k, v]) =>
      `<div class="score-row"><span class="score-key">${k}</span><span class="score-val">${fmtScore(v)}</span></div>`
    ).join('') +
    `<div class="score-row"><span class="score-key" style="font-weight:700">best</span><span class="score-val" style="color:var(--purple-light)">${fmtScore(c.best_score)}</span></div>`;
  } else {
    scoresSection.style.display = 'none';
  }

  const evidenceSection = $('detail-evidence-section');
  if (evidenceSection) {
    renderEvidencePanel(c.evidence || {}, c.video_id, frameIdx);
  }

  $('frame-input').value = frameIdx;
  const customVidInput = $('custom-video-input');
  const customKfInput = $('custom-frame-input');
  if (customVidInput) customVidInput.value = c.video_id;
  if (customKfInput) customKfInput.value = frameIdx;
  $('btn-confirm-selection').disabled = false;

  // Tải danh sách toàn bộ keyframes của video lên Timeline Gallery
  loadVideoTimeline(c.video_id, frameIdx);
}

function renderEvidencePanel(ev, videoId, frameIdx) {
  const evidenceSection = $('detail-evidence-section');
  if (!evidenceSection) return;
  evidenceSection.style.display = '';
  const textEl = $('detail-evidence-text');
  if (!textEl) return;

  const batch = ev.batch || (videoId.includes('_') ? `Tập ${videoId.split('_')[0]}` : videoId);
  const summary = ev.video_summary || '';
  const caption = ev.caption || ev.caption_match || '';
  const ocr = ev.ocr || ev.ocr_match || '';
  const transcript = ev.transcript || ev.transcript_match || '';

  let html = `
    <div class="evidence-header-badges">
      <span class="evidence-badge batch-badge">📦 ${batch}</span>
      <span class="evidence-badge video-badge">🎬 ${videoId}</span>
      <span class="evidence-badge frame-badge">🖼️ Frame #${frameIdx}</span>
    </div>
  `;

  if (summary) {
    html += `
      <div class="evidence-block summary-block">
        <div class="evidence-label">📋 Tóm tắt nội dung video:</div>
        <div class="evidence-content">${summary}</div>
      </div>
    `;
  }

  if (caption) {
    html += `
      <div class="evidence-block caption-block">
        <div class="evidence-label">📝 VLM Caption (Mô tả cảnh quay):</div>
        <div class="evidence-content">${caption}</div>
      </div>
    `;
  }

  if (ocr) {
    html += `
      <div class="evidence-block ocr-block">
        <div class="evidence-label">🔤 Chữ trên màn hình (OCR / Logo / Banner):</div>
        <div class="evidence-content">${ocr}</div>
      </div>
    `;
  }

  if (transcript) {
    html += `
      <div class="evidence-block asr-block">
        <div class="evidence-label">🎙️ Lời thoại / Phát thanh (ASR):</div>
        <div class="evidence-content">${transcript}</div>
      </div>
    `;
  }

  const mediaInfo = ev.media_info_match || '';
  if (mediaInfo) {
    html += `
      <div class="evidence-block" style="background:rgba(148,163,184,0.08);border-color:rgba(148,163,184,0.25)">
        <div class="evidence-label" style="color:#94a3b8">ℹ️ Thông tin Video (Media Info):</div>
        <div class="evidence-content">${mediaInfo}</div>
      </div>
    `;
  }

  textEl.innerHTML = html;
}

// ---------------------------------------------------------------------------
// Video Timeline Keyframes Browser
// ---------------------------------------------------------------------------

state.videoKeyframes = [];
state.currentTimelineIdx = 0;
state.currentVideoId = null;

async function loadVideoTimeline(videoId, activeFrame) {
  const section = $('detail-timeline-section');
  const strip = $('timeline-strip');
  const badge = $('timeline-count-badge');
  const slider = $('timeline-slider');
  const info = $('timeline-current-info');
  if (!section || !strip) return;

  section.style.display = '';
  strip.innerHTML = '<div style="color:var(--text-muted);font-size:11px;padding:8px">Đang tải keyframes...</div>';
  state.currentVideoId = videoId;

  try {
    const res = await fetch(`/api/video_keyframes/${encodeURIComponent(videoId)}`);
    const data = await res.json();
    if (!data.ok || !data.keyframes || !data.keyframes.length) {
      section.style.display = 'none';
      return;
    }

    state.videoKeyframes = data.keyframes;
    badge.textContent = `${data.keyframes.length} frames`;
    strip.innerHTML = '';

    if (slider) {
      slider.min = 0;
      slider.max = data.keyframes.length - 1;
    }

    let activeIdx = 0;
    data.keyframes.forEach((kf, idx) => {
      const kfIdx = kf.frame_idx ?? kf.kf_num;
      const kfNum = kf.kf_num ?? kfIdx;
      const isActive = (kfNum === activeFrame || kfIdx === activeFrame);
      if (isActive) activeIdx = idx;

      const item = document.createElement('div');
      item.className = `timeline-item ${isActive ? 'active' : ''}`;
      item.dataset.index = idx;
      const timeStr = (kf.pts_time !== null && kf.pts_time !== undefined) ? `${Number(kf.pts_time).toFixed(1)}s` : `#${kfNum}`;
      item.title = `Keyframe #${kfNum} | Frame idx: ${kfIdx} (${timeStr})`;
      
      const imgUrl = keyframeUrl(videoId, kfNum);
      item.innerHTML = `
        <img src="${imgUrl}" alt="f${kfNum}" loading="lazy"/>
        <div class="timeline-item-label">${timeStr}</div>
      `;

      item.addEventListener('click', () => {
        selectTimelineKeyframe(idx);
      });

      strip.appendChild(item);
    });

    state.currentTimelineIdx = activeIdx;
    if (slider) slider.value = activeIdx;
    updateTimelineInfo(activeIdx);

    // Tự động cuộn đến keyframe đang chọn
    setTimeout(() => {
      const activeEl = strip.querySelector('.timeline-item.active');
      if (activeEl) {
        activeEl.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
      }
    }, 100);
  } catch (e) {
    section.style.display = 'none';
  }
}

function selectTimelineKeyframe(idx) {
  if (!state.videoKeyframes || idx < 0 || idx >= state.videoKeyframes.length) return;
  state.currentTimelineIdx = idx;
  const kf = state.videoKeyframes[idx];
  const kfIdx = kf.frame_idx ?? kf.kf_num;
  const kfNum = kf.kf_num ?? kfIdx;

  const strip = $('timeline-strip');
  if (strip) {
    strip.querySelectorAll('.timeline-item').forEach((el, i) => {
      el.classList.toggle('active', i === idx);
    });
    const activeEl = strip.querySelector(`.timeline-item[data-index="${idx}"]`);
    if (activeEl) {
      activeEl.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
    }
  }

  const slider = $('timeline-slider');
  if (slider) slider.value = idx;

  const img = $('preview-img');
  if (img && state.currentVideoId) {
    img.src = keyframeUrl(state.currentVideoId, kfNum);
    img.alt = state.currentVideoId;
  }
  $('frame-input').value = kfIdx;
  const customKfInput = $('custom-frame-input');
  if (customKfInput) customKfInput.value = kfIdx;
  updateTimelineInfo(idx);

  // Đồng bộ frame mới vào candidate đang chọn (bên trái)
  if (state.selected !== null && state.candidates[state.selected]) {
    const c = state.candidates[state.selected];
    if (c.video_id === state.currentVideoId) {
      c.representative_frames = [kfNum];
      c.start_frame = kfNum;
      c.end_frame = kfNum;
      
      const cardEl = document.querySelector(`.candidate-card[data-idx="${state.selected}"]`);
      if (cardEl) {
        const cardImg = cardEl.querySelector('.card-thumb img');
        if (cardImg) cardImg.src = keyframeUrl(state.currentVideoId, kfNum);
        const cardFrame = cardEl.querySelector('.card-frame');
        if (cardFrame) cardFrame.innerHTML = `frame ${kfNum} &mdash; ${kfNum}&rarr;${kfNum}`;
      }
    }
  }

  // Cập nhật bằng chứng (Caption & OCR) cho khung hình đang duyệt
  if (state.currentVideoId) {
    fetch(`/api/evidence/${state.currentVideoId}/${kfNum}`)
      .then(r => r.json())
      .then(data => {
        if (data.ok) renderEvidencePanel(data, state.currentVideoId, kfNum);
      })
      .catch(() => {});
  }
}

function updateTimelineInfo(idx) {
  const info = $('timeline-current-info');
  if (!info || !state.videoKeyframes[idx]) return;
  const kf = state.videoKeyframes[idx];
  const kfNum = kf.kf_num ?? kf.frame_idx;
  const timeStr = (kf.pts_time !== null && kf.pts_time !== undefined) ? `${Number(kf.pts_time).toFixed(1)}s` : `#${kfNum}`;
  info.textContent = `#${kfNum} (${timeStr})`;
}

function stepTimeline(delta) {
  if (!state.videoKeyframes.length) return;
  const newIdx = Math.max(0, Math.min(state.videoKeyframes.length - 1, state.currentTimelineIdx + delta));
  selectTimelineKeyframe(newIdx);
}

function onTimelineSliderInput(val) {
  selectTimelineKeyframe(parseInt(val, 10));
}

function toggleTimelineGrid() {
  const strip = $('timeline-strip');
  const btn = $('btn-timeline-grid');
  if (!strip) return;
  const isGrid = strip.classList.toggle('grid-mode');
  if (btn) btn.textContent = isGrid ? '═ Dải ngang' : '⊞ Lưới';
}

// Phím tắt mũi tên trái/phải để tua frame
document.addEventListener('keydown', (e) => {
  if (document.activeElement && ['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) return;
  if (e.key === 'ArrowLeft') {
    e.preventDefault();
    stepTimeline(-1);
  } else if (e.key === 'ArrowRight') {
    e.preventDefault();
    stepTimeline(1);
  }
});

// ---------------------------------------------------------------------------
// Lightbox Fullscreen Viewer
// ---------------------------------------------------------------------------

function openLightbox(src, title) {
  closeLightbox();
  const modal = document.createElement('div');
  modal.id = 'active-lightbox';
  modal.className = 'lightbox-modal';
  modal.innerHTML = `
    <div class="lightbox-content" onclick="event.stopPropagation()">
      <button class="lightbox-close" onclick="closeLightbox()" title="Đóng (ESC)">✕</button>
      <img src="${src}" class="lightbox-img" alt="${title}"/>
      <div class="lightbox-info">${title}</div>
    </div>`;
  modal.addEventListener('click', closeLightbox);
  document.body.appendChild(modal);
}

function closeLightbox() {
  const el = $('active-lightbox');
  if (el) el.remove();
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeLightbox();
});

// Gắn sự kiện click zoom cho ảnh preview bên phải
document.addEventListener('DOMContentLoaded', () => {
  const previewWrap = $('preview-img-wrap');
  if (previewWrap) {
    previewWrap.addEventListener('click', () => {
      const img = $('preview-img');
      if (img && img.src && img.style.display !== 'none') {
        openLightbox(img.src, `${img.alt} (Frame #${$('frame-input').value || 0})`);
      }
    });
  }
});

// ---------------------------------------------------------------------------
// Confirm selection
// ---------------------------------------------------------------------------

function confirmSelection() {
  if (state.selected === null) return;
  const c = state.candidates[state.selected];
  const frameInput = parseInt($('frame-input').value, 10);
  const frame = isNaN(frameInput) ? (c.representative_frames[0] ?? c.start_frame) : frameInput;
  const answer = $('answer-input') ? $('answer-input').value.trim() : '';
  const queryId = $('query-id-input').value.trim() || 'q1';

  const existing = state.selections.findIndex(s => s.video_id === c.video_id && s.queryId === queryId);
  if (existing >= 0) state.selections.splice(existing, 1);

  state.selections.push({ video_id: c.video_id, frames: [frame], answer, queryId, task: state.task, rank: c.rank });
  renderSelectionsList();
  toast(`Đã thêm #${c.rank} ${c.video_id} frame ${frame}`, 'success');
}

function addAllTopKToExport() {
  if (!state.candidates || !state.candidates.length) {
    toast('Chưa có kết quả tìm kiếm nào!', 'warning');
    return;
  }
  const queryId = $('query-id-input').value.trim() || 'q1';
  const answer = ($('answer-input') ? $('answer-input').value.trim() : '');

  // Lọc bỏ các kết quả cũ của câu queryId này (để nạp danh sách mới nhất)
  state.selections = state.selections.filter(s => s.queryId !== queryId);

  // Thêm toàn bộ candidates theo đúng thứ tự xếp hạng (tối đa 100)
  state.candidates.forEach((c, idx) => {
    let frame;
    // Nếu ứng viên đang được chọn và người dùng đã chỉnh ô frame-input thì lấy frame người dùng chọn
    if (state.selected === idx && $('frame-input').value) {
      const parsed = parseInt($('frame-input').value, 10);
      frame = isNaN(parsed) ? (c.representative_frames[0] ?? c.start_frame) : parsed;
    } else {
      frame = c.representative_frames[0] ?? c.start_frame;
    }
    state.selections.push({
      video_id: c.video_id,
      frames: [frame],
      answer: answer,
      queryId: queryId,
      task: state.task,
      rank: c.rank || (idx + 1),
    });
  });

  renderSelectionsList();
  toast(`⚡ Đã đưa toàn bộ ${state.candidates.length} kết quả vào Export cho ${queryId}!`, 'success');
}

function removeSelection(idx) {
  state.selections.splice(idx, 1);
  renderSelectionsList();
  renderExportTable();
}

function removeQueryCluster(qid) {
  if (!qid) return;
  const count = state.selections.filter(s => s.queryId === qid).length;
  if (!count) {
    toast(`Không tìm thấy kết quả nào của cụm "${qid}"`, 'info');
    return;
  }
  if (confirm(`Bạn có chắc chắn muốn xóa toàn bộ ${count} kết quả của cụm câu "${qid}"?`)) {
    state.selections = state.selections.filter(s => s.queryId !== qid);
    renderSelectionsList();
    renderExportTable();
    toast(`Đã xóa sạch cụm câu ${qid} (${count} kết quả)`, 'success');
  }
}

function removeCurrentQueryCluster() {
  const curQid = $('query-id-input') ? $('query-id-input').value.trim() : '';
  if (!curQid) {
    toast('Chưa có mã câu hỏi nào', 'warning');
    return;
  }
  removeQueryCluster(curQid);
}

function renderSelectionsList() {
  const list = $('selections-list');
  if (!list) return;
  $('sel-count').textContent = state.selections.length;
  if (!state.selections.length) {
    list.innerHTML = '<div style="color:var(--text-muted);font-size:12px;text-align:center;padding:12px">Chưa có lựa chọn nào</div>';
    return;
  }

  // Nhóm các kết quả theo queryId
  const byQuery = {};
  state.selections.forEach((s, idx) => {
    const q = s.queryId || 'query-1';
    if (!byQuery[q]) byQuery[q] = [];
    byQuery[q].push({ item: s, origIndex: idx });
  });

  const curQid = $('query-id-input') ? $('query-id-input').value.trim() : '';

  let html = `
    <div class="cluster-actions-bar">
      <button class="btn-cluster-del" onclick="removeCurrentQueryCluster()" title="Xóa toàn bộ kết quả của câu đang mở">
        🗑️ Xóa cụm "${curQid || 'câu này'}"
      </button>
      <button class="btn-cluster-del del-all" onclick="clearAllSelections()" title="Xóa sạch toàn bộ mọi câu">
        🗑️ Xóa tất cả
      </button>
    </div>
  `;

  const sortedQids = Object.keys(byQuery).sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }));

  sortedQids.forEach(qid => {
    const items = byQuery[qid];
    const isCurrent = (qid === curQid);
    html += `
      <div class="selection-cluster-card ${isCurrent ? 'current-cluster' : ''}">
        <div class="cluster-header">
          <div class="cluster-title">
            <span class="badge ${isCurrent ? 'badge-purple' : 'badge-gray'}">${qid}</span>
            <span class="cluster-count">${items.length} video</span>
          </div>
          <button class="btn-del-cluster" onclick="removeQueryCluster('${qid}')" title="Xóa toàn bộ cụm ${qid}">🗑️ Xóa cụm</button>
        </div>
        <div class="cluster-items-preview">
          ${items.slice(0, 4).map(({ item, origIndex }) => `
            <div class="selection-item compact">
              <div class="selection-rank">${item.rank || (origIndex + 1)}</div>
              <div class="selection-info">${item.video_id} <span style="color:var(--text-muted)">f${item.frames[0]}</span></div>
              <button class="selection-del" onclick="removeSelection(${origIndex})" title="Xoá riêng dòng này">✕</button>
            </div>
          `).join('')}
          ${items.length > 4 ? `<div class="cluster-more">+ ${items.length - 4} video khác trong cụm...</div>` : ''}
        </div>
      </div>
    `;
  });

  list.innerHTML = html;
}

// ---------------------------------------------------------------------------
// Iterative retrieval
// ---------------------------------------------------------------------------

function iterStart() {
  if (!state.candidates.length) {
    toast('Hãy tìm kiếm trước ở tab Tìm kiếm', 'warning');
    switchView('search');
    return;
  }
  state.iterCandidates = [...state.candidates];
  state.iterCursor = 0;
  state.iterRound = 1;
  state.iterRunning = true;
  state.iterMatchedList = [];
  state.iterUnsureList = [];
  state.iterExcluded = new Set();
  state.iterVerdict = {};

  $('btn-iter-start').disabled = true;
  $('btn-iter-finish').disabled = false;
  ['btn-iter-prev', 'btn-iter-next', 'btn-iter-skip'].forEach(id => { $(id).disabled = false; });
  $('iter-status-badge').textContent = 'Đang chạy';

  updateIterRoundBadge();
  buildRoundProgress();
  iterShowCurrent();
}

function iterFinish() {
  if (!state.iterRunning) return;
  state.iterRunning = false;
  $('btn-iter-start').disabled = false;
  $('btn-iter-finish').disabled = true;
  ['btn-iter-prev', 'btn-iter-next', 'btn-iter-skip'].forEach(id => { $(id).disabled = true; });
  $('iter-status-badge').textContent = 'Hoàn thành';

  const queryId = $('query-id-input').value.trim() || 'q1';
  state.iterMatchedList.forEach(c => {
    const frame = c.representative_frames[0] ?? c.start_frame;
    const existing = state.selections.findIndex(s => s.video_id === c.video_id && s.queryId === queryId);
    if (existing < 0) {
      state.selections.push({ video_id: c.video_id, frames: [frame], answer: '', queryId, task: state.task, rank: c.rank });
    }
  });
  renderSelectionsList();
  toast(`Iterative xong: ${state.iterMatchedList.length} matched, ${state.iterUnsureList.length} unsure`, 'success');
}

function iterVerdict(verdict) {
  if (!state.iterRunning || !state.iterCandidates.length) return;
  const c = state.iterCandidates[state.iterCursor];
  const key = candidateKey(c);
  state.iterVerdict[key] = verdict;

  if (verdict === 'matched') {
    if (!state.iterMatchedList.find(m => candidateKey(m) === key)) state.iterMatchedList.push(c);
    state.iterUnsureList = state.iterUnsureList.filter(m => candidateKey(m) !== key);
  } else if (verdict === 'not_matched') {
    state.iterExcluded.add(key);
    state.iterMatchedList = state.iterMatchedList.filter(m => candidateKey(m) !== key);
    state.iterUnsureList = state.iterUnsureList.filter(m => candidateKey(m) !== key);
  } else if (verdict === 'unsure') {
    if (!state.iterUnsureList.find(m => candidateKey(m) === key)) state.iterUnsureList.push(c);
    state.iterMatchedList = state.iterMatchedList.filter(m => candidateKey(m) !== key);
  }

  updateIterStats();
  renderIterLists();
  iterNav(1);
}

function iterNav(dir) {
  if (!state.iterRunning) return;
  const len = state.iterCandidates.length;
  if (!len) return;
  if (dir === 0) { iterNav(1); return; }
  const next = Math.max(0, Math.min(len - 1, state.iterCursor + dir));
  if (next === state.iterCursor && dir > 0) {
    toast('Đã đến cuối danh sách', 'info');
    return;
  }
  state.iterCursor = next;
  iterShowCurrent();
}

function iterShowCurrent() {
  const c = state.iterCandidates[state.iterCursor];
  if (!c) return;

  const frameIdx = c.representative_frames[0] ?? c.start_frame;
  $('iter-video-id').textContent = c.video_id;
  $('iter-frame').textContent = frameIdx;
  $('sc-clip').textContent = fmtScore(c.scores && c.scores.clip);
  $('sc-siglip').textContent = fmtScore(c.scores && c.scores.siglip);
  $('sc-fused').textContent = fmtScore(c.best_score);

  const img = $('iter-preview-img');
  const placeholder = $('iter-preview-placeholder');
  img.style.display = 'none';
  placeholder.style.display = 'flex';
  img.onload = () => { img.style.display = 'block'; placeholder.style.display = 'none'; };
  img.onerror = () => { img.style.display = 'none'; placeholder.style.display = 'flex'; };
  img.src = keyframeUrl(c.video_id, frameIdx);

  const key = candidateKey(c);
  const v = state.iterVerdict[key];
  $('btn-vb-matched').classList.toggle('active', v === 'matched');
  $('btn-vb-not').classList.toggle('active', v === 'not_matched');
  $('btn-vb-unsure').classList.toggle('active', v === 'unsure');

  updateIterRoundBadge();
}

function updateIterRoundBadge() {
  const total = state.iterCandidates.length;
  const cur = total ? state.iterCursor + 1 : 0;
  $('iter-round-badge').textContent = `${cur}/${total}`;
  $('btn-iter-prev').disabled = !state.iterRunning || state.iterCursor <= 0;
  $('btn-iter-next').disabled = !state.iterRunning || state.iterCursor >= (state.iterCandidates.length - 1);
  $('btn-iter-skip').disabled = !state.iterRunning;
}

function updateIterStats() {
  const matched = state.iterMatchedList.length;
  const unsure = state.iterUnsureList.length;
  const excluded = state.iterExcluded.size;
  $('iter-matched-count').textContent = matched;
  $('iter-unsure-count').textContent = unsure;
  $('iter-excluded-count').textContent = excluded;
  $('iter-matched-badge').textContent = matched;
  $('iter-unsure-badge').textContent = unsure;
  $('iter-excluded-badge').textContent = excluded;
}

function renderIterLists() {
  function renderList(containerId, items, colorVar) {
    const el = $(containerId);
    if (!items.length) {
      el.innerHTML = '<div style="padding:16px;color:var(--text-muted);font-size:12px;text-align:center">Chưa có</div>';
      return;
    }
    el.innerHTML = items.map(c => {
      const frameIdx = c.representative_frames[0] ?? c.start_frame;
      return `<div style="display:flex;justify-content:space-between;align-items:center;padding:8px 14px;border-bottom:1px solid var(--border)">
        <span style="font-family:'JetBrains Mono',monospace;font-size:12px;color:${colorVar}">${c.video_id}</span>
        <span style="font-size:11px;color:var(--text-muted)">f${frameIdx} · ${fmtScore(c.best_score)}</span>
      </div>`;
    }).join('');
  }
  renderList('iter-matched-list', state.iterMatchedList, 'var(--green)');
  renderList('iter-unsure-list', state.iterUnsureList, 'var(--amber)');
}

function buildRoundProgress() {
  const container = $('round-progress');
  container.innerHTML = '';
  for (let i = 1; i <= state.iterMaxRounds; i++) {
    if (i > 1) {
      const arrow = document.createElement('span');
      arrow.className = 'round-arrow';
      arrow.textContent = '→';
      container.appendChild(arrow);
    }
    const step = document.createElement('div');
    step.className = 'round-step';
    const pill = document.createElement('div');
    pill.className = 'round-pill' + (i < state.iterRound ? ' done' : i === state.iterRound ? ' active' : '');
    pill.textContent = `Round ${i}`;
    step.appendChild(pill);
    container.appendChild(step);
  }
}

// ---------------------------------------------------------------------------
// Export view
// ---------------------------------------------------------------------------

function renderExportTable() {
  const tbody = $('export-tbody');
  const count = $('export-query-count');
  const clustersBar = $('export-clusters-bar');

  if (!state.selections.length) {
    if (tbody) tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:var(--text-muted);padding:32px">Chưa có kết quả. Hãy tìm kiếm và xác nhận lựa chọn trước.</td></tr>';
    if (count) count.textContent = '0 queries';
    if (clustersBar) clustersBar.style.display = 'none';
    refreshPreview();
    return;
  }

  const byQuery = {};
  state.selections.forEach(s => { if (!byQuery[s.queryId]) byQuery[s.queryId] = []; byQuery[s.queryId].push(s); });
  const queryIds = Object.keys(byQuery).sort((a, b) => a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' }));
  if (count) count.textContent = `${queryIds.length} quer${queryIds.length !== 1 ? 'ies' : 'y'}`;

  // Render thanh Quản lý theo cụm câu hỏi
  if (clustersBar) {
    clustersBar.style.display = 'block';
    clustersBar.innerHTML = `
      <div class="clusters-bar-header">📦 Quản lý xóa theo cụm câu hỏi (${queryIds.length} cụm):</div>
      <div class="clusters-chip-list">
        ${queryIds.map(qid => {
          const list = byQuery[qid];
          const task = list[0]?.task || 'kis';
          return `
            <div class="cluster-chip">
              <span class="chip-task">${task.toUpperCase()}</span>
              <span class="chip-qid">${qid}</span>
              <span class="chip-count">(${list.length} dòng)</span>
              <button class="chip-del-btn" onclick="removeQueryCluster('${qid}')" title="Xóa toàn bộ cụm ${qid}">🗑️ Xóa cụm</button>
            </div>
          `;
        }).join('')}
      </div>
    `;
  }

  if (tbody) {
    tbody.innerHTML = state.selections.map((s, i) => `
      <tr>
        <td style="font-family:'JetBrains Mono',monospace;font-size:12px">
          ${s.queryId}
          <button class="btn-translate" style="margin-left:4px;font-size:9px;padding:1px 5px;color:var(--red)" onclick="removeQueryCluster('${s.queryId}')" title="Xóa toàn bộ cụm câu ${s.queryId}">🗑️</button>
        </td>
        <td><span class="badge badge-purple">${s.task.toUpperCase()}</span></td>
        <td style="font-family:'JetBrains Mono',monospace;font-size:12px;color:var(--cyan)">${s.video_id}</td>
        <td style="font-family:'JetBrains Mono',monospace;font-size:12px">${s.frames.join(', ')}</td>
        <td>
          <span class="badge badge-green">✓ Ready</span>
          <button class="btn-translate" style="margin-left:6px;font-size:10px;padding:2px 8px" onclick="removeExportRow(${i})" title="Xóa riêng dòng này">✕</button>
        </td>
      </tr>`).join('');
  }

  refreshPreview();
}

function removeExportRow(idx) {
  state.selections.splice(idx, 1);
  renderSelectionsList();
  renderExportTable();
}

function clearAllSelections() {
  if (!state.selections.length) return;
  if (confirm('Bạn có chắc chắn muốn xóa toàn bộ các lựa chọn đã lưu?')) {
    state.selections = [];
    renderSelectionsList();
    renderExportTable();
    toast('Đã xóa toàn bộ lựa chọn', 'info');
  }
}

function refreshPreview() {
  const preview = $('csv-preview');
  if (!state.selections.length) { preview.textContent = '— chưa có dữ liệu —'; return; }
  const lines = state.selections.map(s => {
    const vid = s.video_id.replace(/\.mp4$/, '');
    const parts = [vid, ...s.frames.map(String)];
    if (s.answer) parts.push(s.answer);
    return parts.join(',');
  });
  preview.textContent = lines.join('\n');
}

async function doExport() {
  if (!state.selections.length) { toast('Chưa có lựa chọn để export', 'warning'); return; }
  const exportQueryId = $('export-query-id').value.trim() || $('query-id-input').value.trim() || 'q1';
  const rows = state.selections.map(s => ({
    video_id: s.video_id,
    frames: s.frames,
    answer: s.answer || '',
    query_id: s.queryId || exportQueryId,
  }));

  setLoading('btn-export', true);
  try {
    const res = await fetch('/api/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query_id: exportQueryId, task: state.task, rows }),
    });
    if (!res.ok) { const err = await res.json(); throw new Error(err.detail || 'Export failed'); }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'submission.zip';
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
    toast('Đã tải xuống submission.zip đúng chuẩn BTC!', 'success');
  } catch (e) {
    toast(e.message, 'error');
  } finally {
    setLoading('btn-export', false);
  }
}

// ---------------------------------------------------------------------------
// Keyboard shortcuts
// ---------------------------------------------------------------------------

document.addEventListener('keydown', (e) => {
  const tag = document.activeElement.tagName.toLowerCase();
  if (['input', 'textarea'].includes(tag)) return;
  const iterActive = $('view-iterative').classList.contains('active');
  if (iterActive && state.iterRunning) {
    if (e.key === 'm' || e.key === 'M') { iterVerdict('matched'); return; }
    if (e.key === 'n' || e.key === 'N') { iterVerdict('not_matched'); return; }
    if (e.key === 'u' || e.key === 'U') { iterVerdict('unsure'); return; }
    if (e.key === 'ArrowLeft') { e.preventDefault(); iterNav(-1); return; }
    if (e.key === 'ArrowRight') { e.preventDefault(); iterNav(1); return; }
  }
});

// ---------------------------------------------------------------------------
// Toast styles
// ---------------------------------------------------------------------------

(function injectToastStyles() {
  const style = document.createElement('style');
  style.textContent = `
    #toast-container {
      position: fixed; bottom: 24px; right: 24px;
      display: flex; flex-direction: column; gap: 8px;
      z-index: 9999; pointer-events: none;
    }
    .toast {
      padding: 11px 18px; border-radius: 10px; font-size: 13px;
      font-weight: 500; color: #fff; opacity: 0;
      transform: translateY(8px);
      transition: opacity 0.25s, transform 0.25s;
      backdrop-filter: blur(12px); pointer-events: auto; max-width: 320px;
    }
    .toast.show { opacity: 1; transform: translateY(0); }
    .toast-info    { background: rgba(30,35,55,0.95); border: 1px solid rgba(124,58,237,0.3); }
    .toast-success { background: rgba(20,40,30,0.95); border: 1px solid rgba(34,197,94,0.35); }
    .toast-warning { background: rgba(40,30,15,0.95); border: 1px solid rgba(245,158,11,0.35); }
    .toast-error   { background: rgba(40,15,15,0.95); border: 1px solid rgba(239,68,68,0.35); }
  `;
  document.head.appendChild(style);
})();

function initPanelResizer() {
  const resizer = $('panel-resizer');
  const panel = $('detail-panel');
  if (!resizer || !panel) return;

  const savedWidth = localStorage.getItem('detail_panel_width');
  if (savedWidth) {
    panel.style.width = `${savedWidth}px`;
  }

  let isDragging = false;
  let startX = 0;
  let startWidth = 0;

  resizer.addEventListener('mousedown', (e) => {
    isDragging = true;
    startX = e.clientX;
    startWidth = panel.getBoundingClientRect().width;
    resizer.classList.add('dragging');
    document.body.style.cursor = 'col-resize';
    document.body.style.userSelect = 'none';
  });

  document.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    const deltaX = startX - e.clientX; // Kéo sang trái -> tăng width
    const newWidth = Math.max(260, Math.min(window.innerWidth * 0.65, startWidth + deltaX));
    panel.style.width = `${newWidth}px`;
  });

  document.addEventListener('mouseup', () => {
    if (isDragging) {
      isDragging = false;
      resizer.classList.remove('dragging');
      document.body.style.cursor = '';
      document.body.style.userSelect = '';
      const w = parseInt(panel.style.width, 10);
      if (w) localStorage.setItem('detail_panel_width', w);
    }
  });
}

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

document.addEventListener('DOMContentLoaded', () => {
  checkStatus();
  setInterval(checkStatus, 30000);
  initPresetDropdown();
  selectTask('kis');
  initPanelResizer();
  $('query-id-input').addEventListener('input', () => {
    $('export-query-id').value = $('query-id-input').value;
  });

  const frameInput = $('frame-input');
  if (frameInput) {
    frameInput.addEventListener('input', (e) => {
      const val = parseInt(e.target.value, 10);
      if (!isNaN(val) && state.selected !== null && state.candidates[state.selected]) {
        const c = state.candidates[state.selected];
        c.representative_frames = [val];
        c.start_frame = val;
        c.end_frame = val;
        const cardEl = document.querySelector(`.candidate-card[data-idx="${state.selected}"]`);
        if (cardEl) {
          const cardImg = cardEl.querySelector('.card-thumb img');
          if (cardImg) cardImg.src = keyframeUrl(c.video_id, val);
          const cardFrame = cardEl.querySelector('.card-frame');
          if (cardFrame) cardFrame.innerHTML = `frame ${val} &mdash; ${val}&rarr;${val}`;
        }
        const img = $('preview-img');
        if (img) img.src = keyframeUrl(c.video_id, val);
        const customKfInput = $('custom-frame-input');
        if (customKfInput) customKfInput.value = val;
      }
    });
  }

  let customChangeTimeout = null;
  function triggerLiveCustomUpdate() {
    if (customChangeTimeout) clearTimeout(customChangeTimeout);
    customChangeTimeout = setTimeout(() => {
      applyCustomVideoAndFrame(true);
    }, 250);
  }

  const customVidInput = $('custom-video-input');
  if (customVidInput) {
    customVidInput.addEventListener('input', triggerLiveCustomUpdate);
    customVidInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') applyCustomVideoAndFrame(false);
    });
  }
  const customKfInput = $('custom-frame-input');
  if (customKfInput) {
    customKfInput.addEventListener('input', triggerLiveCustomUpdate);
    customKfInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter') applyCustomVideoAndFrame(false);
    });
  }
});
