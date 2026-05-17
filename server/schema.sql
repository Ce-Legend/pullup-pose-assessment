-- 用户表
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  username TEXT NOT NULL UNIQUE,
  display_name TEXT NOT NULL,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('student', 'teacher')),
  is_active INTEGER NOT NULL DEFAULT 1,
  created_at TEXT NOT NULL
);

-- 标准视频管理表
CREATE TABLE IF NOT EXISTS standard_videos (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  name TEXT NOT NULL,                                          -- 视频名称（如"正手正面标准"）
  view_angle TEXT NOT NULL CHECK (view_angle IN ('front', 'side', 'angle')),  -- 视角类型
  action TEXT NOT NULL DEFAULT 'pullup',                       -- 动作类型
  file_path TEXT NOT NULL,                                     -- 文件路径
  version TEXT NOT NULL,                                       -- 版本hash（前12位）
  is_active INTEGER NOT NULL DEFAULT 0,                        -- 是否当前激活（每个视角只能有一个激活）
  uploaded_by INTEGER,                                         -- 上传者用户ID
  created_at TEXT NOT NULL,
  FOREIGN KEY (uploaded_by) REFERENCES users(id)
);

-- 确保每个视角+动作组合只有一个激活的标准视频
CREATE UNIQUE INDEX IF NOT EXISTS idx_standard_active ON standard_videos(view_angle, action) WHERE is_active = 1;

-- 分析记录（上传即创建）
CREATE TABLE IF NOT EXISTS analyses (
  id TEXT PRIMARY KEY,
  user_id INTEGER NOT NULL,
  created_at TEXT NOT NULL,
  finished_at TEXT,
  status TEXT NOT NULL CHECK (status IN ('queued','running','succeeded','failed')),
  progress INTEGER NOT NULL DEFAULT 0,
  action TEXT NOT NULL,
  view TEXT NOT NULL,
  standard_version TEXT NOT NULL,
  standard_video_id INTEGER,                                   -- 关联使用的标准视频
  compare_mode TEXT NOT NULL DEFAULT 'standard',               -- 对比模式: standard(标准视频) 或 history(历史动作)
  compare_analysis_id TEXT,                                    -- 如果是历史对比，关联的历史分析ID
  upload_filename TEXT NOT NULL,
  upload_path TEXT NOT NULL,
  duration_ms INTEGER,
  score_total INTEGER,
  diff_joint TEXT,
  diff_time_ms INTEGER,
  result_json_path TEXT,
  image_standard_path TEXT,
  image_student_path TEXT,
  error_code TEXT,
  error_message TEXT,
  FOREIGN KEY (user_id) REFERENCES users(id),
  FOREIGN KEY (standard_video_id) REFERENCES standard_videos(id),
  FOREIGN KEY (compare_analysis_id) REFERENCES analyses(id)
);

CREATE INDEX IF NOT EXISTS idx_analyses_user_created ON analyses(user_id, created_at DESC);

