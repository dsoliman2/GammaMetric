-- Migration 001: Enable Row Level Security on all public tables
--
-- Context: App connects via SQLAlchemy using the DATABASE_URL (service_role
-- level connection). No Supabase JS client or anon key is used in the frontend.
-- Strategy: Enable RLS on all tables; grant unrestricted access to service_role
-- (which the backend uses); deny anon role entirely.
--
-- Run this in the Supabase SQL editor.

-- ── 1. Enable RLS on all tables ───────────────────────────────────────────

ALTER TABLE public.users                   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.analysis_results        ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.drift_alerts            ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.study_results           ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.reader_study_participants ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.reader_study_responses  ENABLE ROW LEVEL SECURITY;

-- ── 2. Drop any existing policies (idempotent re-run safety) ──────────────

DO $$ DECLARE
  r RECORD;
BEGIN
  FOR r IN SELECT schemaname, tablename, policyname
           FROM pg_policies
           WHERE schemaname = 'public'
             AND tablename IN ('users','analysis_results','drift_alerts',
                               'study_results','reader_study_participants',
                               'reader_study_responses')
  LOOP
    EXECUTE format('DROP POLICY IF EXISTS %I ON %I.%I',
                   r.policyname, r.schemaname, r.tablename);
  END LOOP;
END $$;

-- ── 3. Service role bypass policies (backend SQLAlchemy connection) ────────
-- service_role bypasses RLS by default in Supabase, but explicit policies
-- are added here for clarity and in case that default ever changes.

CREATE POLICY "service_role_all" ON public.users
  FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "service_role_all" ON public.analysis_results
  FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "service_role_all" ON public.drift_alerts
  FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "service_role_all" ON public.study_results
  FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "service_role_all" ON public.reader_study_participants
  FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "service_role_all" ON public.reader_study_responses
  FOR ALL TO service_role USING (true) WITH CHECK (true);

-- ── 4. Explicitly deny anon role (belt-and-suspenders) ────────────────────
-- No policies are created for anon, so anon sees zero rows on all tables.
-- This is the RLS default when no matching policy exists, but explicit
-- REVOKE makes it auditable.

REVOKE ALL ON public.users                    FROM anon;
REVOKE ALL ON public.analysis_results         FROM anon;
REVOKE ALL ON public.drift_alerts             FROM anon;
REVOKE ALL ON public.study_results            FROM anon;
REVOKE ALL ON public.reader_study_participants FROM anon;
REVOKE ALL ON public.reader_study_responses   FROM anon;

-- ── 5. Handle sensitive columns on reader_study_participants ───────────────
-- Supabase flagged this table for sensitive column exposure.
-- Revoke column-level SELECT from anon and authenticated roles
-- (backend accesses via service_role which is unaffected).

REVOKE SELECT ON public.reader_study_participants FROM authenticated;
REVOKE SELECT ON public.reader_study_participants FROM anon;

-- ── 6. Verify ─────────────────────────────────────────────────────────────
-- Run this SELECT after applying to confirm RLS is on for all tables:
--
-- SELECT tablename, rowsecurity
-- FROM pg_tables
-- WHERE schemaname = 'public'
--   AND tablename IN ('users','analysis_results','drift_alerts',
--                     'study_results','reader_study_participants',
--                     'reader_study_responses');
--
-- All rows should show rowsecurity = true.
