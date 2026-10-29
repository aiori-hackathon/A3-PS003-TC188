import { lazy, Suspense } from "react";
import { BrowserRouter, Route, Routes } from "react-router";
import { Shell } from "./components/layout/Shell";
import { Skeleton } from "./components/ui";
import { StudyProvider } from "./lib/data";

const Overview = lazy(() => import("./pages/Overview"));
const Forecast = lazy(() => import("./pages/Forecast"));
const Performance = lazy(() => import("./pages/Performance"));
const Federated = lazy(() => import("./pages/Federated"));
const Methodology = lazy(() => import("./pages/Methodology"));
const About = lazy(() => import("./pages/About"));
const NotFound = lazy(() => import("./pages/NotFound"));

export default function App() {
  return (
    <BrowserRouter>
      <StudyProvider>
        <Shell>
          <Suspense fallback={<Skeleton className="h-[520px]" label="Loading page" />}>
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/forecast" element={<Forecast />} />
              <Route path="/performance" element={<Performance />} />
              <Route path="/federated" element={<Federated />} />
              <Route path="/methodology" element={<Methodology />} />
              <Route path="/about" element={<About />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </Suspense>
        </Shell>
      </StudyProvider>
    </BrowserRouter>
  );
}
