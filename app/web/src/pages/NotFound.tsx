import { Compass } from "lucide-react";
import { ButtonLink, Card, EmptyState } from "../components/ui";

export default function NotFound() {
  return (
    <Card>
      <h1 tabIndex={-1} className="sr-only">
        Page not found
      </h1>
      <EmptyState icon={<Compass className="size-5" aria-hidden />} title="This page does not exist">
        The address may be mistyped. Every section is listed in the navigation.
      </EmptyState>
      <div className="flex justify-center pb-10">
        <ButtonLink to="/" variant="primary">
          Go to the overview
        </ButtonLink>
      </div>
    </Card>
  );
}
