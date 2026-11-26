// Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Factories/Factory.h"
#include "IREEModelDataImportFactory.generated.h"

UCLASS()
class SCHOLAEDITOR_API UIREEModelDataImportFactory : public UFactory
{
	GENERATED_BODY()

public:
	UIREEModelDataImportFactory();

	virtual UObject* FactoryCreateFile(
		UClass* InClass,
		UObject* InParent,
		FName InName,
		EObjectFlags Flags,
		const FString& Filename,
		const TCHAR* Parms,
		FFeedbackContext* Warn,
		bool& bOutOperationCanceled) override;

	virtual bool FactoryCanImport(const FString& Filename) override;
};